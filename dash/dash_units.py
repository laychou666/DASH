"""
DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression

Core algorithm: audio-driven semantic chunking + cross-modal token compression.
Training-free, built on boundary detection inspired by H-Net (arXiv:2507.07955).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundaryPredictionOutput:
    boundary_prob: torch.Tensor   # (N,)  p_boundary = (1 - cos_sim) / 2
    boundary_mask: torch.Tensor   # (N,)  bool
    cos_sim: torch.Tensor         # (N-1,)


# ---------------------------------------------------------------------------
# VidCom2-style auxiliary functions
# ---------------------------------------------------------------------------

def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Keep low-variance channels for stable similarity (VidCom2)."""
    with torch.no_grad():
        variances = x.var(dim=0, unbiased=False)
        k = max(1, min(int(x.shape[-1] * ratio), x.shape[-1]))
        _, topk_idx = torch.topk(variances, k=k, largest=False)
        topk_idx_sorted = topk_idx.sort().values
    return x[:, topk_idx_sorted]


def compute_multi_scale_gaussian(
    x: torch.Tensor,
    center: torch.Tensor,
    alphas: Optional[List[float]] = None,
) -> torch.Tensor:
    """Multi-scale Gaussian kernel similarity: sum_a exp(-||x-c||^2 / 2a)."""
    if alphas is None:
        alphas = [0.125, 0.25, 0.5, 1.0, 2.0]
    with torch.no_grad():
        dist_sq = ((x - center) ** 2).sum(dim=-1)
        scores = torch.stack(
            [torch.exp(-dist_sq / (2 * a)) for a in alphas], dim=0
        )
    return scores.sum(dim=0)


def compute_token_uniqueness_gaussian(
    features: torch.Tensor,
    alphas: Optional[List[float]] = None,
    use_channel_selection: bool = True,
    channel_ratio: float = 0.5,
) -> torch.Tensor:
    """Token uniqueness = 1 - normalized Gaussian similarity to global mean."""
    if alphas is None:
        alphas = [0.125, 0.25, 0.5, 1.0, 2.0]
    with torch.no_grad():
        feat = (select_low_var_channels(features, channel_ratio)
                if use_channel_selection and features.shape[-1] > 10
                else features)
        feat_norm = F.normalize(feat, dim=-1)
        center = feat_norm.mean(dim=0, keepdim=True)
        sim = compute_multi_scale_gaussian(feat_norm, center, alphas)
        max_sim = sim.max()
        uniqueness = 1.0 - (sim / max_sim if max_sim > 1e-6 else sim)
    return uniqueness


# ---------------------------------------------------------------------------
# Audio boundary detection
# ---------------------------------------------------------------------------

def compute_audio_boundaries(
    audio_features: torch.Tensor,
    threshold: float = 0.4,
    min_chunk_size: int = 30,
) -> Tuple[List[int], BoundaryPredictionOutput]:
    """Detect semantic boundaries in audio via adjacent-token cosine similarity."""
    if audio_features.dim() == 2:
        audio_features = audio_features.unsqueeze(0)

    B, N, D = audio_features.shape
    device = audio_features.device

    if N <= 1:
        return [N], BoundaryPredictionOutput(
            torch.ones(N, device=device),
            torch.ones(N, dtype=torch.bool, device=device),
            torch.zeros(0, device=device),
        )

    a_norm = F.normalize(audio_features, p=2, dim=-1)
    cos_sim = (a_norm[:, :-1] * a_norm[:, 1:]).sum(dim=-1)
    boundary_prob = F.pad(
        torch.clamp((1.0 - cos_sim) / 2.0, 0.0, 1.0), (1, 0), value=1.0
    )

    boundaries = [0]
    last_cut = 0
    for t in range(N - 1):
        if cos_sim[0, t] < threshold and (t + 1 - last_cut) >= min_chunk_size:
            boundaries.append(t + 1)
            last_cut = t + 1
    if boundaries[-1] != N:
        boundaries.append(N)

    boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)
    for idx in boundaries[:-1]:
        if idx < N:
            boundary_mask[idx] = True

    return boundaries, BoundaryPredictionOutput(
        boundary_prob.squeeze(0), boundary_mask, cos_sim.squeeze(0)
    )


def map_boundaries_to_video(
    audio_boundaries: List[int],
    n_audio_tokens: int,
    n_video_tokens: int,
) -> List[int]:
    """Linear-scale audio boundary indices to the video token domain."""
    if n_audio_tokens == 0 or not audio_boundaries:
        return [0, n_video_tokens]

    scale = n_video_tokens / n_audio_tokens
    vb = sorted(set(min(int(idx * scale), n_video_tokens) for idx in audio_boundaries))
    if vb[0] != 0:
        vb = [0] + vb
    if vb[-1] != n_video_tokens:
        vb.append(n_video_tokens)
    return vb


# ---------------------------------------------------------------------------
# Training-free boundary prediction (H-Net style)
# ---------------------------------------------------------------------------

def compute_boundary_prediction(
    hidden_states: torch.Tensor,
    threshold: float = 0.5,
) -> BoundaryPredictionOutput:
    """boundary_prob(t) = (1 - cos_sim(h_{t-1}, h_t)) / 2, threshold -> mask."""
    squeeze_batch = False
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
        squeeze_batch = True

    B, N, D = hidden_states.shape
    device = hidden_states.device

    if N <= 1:
        bp = torch.ones(B, N, device=device)
        bm = torch.ones(B, N, dtype=torch.bool, device=device)
        cs = torch.zeros(B, 0, device=device)
        if squeeze_batch:
            bp, bm, cs = bp.squeeze(0), bm.squeeze(0), cs.squeeze(0)
        return BoundaryPredictionOutput(bp, bm, cs)

    h = F.normalize(hidden_states, p=2, dim=-1)
    cos_sim = torch.einsum("b l d, b l d -> b l", h[:, :-1], h[:, 1:])
    boundary_prob = F.pad(
        torch.clamp((1.0 - cos_sim) / 2.0, 0.0, 1.0), (1, 0), value=1.0
    )
    boundary_mask = boundary_prob > threshold

    if squeeze_batch:
        boundary_prob = boundary_prob.squeeze(0)
        boundary_mask = boundary_mask.squeeze(0)
        cos_sim = cos_sim.squeeze(0)

    return BoundaryPredictionOutput(boundary_prob, boundary_mask, cos_sim)


# ---------------------------------------------------------------------------
# Audio compression: three-signal fusion (boundary + uniqueness + attention)
# ---------------------------------------------------------------------------

def dash_audio_attn(
    audio_feature: torch.Tensor,
    video_feature: Optional[torch.Tensor],
    attn_logits: torch.Tensor,
    merging_ratio: float = 0.5,
    contextual_ratio: float = 0.03,
    g: int = 3,
    use_boundary_guidance: bool = True,
) -> Tuple[torch.Tensor, Dict[int, List[int]], BoundaryPredictionOutput]:
    """Select and merge audio tokens. Returns (keep_mask, merge_plan, boundary)."""
    device = attn_logits.device

    def to_seq(x):
        if x is None:
            return None
        return x.mean(0) if x.dim() == 3 else x

    a = to_seq(audio_feature).to(device)
    v = to_seq(video_feature).to(device) if video_feature is not None else None
    N = a.size(0)

    if N == 0:
        empty = BoundaryPredictionOutput(
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.bool, device=device),
            torch.zeros(0, device=device),
        )
        return torch.zeros(0, dtype=torch.bool, device=device), {}, empty

    boundary_output = compute_boundary_prediction(a)
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    dominant_num = int(max(0, min(N, round((1.0 - merging_ratio) * N))))

    if dominant_num > 0:
        if use_boundary_guidance and boundary_output.boundary_mask.any():
            # Three-signal fusion: boundary(0.4) + uniqueness(0.3) + attention(0.3)
            bscore = boundary_output.boundary_prob / (boundary_output.boundary_prob.max() + 1e-6)
            uscore = compute_token_uniqueness_gaussian(a)
            ascore = attn_logits / (attn_logits.max() + 1e-6)
            combined = 0.4 * bscore + 0.3 * uscore + 0.3 * ascore
            _, topk_indices = torch.topk(combined, dominant_num)
            keep_mask[topk_indices] = True
        else:
            _, topk = torch.topk(attn_logits, dominant_num)
            keep_mask[topk] = True

    # Contextual anchor selection + merge plan
    all_idx = torch.arange(N, device=device)
    remaining = all_idx[~keep_mask]
    contextual_num = int(max(0, round(contextual_ratio * N)))
    merge_plan: Dict[int, List[int]] = {}

    if remaining.numel() > 0 and contextual_num > 0:
        contextual_num = min(contextual_num, remaining.numel())
        step = max(1, remaining.numel() // contextual_num)
        init_pos = torch.arange(0, remaining.numel(), step, device=device)[:contextual_num]
        anchors = remaining[init_pos]
        keep_mask[anchors] = True

        rem_local_mask = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
        rem_local_mask[init_pos] = False
        pool_global = remaining[torch.arange(remaining.numel(), device=device)[rem_local_mask]]

        if pool_global.numel() > 0 and g > 0:
            a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
            sim_aa = a_norm[pool_global] @ a_norm[anchors].T
            assign = sim_aa.argmax(dim=1)

            if v is not None and v.numel() > 0:
                v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
                scores = (a_norm[pool_global] @ v_norm.T).max(dim=1).values
            else:
                scores = sim_aa.max(dim=1).values

            for c in range(anchors.numel()):
                mask_c = (assign == c)
                cand = pool_global[mask_c]
                if cand.numel() == 0:
                    merge_plan[int(anchors[c].item())] = []
                    continue
                topg = min(g, cand.numel())
                _, sel = torch.topk(scores[mask_c], topg, largest=True)
                merge_plan[int(anchors[c].item())] = cand[sel].tolist()

    return keep_mask, merge_plan, boundary_output


# ---------------------------------------------------------------------------
# Interleaved spatio-temporal merging (ISTM) with boundary awareness
# ---------------------------------------------------------------------------

def dash_istm(
    video_feature: torch.Tensor,
    num_tokens_per_frame: int = 196,
    merging_ratio: float = 0.7,
    boundary_output: Optional[BoundaryPredictionOutput] = None,
) -> torch.Tensor:
    """Even frames: spatial pruning (DPCKNN). Odd frames: temporal pruning."""
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)

    def dpcknn(tokens, keep_rate=0.5, k=5):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        with torch.no_grad():
            normed = F.normalize(tokens, dim=1)
            sim = torch.mm(normed, normed.T)
            sim.fill_diagonal_(-float('inf'))
            knn_vals, _ = torch.topk(sim, k, dim=1)
            knn_dist = knn_vals.mean(dim=1)
            selected = torch.topk(-knn_dist, num_keep, largest=True).indices
        return selected

    keep_ratio = 1 - merging_ratio

    for t in range(num_frames):
        start = t * num_tokens_per_frame
        end = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start:end]

        # Boundary frames get less compression
        fkr = keep_ratio
        if boundary_output is not None and t < len(boundary_output.boundary_mask):
            if boundary_output.boundary_mask[t]:
                fkr = keep_ratio + (1 - keep_ratio) * 0.3 * boundary_output.boundary_prob[t].item()

        num_keep = int(num_tokens_per_frame * fkr)
        if t % 2 == 0:
            keep_idx = (dpcknn(tokens, keep_rate=fkr, k=5)
                        if num_keep < num_tokens_per_frame
                        else torch.arange(num_tokens_per_frame, device=tokens.device))
        else:
            prev = video_feature[(t - 1) * num_tokens_per_frame : start]
            similarity = F.cosine_similarity(
                F.normalize(tokens, p=2, dim=1),
                F.normalize(prev, p=2, dim=1), dim=1,
            )
            keep_idx = (similarity.topk(num_keep, largest=False).indices
                        if num_keep < num_tokens_per_frame
                        else torch.arange(num_tokens_per_frame, device=tokens.device))

        mask[start:end][keep_idx] = True

    return mask


# ---------------------------------------------------------------------------
# DASH main entry point
# ---------------------------------------------------------------------------

def dash(
    input_embeds: torch.Tensor,
    attn_logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    merging_ratio_audio: float = 0.5,
    merging_ratio_v: float = 0.5,
    contextual_ratio: float = 0.05,
    g: int = 3,
    use_adaptive_grouping: bool = True,
    audio_guidance_strength: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Run the full DASH pipeline: audio boundary detection -> cross-modal
    segmentation -> audio/video token compression."""
    device = input_embeds.device

    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids

    video_indices = torch.nonzero(flat_ids == video_token_id, as_tuple=True)[0].to(device)
    audio_indices = torch.nonzero(flat_ids == audio_token_id, as_tuple=True)[0].to(device)
    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]
    video_token_per_frame = (
        video_feature.shape[0] // num_input_frames if num_input_frames > 0 else 196
    )

    debug_info = {}
    audio_boundaries = None
    video_boundaries = None

    # --- Step 1: Audio-triggered visual segmentation ---
    if use_adaptive_grouping and audio_feature.shape[0] > 0:
        audio_boundaries, audio_boundary_output = compute_audio_boundaries(
            audio_feature, threshold=0.4, min_chunk_size=30,
        )
        debug_info['audio_boundaries'] = audio_boundaries
        debug_info['audio_boundary_output'] = audio_boundary_output

        if video_feature.shape[0] > 0:
            video_boundaries = map_boundaries_to_video(
                audio_boundaries, audio_feature.shape[0], video_feature.shape[0],
            )
            debug_info['video_boundaries'] = video_boundaries

    # --- Step 2: Audio compression ---
    audio_mask, merge_plan, audio_boundary = dash_audio_attn(
        audio_feature=audio_feature,
        video_feature=video_feature,
        attn_logits=attn_logits,
        merging_ratio=merging_ratio_audio,
        contextual_ratio=contextual_ratio,
        g=g,
        use_boundary_guidance=use_adaptive_grouping,
    )
    debug_info['audio_boundary'] = audio_boundary

    # Token merging
    if g > 0 and len(merge_plan) > 0:
        a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
        v_norm = (video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6)
                  if video_feature.numel() > 0 else None)

        for anchor_idx, merge_list in merge_plan.items():
            if not merge_list:
                continue
            merge_rel = torch.tensor(merge_list, device=device, dtype=torch.long)
            if v_norm is not None:
                scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
            else:
                scores = (a_norm[merge_rel] @ a_norm[anchor_idx].unsqueeze(-1)).squeeze(-1)
            w = torch.softmax(scores, dim=0)
            new_anchor = (audio_feature[anchor_idx]
                          + (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(0)) / (1.0 + w.sum())
            flat_embeds[audio_indices[anchor_idx]] = new_anchor

    # --- Step 3: Audio-driven video compression ---
    # Per-segment ISTM; compression rate guided by audio retention in each segment.
    if use_adaptive_grouping and video_boundaries is not None and len(video_boundaries) > 1:
        num_video_tokens = video_feature.shape[0]
        video_group_masks = []
        start_idx = 0

        for seg_idx, end_idx in enumerate(video_boundaries[1:]):
            if start_idx >= num_video_tokens:
                break
            end_idx = min(end_idx, num_video_tokens)
            segment_video = video_feature[start_idx:end_idx]
            segment_len = segment_video.shape[0]

            if segment_len == 0:
                start_idx = end_idx
                continue

            # Audio retention -> adaptive compression rate
            audio_retention = 0.5
            if seg_idx < len(audio_boundaries) - 1:
                a_start = audio_boundaries[seg_idx]
                a_end = (audio_boundaries[seg_idx + 1]
                         if seg_idx + 1 < len(audio_boundaries)
                         else audio_feature.shape[0])
                if a_end > a_start and a_end <= audio_mask.shape[0]:
                    audio_retention = audio_mask[a_start:a_end].float().mean().item()

            adaptive_ratio = max(0.1, min(0.95,
                merging_ratio_v + 0.1 * (0.5 - audio_retention)))

            if segment_len >= video_token_per_frame * 2:
                seg_mask = dash_istm(segment_video, video_token_per_frame, adaptive_ratio)
            else:
                seg_mask = torch.ones(segment_len, dtype=torch.bool, device=device)

            video_group_masks.append(seg_mask)
            start_idx = end_idx

        if video_group_masks:
            video_mask = torch.cat(video_group_masks, dim=0)
            diff = video_feature.shape[0] - video_mask.shape[0]
            if diff > 0:
                video_mask = torch.cat([video_mask,
                    torch.ones(diff, dtype=torch.bool, device=device)])
            elif diff < 0:
                video_mask = video_mask[:video_feature.shape[0]]
        else:
            video_mask = torch.ones(video_feature.shape[0], dtype=torch.bool, device=device)
    else:
        # Fallback without audio: more conservative compression
        fallback_ratio = max(0.35, merging_ratio_v - 0.15)
        video_mask = dash_istm(video_feature, video_token_per_frame, fallback_ratio)

    # --- Step 4: Global mask ---
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    if video_mask.shape[0] == video_indices.shape[0]:
        global_mask[video_indices] = video_mask
    if audio_mask.shape[0] == audio_indices.shape[0]:
        global_mask[audio_indices] = audio_mask

    if is_batched:
        input_embeds_out = flat_embeds.reshape(B, L, D)
    else:
        input_embeds_out = flat_embeds

    return input_embeds_out, global_mask, debug_info
