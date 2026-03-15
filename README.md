
# DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression

[Bingzhou Li](https://laychou666.github.io/), [Tao Huang](https://taohuang.info/), "DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression"

#### News

- **2026-03-15:** This repo is released.

![overview](figures/overview.png)

> **Abstract:** Omnimodal large language models (OmniLLMs) jointly process audio and visual streams, but the resulting long multimodal token sequences make inference prohibitively expensive. Existing compression methods typically rely on fixed window partitioning and attention-based pruning, which overlook the piecewise semantic structure of audio-visual signals and become fragile under aggressive token reduction. We propose Dynamic Audio-driven Semantic cHunking (DASH), a training-free framework that aligns token compression with semantic structure. DASH treats audio embeddings as a semantic anchor and detects boundary candidates via cosine-similarity discontinuities, inducing dynamic, variable-length segments that approximate the underlying piecewise-coherent organization of the sequence. These boundaries are projected onto video tokens to establish explicit cross-modal segmentation. Within each segment, token retention is determined by a tri-signal importance estimator that fuses structural boundary cues, representational distinctiveness, and attention-based salience, mitigating the sparsity bias of attention-only selection. This structure-aware allocation preserves transition-critical tokens while reducing redundant regions. Extensive experiments on AVUT, VideoMME, and WorldSense demonstrate that DASH maintains superior accuracy while achieving higher compression ratios compared to prior methods.

## ⚒️ TODO

* [x] Release core code 
* [ ] Release paper
* [ ] Release all evaluation scripts

## Install

#### 1. Clone this repository:
```bash
git clone https://github.com/laychou666/DASH.git
cd DASH
```

#### 2. Install dependencies:
```bash
conda create -n dash python=3.10 -y
conda activate dash
pip install --upgrade pip
bash setup.sh

cd lmms-eval
pip install -e .

pip install flash-attn --no-build-isolation
```

## Quick Start

Run a quick demo of DASH with Qwen2.5-Omni:
```bash
python demo.py --dash
```

Set compression parameters:
```bash
python demo.py --dash --rho_audio 0.45 --rho_video 0.76
```

## Evaluation

#### VideoMME (via lmms-eval)
We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit. Specify DASH parameters in **eval.sh**:
```bash
bash eval.sh
```

## Method Overview

DASH operates in four stages:

1. **Audio Boundary Detection**: Detects semantic breakpoints in audio tokens via training-free cosine similarity analysis.
2. **Boundary-to-Video Mapping**: Projects audio boundaries onto video tokens through linear temporal scaling.
3. **Audio Compression**: Selects informative audio tokens using three-signal fusion (boundary probability + Gaussian uniqueness + attention scores).
4. **Video Compression**: Applies per-segment interleaved spatio-temporal merging (ISTM) with audio-guided compression rates.

## Acknowledgement

This project builds upon [OmniZip](https://github.com/KD-TAO/OmniZip), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni). Thanks for their awesome work.


## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{dash2026,
  title={DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression},
  author={},
  journal={},
  year={2026}
}
```
