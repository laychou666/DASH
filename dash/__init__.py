"""
DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression

Modules:
- dash_units: Core DASH algorithm (audio-driven chunking + cross-modal compression)
- modeling_qwen2_5_omni: Qwen2.5-Omni model integration with DASH
"""

from dash.dash_units import (
    dash,
    dash_audio_attn,
    dash_istm,
    compute_boundary_prediction,
    compute_audio_boundaries,
    map_boundaries_to_video,
    BoundaryPredictionOutput,
)

__all__ = [
    "dash",
    "dash_audio_attn",
    "dash_istm",
    "compute_boundary_prediction",
    "compute_audio_boundaries",
    "map_boundaries_to_video",
    "BoundaryPredictionOutput",
]

__version__ = "0.1.0"
