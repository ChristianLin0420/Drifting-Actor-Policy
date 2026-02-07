"""
Drifting-VLA Model Components
=============================

This module contains all neural network components for Drifting-VLA:

- Vision Encoder: DINOv2 for extracting visual features
- Language Encoder: CLIP text encoder for language understanding
- Fusion: Cross-attention fusion of modalities
- DiT Transformer: Drifting-style transformer blocks
- Action Decoder: MLP heads for action prediction
- Feature Encoder: Action sequence feature extractor for multi-scale loss
- DriftingVLA: Complete model assembly
"""

from drifting_vla.models.vision_encoder import DINOv2Encoder, VisionEncoderConfig
from drifting_vla.models.language_encoder import CLIPLanguageEncoder, LanguageEncoderConfig
from drifting_vla.models.fusion import CrossAttentionFusion, FusionConfig
from drifting_vla.models.dit import DiTBlock, DiTTransformer, DiTConfig
from drifting_vla.models.action_decoder import ActionDecoder, ActionDecoderConfig
from drifting_vla.models.feature_encoder import ActionFeatureEncoder, FeatureEncoderConfig
from drifting_vla.models.drifting_vla import DriftingVLA, DriftingVLAConfig

__all__ = [
    # Vision
    "DINOv2Encoder",
    "VisionEncoderConfig",
    # Language
    "CLIPLanguageEncoder", 
    "LanguageEncoderConfig",
    # Fusion
    "CrossAttentionFusion",
    "FusionConfig",
    # Transformer
    "DiTBlock",
    "DiTTransformer",
    "DiTConfig",
    # Action
    "ActionDecoder",
    "ActionDecoderConfig",
    # Feature
    "ActionFeatureEncoder",
    "FeatureEncoderConfig",
    # Main model
    "DriftingVLA",
    "DriftingVLAConfig",
]


