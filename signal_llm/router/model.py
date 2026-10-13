"""
Student Model (INT4 + GLU Head)

This is the core production artifact. It uses `torchao` to quantize the transformer backbone in-place.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_torchao_compat() -> bool:
    """Apply the torchao 0.4 compatibility shim lazily.

    transformers imports ``torchao.quantization.Int4WeightOnlyConfig``, which
    torchao 0.4.0 does not provide. The shim is applied when a model is
    instantiated rather than at import time, so importing this module has no
    global side effects.

    Returns True when torchao is importable (shim applied if needed).
    """
    try:
        import torchao.quantization as tq
    except ImportError:
        return False
    if not hasattr(tq, "Int4WeightOnlyConfig"):

        class Int4WeightOnlyConfig:
            def __init__(self, *args, **kwargs):
                pass

        tq.Int4WeightOnlyConfig = Int4WeightOnlyConfig
    return True


# Heavy optional dependency: imported lazily at first model construction so
# this module (and everything importing it) stays importable without
# sentence-transformers installed.
SentenceTransformer = None


def _get_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST

        SentenceTransformer = _ST
    return SentenceTransformer


try:
    # Try torchao API - use int4_weight_only helper function
    from torchao.quantization import int4_weight_only, quantize_

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

from .features import LinguisticFeatureExtractor  # noqa: E402


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function: x * sigmoid(x)

    Citation: Ramachandran et al. (2017) "Searching for Activation Functions" (arXiv:1710.05941)
    - Outperforms ReLU in deep networks by providing smoother gradients
    """
    return x * torch.sigmoid(x)


class SwiGLUTransform(nn.Module):
    """
    Core SwiGLU transformation: Linear -> SwiGLU (no LayerNorm)

    Shared component used by both Post-LN and Pre-LN residual blocks.
    This extracts the common SwiGLU logic to enable code reuse.

    Scientific Citations (same as SwiGLUBlock):
    1. SwiGLU: Chowdhery et al. (2022) PaLM paper (arXiv:2204.02311)
    2. GLU: Dauphin et al. (2017) GLU paper (arXiv:1612.08083)
    3. Swish: Ramachandran et al. (2017) Activation search (arXiv:1710.05941)
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear projection: splits into value and gate (2x output_dim)
        self.projection = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear -> SwiGLU (no LayerNorm)

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        # Project to 2x output_dim and split into value and gate
        projected = self.projection(x)
        val, gate = projected.chunk(2, dim=-1)

        # SwiGLU: val * Swish(gate) where Swish(x) = x * sigmoid(x)
        return val * swish(gate)  # type: ignore[no-any-return]


class SwiGLUBlock(nn.Module):
    """
    Modular SwiGLU block: Linear -> SwiGLU -> LayerNorm

    PHASE 1: Base component for SwiGLU + LayerNorm

    Scientific Citations (10 sources - see GatedFusionHead docstring for full list):
    1. SwiGLU: Chowdhery et al. (2022) PaLM paper (arXiv:2204.02311)
    2. GLU: Dauphin et al. (2017) GLU paper (arXiv:1612.08083)
    3. Swish: Ramachandran et al. (2017) Activation search (arXiv:1710.05941)
    4. LayerNorm: Ba et al. (2016) LayerNorm paper (arXiv:1607.06450)
    5-10. See GatedFusionHead for additional citations

    This is a reusable, modular component that can be stacked to create deeper networks.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use shared SwiGLU transformation
        self.swiglu = SwiGLUTransform(input_dim, output_dim)
        # LayerNorm for stability and variance matching
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear -> SwiGLU -> LayerNorm

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        # SwiGLU transformation (shared code)
        gated = self.swiglu(x)

        # LayerNorm for stability
        return self.layer_norm(gated)  # type: ignore[no-any-return]


class ResidualSwiGLUBlock(nn.Module):
    """
    Post-LN Residual SwiGLU block: SwiGLU -> LayerNorm -> + residual

    PHASE 2 IMPROVEMENT: Residual Connections (Post-LN style)

    Architecture: x -> SwiGLU -> LayerNorm -> + residual -> output

    Scientific Citations (3+ sources for residual connections):
    1. ResNet: He et al. (2016) "Deep Residual Learning for Image Recognition" (arXiv:1512.03385)
       - Residual connections enable training of deeper networks without degradation
       - Formula: F(x) + x where F(x) is learned residual function
    2. Identity Mappings: He et al. (2016) "Identity Mappings in Deep Residual Networks" (arXiv:1603.05027)
       - Identity skip connections provide optimal gradient flow
       - Direct identity mapping (when dimensions match) is optimal
    3. Transformer Residuals: Vaswani et al. (2017) "Attention Is All You Need" (arXiv:1706.03762)
       - Residual connections around attention and feedforward layers are essential
       - Enables gradient flow through deep transformer stacks

    Additional Supporting Citations:
    - Gradient Flow: Veit et al. (2016) "Residual Networks Behave Like Ensembles" (arXiv:1605.06431)
    - Training Stability: Xiong et al. (2020) "On Layer Normalization" (arXiv:2002.04745)

    Implementation:
    - If input_dim == output_dim: Direct residual connection (identity mapping) - optimal gradient flow
    - If input_dim != output_dim: Project input to output_dim for residual - matches dimensions
    - Formula: output = LayerNorm(SwiGLU(x)) + residual(x)
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use shared SwiGLU transformation
        self.swiglu = SwiGLUTransform(input_dim, output_dim)
        # LayerNorm AFTER SwiGLU (Post-LN style)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Residual projection: only needed if dimensions don't match
        self.residual_proj: nn.Linear | None = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SwiGLU -> LayerNorm -> + residual (Post-LN style)

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        # SwiGLU transformation (shared code)
        gated = self.swiglu(x)

        # LayerNorm AFTER SwiGLU (Post-LN style)
        out = self.layer_norm(gated)  # type: ignore[no-any-return]

        # Residual connection
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        return out + residual  # type: ignore[no-any-return]


class PreLNResidualSwiGLUBlock(nn.Module):
    """
    Pre-LN Residual SwiGLU block: LayerNorm -> SwiGLU -> + residual

    PHASE 2 IMPROVEMENT: Residual Connections (Pre-LN style)

    Architecture: x -> LayerNorm -> SwiGLU -> + residual -> output

    Scientific Citations:
    - Pre-LN vs Post-LN: Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture" (arXiv:2002.04745)
      - Pre-LN provides better gradient flow than Post-LN
      - LayerNorm before transformation improves training stability
    - Same residual connection citations as PostLNResidualSwiGLUBlock

    Implementation:
    - If input_dim == output_dim: Direct residual connection (identity mapping) - optimal gradient flow
    - If input_dim != output_dim: Project input to output_dim for residual - matches dimensions
    - Formula: output = SwiGLU(LayerNorm(x)) + residual(x)
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LayerNorm BEFORE SwiGLU (Pre-LN style)
        self.layer_norm = nn.LayerNorm(input_dim)
        # Use shared SwiGLU transformation
        self.swiglu = SwiGLUTransform(input_dim, output_dim)

        # Residual projection: only needed if dimensions don't match
        self.residual_proj: nn.Linear | None = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: LayerNorm -> SwiGLU -> + residual (Pre-LN style)

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        # LayerNorm BEFORE SwiGLU (Pre-LN style)
        x_norm = self.layer_norm(x)

        # SwiGLU transformation (shared code)
        out = self.swiglu(x_norm)

        # Residual connection (use original x, not normalized)
        residual = self.residual_proj(x) if self.residual_proj is not None else x

        return out + residual  # type: ignore[no-any-return]


class GatedFusionHead(nn.Module):
    """
    PHASE 1 + PHASE 2: Modular, configurable head with SwiGLU, LayerNorm, Residuals, and Depth

    PHASE 1 (Base): SwiGLU + LayerNorm
    PHASE 2 (Advanced): Residual connections, multiple layers, dropout, configurable width

    Scientific Citations - PHASE 1 (10 sources):
    1. SwiGLU: Chowdhery et al. (2022) "PaLM: Scaling Language Modeling with Pathways" (arXiv:2204.02311)
    2. GLU: Dauphin et al. (2017) "Language Modeling with Gated Convolutional Networks" (arXiv:1612.08083)
    3. Swish: Ramachandran et al. (2017) "Searching for Activation Functions" (arXiv:1710.05941)
    4. LayerNorm: Ba et al. (2016) "Layer Normalization" (arXiv:1607.06450)
    5. LLaMA: Touvron et al. (2023) "LLaMA: Open and Efficient Foundation Language Models" (arXiv:2302.13971)
    6. Transformer: Vaswani et al. (2017) "Attention Is All You Need" (arXiv:1706.03762)
    7. Normalization: Ioffe & Szegedy (2015) "Batch Normalization" (arXiv:1502.03167)
    8. SwiGLU Details: Shazeer (2020) "GLU Variants Improve Transformer" (referenced in PaLM)
    9. Variance Matching: Santurkar et al. (2018) "How Does Batch Normalization Help Optimization?" (arXiv:1805.11604)
    10. RMSNorm: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization" (arXiv:1910.07467)

    Scientific Citations - PHASE 2:

    IMPROVEMENT 1: Residual Connections (3+ sources)
    1. ResNet: He et al. (2016) "Deep Residual Learning" (arXiv:1512.03385)
       - Residual connections enable training deeper networks without degradation
    2. Identity Mappings: He et al. (2016) "Identity Mappings" (arXiv:1603.05027)
       - Identity skip connections provide optimal gradient flow
    3. Transformer Residuals: Vaswani et al. (2017) "Attention Is All You Need" (arXiv:1706.03762)
       - Residual connections essential for transformer training

    IMPROVEMENT 2: Deeper Networks / Multiple Layers (3+ sources)
    1. Going Deeper: Szegedy et al. (2015) "Going Deeper with Convolutions" (arXiv:1409.4842)
       - Deeper networks (22 layers) achieve better accuracy than shallower networks
    2. Universal Transformers: Dehghani et al. (2019) "Universal Transformers" (arXiv:1807.03819)
       - Deeper transformer layers improve performance, residuals essential for depth
    3. Highway Networks: Srivastava et al. (2015) "Highway Networks" (arXiv:1505.00387)
       - Gated residual connections enable training networks with 100+ layers

    IMPROVEMENT 3: Dropout Regularization (3+ sources)
    1. Dropout Original: Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Overfitting"
       - Dropout prevents overfitting by randomly setting neurons to zero during training
    2. Dropout in Deep Learning: Hinton et al. (2012) "Improving neural networks" (arXiv:1207.0580)
       - Dropout improves generalization by preventing co-adaptation of neurons
    3. Transformer Dropout: Vaswani et al. (2017) "Attention Is All You Need" (arXiv:1706.03762)
       - Dropout applied to feedforward layers improves transformer generalization

    IMPROVEMENT 4: Configurable Hidden Dimension (3+ sources)
    1. EfficientNet: Tan & Le (2019) "EfficientNet: Rethinking Model Scaling" (arXiv:1905.11946)
       - Compound scaling (width x depth) outperforms scaling single dimension
    2. Scaling Laws: Kaplan et al. (2020) "Scaling Laws for Neural Language Models" (arXiv:2001.08361)
       - Model performance scales with width (hidden dimension) following power laws
    3. Width Expressiveness: Lu et al. (2017) "The Expressive Power of Neural Networks" (arXiv:1709.02540)
       - Width increases model expressiveness and capacity

    Architecture:
    - Modular design using SwiGLUBlock and ResidualSwiGLUBlock components
    - Supports 1-3 layers with optional residual connections
    - Configurable hidden dimension and dropout
    - Backward compatible with Phase 1 (num_layers=1, use_residual=False)

    We use SwiGLU to allow linguistic features to 'gate' the semantic embedding.
    If the text is semantically simple but linguistically complex (archaic words),
    the gate allows the linguistic signal to override.
    """

    def __init__(
        self,
        embedding_dim: int,
        linguistic_dim: int,
        model_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 1,
        use_residual: bool = False,
        residual_style: str = "post_ln",
        enable_dropout: bool | None = None,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            embedding_dim: Dimension of semantic embeddings
            linguistic_dim: Dimension of linguistic features (typically 3)
            model_dim: Dimension of model features (typically 2: model_a_elo, model_b_elo)
                       ⚠️  CRITICAL: DO NOT SET TO 0 OR REMOVE THIS PARAMETER! ⚠️
                       Without model features, the model cannot learn which models are competing.
                       This caused hours of debugging when it was missing (stuck at 50% accuracy).
            hidden_dim: Hidden dimension for the head (default: 256)
            num_layers: Number of SwiGLU layers (1-3, default: 1 for Phase 1 compatibility)
            use_residual: Whether to use residual connections (default: False for Phase 1)
            residual_style: Residual style - "post_ln" (LayerNorm after SwiGLU) or "pre_ln" (LayerNorm before SwiGLU)
                           Default: "post_ln" for backward compatibility
                           Pre-LN may provide better gradient flow (Xiong et al., 2020)
                           Only used when use_residual: true
            enable_dropout: Whether to enable dropout (default: None = auto-detect from dropout_rate > 0.0)
                           Explicitly set to False to disable dropout even if dropout_rate > 0.0
                           Useful for hyperparameter tuning to clearly separate enable/disable from rate
            dropout_rate: Dropout rate for regularization (0.0-0.2, default: 0.0)
                         Only used when enable_dropout: true (or when enable_dropout is None and dropout_rate > 0.0)
        """
        super().__init__()
        # ============================================================================
        # ⚠️  CRITICAL: model_dim MUST BE INCLUDED IN input_dim! ⚠️
        # ============================================================================
        # Input dimensions:
        #   - embedding_dim: Semantic meaning of the prompt (768 for embeddinggemma-300m)
        #   - linguistic_dim: Linguistic complexity (grade level, length, polysyllables)
        #   - model_dim: WHICH MODELS ARE BEING COMPARED (model_a_elo, model_b_elo)
        #
        # Without model_dim, the model is asked "Who won?" without knowing "Who played?"
        # This results in 50% accuracy (random guessing) because the model has no way
        # to learn that GPT-4 beats GPT-3.5, or any other model-specific patterns.
        # ============================================================================
        self.input_dim = embedding_dim + linguistic_dim + model_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.residual_style = residual_style

        # Validate dimensions and layer counts (public boundary contract).
        import math as _math

        for dim_name, dim_value in (
            ("embedding_dim", embedding_dim),
            ("linguistic_dim", linguistic_dim),
            ("model_dim", model_dim),
            ("hidden_dim", hidden_dim),
        ):
            if not isinstance(dim_value, int) or dim_value < 1:
                raise ValueError(f"{dim_name} must be a positive int, got {dim_value!r}")
        if not isinstance(num_layers, int) or not 1 <= num_layers <= 3:
            raise ValueError(f"num_layers must be within [1, 3], got {num_layers!r}")
        if not isinstance(dropout_rate, (int, float)) or not _math.isfinite(dropout_rate):
            raise ValueError(f"dropout_rate must be a finite number, got {dropout_rate!r}")

        # Validate residual_style
        if residual_style not in ["post_ln", "pre_ln"]:
            raise ValueError(
                f"residual_style must be 'post_ln' or 'pre_ln', got '{residual_style}'"
            )

        # Auto-detect enable_dropout from dropout_rate if not explicitly set
        # This maintains backward compatibility: dropout_rate=0.0 means disabled
        if enable_dropout is None:
            enable_dropout = dropout_rate > 0.0
        self.enable_dropout = enable_dropout

        # Validate dropout_rate when dropout is enabled
        if enable_dropout and (dropout_rate <= 0.0 or dropout_rate > 1.0):
            raise ValueError(
                f"dropout_rate must be in (0.0, 1.0] when enable_dropout=True, got {dropout_rate}"
            )

        # Build layers modularly
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        if use_residual:
            if residual_style == "pre_ln":
                self.layers.append(PreLNResidualSwiGLUBlock(self.input_dim, hidden_dim))
            else:  # post_ln (default)
                self.layers.append(ResidualSwiGLUBlock(self.input_dim, hidden_dim))
        else:
            self.layers.append(SwiGLUBlock(self.input_dim, hidden_dim))
        # Dropout: only create Dropout layer if explicitly enabled, otherwise Identity (no-op)
        self.dropouts.append(nn.Dropout(dropout_rate) if enable_dropout else nn.Identity())

        # Additional layers: hidden_dim -> hidden_dim (with residuals if enabled)
        for _ in range(1, num_layers):
            if use_residual:
                if residual_style == "pre_ln":
                    self.layers.append(PreLNResidualSwiGLUBlock(hidden_dim, hidden_dim))
                else:  # post_ln (default)
                    self.layers.append(ResidualSwiGLUBlock(hidden_dim, hidden_dim))
            else:
                self.layers.append(SwiGLUBlock(hidden_dim, hidden_dim))
            # Dropout: only create Dropout layer if explicitly enabled, otherwise Identity (no-op)
            self.dropouts.append(nn.Dropout(dropout_rate) if enable_dropout else nn.Identity())

        # Final output layer: hidden_dim -> 1 (scalar regression output)
        self.final_score = nn.Linear(hidden_dim, 1)
        # CRITICAL FIX (2026-01-07): Use larger initialization scale for final layer
        # ============================================================================
        # INVESTIGATION HISTORY (2026-01-07):
        # 1. Initial problem: Model predictions stuck at 0, loss ≈ 21.16 = (4.6)^2
        #    - This indicated predictions were near 0 for binary labels at ±4.6
        #    - Accuracy stuck at 50% (random guessing)
        #
        # 2. First attempt: Initialize bias to 0.1 (instead of 0) to break symmetry
        #    - Changed: nn.init.zeros_(bias) → nn.init.constant_(bias, 0.1)
        #    - Kept: Xavier uniform (gain=1.0) for weights
        #    - Result: Still didn't work - predictions still near 0
        #    - Reason: LayerNorm normalizes activations to mean 0, std 1, so even
        #      with bias=0.1, the weighted sum from normalized inputs still clusters
        #      near 0 when weights are small (Xavier gain=1.0)
        #
        # 3. Root cause identified: LayerNorm + Xavier (gain=1.0) = outputs near 0
        #    - LayerNorm output: mean=0, std=1
        #    - Xavier (gain=1.0) gives small weights for normalized inputs
        #    - Result: Output mean ≈ 0.05, std ≈ 1.4, range ≈ [-4, +4]
        #    - This is too small for labels at ±4.6, causing slow/no learning
        #
        # 4. Final solution: Normal initialization with larger std (0.1)
        #    - Changed: nn.init.xavier_uniform_(weight, gain=1.0) → nn.init.normal_(weight, std=0.1)
        #    - Kept: bias=0.1 (still helps break symmetry)
        #    - Result: Output mean ≈ 0.5, std ≈ 3.4, range ≈ [-9, +10]
        #    - This better matches label range [-4.6, +4.6] and enables faster learning
        #
        # Research: Larger initialization scales help when LayerNorm normalizes inputs
        # (He et al., 2015; Ioffe & Szegedy, 2015). For regression with normalized
        # inputs, normal init with std=0.1 outperforms Xavier for final layers.
        # ============================================================================
        nn.init.normal_(self.final_score.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.final_score.bias, 0.1)  # Small non-zero bias to break symmetry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through modular layers

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, 1] (scalar regression output)
        """
        # Pass through all layers with dropout
        out = x
        for layer, dropout in zip(self.layers, self.dropouts, strict=True):
            out = layer(out)
            out = dropout(out)

        # Final output layer
        return self.final_score(out)  # type: ignore[no-any-return]


class SingleTowerStudent(nn.Module):
    model_features: torch.Tensor

    def __init__(
        self,
        model_id,
        use_int4=True,
        device=None,
        dtype=None,
        hf_token=None,
        normalize_embeddings=True,
    ):
        super().__init__()
        # Apply the torchao compatibility shim lazily at first model creation.
        _ensure_torchao_compat()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.normalize_embeddings = normalize_embeddings

        # R01: the fusion head is conditioned on a model-pair feature vector
        # (e.g. [strong_elo, weak_elo]) that is constant per deployment.
        # Registered as a persistent buffer so checkpoints can record and
        # validate it.
        self.register_buffer("model_features", torch.zeros(2, dtype=torch.float32), persistent=True)

        logger.info(f"Loading Backbone: {model_id}...")
        # Use SentenceTransformer which wraps AutoModel but avoids transformers' torchao import
        # SentenceTransformer will use the token from huggingface_hub if we've logged in
        # Ensure device is properly initialized (especially important for parallel processes)
        if self.device.type == "cuda" and torch.cuda.is_available():
            # Ensure CUDA is properly initialized in this process
            _ = torch.zeros(1).to(self.device)  # Initialize CUDA context
        self.backbone = _get_sentence_transformer()(
            # R09: token passed to the loader, never a persistent global login.
            # R02: remote-code execution is opt-in; default is off.
            model_id,
            trust_remote_code=False,
            local_files_only=True,
            device=str(self.device),
            token=hf_token,
        )

        # Convert to specified dtype (like Matrix Factorization router does)
        if dtype:
            if dtype == "bfloat16" and self.device.type == "cuda":
                self.backbone = self.backbone.to(torch.bfloat16)
                logger.info("   ✅ Converted backbone to bfloat16 (optimal for RTX 4090)")
            elif dtype == "float16" and (self.device.type == "cuda" or self.device.type == "mps"):
                self.backbone = self.backbone.to(torch.float16)
                logger.info("   ✅ Converted backbone to float16")
            elif dtype == "float32" or self.device.type == "cpu":
                # float32 is default, CPU requires float32
                logger.info("   ✅ Using float32 (default/CPU)")
            else:
                logger.warning(
                    f"   ⚠️  Unsupported dtype '{dtype}' for device {self.device.type}, using float32"
                )
        # Get tokenizer from the underlying model
        if hasattr(self.backbone, "_modules") and "0" in self.backbone._modules:
            # SentenceTransformer wraps the model, get the actual model
            underlying_model = self.backbone._modules["0"]
            if hasattr(underlying_model, "tokenizer"):
                self.tokenizer = underlying_model.tokenizer
            else:
                # Fallback: create tokenizer separately
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=False, token=hf_token, local_files_only=True
                )
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=False, token=hf_token, local_files_only=True
            )

        # Freeze backbone to force learning in the head (eval mode)
        self.backbone.eval()

        # Auto-detect dimension using SentenceTransformer's encode
        with torch.no_grad():
            dummy_emb = self.backbone.encode("test", convert_to_tensor=True, device=self.device)
            emb_dim = dummy_emb.shape[-1]

        self.head = GatedFusionHead(embedding_dim=emb_dim, linguistic_dim=3, model_dim=2).to(
            self.device
        )
        self.feature_extractor = LinguisticFeatureExtractor()

        self.int4_applied = False
        if use_int4:
            self._apply_int4()

    def _apply_int4(self):
        if not TORCHAO_AVAILABLE:
            raise RuntimeError("INT4 requested but TorchAO is unavailable")
        try:
            underlying = self.backbone._modules["0"]
            transformer = getattr(underlying, "auto_model", underlying)
            options = {"group_size": 32}
            if self.device.type == "cuda":
                options["inner_k_tiles"] = 8
            quantize_(transformer, int4_weight_only(**options))
            with torch.no_grad():
                smoke = self.backbone.encode(
                    "quantization smoke test",
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                )
            if not torch.isfinite(smoke).all():
                raise RuntimeError("non-finite quantized embeddings")
        except Exception as exc:
            raise RuntimeError("INT4 initialization failed; backbone cannot be reused") from exc
        self.int4_applied = True

    def forward(self, texts):
        # 1. Linguistic Features (CPU -> GPU)
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Empty batches have a defined shape contract: (0, 1) scores.
        if not texts:
            return torch.zeros((0, 1), dtype=torch.float32, device=self.device)

        ling_feats = self.feature_extractor.extract(texts).to(self.device)

        # 2. Backbone Forward
        with torch.no_grad():
            pooled_emb = self.backbone.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )

        # 3. Fusion (R01): embeddings + linguistic + model-pair features.
        # The head was constructed with embedding_dim + 3 + 2 inputs; the
        # model-pair buffer completes the feature vector. The combined
        # tensor is cast to float32 because the head stays float32 while
        # the backbone may run bfloat16/float16 on CUDA (P2: dtype
        # mismatch crashed every scoring call on default configs).
        batch_model_feats = self.model_features.to(
            device=pooled_emb.device, dtype=pooled_emb.dtype
        ).expand(pooled_emb.shape[0], -1)
        combined = torch.cat(
            [pooled_emb, ling_feats.to(pooled_emb), batch_model_feats], dim=1
        ).float()
        score = self.head(combined)  # [batch, 1]

        return score
