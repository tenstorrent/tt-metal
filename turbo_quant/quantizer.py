"""TurboQuant MSE and inner-product optimized quantizers."""

from __future__ import annotations

import math
import torch

from turbo_quant.rotation import generate_rotation_matrix
from turbo_quant.codebook import get_codebook


class TurboQuantMSE:
    """MSE-optimized TurboQuant quantizer (Algorithm 1).

    Quantizes vectors by:
      1. Rotating via a random orthogonal matrix
      2. Normalizing to the unit sphere
      3. Applying per-coordinate Lloyd-Max quantization
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        self.rotation = generate_rotation_matrix(head_dim, seed=seed, device=device, dtype=dtype)
        self.rotation_t = self.rotation.t()
        self.codebook = get_codebook(head_dim, bits, device=device, dtype=dtype)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape [..., head_dim].

        Returns:
            (indices, norms) where:
              - indices: uint8 tensor of shape [..., head_dim] with centroid indices
              - norms: float tensor of shape [..., 1] with L2 norms
        """
        # Rotate
        y = x @ self.rotation  # [..., head_dim]

        # Compute and store norms
        norms = y.norm(dim=-1, keepdim=True)  # [..., 1]

        # Normalize to unit sphere
        y_hat = y / (norms + 1e-10)

        # Quantize per coordinate
        indices = self.codebook.quantize(y_hat)  # [..., head_dim]
        indices = indices.to(torch.uint8)

        return indices, norms.to(self.dtype)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize back to original space.

        Args:
            indices: uint8 tensor of centroid indices, shape [..., head_dim].
            norms: float tensor of norms, shape [..., 1].

        Returns:
            Reconstructed tensor of shape [..., head_dim].
        """
        # Retrieve centroid values
        y_hat = self.codebook.dequantize(indices.long())  # [..., head_dim]

        # Rescale
        y = y_hat * norms

        # Rotate back
        x_rec = y @ self.rotation_t  # [..., head_dim]
        return x_rec.to(self.dtype)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TurboQuantMSE":
        if device is not None:
            self.device = device
            self.rotation = self.rotation.to(device=device)
            self.rotation_t = self.rotation_t.to(device=device)
            self.codebook.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self.rotation = self.rotation.to(dtype=dtype)
            self.rotation_t = self.rotation_t.to(dtype=dtype)
            self.codebook.to(dtype=dtype)
        return self


class TurboQuantProd:
    """Inner-product optimized TurboQuant quantizer (Algorithm 2).

    Uses MSE quantizer at (b-1) bits plus 1-bit QJL on the residual
    to produce unbiased inner-product estimates.
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        if bits < 2:
            raise ValueError("TurboQuantProd requires at least 2 bits (1 for MSE + 1 for QJL)")

        # MSE quantizer at (b-1) bits
        self.mse_quantizer = TurboQuantMSE(
            head_dim=head_dim,
            bits=bits - 1,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        # Random projection matrix for QJL
        gen = torch.Generator(device="cpu").manual_seed(seed + 1000)
        self.S = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
        self.S = self.S.to(device=device, dtype=dtype)
        self.S_t = self.S.t()

        self._qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape [..., head_dim].

        Returns:
            (mse_indices, mse_norms, qjl_signs, residual_norms) where:
              - mse_indices: uint8 tensor [..., head_dim]
              - mse_norms: float tensor [..., 1]
              - qjl_signs: int8 tensor [..., head_dim] with values ±1
              - residual_norms: float tensor [..., 1]
        """
        # MSE quantize at (b-1) bits
        mse_indices, mse_norms = self.mse_quantizer.quantize(x)

        # Compute residual
        x_rec = self.mse_quantizer.dequantize(mse_indices, mse_norms)
        residual = x - x_rec

        # QJL: 1-bit quantization of projected residual
        projected = residual @ self.S.t()  # [..., head_dim]
        qjl_signs = torch.sign(projected).to(torch.int8)
        qjl_signs[qjl_signs == 0] = 1  # Map zeros to +1

        # Store residual norm
        residual_norms = residual.norm(dim=-1, keepdim=True).to(self.dtype)

        return mse_indices, mse_norms, qjl_signs, residual_norms

    def dequantize(
        self,
        mse_indices: torch.Tensor,
        mse_norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize back to original space.

        Args:
            mse_indices: uint8 centroid indices [..., head_dim].
            mse_norms: float norms [..., 1].
            qjl_signs: int8 signs [..., head_dim].
            residual_norms: float residual norms [..., 1].

        Returns:
            Reconstructed tensor of shape [..., head_dim].
        """
        # MSE reconstruction
        x_mse = self.mse_quantizer.dequantize(mse_indices, mse_norms)

        # QJL reconstruction: √(π/2)/d · γ · Sᵀ · qjl
        qjl_float = qjl_signs.to(self.dtype)
        x_qjl = self._qjl_scale * residual_norms * (qjl_float @ self.S)

        return (x_mse + x_qjl).to(self.dtype)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TurboQuantProd":
        if device is not None:
            self.device = device
            self.S = self.S.to(device=device)
            self.S_t = self.S_t.to(device=device)
            self.mse_quantizer.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self.S = self.S.to(dtype=dtype)
            self.S_t = self.S_t.to(dtype=dtype)
            self.mse_quantizer.to(dtype=dtype)
        return self


class OutlierAwareTurboQuant:
    """TurboQuant with separate bit allocations for outlier vs normal channels.

    Splits the head_dim channels into two groups after rotation:
      - Outlier channels: quantized at higher bit-width
      - Normal channels: quantized at lower bit-width

    Example: 32 outlier channels at 3 bits + 96 normal channels at 2 bits
    yields an effective 2.5 bits per coordinate.

    Outlier detection modes:
      - "static": fixed first `num_outlier_channels` dimensions after rotation
      - "calibration": detect outliers from calibration data (channels with highest variance)
    """

    def __init__(
        self,
        head_dim: int = 128,
        outlier_bits: int = 3,
        normal_bits: int = 2,
        num_outlier_channels: int = 32,
        outlier_mode: str = "static",
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.head_dim = head_dim
        self.outlier_bits = outlier_bits
        self.normal_bits = normal_bits
        self.num_outlier_channels = num_outlier_channels
        self.num_normal_channels = head_dim - num_outlier_channels
        self.outlier_mode = outlier_mode
        self.device = device
        self.dtype = dtype

        if num_outlier_channels >= head_dim:
            raise ValueError(f"num_outlier_channels ({num_outlier_channels}) must be < head_dim ({head_dim})")

        self.effective_bits = (num_outlier_channels * outlier_bits + self.num_normal_channels * normal_bits) / head_dim

        # Shared rotation matrix — applied before splitting
        self.rotation = generate_rotation_matrix(head_dim, seed=seed, device=device, dtype=dtype)
        self.rotation_t = self.rotation.t()

        # Separate codebooks for each group
        self.outlier_codebook = get_codebook(head_dim, outlier_bits, device=device, dtype=dtype)
        self.normal_codebook = get_codebook(head_dim, normal_bits, device=device, dtype=dtype)

        # Outlier channel indices — will be set by calibration or static init
        self._outlier_idx: torch.Tensor | None = None
        self._normal_idx: torch.Tensor | None = None
        if outlier_mode == "static":
            self._set_static_outlier_indices()

    def _set_static_outlier_indices(self) -> None:
        """Use first `num_outlier_channels` as outliers (arbitrary but deterministic)."""
        self._outlier_idx = torch.arange(self.num_outlier_channels, device=self.device)
        self._normal_idx = torch.arange(self.num_outlier_channels, self.head_dim, device=self.device)

    def calibrate(self, data: torch.Tensor) -> None:
        """Detect outlier channels from calibration data.

        Selects channels with highest variance in the rotated space as outliers,
        since these benefit most from higher bit-width quantization.

        Args:
            data: Calibration tensor of shape [..., head_dim]. Typically a sample
                  of KV vectors from a few forward passes.
        """
        # Rotate to the quantization space
        y = data.reshape(-1, self.head_dim).to(self.dtype) @ self.rotation

        # Compute per-channel variance
        channel_var = y.var(dim=0)  # [head_dim]

        # Top-k by variance are the outliers
        _, top_indices = channel_var.topk(self.num_outlier_channels)
        all_indices = torch.arange(self.head_dim, device=self.device)
        mask = torch.ones(self.head_dim, dtype=torch.bool, device=self.device)
        mask[top_indices] = False

        self._outlier_idx = top_indices.sort().values
        self._normal_idx = all_indices[mask]
        self.outlier_mode = "calibration"

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with outlier-aware channel splitting.

        Args:
            x: Input tensor of shape [..., head_dim].

        Returns:
            (indices, norms, channel_map) where:
              - indices: uint8 tensor [..., head_dim] — packed indices (outlier and normal)
              - norms: float tensor [..., 1] — L2 norms
              - channel_map: not stored per-token, but available via self._outlier_idx
        """
        # Rotate
        y = x @ self.rotation  # [..., head_dim]

        # Compute norms
        norms = y.norm(dim=-1, keepdim=True)  # [..., 1]
        y_hat = y / (norms + 1e-10)

        # Split channels and quantize each group with its own codebook
        indices = torch.empty(*y_hat.shape[:-1], self.head_dim, dtype=torch.uint8, device=self.device)
        indices[..., self._outlier_idx] = self.outlier_codebook.quantize(y_hat[..., self._outlier_idx]).to(torch.uint8)
        indices[..., self._normal_idx] = self.normal_codebook.quantize(y_hat[..., self._normal_idx]).to(torch.uint8)

        return indices, norms.to(self.dtype)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize with outlier-aware channel splitting.

        Args:
            indices: uint8 centroid indices [..., head_dim].
            norms: float norms [..., 1].

        Returns:
            Reconstructed tensor of shape [..., head_dim].
        """
        y_hat = torch.empty(*indices.shape[:-1], self.head_dim, dtype=self.dtype, device=self.device)

        y_hat[..., self._outlier_idx] = self.outlier_codebook.dequantize(indices[..., self._outlier_idx].long())
        y_hat[..., self._normal_idx] = self.normal_codebook.dequantize(indices[..., self._normal_idx].long())

        # Rescale and rotate back
        y = y_hat * norms
        x_rec = y @ self.rotation_t
        return x_rec.to(self.dtype)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> "OutlierAwareTurboQuant":
        if device is not None:
            self.device = device
            self.rotation = self.rotation.to(device=device)
            self.rotation_t = self.rotation_t.to(device=device)
            self.outlier_codebook.to(device=device)
            self.normal_codebook.to(device=device)
            if self._outlier_idx is not None:
                self._outlier_idx = self._outlier_idx.to(device=device)
                self._normal_idx = self._normal_idx.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self.rotation = self.rotation.to(dtype=dtype)
            self.rotation_t = self.rotation_t.to(dtype=dtype)
            self.outlier_codebook.to(dtype=dtype)
            self.normal_codebook.to(dtype=dtype)
        return self
