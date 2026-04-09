"""Lloyd-Max codebook construction for TurboQuant's Beta distribution."""

import math
import torch
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad


def beta_pdf(x: float, d: int) -> float:
    """PDF of a coordinate of a uniformly random point on the d-sphere.

    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
    Defined on [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    coeff = gamma_fn(d / 2.0) / (math.sqrt(math.pi) * gamma_fn((d - 1) / 2.0))
    return coeff * (1.0 - x * x) ** ((d - 3) / 2.0)


def lloyd_max(d: int, num_levels: int, max_iter: int = 500, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max centroids and boundaries for the Beta distribution.

    Args:
        d: Dimension (determines the Beta distribution shape).
        num_levels: Number of quantization levels (2^b).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        (centroids, boundaries) — centroids shape [num_levels], boundaries shape [num_levels+1].
    """
    pdf = lambda x: beta_pdf(x, d)

    # Initialize centroids uniformly in [-1, 1]
    centroids = np.linspace(-1 + 1 / num_levels, 1 - 1 / num_levels, num_levels)

    for _ in range(max_iter):
        # Compute boundaries as midpoints
        boundaries = np.empty(num_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(num_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        # Update centroids as conditional expectations
        new_centroids = np.empty(num_levels)
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            numerator, _ = quad(lambda x: x * pdf(x), lo, hi)
            denominator, _ = quad(pdf, lo, hi)
            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2.0

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Recompute final boundaries
    boundaries = np.empty(num_levels + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(num_levels - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return centroids, boundaries


class TurboQuantCodebook:
    """Precomputed codebook for a given dimension and bit-width."""

    def __init__(self, d: int, bits: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.d = d
        self.bits = bits
        self.num_levels = 1 << bits

        centroids_np, boundaries_np = lloyd_max(d, self.num_levels)
        self.centroids = torch.tensor(centroids_np, device=device, dtype=dtype)
        self.boundaries = torch.tensor(boundaries_np, device=device, dtype=dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map each coordinate to its nearest centroid index.

        Args:
            x: Tensor of any shape, values expected in ~[-1, 1] (unit-sphere coordinates).

        Returns:
            Integer tensor of same shape with values in [0, num_levels).
        """
        # Use boundary-based bucketing (faster than distance computation)
        # boundaries: [num_levels+1], we check which bucket each value falls into
        idx = torch.bucketize(x, self.boundaries[1:-1])
        return idx.clamp(0, self.num_levels - 1)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """Map centroid indices back to centroid values.

        Args:
            idx: Integer tensor of centroid indices.

        Returns:
            Tensor of centroid values with same shape.
        """
        return self.centroids[idx]

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TurboQuantCodebook":
        if device is not None:
            self.centroids = self.centroids.to(device=device)
            self.boundaries = self.boundaries.to(device=device)
        if dtype is not None:
            self.centroids = self.centroids.to(dtype=dtype)
            self.boundaries = self.boundaries.to(dtype=dtype)
        return self


# Cache of precomputed codebooks keyed by (d, bits)
_codebook_cache: dict[tuple[int, int], TurboQuantCodebook] = {}


def get_codebook(
    d: int, bits: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> TurboQuantCodebook:
    """Get or create a cached codebook for the given parameters."""
    key = (d, bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = TurboQuantCodebook(d, bits, device="cpu", dtype=torch.float64)
    cb = _codebook_cache[key]
    return cb.to(device=device, dtype=dtype)
