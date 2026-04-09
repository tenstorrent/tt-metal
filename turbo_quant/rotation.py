"""Random orthogonal rotation matrix generation for TurboQuant."""

import torch


def generate_rotation_matrix(
    d: int,
    seed: int = 42,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a uniformly random orthogonal matrix via QR decomposition.

    Args:
        d: Dimension of the rotation matrix (head_dim).
        seed: Random seed for reproducibility.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Orthogonal matrix Π of shape [d, d].
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    # Ensure uniform Haar distribution by fixing signs via diagonal of R
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device=device, dtype=dtype)
