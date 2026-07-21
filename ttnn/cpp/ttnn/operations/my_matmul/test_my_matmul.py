"""Quick correctness check for ttnn.my_matmul against a torch golden."""
import torch
import ttnn


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    # M, K, N = 1024, 1024, 1024
    # M, K, N = 2048, 2048, 2048
    # M, K, N = 4096, 4096, 4096
    M, K, N = 4096, 16384, 4096
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    golden = a.float() @ b.float()

    device = ttnn.open_device(device_id=0)
    try:
        ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

        tc = ttnn.my_matmul(ta, tb)
        result = ttnn.to_torch(tc)

        print(f"shapes: A{tuple(a.shape)} @ B{tuple(b.shape)} -> C{tuple(result.shape)}")
        p = pcc(golden, result)
        print(f"PCC vs torch golden: {p:.6f}")
        assert result.shape == golden.shape, f"shape mismatch {result.shape} vs {golden.shape}"
        assert p > 0.99, f"PCC too low: {p}"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
