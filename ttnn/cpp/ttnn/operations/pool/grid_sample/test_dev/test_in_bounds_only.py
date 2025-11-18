import ttnn
import torch
import torch.nn.functional as F


def test_in_bounds_only():
    """Test with grid that only samples in-bounds coordinates"""
    device = ttnn.open_device(device_id=0)

    try:
        batch_size = 2
        channels = 64
        H, W = 16, 16

        input_torch = torch.zeros((batch_size, channels, H, W), dtype=torch.float32)

        # Set unique values for each height row
        for b in range(batch_size):
            for h in range(H):
                input_torch[b, :, h, :] = float(b * 1000 + h * 10)

        # Create a grid with 8x8 sampling that stays well within bounds
        grid_h, grid_w = 8, 8
        grid_torch = torch.zeros((batch_size, grid_h, grid_w, 2), dtype=torch.float32)

        # Sample from center region (pixels 4-11) which should all be in bounds
        for b in range(batch_size):
            for gh in range(grid_h):
                for gw in range(grid_w):
                    # Sample from pixels 4+gh, 4+gw (range [4,11])
                    h_pix = 4 + gh
                    w_pix = 4 + gw
                    # Convert to normalized coords
                    h_norm = (h_pix - 7.5) / 8.0
                    w_norm = (w_pix - 7.5) / 8.0
                    grid_torch[b, gh, gw, 0] = w_norm
                    grid_torch[b, gh, gw, 1] = h_norm

        print(f"Sampling from pixels [4,11] x [4,11] (all in-bounds)")
        print(f"Expected batch 0 values: 40, 50, ..., 110")
        print(f"Expected batch 1 values: 1040, 1050, ..., 1110")

        # PyTorch reference
        expected_torch = F.grid_sample(
            input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False
        )
        expected_nhwc = expected_torch.permute(0, 2, 3, 1).contiguous()

        print("\nPyTorch output batch 0 (first channel, all 8x8):")
        for gh in range(grid_h):
            row = " ".join([f"{expected_nhwc[0, gh, gw, 0].item():4.0f}" for gw in range(grid_w)])
            print(f"  Row {gh}: {row}")

        # TTNN
        input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
        input_ttnn = ttnn.from_torch(input_nhwc, device=device, dtype=ttnn.bfloat16)
        grid_ttnn = ttnn.from_torch(grid_torch, device=device, dtype=ttnn.bfloat16)

        result_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest")
        result_torch = ttnn.to_torch(result_ttnn)

        print("\nTTNN output batch 0 (first channel, all 8x8):")
        for gh in range(grid_h):
            row = " ".join([f"{result_torch[0, gh, gw, 0].item():4.0f}" for gw in range(grid_w)])
            print(f"  Row {gh}: {row}")

        # Check if outputs match
        max_diff = torch.max(torch.abs(expected_nhwc - result_torch)).item()
        print(f"\nMax difference: {max_diff:.6f}")

        if max_diff < 0.1:
            print("✓ PASS: In-bounds sampling is correct")
        else:
            print("✗ FAIL: In-bounds sampling has errors")
            print("\nFirst 10 mismatches:")
            count = 0
            for b in range(batch_size):
                for gh in range(grid_h):
                    for gw in range(grid_w):
                        exp = expected_nhwc[b, gh, gw, 0].item()
                        got = result_torch[b, gh, gw, 0].item()
                        if abs(exp - got) >= 0.1:
                            print(f"  Batch{b} Grid[{gh},{gw}]: expected={exp:.1f}, got={got:.1f}")
                            count += 1
                            if count >= 10:
                                return

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_in_bounds_only()
