import ttnn
import torch
import torch.nn.functional as F


def test_split_reader_output_order():
    """Test that split reader produces correct output order"""
    device = ttnn.open_device(device_id=0)

    try:
        # Create simple input where each channel has a unique value
        # so we can track which channel ended up where
        batch_size = 2
        channels = 64
        H, W = 16, 16

        input_torch = torch.zeros((batch_size, channels, H, W), dtype=torch.float32)

        # Set unique values for each height row so we can identify sampling
        # Batch 0: rows have values 0, 10, 20, ...
        # Batch 1: rows have values 1000, 1010, 1020, ...
        for b in range(batch_size):
            for h in range(H):
                input_torch[b, :, h, :] = float(b * 1000 + h * 10)

        # Create a larger grid to test multiple grid points
        grid_h, grid_w = 12, 12
        grid_torch = torch.zeros((batch_size, grid_h, grid_w, 2), dtype=torch.float32)

        # Create a random grid (like the failing unit tests)
        torch.manual_seed(0)
        grid_torch = torch.rand((batch_size, grid_h, grid_w, 2), dtype=torch.float32) * 2.0 - 1.0

        print(f"Input batch 0: rows have values 0, 10, 20, ..., {(H-1)*10}")
        print(f"Input batch 1: rows have values 1000, 1010, 1020, ..., {1000 + (H-1)*10}")
        print(f"Grid size: {grid_h}x{grid_w}")
        print(f"Grid is random with values in range [-1, 1]")

        # PyTorch reference
        expected_torch = F.grid_sample(
            input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False
        )
        expected_nhwc = expected_torch.permute(0, 2, 3, 1).contiguous()

        print("\nPyTorch output batch 0 (first channel, first 3x3 grid):")
        for gh in range(min(3, grid_h)):
            row = " ".join([f"{expected_nhwc[0, gh, gw, 0].item():5.0f}" for gw in range(min(3, grid_w))])
            print(f"  Row {gh}: {row}")

        print("\nPyTorch output batch 1 (first channel, first 3x3 grid):")
        for gh in range(min(3, grid_h)):
            row = " ".join([f"{expected_nhwc[1, gh, gw, 0].item():5.0f}" for gw in range(min(3, grid_w))])
            print(f"  Row {gh}: {row}")

        # TTNN
        input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
        input_ttnn = ttnn.from_torch(input_nhwc, device=device, dtype=ttnn.bfloat16)
        grid_ttnn = ttnn.from_torch(grid_torch, device=device, dtype=ttnn.bfloat16)

        result_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest")
        result_torch = ttnn.to_torch(result_ttnn)

        print("\nTTNN output batch 0 (first channel, first 3x3 grid):")
        for gh in range(min(3, grid_h)):
            row = " ".join([f"{result_torch[0, gh, gw, 0].item():5.0f}" for gw in range(min(3, grid_w))])
            print(f"  Row {gh}: {row}")

        print("\nTTNN output batch 1 (first channel, first 3x3 grid):")
        for gh in range(min(3, grid_h)):
            row = " ".join([f"{result_torch[1, gh, gw, 0].item():5.0f}" for gw in range(min(3, grid_w))])
            print(f"  Row {gh}: {row}")

        # Check if outputs match
        max_diff = torch.max(torch.abs(expected_nhwc - result_torch)).item()
        print(f"\nMax difference: {max_diff:.6f}")

        if max_diff < 0.1:
            print("✓ PASS: Split reader output order is correct")
        else:
            print("✗ FAIL: Split reader output order is incorrect")
            # Print first few mismatches
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
                                break
                    if count >= 10:
                        break
                if count >= 10:
                    break

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_split_reader_output_order()
