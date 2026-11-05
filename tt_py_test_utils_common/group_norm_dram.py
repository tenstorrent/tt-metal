# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utils_for_testing import assert_numeric_metrics

STATISTICS_MODES = ("tile_reduction", "two_pass")


def _use_two_pass_statistics(statistics_mode):
    if statistics_mode not in STATISTICS_MODES:
        raise ValueError(f"Unknown GroupNorm statistics mode: {statistics_mode!r}")
    return statistics_mode == "two_pass"


# perf_test_mode is used to skip the torch execution and pcc comparison, and always runs the operation once
def run_group_norm_DRAM(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    statistics_mode,
    use_input_mask,
    perf_test_mode=False,
    specify_grid=True,
    input_layout=ttnn.TILE_LAYOUT,
    output_layout=ttnn.TILE_LAYOUT,
):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    if specify_grid:
        #   Use the explicit user-chosen grid.
        grid_for_params = grid_size
    else:
        # Exercises the C++ automatic grid selection.
        # We must prepare gamma/beta/mask for that *same* auto-selected
        # grid; otherwise their shapes (driven by num_virtual_cols/num_virtual_rows)
        # would not match the grid the op actually picks at runtime.
        grid_for_params = ttnn.determine_expected_group_norm_dram_grid_size(
            device=device,
            num_channels=C,
            num_groups=num_groups,
            input_nhw=N * H * W,
            num_batches=N,
        )

    use_welford = _use_two_pass_statistics(statistics_mode)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    if not perf_test_mode:
        torch_output_tensor = torch.nn.functional.group_norm(
            torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-12
        )
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if input_layout == ttnn.TILE_LAYOUT:
        input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    gn_input_tensor = input_tensor_tilized if input_layout == ttnn.TILE_LAYOUT else input_tensor_row_major

    # Create dram group norm params
    [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_for_params, return_mask=True
    )

    # groupnorm

    num_itr = 2  # second iteration to help catch potential runtime args issue.

    if C > 512 or N > 2 or perf_test_mode:
        num_itr = 1  # one iter if it is too slow
    for _ in range(num_itr):
        output_tensor = ttnn.group_norm(
            gn_input_tensor,
            num_groups=num_groups,
            input_mask=input_mask_tensor if use_input_mask else None,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=output_layout,
            core_grid=grid_size if specify_grid else None,
            inplace=False,
            num_out_blocks=num_out_blocks if specify_grid else None,
            use_welford=use_welford,
        )
        ttnn.synchronize_device(device)

    if not perf_test_mode:
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        pcc_threshold = 0.999
        rtol = 0.060

        if use_welford:
            atol = 0.043
            frobenius_threshold = 0.01
        else:
            if specify_grid:
                atol = 0.069
                frobenius_threshold = 0.025
            else:
                # Not using Welford + auto-grid: the op also picks num_out_blocks via the
                # heuristic in the program factory, which generally differs from
                # the explicit num_out_blocks used in the specify_grid=True branch.
                # Different num_out_blocks chunks the per-block partial reductions
                # differently, producing visible bfloat16 rounding drift on the
                # tile-reduction mean/variance path.
                atol = 0.085
                frobenius_threshold = 0.030

        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


def run_group_norm_DRAM_with_input_mask(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    statistics_mode,
    specify_grid=True,
    perf_test_mode=False,
):
    run_group_norm_DRAM(
        device,
        N,
        C,
        H,
        W,
        num_groups,
        num_out_blocks,
        cores_y,
        cores_x,
        statistics_mode,
        use_input_mask=True,
        perf_test_mode=perf_test_mode,
        specify_grid=specify_grid,
    )
