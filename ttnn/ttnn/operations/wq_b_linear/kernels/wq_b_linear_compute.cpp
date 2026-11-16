// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// y block (block_m_tiles x core_n_tiles) = x (resident, K-blocked) @ W (streamed, K-blocked),
// accumulated in place in cb_y_tiles via packer L1 accumulation.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t sb_h = get_compile_time_arg_val(5);
    constexpr uint32_t sb_w = get_compile_time_arg_val(6);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(8);

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_stream, cb_y_tiles);

    CircularBuffer x_buf(cb_x_resident);
    CircularBuffer w_buf(cb_w_stream);
    CircularBuffer y_buf(cb_y_tiles);

    constexpr uint32_t y_block_tiles = block_m_tiles * core_n_tiles;
    y_buf.reserve_back(y_block_tiles);
    matmul_block<
        false,
        true,
        LastBlockTarget::Interm,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        NoPostCompute,
        NoPreKBlock,
        NoPostKBlock,
        0,
        NoKBlockInnerDimFn,
        NoIn0Source,
        NoIn1BaseOffset,
        true>(
        x_buf,
        w_buf,
        y_buf,
        y_buf,
        MatmulBlockShape::of(block_m_tiles / sb_h, core_n_tiles / sb_w, sb_h, sb_w, k_block_tiles, num_k_blocks));
    y_buf.push_back(y_block_tiles);
}
