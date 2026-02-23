// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace eltwise_dram_optimized {

/* RUNTIME ARGS */

struct EltwiseReaderArgs {
    uint32_t a_tensor_base_addr;
    uint32_t b_tensor_base_addr;
    uint32_t tile_ofs;
    uint32_t num_tiles;
    uint32_t tile_stride;
    uint32_t vc;
};

struct EltwiseComputeArgs {
    uint32_t num_tiles;
    uint32_t vc;
};
struct EltwiseWriterArgs {
    uint32_t dst_base_addr;
    uint32_t tile_ofs;
    uint32_t num_tiles;
    uint32_t tile_stride;
    uint32_t vc;
};

/* Compile-time ARGS */

struct EltwiseReaderCTArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t num_tiles_per_cycle;
};

struct EltwiseComputeCTArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t output_cb;
    uint32_t num_tiles_per_cycle;
};

struct EltwiseWriterCTArgs {
    uint32_t cb_dst;
    uint32_t num_tiles_per_cycle;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseReaderArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseReaderCTArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseComputeArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseComputeCTArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseWriterArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseWriterCTArgs>);

}  // namespace eltwise_dram_optimized
