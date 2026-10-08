// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the all-gather MoE block's program factories (route plan, local reduce, row adds, untilizes).
// Every program takes common runtime args only: tensors as Buffer* bindings (a program-cache hit only patches their
// addresses), scalars derived from shapes / attributes. A core finds its index (y * grid x + x over the logical
// worker grid) and work range on device (kernels/core_range.hpp).

#include <cstdint>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag {

inline constexpr const char* KERNEL_DIR =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_ag/device/kernels/";
inline constexpr uint32_t NONE = 0xFFFFFFFFu;

inline uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// Items per core when `total` is split over `parts` cores in `align` multiples (core i: [i per, i per + per) clipped).
inline uint32_t per_core(uint32_t total, uint32_t parts, uint32_t align) {
    return round_up((total + parts - 1) / parts, align);
}

// Tokens per core are a multiple of this so a core's y_slot block (K uint32 per token) starts 64 B aligned in DRAM
// (the NoC moves DRAM data at 64 B alignment): 2 at top-8.
inline uint32_t token_align(uint32_t K) {
    uint32_t g = 16, k = K;  // gcd(K, 16)
    while (k) {
        const uint32_t r = g % k;
        g = k;
        k = r;
    }
    const uint32_t a = 16 / g;
    return a % 2 ? 2 * a : a;  // lcm(a, 2)
}

// The whole logical worker grid (row major: core index = y * grid.x + x).
struct Grid {
    CoreCoord size;
    uint32_t cores;
    CoreRangeSet range;
};

inline Grid worker_grid(const Tensor& t) {
    const auto size = t.device()->compute_with_storage_grid_size();
    return {size, static_cast<uint32_t>(size.x * size.y), CoreRangeSet(CoreRange({0, 0}, {size.x - 1, size.y - 1}))};
}

inline tt::tt_metal::CBDescriptor cb_desc(
    uint32_t index, uint32_t total_size, uint32_t page_size, tt::DataFormat fmt, const CoreRangeSet& cores) {
    return tt::tt_metal::CBDescriptor{
        .total_size = total_size,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = fmt,
            .page_size = page_size,
        }}},
    };
}

// Scratch CB (uint32 format, one page).
inline tt::tt_metal::CBDescriptor scratch_cb(uint32_t index, uint32_t size, const CoreRangeSet& cores) {
    return cb_desc(index, size, size, tt::DataFormat::UInt32, cores);
}

// bf16 "tile" pages (2 KB): row segments of 1024 elements or 32 x 32 tiles.
inline tt::tt_metal::CBDescriptor tile_cb(
    uint32_t index,
    uint32_t tiles,
    const CoreRangeSet& cores,
    tt::DataFormat fmt = tt::DataFormat::Float16_b,
    uint32_t page = 2048) {
    return cb_desc(index, tiles * page, page, fmt, cores);
}

inline tt::tt_metal::KernelDescriptor kernel_desc(
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> compile_time_args,
    tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
    tt::tt_metal::KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(compile_time_args);
    k.config = std::move(config);
    return k;
}

inline tt::tt_metal::DataMovementConfigDescriptor dm_config(uint32_t processor, uint32_t noc) {
    return tt::tt_metal::DataMovementConfigDescriptor{
        .processor =
            processor ? tt::tt_metal::DataMovementProcessor::RISCV_1 : tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = noc ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0,
    };
}

inline tt::tt_metal::ComputeConfigDescriptor fp32_compute_config() {
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = true};
}

// Validation helpers (TT_FATAL with the op name).
void check_dram_interleaved(const Tensor& t, const char* op, const char* name);
void check_row_major(const Tensor& t, tt::tt_metal::DataType dtype, const char* op, const char* name);
void check_single_row(const Tensor& t, const char* op, const char* name);  // [.., 1, X]: one DRAM page

// Rows of a row-major tensor (product of all dims but the last).
uint32_t rm_rows(const Tensor& t);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag
