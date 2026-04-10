// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fill_pad_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <array>
#include <bit>

namespace ttnn::prim::detail {

inline const std::map<ttnn::DataType, uint32_t> data_type_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT16, 2},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::INT32, 4},
};

// Packs fill_value into the uint32 bit pattern expected by the reader/compute kernels.
// For 2-byte types both halves of the 32-bit word carry the same 16-bit value;
// for 4-byte types the full 32-bit pattern is used.
inline uint32_t pack_fill_value(ttnn::DataType dtype, float fill_value) {
    switch (dtype) {
        case ttnn::DataType::BFLOAT16: {
            const uint16_t bits = static_cast<uint16_t>(std::bit_cast<uint32_t>(fill_value) >> 16);
            return (static_cast<uint32_t>(bits) << 16) | bits;
        }
        case ttnn::DataType::UINT16: {
            const uint16_t bits = static_cast<uint16_t>(static_cast<int32_t>(fill_value) & 0xFFFF);
            return (static_cast<uint32_t>(bits) << 16) | bits;
        }
        case ttnn::DataType::FLOAT32: return std::bit_cast<uint32_t>(fill_value);
        default:  // UINT32, INT32
            return static_cast<uint32_t>(static_cast<int32_t>(fill_value));
    }
}

// DataFormat string for where_tile<> template parameter.
inline std::string get_where_data_fmt(ttnn::DataType dtype) {
    switch (dtype) {
        case ttnn::DataType::FLOAT32: return "DataFormat::Float32";
        case ttnn::DataType::UINT32: return "DataFormat::UInt32";
        case ttnn::DataType::INT32: return "DataFormat::Int32";
        default: return "DataFormat::Float16_b";  // BFLOAT16, UINT16
    }
}

// DataFormat string for fill_tile_int<> template parameter (integer types only).
inline std::string get_fill_int_fmt(ttnn::DataType dtype) {
    switch (dtype) {
        case ttnn::DataType::UINT32: return "DataFormat::UInt32";
        case ttnn::DataType::INT32: return "DataFormat::Int32";
        default: return "DataFormat::UInt16";  // UINT16
    }
}

}  // namespace ttnn::prim::detail

namespace ttnn::prim {

struct FillPadSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

// Handles DRAM tensors (Interleaved + DRAM sharded).
//
// Work splitting is unified across three border-tile phases (right / bottom /
// corner). All border tiles across all slices are enumerated into a single
// global index space and handed to split_work_to_cores at tile granularity,
// so 2D / small-N_slices shapes (e.g. 4097x4097) still fan out across the
// full compute grid. Each core gets per-phase (start, num) pairs; phases
// with num == 0 are skipped. A single compute kernel binary covers all cores
// (CT has_right_pad / has_bottom_pad gate the phase branches at compile time).
struct FillPadProgramFactory {
    using shared_variables_t = FillPadSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FillPadParams& operation_attributes,
        const FillPadInputs& tensor_args,
        Tensor& tensor_return_value);
};

struct FillPadL1ShardedSharedVariables {
    // Indexed by has_right_pad (0/1); 0 means that group is unused.
    std::array<tt::tt_metal::KernelHandle, 2> reader_kernel_ids = {0, 0};
    std::array<tt::tt_metal::KernelHandle, 2> writer_kernel_ids = {0, 0};
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    std::vector<CoreCoord> active_cores;
    std::vector<uint32_t> active_core_rp;  // has_right_pad per active core (for override_runtime_arguments)
};

// Handles all L1-sharded tensors (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED).
// Unlike `FillPadProgramFactory` which gives each core a comparable number of borders,
// with FillPadL1ShardedProgramFactory, each core only processes its own local L1 data (no balancing).
struct FillPadL1ShardedProgramFactory {
    using shared_variables_t = FillPadL1ShardedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FillPadParams& operation_attributes,
        const FillPadInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
