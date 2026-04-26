// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct TypecastParams {
    const tt::tt_metal::DataType input_dtype;
    const tt::tt_metal::DataType output_dtype;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
    const bool bfp8_pack_precise = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
    const std::optional<Layout> output_layout = std::nullopt;
};

struct TypecastInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

// ── Shared helpers for typecast program factories ────────────────────────────

inline constexpr const char* TYPECAST_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

// Build the TYPECAST_LLK_INIT / TYPECAST_LLK define strings for the compute kernel.
inline std::map<std::string, std::string> make_typecast_compute_defines(
    tt::tt_metal::DataType input_dtype, tt::tt_metal::DataType output_dtype) {
    const auto in_fmt = static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(input_dtype));
    const auto out_fmt = static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(output_dtype));
    return {
        {"TYPECAST_LLK_INIT", fmt::format("typecast_tile_init<{}u, {}u>", in_fmt, out_fmt)},
        {"TYPECAST_LLK", fmt::format("typecast_tile<{}u, {}u>", in_fmt, out_fmt)},
    };
}

// Descriptor variant (vector of pairs).
inline tt::tt_metal::KernelDescriptor::Defines make_typecast_compute_defines_desc(
    tt::tt_metal::DataType input_dtype, tt::tt_metal::DataType output_dtype) {
    const auto in_fmt = static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(input_dtype));
    const auto out_fmt = static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(output_dtype));
    return {
        {"TYPECAST_LLK_INIT", fmt::format("typecast_tile_init<{}u, {}u>", in_fmt, out_fmt)},
        {"TYPECAST_LLK", fmt::format("typecast_tile<{}u, {}u>", in_fmt, out_fmt)},
    };
}

}  // namespace ttnn::prim
