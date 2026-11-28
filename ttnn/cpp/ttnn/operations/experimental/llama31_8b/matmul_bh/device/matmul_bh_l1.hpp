// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "ttnn/operations/core/program_cache_l1.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_1d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_2d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/minimal.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh {

template <typename Buffers>
std::uint64_t dfb_l1_bytes(const Buffers& buffers) {
    std::vector<std::decay_t<decltype(buffers.begin()->unique_id)>> counted;
    std::uint64_t total = 0;
    for (const auto& dfb : buffers) {
        const auto& aliases = dfb.advanced_options.alias_with;
        const bool shared = std::any_of(aliases.begin(), aliases.end(), [&](const auto& name) {
            return std::find(counted.begin(), counted.end(), name) != counted.end();
        });
        if (!shared) {
            total += std::uint64_t{dfb.entry_size} * dfb.num_entries;
        }
        counted.push_back(dfb.unique_id);
    }
    return total;
}

inline tt::DataFormat format_of(DataType dtype) { return tt::tt_metal::datatype_to_dataformat_converter(dtype); }

inline tt::DataFormat interm_format_of(std::uint32_t code, tt::DataFormat output) {
    return code == 2 ? tt::DataFormat::Float32 : code == 1 ? tt::DataFormat::Float16_b : output;
}

inline std::uint64_t l1_bytes_1d(
    const mcast_1d::Mcast1DPlan& plan, const ttnn::Tensor& a, const ttnn::Tensor& b, DataType out_dtype) {
    const auto in0_tile = a.tensor_spec().tile();
    const auto in1_tile = b.tensor_spec().tile();
    const auto out_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    const auto align = tt::tt_metal::hal::get_dram_alignment();
    const auto out_format = format_of(out_dtype);
    const auto interm_format = interm_format_of(plan.interm0_format, out_format);
    std::uint64_t bytes =
        std::uint64_t{plan.in0_dfb_tiles} * tt::align(in0_tile.get_tile_size(format_of(a.dtype())), align) +
        std::uint64_t{plan.in1_dfb_tiles} * tt::align(in1_tile.get_tile_size(format_of(b.dtype())), align) +
        std::uint64_t{plan.out_dfb_tiles} * out_tile.get_tile_size(out_format);
    if (interm_format != out_format) {
        bytes += std::uint64_t{plan.interm0_dfb_tiles} * out_tile.get_tile_size(interm_format);
    }
    return bytes;
}

inline std::uint64_t l1_bytes_2d(
    const mcast_2d::Mcast2DPlan& plan, const ttnn::Tensor& a, const ttnn::Tensor& b, DataType out_dtype) {
    const auto in0_tile = a.tensor_spec().tile();
    const auto in1_tile = b.tensor_spec().tile();
    const auto out_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    const auto swiglu_format = format_of(out_dtype);
    const auto out_format = plan.swiglu ? tt::DataFormat::Float16_b : swiglu_format;
    std::uint64_t out_pair = std::uint64_t{plan.out_entries} * out_tile.get_tile_size(out_format);
    if (!plan.share_out_interm) {
        out_pair += std::uint64_t{plan.interm0_entries} *
                    out_tile.get_tile_size(interm_format_of(plan.interm0_format, out_format));
    }
    std::uint64_t bytes =
        std::uint64_t{plan.in0_entries} *
            tt::align(in0_tile.get_tile_size(format_of(a.dtype())), tt::tt_metal::hal::get_dram_alignment()) +
        std::uint64_t{plan.in1_entries} * in1_tile.get_tile_size(format_of(b.dtype())) + out_pair;
    if (plan.out_writer_on_in0) {
        bytes += out_pair;
        if (plan.swiglu) {
            bytes += std::uint64_t{plan.swiglu_entries} * out_tile.get_tile_size(swiglu_format);
        }
    }
    return bytes;
}

inline std::uint64_t l1_bytes_minimal(
    const ttnn::experimental::prim::llama31_8b_matmul_bh::MinimalPlan& plan,
    DataType in0_dtype,
    DataType in1_dtype,
    DataType out_dtype,
    bool fp32_dest_acc_en) {
    return std::uint64_t{plan.in0_cb_tiles} * tt::tile_size(format_of(in0_dtype)) +
           std::uint64_t{plan.in1_cb_tiles} * tt::tile_size(format_of(in1_dtype)) +
           std::uint64_t{plan.out_cb_tiles} * tt::tile_size(format_of(out_dtype)) +
           std::uint64_t{plan.interm_cb_tiles} *
               tt::tile_size(fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
}

inline bool l1_fits(const ttnn::Tensor& on, std::uint64_t bytes) {
    return bytes <= ttnn::operations::core::available_program_l1_capacity(on.device());
}

}  // namespace ttnn::prim::llama31_8b_matmul_bh
