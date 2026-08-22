// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Opt-in TILE-layout last-dim argmax on the pack RISC's RVV (Zve32f) unit —
// Blackhole only, single core. See kernels/argmax_rvv_tile_compute.cpp for
// the algorithm and semantics notes. Unlike the other argmax paths, this one
// launches a compute kernel: the unpack/math threads are no-ops and the pack
// thread does the whole scan, so the dataflow RISC only streams tiles and
// writes results.

#include "argmax_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ArgMaxRvvTileProgramFactory::create_descriptor(
    const ArgmaxParams& /*operation_attributes*/, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const bool has_maxval = tensor_args.optional_maxval_tensor.has_value();

    const auto& padded_shape = input.padded_shape();
    const uint32_t rank = padded_shape.size();
    const uint32_t logical_rank = input.logical_shape().size();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();

    const uint32_t w_tiles = padded_shape[rank - 1] / tile_width;
    const uint32_t h_tiles = padded_shape[rank - 2] / tile_height;
    const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
    const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;
    const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);

    const uint32_t src_page_size = input.tensor_spec().compute_page_size_bytes();
    const uint32_t dst_page_size = output.tensor_spec().compute_page_size_bytes();
    const uint32_t val_page_size =
        has_maxval ? tensor_args.optional_maxval_tensor->mesh_tensor().tensor_spec().compute_page_size_bytes()
                   : dst_page_size;
    const uint32_t out_page_elems = dst_page_size / sizeof(uint32_t);

    // Chunked double-buffered input streaming: the compute-side scan of chunk
    // k overlaps the NOC staging of chunk k+1.
    const uint32_t chunk_pages = std::min<uint32_t>(64, w_tiles);
    const uint32_t in_cb_pages = 2 * chunk_pages;

    ProgramDescriptor desc;
    const CoreCoord core{0, 0};
    const CoreRangeSet all_cores(CoreRange(core, core));

    constexpr auto cb_in = tt::CBIndex::c_0;         // input tiles (ring)
    constexpr auto cb_res_idx = tt::CBIndex::c_1;    // per-pass index results (u32[32])
    constexpr auto cb_res_val = tt::CBIndex::c_2;    // per-pass maxval results (bf16[32])
    constexpr auto cb_stage_idx = tt::CBIndex::c_3;  // output-page staging (indices)
    constexpr auto cb_stage_val = tt::CBIndex::c_4;  // output-page staging (max values)

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pages * src_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in),
            .data_format = in_df,
            .page_size = src_page_size,
        }}},
    });
    constexpr uint32_t res_idx_page = 32 * sizeof(uint32_t);
    constexpr uint32_t res_val_page = 32 * sizeof(uint16_t);
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * res_idx_page,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_res_idx),
            .data_format = tt::DataFormat::UInt32,
            .page_size = res_idx_page,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * res_val_page,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_res_val),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = res_val_page,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = dst_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stage_idx),
            .data_format = tt::DataFormat::UInt32,
            .page_size = dst_page_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = val_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stage_val),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = val_page_size,
        }}},
    });

    // Reader (data-movement RISC): streams tiles, then writes collected results.
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_idx),
        static_cast<uint32_t>(cb_res_val),
        static_cast<uint32_t>(cb_stage_idx),
        static_cast<uint32_t>(cb_stage_val),
        src_page_size,
        chunk_pages,
        in_cb_pages,
        w_tiles,
        h_tiles,
        h_logical,
        outer_dim_units,
        out_page_elems,
        dst_page_size,
        val_page_size,
        static_cast<uint32_t>(has_maxval),
    };
    TensorAccessorArgs(input).append_to(reader_ct_args);
    TensorAccessorArgs(output).append_to(reader_ct_args);
    if (has_maxval) {
        TensorAccessorArgs(tensor_args.optional_maxval_tensor->mesh_tensor()).append_to(reader_ct_args);
    } else {
        // Placeholder so the kernel's third accessor always has args to parse;
        // it is never used when has_maxval == false.
        TensorAccessorArgs(output).append_to(reader_ct_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_rvv_tile.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    {
        KernelDescriptor::RTArgList args;
        args.push_back(input);
        args.push_back(output);
        if (has_maxval) {
            args.push_back(tensor_args.optional_maxval_tensor->mesh_tensor());
        }
        reader_desc.emplace_runtime_args(core, args);
    }
    desc.kernels.push_back(std::move(reader_desc));

    // Compute kernel: unpack/math no-op, pack thread runs the RVV scan.
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_rvv_tile_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_idx),
        static_cast<uint32_t>(cb_res_val),
        chunk_pages,
        w_tiles,
        h_tiles,
        h_logical,
        outer_dim_units,
    };
    // enable_trisc2_rvv: compile this kernel's TRISC2 (pack) TU with the Zve32f extension —
    // the in-tree opt-in that makes the RVV scan compile in a stock build.
    compute_desc.config = ComputeConfigDescriptor{.enable_trisc2_rvv = true};
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
