// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunked_gated_delta_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor ChunkedGatedDeltaDeviceOperation::SingleCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& g_exp = tensor_args.g_exp;
    const auto& factor = tensor_args.factor;
    const auto& bktv = tensor_args.bktv;
    const auto& state = tensor_args.state;
    auto& all_states_tensor = tensor_return_value;

    const uint32_t total_num_heads = operation_attributes.total_num_heads;
    const uint32_t seq_len = operation_attributes.seq_len;
    const uint32_t dim_k = operation_attributes.dim_k;
    const uint32_t dim_v = operation_attributes.dim_v;

    tt::DataFormat data_format = datatype_to_dataformat_converter(g_exp.dtype());
    uint32_t single_tile_size = tile_size(data_format);

    uint32_t num_tiles = g_exp.physical_volume() / constants::TILE_HW;

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    ProgramDescriptor desc;

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t g_cb_index = CBIndex::c_1;
    constexpr uint32_t factor_cb_index = CBIndex::c_2;
    constexpr uint32_t bktv_cb_index = CBIndex::c_3;
    constexpr uint32_t projected_cb_index = CBIndex::c_4;
    constexpr uint32_t output_cb_index = CBIndex::c_5;
    // current_state_cb holds the recurrent state copied back from output_cb by the
    // writer after each (head, seq_idx). Compute reads it as the state source for
    // every seq_idx > 0 (seq_idx == 0 still reads from src0_cb_index, which is
    // the reader-pushed initial state for the head).
    constexpr uint32_t current_state_cb_index = CBIndex::c_6;

    uint32_t current_state_num_tiles = (state.padded_shape()[2] * state.padded_shape()[3]) / constants::TILE_HW;

    log_info(
        tt::LogOp,
        "current_state CB index = {}, tile size = {}, num_tiles = {}",
        src0_cb_index,
        single_tile_size,
        current_state_num_tiles);
    desc.cbs.push_back(CBDescriptor{
        .total_size = current_state_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t g_num_tiles = (g_exp.padded_shape()[2] * g_exp.padded_shape()[3]) / constants::TILE_HW;
    log_info(tt::LogOp, "g CB index = {}, tile size = {}, num_tiles = {}", g_cb_index, single_tile_size, g_num_tiles);
    desc.cbs.push_back(CBDescriptor{
        .total_size = g_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = g_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t factor_num_tiles = (factor.padded_shape()[2] * factor.padded_shape()[3]) / constants::TILE_HW;
    log_info(
        tt::LogOp,
        "factor CB index = {}, tile size = {}, num_tiles = {}",
        factor_cb_index,
        single_tile_size,
        factor_num_tiles);
    desc.cbs.push_back(CBDescriptor{
        .total_size = factor_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = factor_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t bktv_num_tiles = (bktv.padded_shape()[2] * bktv.padded_shape()[3]) / constants::TILE_HW;
    log_info(
        tt::LogOp,
        "bktv CB index = {}, tile size = {}, num_tiles = {}",
        bktv_cb_index,
        single_tile_size,
        bktv_num_tiles);
    desc.cbs.push_back(CBDescriptor{
        .total_size = bktv_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = bktv_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    log_info(
        tt::LogOp,
        "projected CB index = {}, tile size = {}, num_tiles = {}",
        projected_cb_index,
        single_tile_size,
        current_state_num_tiles);
    desc.cbs.push_back(CBDescriptor{
        .total_size = current_state_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = projected_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t output_buffering = 3;
    log_info(
        tt::LogOp,
        "output CB index = {}, tile size = {}, num_tiles = {}",
        output_cb_index,
        single_tile_size,
        current_state_num_tiles * output_buffering);
    desc.cbs.push_back(CBDescriptor{
        .total_size = current_state_num_tiles * single_tile_size * output_buffering,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Recurrent state CB. Producer is the writer (NOC-copies tiles from output_cb
    // into this CB after each non-final seq_idx). Consumer is the compute kernel
    // (reads state for the next seq_idx). Two slots allow the writer to start
    // filling state[i+1] while compute is still processing state[i].
    constexpr uint32_t current_state_buffering = 2;
    log_info(
        tt::LogOp,
        "current_state CB index = {}, tile size = {}, num_tiles = {}",
        current_state_cb_index,
        single_tile_size,
        current_state_num_tiles * current_state_buffering);
    desc.cbs.push_back(CBDescriptor{
        .total_size = current_state_num_tiles * single_tile_size * current_state_buffering,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = current_state_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        g_cb_index, factor_cb_index, bktv_cb_index, src0_cb_index, seq_len, dim_k, dim_v};
    TensorAccessorArgs(*g_exp.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*factor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*bktv.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*state.buffer()).append_to(reader_compile_time_args);

    log_info(tt::LogOp, "reader compile time args = {}", reader_compile_time_args);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/chunked_gated_delta/device/kernels/dataflow/"
        "reader_chunked_gated_delta.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, current_state_cb_index, seq_len, dim_k, dim_v};
    TensorAccessorArgs(*all_states_tensor.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/chunked_gated_delta/device/kernels/dataflow/"
        "writer_chunked_gated_delta.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};
    log_info(tt::LogOp, "writer compile time args = {}", writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index,
        g_cb_index,
        factor_cb_index,
        bktv_cb_index,
        projected_cb_index,
        output_cb_index,
        current_state_cb_index,
        seq_len,
        current_state_num_tiles,
        factor_num_tiles,
        dim_k,
        dim_v};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/chunked_gated_delta/device/kernels/compute/"
        "compute_chunked_gated_delta.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    log_info(tt::LogOp, "compute compile time args = {}", compute_compile_time_args);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Single-core variant: this core owns every head; head_offset = 0.
        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                g_exp.buffer()->address(),
                factor.buffer()->address(),
                bktv.buffer()->address(),
                state.buffer()->address(),
                total_num_heads,
                0u});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{all_states_tensor.buffer()->address(), total_num_heads, 0u});

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{total_num_heads});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
