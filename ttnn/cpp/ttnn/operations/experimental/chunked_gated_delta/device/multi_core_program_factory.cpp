// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core program factory for chunked_gated_delta.
//
// Parallelization strategy
// ------------------------
// Each (head, seq_len) recurrence is independent of every other head, so we
// parallelize purely along the head dimension. ``total_num_heads`` items of
// work are split across the storage-grid cores via ``split_work_to_cores``,
// which yields one or two contiguous head-count groups so the work divides as
// evenly as possible.
//
// Per core we run the same reader/writer/compute kernel binaries as the
// single-core factory. Each core knows its slice of heads through two runtime
// args:
//   - ``num_heads``   : how many heads this core processes
//   - ``head_offset`` : index of this core's first head in the global tensor
//
// The kernels translate ``head_local + head_offset`` into the absolute DRAM
// page IDs, so all on-device CB and recurrence logic is identical to the
// single-core case and is simply replicated per core.

#include "chunked_gated_delta_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor ChunkedGatedDeltaDeviceOperation::MultiCore::create_descriptor(
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

    IDevice* device = g_exp.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // One unit of work == one head. ``split_work_to_cores`` produces two head
    // groups so the head count divides as evenly as possible across the grid.
    auto [num_cores, all_cores, core_group_1, core_group_2, num_heads_per_core_group_1, num_heads_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, total_num_heads);

    log_info(
        tt::LogOp,
        "chunked_gated_delta multi-core: total_num_heads={}, num_cores={}, "
        "group_1: {} heads/core, group_2: {} heads/core",
        total_num_heads,
        num_cores,
        num_heads_per_core_group_1,
        num_heads_per_core_group_2);

    ProgramDescriptor desc;

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t g_cb_index = CBIndex::c_1;
    constexpr uint32_t factor_cb_index = CBIndex::c_2;
    constexpr uint32_t bktv_cb_index = CBIndex::c_3;
    constexpr uint32_t projected_cb_index = CBIndex::c_4;
    constexpr uint32_t output_cb_index = CBIndex::c_5;
    // See single_core_program_factory.cpp for the CB roles. All CBs are sized
    // per-head (one head's worth of tiles) and are allocated on every active
    // core via ``all_cores`` -- each core runs an independent head sub-loop.
    constexpr uint32_t current_state_cb_index = CBIndex::c_6;

    uint32_t current_state_num_tiles = (state.padded_shape()[2] * state.padded_shape()[3]) / constants::TILE_HW;

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
    desc.cbs.push_back(CBDescriptor{
        .total_size = bktv_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = bktv_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

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
    desc.cbs.push_back(CBDescriptor{
        .total_size = current_state_num_tiles * single_tile_size * output_buffering,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t current_state_buffering = 2;
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

    // Walk the cores in row-major order and hand each one a contiguous run of
    // heads. ``head_offset`` is the running prefix sum, so concatenating
    // [head_offset, head_offset + num_heads) across cores reconstructs
    // [0, total_num_heads).
    uint32_t head_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_heads_for_this_core = 0;
        if (core_group_1.contains(core)) {
            num_heads_for_this_core = num_heads_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_heads_for_this_core = num_heads_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                g_exp.buffer()->address(),
                factor.buffer()->address(),
                bktv.buffer()->address(),
                state.buffer()->address(),
                num_heads_for_this_core,
                head_offset});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                all_states_tensor.buffer()->address(), num_heads_for_this_core, head_offset});

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_heads_for_this_core});

        head_offset += num_heads_for_this_core;
    }
    TT_FATAL(
        head_offset == total_num_heads,
        "chunked_gated_delta multi-core: assigned {} heads across {} cores but expected {}",
        head_offset,
        num_cores,
        total_num_heads);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
