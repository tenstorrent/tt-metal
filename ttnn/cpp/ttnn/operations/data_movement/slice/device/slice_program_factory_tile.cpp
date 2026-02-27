// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

// Helper functions used by SliceTileProgramFactory
namespace {

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tile(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim) {
    auto* const input_buffer = input_tensor.buffer();
    auto* const output_buffer = output_tensor.buffer();
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    const auto set_common_reader_args =
        [&](uint32_t* reader_common_args, uint32_t* num_unpadded_tiles_per_dim, uint32_t* num_padded_tiles_per_dim)
            __attribute__((always_inline)) {
                reader_common_args[0] = input_buffer->address();
                num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
                num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
                num_padded_tiles_per_dim[0] = num_padded_Xt;
                num_padded_tiles_per_dim[1] = num_padded_Yt;
                accumulated_total_per_dim[0] = num_total_Xt;
                accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
                for (int32_t i = 2; i < num_dims; ++i) {
                    uint32_t num_unpadded_dim = output_shape[-(i + 1)];
                    uint32_t num_total_dim = input_shape[-(i + 1)];
                    uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
                    num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
                    num_padded_tiles_per_dim[i] = num_padded_dim;
                    accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
                }
            };

    const auto set_reader_rt_args = [&](uint32_t* reader_rt_args,
                                        const uint32_t* num_unpadded_tiles_per_dim,
                                        const uint32_t* /*num_padded_tiles_per_dim*/,
                                        const uint32_t& num_tiles_per_core,
                                        const uint32_t& start_offset,
                                        const uint32_t& num_tiles_written) __attribute__((always_inline)) {
        reader_rt_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_rt_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_rt_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_rt_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_rt_args[0] = start_id;
        reader_rt_args[1] = num_tiles_per_core;
    };

    if constexpr (initialize_args) {
        std::vector<uint32_t> reader_common_args(1 + (num_dims * 2));
        uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
        uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
        SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);
    }
    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    if constexpr (!initialize_args) {
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            if constexpr (initialize_args) {
                std::vector<uint32_t> reader_kernel_args(2 + num_dims, 0);
                std::vector<uint32_t> writer_kernel_args(3, 0);
                tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
            } else {
                auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
                reader_kernel_args[1] = 0;
                auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
                writer_kernel_args[1] = 0;
            }
            continue;
        }

        if constexpr (initialize_args) {
            std::vector<uint32_t> reader_kernel_args(2 + num_dims);
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
        } else {
            auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
        }

        if constexpr (initialize_args) {
            const std::array writer_kernel_args = {output_buffer->address(), num_tiles_per_core, num_tiles_written};
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        } else {
            auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
            writer_kernel_args[1] = num_tiles_per_core;
            writer_kernel_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }
}
}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

// Slice Tile Program Factory implementation
SliceTileProgramFactory::cached_program_t SliceTileProgramFactory::create(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::uint32_t num_dims = static_cast<std::uint32_t>(input.padded_shape().rank());

    // Reader compile-time args (CB index via named arg, positional starts at num_dims)
    std::vector<uint32_t> reader_compile_time_args = {num_dims};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::unordered_map<std::string, uint32_t> reader_named_args = {{"cb_in", src0_cb_index}};

    // Writer compile-time args (CB index via named arg, positional starts at TensorAccessorArgs)
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    std::unordered_map<std::string, uint32_t> writer_named_args = {{"cb_out", src0_cb_index}};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_args));

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    ttnn::operations::data_movement::set_slice_runtime_args_tile<true>(
        input,
        output,
        args.slice_start,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        program,
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        accumulated_total_per_dim);

    return {
        std::move(program),
        {unary_reader_kernel_id,
         unary_writer_kernel_id,
         compute_with_storage_grid_size,
         args.sub_core_grids,
         accumulated_total_per_dim}};
}

void SliceTileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const Tensor& src_tensor = tensor_args.input;
    const Tensor& dst_tensor = output;
    const auto& slice_start = args.slice_start;
    const auto& sub_core_grids = cached_program.shared_variables.sub_core_grids;
    const auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;
    uint32_t num_unpadded_tiles = dst_tensor.physical_volume() / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    ttnn::operations::data_movement::set_slice_runtime_args_tile<false>(
        src_tensor,
        dst_tensor,
        slice_start,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        cached_program.program,
        cached_program.shared_variables.unary_reader_kernel_id,
        cached_program.shared_variables.unary_writer_kernel_id,
        cached_program.shared_variables.accumulated_total_per_dim);
}

tt::tt_metal::ProgramDescriptor SliceTileProgramFactory::create_descriptor(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<CoreRangeSet>& core_range_set) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        core_range_set.has_value() ? tt::tt_metal::split_work_to_cores(core_range_set.value(), num_unpadded_tiles)
        : args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // --- CB Descriptor ---
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    tt::tt_metal::ProgramDescriptor program_descriptor;

    tt::tt_metal::CBDescriptor cb_desc;
    cb_desc.total_size = num_input_tiles * single_tile_size;
    cb_desc.core_ranges = all_cores;
    cb_desc.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(src0_cb_index),
        .data_format = cb_data_format,
        .page_size = single_tile_size});
    program_descriptor.cbs.push_back(std::move(cb_desc));

    // --- Reader Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> reader_compile_time_args = {num_dims};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    // Reader common runtime args: [src_addr, num_unpadded_per_dim..., num_padded_per_dim...]
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> reader_common_args(1 + (num_dims * 2));
    reader_common_args[0] = src0_buffer->address();
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    // Reader per-core runtime args: [start_id, num_tiles, id_per_dim...]
    tt::tt_metal::KernelDescriptor::RuntimeArgs reader_runtime_args;
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            std::vector<uint32_t> reader_args(2 + num_dims, 0);
            reader_runtime_args.emplace_back(core, std::move(reader_args));
            continue;
        }

        std::vector<uint32_t> reader_args(2 + num_dims);
        // Compute per-dim indices for this core's starting position
        reader_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_args[0] = start_id;
        reader_args[1] = num_tiles_per_core;

        reader_runtime_args.emplace_back(core, std::move(reader_args));
        num_tiles_written += num_tiles_per_core;
    }

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.named_compile_time_args = {{"cb_in", src0_cb_index}};
    reader_kernel_desc.runtime_args = std::move(reader_runtime_args);
    reader_kernel_desc.common_runtime_args = std::move(reader_common_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // --- Writer Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Writer per-core runtime args: [dst_addr, num_tiles, start_id]
    tt::tt_metal::KernelDescriptor::RuntimeArgs writer_runtime_args;
    num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            writer_runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0});
            continue;
        }

        writer_runtime_args.emplace_back(
            core, std::vector<uint32_t>{dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = writer_compile_time_args;
    writer_kernel_desc.named_compile_time_args = {{"cb_out", src0_cb_index}};
    writer_kernel_desc.runtime_args = std::move(writer_runtime_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    return program_descriptor;
}

}  // namespace ttnn::prim
