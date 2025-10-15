// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/fused/device/materialize_device_operation.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

namespace metal = tt::tt_metal;
namespace fused = ttnn::operations::fused;

template <typename F>
void set_or_update_runtime_arguments(
    metal::Program& program,
    const fused::MaterializeDeviceOperation::ProgramFactory::shared_variables_t& shared_variables,
    const fused::MaterializeDeviceOperation::operation_attributes_t& operation_attributes,
    const fused::MaterializeDeviceOperation::tensor_args_t& tensor_args,
    fused::MaterializeDeviceOperation::tensor_return_value_t& output_tensor,
    F handle_args) {
    const auto num_tiles = output_tensor.physical_volume() / output_tensor.tensor_spec().tile().get_tile_hw();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(operation_attributes.worker_grid, num_tiles);

    const auto set_runtime_args_for =
        [&](const CoreRangeSet& group, uint32_t num_tiles_per_core_group, uint32_t group_start_id = 0) {
            ttsl::SmallVector<std::uint32_t> reader_runtime_args{num_tiles_per_core_group, group_start_id};
            ttsl::SmallVector<std::uint32_t> writer_runtime_args{
                num_tiles_per_core_group, group_start_id, output_tensor.buffer()->address()};
            ttsl::SmallVector<std::uint32_t> compute_runtime_args{num_tiles_per_core_group};

            for (const auto& tensor : tensor_args.input_tensors) {
                reader_runtime_args.push_back(tensor.buffer()->address());
            }

            const std::span params = operation_attributes.params;
            compute_runtime_args.insert(compute_runtime_args.end(), params.begin(), params.end());

            for (const auto& range : group.ranges()) {
                for (const auto& core : range) {
                    handle_args(program, shared_variables.reader_kernel_id, core, reader_runtime_args);
                    handle_args(program, shared_variables.writer_kernel_id, core, writer_runtime_args);
                    handle_args(program, shared_variables.compute_kernel_id, core, compute_runtime_args);
                    group_start_id += num_tiles_per_core_group;
                    constexpr auto group_start_index = 1;
                    reader_runtime_args[group_start_index] = group_start_id;
                    writer_runtime_args[group_start_index] = group_start_id;
                }
            }

            return group_start_id;
        };
    const auto start_id_group_2 = set_runtime_args_for(core_group_1, num_tiles_per_core_group_1);
    set_runtime_args_for(core_group_2, num_tiles_per_core_group_2, start_id_group_2);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::fused {

MaterializeDeviceOperation::ProgramFactory::cached_program_t MaterializeDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    namespace metal = tt::tt_metal;

    constexpr uint32_t tiles_per_cycle = 1;
    const auto& all_device_cores = operation_attributes.worker_grid;
    const auto data_format = metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const auto fp32_dest_acc_en = data_format == tt::DataFormat::Float32 or data_format == tt::DataFormat::Int32 or
                                  data_format == tt::DataFormat::UInt32;
    const auto tile_size = tt::tile_size(data_format);
    auto program = metal::CreateProgram();
    ttsl::SmallVector<metal::CBHandle> cbs;

    for (std::size_t index = 0; index < operation_attributes.circular_buffers; ++index) {
        // TODO handle heterogenous data formats for circular buffers
        cbs.push_back(metal::CreateCircularBuffer(
            program,
            all_device_cores,
            metal::CircularBufferConfig(2 * tile_size, {{tt::CBIndex(index), data_format}})
                .set_page_size(index, tile_size)));
    }

    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> writer_common_runtime_args;
    std::vector<uint32_t> compute_compile_time_args;

    constexpr auto pack_into = [](const metal::distributed::MeshBuffer& buffer,
                                  tt::CBIndex cb_index,
                                  std::vector<uint32_t>& compile_time_args,
                                  std::vector<uint32_t>& common_runtime_args) {
        using enum tensor_accessor::ArgConfig;
        constexpr auto args_config = RuntimeNumBanks | RuntimeTensorShape | RuntimeShardShape | RuntimeBankCoords;
        compile_time_args.push_back(static_cast<uint32_t>(cb_index));
        compile_time_args.push_back(common_runtime_args.size());
        metal::TensorAccessorArgs(buffer, args_config).append_to(compile_time_args, common_runtime_args);
    };

    reader_compile_time_args.push_back(tiles_per_cycle);
    reader_compile_time_args.push_back(operation_attributes.inputs.size());

    for (const auto [cb, tensor] : operation_attributes.inputs) {
        pack_into(
            *tensor_args.input_tensors[tensor].mesh_buffer(), cb, reader_compile_time_args, reader_common_runtime_args);
    }

    writer_compile_time_args.push_back(tiles_per_cycle);
    pack_into(
        *output_tensor.mesh_buffer(),
        operation_attributes.output,
        writer_compile_time_args,
        writer_common_runtime_args);

    compute_compile_time_args.push_back(tiles_per_cycle);

    auto shared_variables = shared_variables_t{
        .reader_kernel_id = metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/fused/device/kernels/dataflow/reader.cpp",
            all_device_cores,
            metal::ReaderDataMovementConfig(std::move(reader_compile_time_args))),
        .writer_kernel_id = metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/fused/device/kernels/dataflow/writer.cpp",
            all_device_cores,
            metal::WriterDataMovementConfig(std::move(writer_compile_time_args))),
        .compute_kernel_id = metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/fused/device/kernels/compute/materialize_no_bcast.cpp",
            all_device_cores,
            metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = {NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default},
                .compile_args = std::move(compute_compile_time_args),
                .defines = {{"COMPUTE_TILES()", operation_attributes.compute_kernel_source}}}),
        .cbs = std::move(cbs),
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        shared_variables,
        operation_attributes,
        tensor_args,
        output_tensor,
        [](Program& program, metal::KernelHandle kernel_id, CoreCoord core, std::span<const std::uint32_t> args) {
            metal::SetRuntimeArgs(program, kernel_id, core, args);
        });

    return {std::move(program), std::move(shared_variables)};
}

void MaterializeDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    namespace metal = tt::tt_metal;

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables,
        operation_attributes,
        tensor_args,
        output_tensor,
        [](Program& program, metal::KernelHandle kernel_id, CoreCoord core, std::span<const std::uint32_t> args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        });
}

}  // namespace ttnn::operations::fused
