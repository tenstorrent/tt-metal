// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_same_height_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

template <bool is_reader>
operation::ProgramWithCallbacks reshard_multi_core_same_height(const Tensor& input, Tensor& output) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

    const auto local_core_type = local_tensor.buffer()->core_type();
    const auto remote_core_type = remote_tensor.buffer()->core_type();
    bool interface_with_dram = (remote_core_type == CoreType::DRAM);
    const auto local_cores = corerange_to_cores(
        local_shard_spec.grid, std::nullopt, local_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    const auto remote_cores = corerange_to_cores(
        remote_shard_spec.grid, std::nullopt, remote_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());
    const uint32_t element_size = tt::datum_size(data_format);

    TT_FATAL(local_tensor.layout() == Layout::ROW_MAJOR, "Expected row major tensor");
    const uint32_t unit_size = local_shard_spec.shape[1] * local_tensor.element_size();  // width * element size
    const uint32_t local_units_per_shard = local_shard_spec.shape[0];                    // height
    const uint32_t remote_units_per_shard = remote_shard_spec.shape[0];                  // height
    const uint32_t total_size = remote_units_per_shard * unit_size;

    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_index, data_format}})
            .set_page_size(cb_index, unit_size)
            .set_globally_allocated_address(*local_tensor.buffer());
    auto cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    const std::string kernel_name =
        is_reader
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_writer.cpp";

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::ReaderDataMovementConfig({cb_index, interface_with_dram}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::WriterDataMovementConfig({cb_index, interface_with_dram}));

    uint32_t remote_address = remote_tensor.buffer()->address();
    auto remote_buffer_type = remote_tensor.buffer()->buffer_type();

    // Generate all read/write offsets for each core
    auto [runtime_args_for_each_core, total_num_sticks, local_stride_bytes, remote_stride_bytes] =
        compute_width_sharding_reshard_segments(
            local_shard_spec.shape,
            remote_shard_spec.shape,
            local_cores,
            remote_cores,
            remote_buffer_type,
            remote_core_type,
            device,
            element_size);  // local_core_idx -> runtime args[]

    // Split work across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = total_num_sticks / 2;
    const uint32_t total_num_sticks_kernel_1 = total_num_sticks - total_num_sticks_kernel_0;

    // Here all we do is convert pre-computed offsets into vectors so they can be passed as runtime arguments
    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
        std::vector<uint32_t> runtime_args_0 = {
            total_num_sticks_kernel_0,
            local_stride_bytes,
            remote_stride_bytes,
            remote_address,
            args_for_all_segments.size()};
        std::vector<uint32_t> runtime_args_1 = {
            total_num_sticks_kernel_1,
            local_stride_bytes,
            remote_stride_bytes,
            remote_address,
            args_for_all_segments.size()};
        for (const auto& args : args_for_all_segments) {
            const std::vector<uint32_t> segment_kernel_0 = {
                args.write_size, args.read_offset, args.bank_id, args.write_offset};
            runtime_args_0.insert(runtime_args_0.end(), segment_kernel_0.begin(), segment_kernel_0.end());

            // Adjust read and write offsets to the correct stick address because we are splitting work across 2 kernels
            const uint32_t adjusted_read_offset = args.read_offset + total_num_sticks_kernel_0 * local_stride_bytes;
            const uint32_t adjusted_write_offset = args.write_offset + total_num_sticks_kernel_0 * remote_stride_bytes;

            const std::vector<uint32_t> segment_kernel_1 = {
                args.write_size, adjusted_read_offset, args.bank_id, adjusted_write_offset};
            runtime_args_1.insert(runtime_args_1.end(), segment_kernel_1.begin(), segment_kernel_1.end());
        }
        SetRuntimeArgs(program, kernel_id_0, local_cores[core_idx], runtime_args_0);
        SetRuntimeArgs(program, kernel_id_1, local_cores[core_idx], runtime_args_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_0, local_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        const auto& local_tensor = is_reader ? output : input;
        const auto& remote_tensor = is_reader ? input : output;
        uint32_t remote_address = remote_tensor.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : local_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[3] = remote_address;
            runtime_args_1[3] = remote_address;
        }
        UpdateDynamicCircularBufferAddress(program, cb_0, *local_tensor.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// Explicit template instantiations
template operation::ProgramWithCallbacks reshard_multi_core_same_height<true>(const Tensor& input, Tensor& output);
template operation::ProgramWithCallbacks reshard_multi_core_same_height<false>(const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
