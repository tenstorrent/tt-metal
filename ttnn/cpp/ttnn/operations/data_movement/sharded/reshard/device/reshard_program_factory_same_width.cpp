// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_align.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::reshard::program {

template <bool local_is_output>
ReshardSameWidthFactory<local_is_output>::cached_program_t ReshardSameWidthFactory<local_is_output>::create(
    const reshard::ReshardParams& /*operation_attributes*/,
    const reshard::ReshardInputs& tensor_args,
    reshard::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& local_tensor = local_is_output ? output : input;
    const auto& remote_tensor = local_is_output ? input : output;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

    auto remote_core_type = remote_tensor.buffer()->core_type();
    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    constexpr uint32_t cb_scratch_index = tt::CBIndex::c_1;
    auto local_cores = corerange_to_cores(
        local_shard_spec.grid, std::nullopt, local_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto remote_cores = corerange_to_cores(
        remote_shard_spec.grid, std::nullopt, remote_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t unit_size, local_units_per_shard, remote_units_per_shard;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());

    uint32_t num_units = local_tensor.buffer()->num_pages();
    if (local_tensor.layout() == Layout::TILE) {
        unit_size = tt::tile_size(data_format);
        local_units_per_shard = local_shard_spec.numel() / TILE_HW;
        remote_units_per_shard = remote_shard_spec.numel() / TILE_HW;
    } else {
        unit_size = local_shard_spec.shape[1] * local_tensor.element_size();
        local_units_per_shard = local_shard_spec.shape[0];
        remote_units_per_shard = remote_shard_spec.shape[0];
    }
    uint32_t local_unit_size_padded = tt::align(unit_size, local_tensor.buffer()->alignment());
    uint32_t remote_unit_size_padded = tt::align(unit_size, remote_tensor.buffer()->alignment());
    bool unaligned = false;
    if (remote_unit_size_padded != unit_size || local_unit_size_padded != unit_size) {
        unaligned = true;
    }
    const uint32_t total_size = std::min(local_units_per_shard, remote_units_per_shard) * unit_size;
    const std::string kernel_name =
        local_is_output
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_writer.cpp";

    bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program,
        kernel_name,
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            {cb_index,
             interface_with_dram,
             unaligned,
             unit_size,
             local_unit_size_padded,
             remote_unit_size_padded,
             cb_scratch_index}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program,
        kernel_name,
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(
            {cb_index,
             interface_with_dram,
             unaligned,
             unit_size,
             local_unit_size_padded,
             remote_unit_size_padded,
             cb_scratch_index}));

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_index, data_format}})
            .set_page_size(cb_index, unit_size)
            .set_globally_allocated_address(*local_tensor.buffer());
    auto cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    if (unaligned) {
        tt::tt_metal::CircularBufferConfig cb_scratch_config =
            tt::tt_metal::CircularBufferConfig(
                remote_units_per_shard * remote_unit_size_padded, {{cb_scratch_index, data_format}})
                .set_page_size(cb_scratch_index, unit_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scratch_config);
    }

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    uint32_t remote_address = remote_tensor.buffer()->address();
    auto remote_buffer_type = remote_tensor.buffer()->buffer_type();
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];

    std::array<tt::tt_metal::KernelHandle, 2> kernels = {kernel_id_0, kernel_id_1};
    uint32_t local_units_left = num_units;
    for (const auto& core : local_cores) {
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, kernels.size());
        uint32_t local_start_offset = 0;
        for (const auto& kernel_id : kernels) {
            std::vector<uint32_t> kernel_args = {remote_address, 0, 0};
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                uint32_t num_transfers = 0;
                kernel_args[1] = local_start_offset;
                local_start_offset += local_units_to_transfer * unit_size;
                while (local_units_to_transfer > 0) {
                    if (remote_core_units_rem == 0) {
                        remote_core_idx++;
                        remote_core_units_rem = remote_units_per_shard;
                        bank_id = device->allocator()->get_bank_ids_from_logical_core(
                            remote_buffer_type, remote_cores[remote_core_idx])[0];
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    kernel_args.insert(
                        kernel_args.end(),
                        {bank_id,
                         (remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded,
                         units_to_transfer});
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    num_transfers++;
                }
                kernel_args[2] = num_transfers;
            }
            SetRuntimeArgs(program, kernel_id, core, kernel_args);
        }
    }
    return {std::move(program), {kernel_id_0, kernel_id_1, cb_0, local_cores}};
}

template <bool is_reader>
void ReshardSameWidthFactory<is_reader>::override_runtime_arguments(
    cached_program_t& cached_program,
    const reshard::ReshardParams& /*operation_attributes*/,
    const reshard::ReshardInputs& tensor_args,
    reshard::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;
    uint32_t remote_addr = remote_tensor.buffer()->address();
    auto& runtime_args_0_by_core = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.kernel_id_0);
    auto& runtime_args_1_by_core = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.kernel_id_1);
    for (auto core : cached_program.shared_variables.local_cores) {
        auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
        auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
        runtime_args_0[0] = remote_addr;
        runtime_args_1[0] = remote_addr;
    }
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.cb_0, *local_tensor.buffer());
}

// Explicit template instantiations
template struct ReshardSameWidthFactory<true>;
template struct ReshardSameWidthFactory<false>;

}  // namespace ttnn::operations::data_movement::reshard::program
