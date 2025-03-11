// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_crop_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt-metalium/logger.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

uint32_t calculate_payload_size_for_remote_core(
    uint32_t remote_core_idx,
    uint32_t num_remote_cores,
    uint32_t num_units_per_remote_core_pre_crop,
    uint32_t pre_crop_height_per_core,
    uint32_t pre_crop_width,
    uint32_t crop_w,
    uint32_t crop_h) {
    uint32_t pre_crop_num_units = pre_crop_height_per_core * pre_crop_width;
    uint32_t post_crop_num_units = pre_crop_num_units - crop_w * pre_crop_height_per_core * 2;
    if (remote_core_idx == 0 || remote_core_idx == num_remote_cores - 1) {
        // in these cases, also crop the height
        int crop_rows_count = 1;
        if (remote_core_idx == 0 && remote_core_idx == num_remote_cores - 1) {
            crop_rows_count = 2;
        }
        uint32_t post_crop_width = pre_crop_width - crop_w * 2;
        post_crop_num_units -= post_crop_width * crop_rows_count * crop_h;
    }
    return post_crop_num_units;
}

template <bool is_reader>
operation::ProgramWithCallbacks conv_crop_multi_core_same_width(
    const Tensor& input, Tensor& output, int crop_h, int crop_w, int pre_crop_h, int pre_crop_w) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

    auto local_core_type = local_tensor.buffer()->core_type();
    auto remote_core_type = remote_tensor.buffer()->core_type();
    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    auto local_cores = corerange_to_cores(
        local_shard_spec.grid, std::nullopt, local_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto remote_cores = corerange_to_cores(
        remote_shard_spec.grid, std::nullopt, remote_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t unit_size, local_units_per_shard, remote_units_per_shard;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.get_dtype());

    uint32_t num_units = local_tensor.buffer()->num_pages();
    unit_size = local_shard_spec.shape[1] * local_tensor.element_size();
    local_units_per_shard = local_shard_spec.shape[0];
    remote_units_per_shard = remote_shard_spec.shape[0];
    const uint32_t total_size = std::min(local_units_per_shard, remote_units_per_shard) * unit_size;
    const std::string kernel_name =
        is_reader
            ? "ttnn/cpp/ttnn/operations/data_movement/conv_crop/device/kernels/dataflow/conv_crop_same_width_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/conv_crop/device/kernels/dataflow/"
              "conv_crop_same_width_writer.cpp";

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::ReaderDataMovementConfig({cb_index, crop_w, unit_size}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::WriterDataMovementConfig({cb_index, crop_w, unit_size}));

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_index, data_format}})
            .set_page_size(cb_index, unit_size)
            .set_globally_allocated_address(*local_tensor.buffer());
    auto cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    uint32_t remote_core_idx = 0;
    uint32_t pre_crop_height_per_remote_core = remote_shard_spec.shape[0] / pre_crop_w;
    uint32_t remote_core_units_rem = calculate_payload_size_for_remote_core(
        0, remote_cores.size(), remote_units_per_shard, pre_crop_height_per_remote_core, pre_crop_w, crop_w, crop_h);
    remote_units_per_shard = remote_core_units_rem;
    uint32_t remote_core_num_units_in_uncropped_shard = remote_shard_spec.shape[0];
    log_info(
        tt::LogOp,
        "Pre crop num units for first core: {} post crop num_units for first core: {}",
        remote_shard_spec.shape[0],
        remote_core_units_rem);
    uint32_t remote_address = remote_tensor.buffer()->address();
    auto remote_buffer_type = remote_tensor.buffer()->buffer_type();
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];
    uint32_t bank_offset = device->allocator()->get_bank_offset(remote_buffer_type, bank_id);

    std::array<tt::tt_metal::KernelHandle, 2> kernels = {kernel_id_0, kernel_id_1};
    // std::array<tt::tt_metal::KernelHandle, 1> kernels = {kernel_id_0};
    uint32_t local_units_left = num_units;
    bool visited_first_remote_core = false;
    uint32_t post_crop_w = pre_crop_w - crop_w * 2;

    uint32_t offset_in_remote_core = 0;
    for (int i = 0; i < local_cores.size(); i++) {
        auto core = local_cores[i];
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        // log_info(
        //     tt::LogOp,
        //     "Local units remainig for local core {} are: {} local units left are: {}",
        //     i,
        //     local_units_per_core,
        //     local_units_left);
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
                        offset_in_remote_core = 0;
                        remote_core_units_rem = calculate_payload_size_for_remote_core(
                            remote_core_idx,
                            remote_cores.size(),
                            remote_units_per_shard,
                            pre_crop_height_per_remote_core,
                            pre_crop_w,
                            crop_w,
                            crop_h);
                        // log_info(
                        //     tt::LogOp,
                        //     "Remaining units post crop for remote core {} are: {}",
                        //     remote_core_idx,
                        //     remote_core_units_rem);
                        bank_id = device->allocator()->get_bank_ids_from_logical_core(
                            remote_buffer_type, remote_cores[remote_core_idx])[0];
                        bank_offset = device->allocator()->get_bank_offset(remote_buffer_type, bank_id);
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    if (remote_core_idx == 0 && visited_first_remote_core == false) {
                        visited_first_remote_core = true;
                        // Skip first row
                        offset_in_remote_core += pre_crop_w * crop_h;
                    }

                    uint32_t transfered_rows = units_to_transfer / post_crop_w;
                    uint32_t units_to_transfer_per_row = post_crop_w;
                    kernel_args.insert(
                        kernel_args.end(),
                        {bank_id,
                         offset_in_remote_core * unit_size,
                         units_to_transfer_per_row * unit_size,
                         transfered_rows});
                    // log_info(
                    //     tt::LogOp,
                    //     "Offset in remote core bare: {} multiplied with unit size: {}",
                    //     offset_in_remote_core,
                    //     offset_in_remote_core * unit_size);
                    // We want to transfer units_to_transfer post-croped units, convert that to pre-croped units to
                    // calculate offset. Number of rows is the same
                    offset_in_remote_core += transfered_rows * pre_crop_w;
                    // log_info(tt::LogOp, "Units to transfer: {}", units_to_transfer);

                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    num_transfers++;
                }
                kernel_args[2] = num_transfers;
            }
            // log_info(tt::LogOp, "Kernel args for core: {} kernel are: {}", core, kernel_args);
            SetRuntimeArgs(program, kernel_id, core, kernel_args);
        }
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
        uint32_t remote_addr = remote_tensor.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : local_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = remote_addr;
            runtime_args_1[0] = remote_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_0, *local_tensor.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks conv_crop_multi_core(
    const Tensor& input, Tensor& output, int crop_height, int crop_width, int pre_crop_height, int pre_crop_width) {
    return conv_crop_multi_core_same_width<true>(
        input, output, crop_height, crop_width, pre_crop_height, pre_crop_width);
}

}  // namespace ttnn::operations::data_movement::detail
