// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.hpp"

namespace ttnn::operations::sliding_window::halo {

struct HaloDeviceOperation {
    thread_local static std::unordered_map<std::size_t, std::uint32_t> sliding_window_max_out_nsticks_per_core;
    using operation_attributes_t = HaloParams;
    using tensor_args_t = HaloInputs;
    using spec_return_value_t = halo::spec_return_value_t;
    using tensor_return_value_t = halo::tensor_return_value_t;
    using program_factory_t = std::variant<data_movement::program::UntilizeWithHaloProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::sliding_window::halo

namespace ttnn::prim {
ttnn::operations::sliding_window::halo::HaloDeviceOperation::tensor_return_value_t halo(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled,
    bool config_tensors_in_dram);
}  // namespace ttnn::prim
