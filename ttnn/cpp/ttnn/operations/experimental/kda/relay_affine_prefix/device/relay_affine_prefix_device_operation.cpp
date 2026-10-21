// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

void RelayAffinePrefixOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view name = "relay_affine_prefix";
    using tt::tt_metal::DataType;
    kda_factory_detail::check_dtype(in.packed, DataType::BFLOAT16, name, "packed");
    kda_factory_detail::check_dtype(in.initial_state, DataType::FLOAT32, name, "initial_state");
    kda_factory_detail::check_actual_start(in.packed, in.actual_start, name);
    const auto& state = in.initial_state.logical_shape();
    TT_FATAL(state.rank() == 3 && state[1] == state[2], "{}: initial_state must be [B*H, K, K]", name);
    const uint32_t heads = state[0];
    const auto* mesh = in.packed.device();
    TT_FATAL(mesh->shape()[attrs.sequence_parallel_axis] > 2, "{}: needs a ring of more than two SP ranks", name);
    TT_FATAL(
        attrs.num_links > 0 && attrs.cores.size() >= heads + attrs.num_links,
        "{}: needs a core per head, a sender per link",
        name);
    TT_FATAL(
        attrs.semaphore_addresses[0] != 0 && attrs.semaphore_addresses[1] != 0,
        "{}: needs two global semaphores",
        name);
    TT_FATAL(
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= tt::constants::TILE_HW * sizeof(float),
        "{}: the relay sends whole FP32 tiles",
        name);
    for (const auto* landing : {&in.entry_landing, &in.state_staging, &in.final_landing}) {
        const auto& memory = landing->memory_config();
        TT_FATAL(
            landing->dtype() == DataType::FLOAT32 && memory.buffer_type() == tt::tt_metal::BufferType::L1 &&
                memory.shard_spec().has_value() && memory.shard_spec()->grid.num_cores() == heads &&
                memory.shard_spec()->shape[0] == state[1],
            "{}: the landing buffers must be FP32 L1, one {} x {} state per head core",
            name,
            state[1],
            state[1]);
    }
}

RelayAffinePrefixOperation::spec_return_value_t RelayAffinePrefixOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto spec = [&](const tt::tt_metal::MemoryConfig& memory) {
        return tt::tt_metal::TensorSpec(
            in.initial_state.logical_shape(),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory));
    };
    return {spec(attrs.entry_mem_config), spec(ttnn::DRAM_MEMORY_CONFIG)};
}

RelayAffinePrefixOperation::tensor_return_value_t RelayAffinePrefixOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.packed.device()));
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim
