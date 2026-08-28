// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

// The shared reduce-scatter factory sizes its pages with the two-argument tile_size(), which assumes
// the architectural 32x32 tile, and derives the per-packet page counts from it. Keying page_config
// stops a non-standard tile from aliasing onto a cached program, but the program it would compile for
// itself is still mis-sized, so reject it outright. Called from both validators because the op defines
// its own cache-hit validator, which would otherwise drop the check on hits.
void validate_standard_tile(const Matmul_RS::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.rs.input_tensor;
    if (input.layout() != Layout::TILE) {
        return;
    }
    const auto tile = input.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "llama_rs_matmul requires a standard 32x32 tile on the reduce-scatter input, got {}x{}",
        tile.get_height(),
        tile.get_width());
}

}  // namespace

void Matmul_RS::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.second_weight_tensor.has_value()) {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor,
              tensor_args.matmul.weight_tensor,
              tensor_args.second_weight_tensor.value()},
             {std::nullopt},
             {}});
    } else {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {std::nullopt}, {}});
    }
    LlamaReduceScatterDeviceOperation::validate_on_program_cache_hit(operation_attributes.rs_op, tensor_args.rs);
    validate_standard_tile(tensor_args);
}

void Matmul_RS::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.second_weight_tensor.has_value()) {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor,
              tensor_args.matmul.weight_tensor,
              tensor_args.second_weight_tensor.value()},
             {std::nullopt},
             {}});
    } else {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {std::nullopt}, {}});
    }
    LlamaReduceScatterDeviceOperation::validate_on_program_cache_miss(operation_attributes.rs_op, tensor_args.rs);
    validate_standard_tile(tensor_args);
}

Matmul_RS::spec_return_value_t Matmul_RS::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Reduce Scatter shape
    tt::tt_metal::TensorSpec reduce_scatter_output_spec =
        LlamaReduceScatterDeviceOperation::compute_output_specs(operation_attributes.rs_op, tensor_args.rs);
    // Matmul shape
    if (tensor_args.second_weight_tensor.has_value()) {
        auto matmul_output_specs = operation_attributes_t::matmul_device_t::compute_output_specs(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor,
              tensor_args.matmul.weight_tensor,
              tensor_args.second_weight_tensor.value()},
             {}});
        return {matmul_output_specs.at(0), matmul_output_specs.at(1), reduce_scatter_output_spec};
    }
    tt::tt_metal::TensorSpec matmul_output_specs = operation_attributes_t::matmul_device_t::compute_output_specs(
        operation_attributes.matmul, {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {}})[0];
    return {matmul_output_specs, reduce_scatter_output_spec};
}

Matmul_RS::tensor_return_value_t Matmul_RS::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    Tensor rs_output_tensor =
        LlamaReduceScatterDeviceOperation::create_output_tensors(operation_attributes.rs_op, tensor_args.rs);
    if (tensor_args.second_weight_tensor.has_value()) {
        return {tensor_args.matmul_output_tensors.at(0), tensor_args.matmul_output_tensors.at(1), rs_output_tensor};
    }
    Tensor matmul_output_tensor = operation_attributes_t::matmul_device_t::create_output_tensors(
        operation_attributes.matmul, {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {}})[0];
    return {matmul_output_tensor, rs_output_tensor};
}

ttsl::hash::hash_t Matmul_RS::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // The matmul operands need their full specs, not just page configs: the helper derives M/K/N, every
    // CB format and the work split from the real shapes and dtypes. MatmulParams pins only the
    // caller-supplied blocking, because the fused launcher routes through create_matmul_attributes,
    // which (unlike matmul()) never normalizes a nullopt program_config via get_program_config. So two
    // calls that differ only in weight width or dtype would otherwise share a key.
    const auto& mm_in = tensor_args.matmul.input_tensor;
    const auto& mm_w = tensor_args.matmul.weight_tensor;
    // Both halves are tile-aware: the shared reduce-scatter factory derives
    // input_tiles_per_core_width from the input's real tile, and the matmul helper reads in0/in1
    // tiles off its operands to size every block and circular buffer. Keying on each page config
    // keeps tensors that differ only in tile geometry on separate programs.
    // The fused matmul half also compiles from MatmulParams (program_config, compute kernel
    // config, fused activation, output dtype/tile, untilize_out, transpose, global_cb); none of
    // those can be refreshed on a cache hit, so the whole struct must be keyed.
    // second_weight_tensor's presence selects between two structurally different programs: a
    // two-output matmul driven by a MatmulFusedOpSignaler versus a single-output matmul with none,
    // which also changes the output arity that override_runtime_arguments indexes into. The launcher
    // requires exactly one of rs_tensor / second_weight_tensor, so both modes are reachable in one
    // process and the discriminator has to be part of the key.
    return tt::tt_metal::operation::hash_operation<Matmul_RS>(
        tensor_args.second_weight_tensor.has_value(),
        operation_attributes.rs_op.dim,
        operation_attributes.rs_op.cluster_axis,
        operation_attributes.rs_op.ring_devices,
        operation_attributes.rs_op.num_links,
        operation_attributes.rs_op.topology,
        operation_attributes.rs_op.use_noc1_only,
        operation_attributes.matmul,
        // output_mem_config supplies the output shard grid behind the OUTPUT_CORE_XY define and the
        // output/accumulator CB sizes; subdevice_id selects the matmul core pool and mcast origin.
        operation_attributes.rs_op.output_mem_config,
        operation_attributes.rs_op.subdevice_id.has_value()
            ? static_cast<uint32_t>(operation_attributes.rs_op.subdevice_id->get())
            : 0xFFFFFFFFu,
        tensor_args.rs.input_tensor.dtype(),
        tensor_args.rs.input_tensor.memory_config(),
        tensor_args.rs.input_tensor.device()->id(),
        tensor_args.rs.input_tensor.tensor_spec().page_config(),
        // The shared reduce-scatter factory derives ncores_input and the SCHEDULE define from the
        // logical shape, which the hashed memory config does not determine for an over-provisioned
        // shard grid — a configuration this op explicitly supports.
        tensor_args.rs.input_tensor.logical_shape(),
        // The packet buffer's shard grid picks the packet-worker cores and, via restricted_cores, the
        // matmul placement too, so its memory config is structural and not just its page config.
        tensor_args.rs.intermediate_packet_buffer.tensor_spec().page_config(),
        tensor_args.rs.intermediate_packet_buffer.memory_config(),
        mm_in.tensor_spec().page_config(),
        mm_in.padded_shape(),
        mm_in.dtype(),
        mm_in.memory_config(),
        mm_w.tensor_spec().page_config(),
        mm_w.padded_shape(),
        mm_w.dtype(),
        mm_w.memory_config());
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::Matmul_RS::tensor_return_value_t llama_rs_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& rs_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& semaphore,
    uint32_t cluster_axis,
    uint32_t ring_devices,
    uint32_t num_links,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    std::optional<const ttnn::CoreGrid> core_grid,
    bool transpose_a,
    bool transpose_b,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<Tensor>& /*optional_output_tensor*/,
    tt::tt_fabric::Topology topology,
    bool use_noc1_only,
    const std::optional<const ttnn::Tensor>& second_weight_tensor) {
    using OperationType = ttnn::operations::experimental::ccl::Matmul_RS;

    TT_FATAL(
        rs_tensor.has_value() ^ second_weight_tensor.has_value(),
        "Exactly one of rs_tensor or second_weight_tensor must have a value");
    ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation rs_struct{};
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    bool user_run_batched = ttnn::operations::matmul::utilities::is_input_batched(weight_tensor.logical_shape());
    auto matmul_struct = ttnn::prim::create_matmul_attributes(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        {program_config,
         /*bcast_batch=*/std::nullopt,
         memory_config_mm.value_or(input_tensor.memory_config()),
         dtype.value_or(input_tensor.dtype()),
         compute_kernel_config,
         /*untilize_out=*/false,
         user_core_coord,
         ttnn::operations::matmul::utilities::get_fused_activation(activation),
         user_run_batched,
         transpose_a,
         transpose_b,
         output_tile,
         global_cb},
        {});

    std::vector<Tensor> matmul_output_tensors;
    std::optional<const ttnn::Tensor> second_weight_tensor_arg = second_weight_tensor;
    Tensor new_rs_tensor;

    if (second_weight_tensor.has_value()) {
        matmul_output_tensors =
            ttnn::operations::experimental::ccl::Matmul_RS::operation_attributes_t::matmul_device_t::
                create_output_tensors(matmul_struct, {{input_tensor, weight_tensor, second_weight_tensor.value()}});
        new_rs_tensor = matmul_output_tensors.at(0);
    } else {
        new_rs_tensor = rs_tensor.value();
    }

    auto rs_op_attr = ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation::operation_attributes_t{
        .dim = (dim < 0 ? uint32_t(new_rs_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
        .cross_device_semaphore = semaphore,
        .subdevice_id = subdevice_id,
        .cluster_axis = cluster_axis,
        .output_mem_config = memory_config_rs,
        .ring_devices = ring_devices,
        .num_links = num_links,
        .topology = topology,
        .use_noc1_only = use_noc1_only};

    auto rs_tensor_args = ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation::tensor_args_t{
        new_rs_tensor, intermediate_packet_buffer};

    auto operation_attributes = OperationType::operation_attributes_t{
        .rs = rs_struct, .rs_op = std::move(rs_op_attr), .matmul = std::move(matmul_struct)};

    auto tensor_args = OperationType::tensor_args_t{
        .rs = std::move(rs_tensor_args),
        .matmul = {.input_tensor = input_tensor, .weight_tensor = weight_tensor},
        .matmul_output_tensors = std::move(matmul_output_tensors),
        .second_weight_tensor = second_weight_tensor_arg};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
