// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_matmul_async_detail {

AllGatherMatmulAsync create_all_gather_matmul_async_struct(
    const ttnn::AllGatherAsync& all_gather_struct_input,
    const operations::matmul::Matmul& matmul_struct_input,
    const CoreCoord all_gather_core_grid_offset,
    const std::vector<IDevice*>& devices) {
    return ttnn::AllGatherMatmulAsync{
        all_gather_struct_input, matmul_struct_input, all_gather_core_grid_offset, std::move(devices)};
}

}  // namespace all_gather_matmul_async_detail
}  // namespace ccl

void AllGatherMatmulAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "AllGatherMatmulAsync requires 2 input tensors: [input, weight]");
    auto& input_tensor = input_tensors[0];
    auto& weight_tensor = input_tensors[1];
    auto& all_gather_output_tensor = output_tensors.at(0).value();
    // All Gather validate
    this->all_gather_async_struct.validate_with_output_tensors({input_tensor}, {all_gather_output_tensor});
    // Matmul validate.
    this->matmul_struct.validate({all_gather_output_tensor, weight_tensor}, optional_input_tensors, {});
    // All Gather Matmul validate
    TT_FATAL(
        this->all_gather_async_struct.dim == 3, "AllGatherMatmulAsync requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        input_tensor.get_padded_shape()[0] == 1 && input_tensor.get_padded_shape()[1] == 1,
        "AllGatherMatmulAsync requires input tensor to have batch size of 1.");
    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::
                        is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                TT_THROW(
                    "Unsupported MatmulProgramConfig type for AllGatherMatmulAsync. Needs to be 1D or 2D Multicast.");
            }
        },
        this->matmul_struct.program_config.value());

    const auto& all_gather_output_tensor_shard_spec = all_gather_output_tensor.shard_spec();
    if (all_gather_output_tensor_shard_spec.has_value()) {
        const auto& shard_grid = all_gather_output_tensor_shard_spec->grid.bounding_box();
        const auto& shard_grid_start = shard_grid.start_coord;
        const auto& shard_grid_end = shard_grid.end_coord;
        const uint32_t num_all_gather_output_shards = shard_builder::get_sharding_core_count(all_gather_output_tensor);
        TT_FATAL(
            this->all_gather_async_struct.ring_size == num_all_gather_output_shards,
            "AllGatherMatmulAsync requires number of tensor slices to equal the number of output shards of the "
            "all_gather.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherMatmulAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // All Gather shape
    ttnn::TensorSpec all_gather_output_shape =
        this->all_gather_async_struct.compute_output_specs({input_tensors[0]})[0];

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[0], input_tensors[1]}, {})[0];

    return {all_gather_output_shape, matmul_output_specs};
}

std::vector<Tensor> AllGatherMatmulAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    // All Gather output tensor
    auto& all_gather_output_tensor = optional_output_tensors.at(0).value();

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({all_gather_output_tensor, input_tensors[1]})[0];

    return {all_gather_output_tensor, matmul_output_tensor};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherMatmulAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ttnn::ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherMatmulAsync::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].mesh_device();
    ttnn::ccl::SenderRecieverConfig config = ::ttnn::ccl::get_device_sender_receiver_config(
        mesh_device->get_device(mesh_coord),
        this->all_gather_async_struct.devices,
        this->all_gather_async_struct.topology);
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensors[0].device();

    std::vector<IDevice*> devices_to_use = {};
    devices_to_use = this->all_gather_async_struct.devices;

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->all_gather_async_struct.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (this->all_gather_async_struct.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->all_gather_async_struct.ring_size - 1);
            }
            if (i != this->all_gather_async_struct.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (this->all_gather_async_struct.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    // Return the AllGatherMatmulAsync program with callbacks
    return all_gather_matmul_async_multi_core_with_workers(
        input_tensors[0],   // input_tensor
        output_tensors[0],  // all_gather_output_tensor
        input_tensors[1],   // weight_tensor
        output_tensors[1],  // matmul_output_tensor

        /* All Gather Params */
        target_device,
        forward_device,
        backward_device,
        this->all_gather_async_struct.dim,
        this->all_gather_async_struct.num_links,
        this->all_gather_async_struct.ring_size,
        device_index,
        this->all_gather_async_struct.topology,
        this->all_gather_async_struct.semaphore,
        this->all_gather_async_struct.sub_device_id,
        this->all_gather_core_grid_offset,

        /* Matmul Params */
        optional_input_tensors[0],  // Bias
        this->matmul_struct.bcast_batch.value(),
        this->matmul_struct.compute_kernel_config.value(),
        this->matmul_struct.program_config.value(),
        this->matmul_struct.untilize_out);
}

tt::tt_metal::operation::Hash AllGatherMatmulAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    uint32_t semaphore_address = this->all_gather_async_struct.semaphore.at(0).address();

    return tt::tt_metal::operation::hash_operation<AllGatherMatmulAsync>(
        this->all_gather_async_struct.dim,
        this->all_gather_async_struct.num_links,
        this->all_gather_async_struct.ring_size,
        this->all_gather_async_struct.output_mem_config,
        this->all_gather_async_struct.topology,
        this->all_gather_async_struct.sub_device_id,
        this->all_gather_core_grid_offset,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        semaphore_address);
}

namespace operations {
namespace experimental {
namespace ccl {

std::vector<ttnn::Tensor> all_gather_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord all_gather_core_grid_offset,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "AllGatherMatmulAsync is only supported for Fast Dispatch");

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};

    /* AllGather setup */
    ttnn::AllGatherAsync all_gather_async_struct = ttnn::AllGatherAsync(
        devices,
        dim,
        num_links,
        devices.size(),
        memory_config_ag.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        sub_device_id,
        /*cluster_axis=*/std::nullopt,
        false);

    // Create the all gather output tensor used as input (activation) to the matmul
    ttnn::Tensor all_gather_out_tensor =
        all_gather_async_struct.create_output_tensors({input_tensor}, optional_output_tensors)[0];

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.get_logical_shape());
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        all_gather_out_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.get_dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt});

    return tt::tt_metal::operation::run(
        ttnn::ccl::all_gather_matmul_async_detail::create_all_gather_matmul_async_struct(
            /* All Gather Params */
            all_gather_async_struct,
            /* Matmul params */
            matmul_struct,
            /* Fusion params */
            all_gather_core_grid_offset,
            devices),
        {input_tensor, weight_tensor},
        optional_input_tensors,
        optional_output_tensors);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
