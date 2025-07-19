// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn {
namespace ccl {
namespace matmul_reduce_scatter_async_detail {

MatmulReduceScatterAsync create_matmul_reduce_scatter_async_struct(
    const ttnn::ReduceScatterMinimalAsync& reduce_scatter_minimal_struct_input,
    const operations::matmul::Matmul& matmul_struct_input,
    const CoreCoord reduce_scatter_core_grid_offset,
    const std::vector<IDevice*>& devices) {
    return ttnn::MatmulReduceScatterAsync{
        reduce_scatter_minimal_struct_input, matmul_struct_input, reduce_scatter_core_grid_offset, std::move(devices)};
}

}  // namespace matmul_reduce_scatter_async_detail
}  // namespace ccl

void MatmulReduceScatterAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "MatmulReduceScatterAsync requires 2 input tensors: [input, weight]");
    auto& input_tensor = input_tensors[0];
    auto& weight_tensor = input_tensors[1];
    auto& intermediate_tensor = output_tensors.at(0).value();
    auto& reduce_scatter_output_tensor = output_tensors.at(1).value();
    // // Reduce Scatter validate
    // this->reduce_scatter_minimal_async_struct.validate_with_output_tensors(
    //     {}, {intermediate_tensor, reduce_scatter_output_tensor});
    // Matmul validate
    this->matmul_struct.validate({input_tensor, weight_tensor}, optional_input_tensors, {});

    // Matmul Reduce Scatter validate
    TT_FATAL(
        this->reduce_scatter_minimal_async_struct.dim == 3,
        "MatmulReduceScatterAsync requires dim=3 for the AllGather operaitons.");
    if (this->matmul_struct.program_config.has_value()) {
        std::visit(
            [&](const auto& config) {
                using ProgramConfigType = std::decay_t<decltype(config)>;
                if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                    TT_THROW(
                        "Unsupported MatmulProgramConfig type for MatmulReduceScatterAsync. Needs to be 2D Multicast.");
                }
            },
            this->matmul_struct.program_config.value());
    }
}

std::vector<ttnn::TensorSpec> MatmulReduceScatterAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    // Matmul shape
    ttnn::TensorSpec matmul_output_specs = this->matmul_struct.compute_output_specs(input_tensors, {})[0];

    // Reduce Scatter shape
    ttnn::TensorSpec reduce_scatter_output_specs =
        this->reduce_scatter_minimal_async_struct.compute_output_specs(input_tensors)[0];
    return {reduce_scatter_output_specs, matmul_output_specs};
}

std::vector<Tensor> MatmulReduceScatterAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({input_tensors[0], input_tensors[1]})[0];

    // Reduce Scatter output tensor
    auto& intermediate_tensor = optional_output_tensors.at(0).value();
    auto& reduce_scatter_output_tensor = optional_output_tensors.at(1).value();

    return {matmul_output_tensor, intermediate_tensor, reduce_scatter_output_tensor};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks MatmulReduceScatterAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ttnn::ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks MatmulReduceScatterAsync::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].mesh_device();
    ttnn::ccl::SenderRecieverConfig config = ::ttnn::ccl::get_device_sender_receiver_config(
        mesh_device->get_device(mesh_coord),
        this->reduce_scatter_minimal_async_struct.devices,
        this->reduce_scatter_minimal_async_struct.topology);
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensors[0].device();

    std::vector<IDevice*> devices_to_use = {};
    devices_to_use = this->reduce_scatter_minimal_async_struct.devices;

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->reduce_scatter_minimal_async_struct.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (this->reduce_scatter_minimal_async_struct.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->reduce_scatter_minimal_async_struct.ring_size - 1);
            }
            if (i != this->reduce_scatter_minimal_async_struct.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (this->reduce_scatter_minimal_async_struct.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    // Return the MatmulReduceScatterAsync program with callbacks
    return matmul_reduce_scatter_async_multi_core_with_workers(
        input_tensors[0],   // input_tensor
        output_tensors[1],  // persistent_intermediate_tensor
        output_tensors[2],  // reduce_scatter_output_tensor
        input_tensors[1],   // weight_tensor
        output_tensors[0],  // matmul_output_tensor

        /* Reduce Scatter Params */
        target_device,
        forward_device,
        backward_device,
        this->reduce_scatter_minimal_async_struct.dim,
        this->reduce_scatter_minimal_async_struct.num_links,
        this->reduce_scatter_minimal_async_struct.ring_size,
        device_index,
        this->reduce_scatter_minimal_async_struct.topology,
        this->reduce_scatter_minimal_async_struct.semaphore,
        this->reduce_scatter_minimal_async_struct.sub_device_id,
        this->reduce_scatter_core_grid_offset,

        /* Matmul Params */
        optional_input_tensors[0],  // Bias
        this->matmul_struct.bcast_batch.value(),
        this->matmul_struct.compute_kernel_config.value(),
        this->matmul_struct.program_config.value(),
        this->matmul_struct.untilize_out);
}

tt::tt_metal::operation::Hash MatmulReduceScatterAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    uint32_t semaphore_address = this->reduce_scatter_minimal_async_struct.semaphore.at(0).address();

    return tt::tt_metal::operation::hash_operation<MatmulReduceScatterAsync>(
        this->reduce_scatter_minimal_async_struct.dim,
        this->reduce_scatter_minimal_async_struct.num_links,
        this->reduce_scatter_minimal_async_struct.ring_size,
        this->reduce_scatter_minimal_async_struct.output_mem_config,
        this->reduce_scatter_minimal_async_struct.topology,
        this->reduce_scatter_minimal_async_struct.sub_device_id,
        this->reduce_scatter_core_grid_offset,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        semaphore_address);
}

namespace operations {
namespace experimental {
namespace ccl {

std::vector<ttnn::Tensor> matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_rs,
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
        "MatmulReduceScatterAsync is only supported for Fast Dispatch");

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt});

    std::vector<std::optional<Tensor>> optional_output_tensors = {
        persistent_intermediate_buffer, persistent_output_buffer};

    // Create the matmul output tensor used as input (activation) to the reduce scatter
    ttnn::Tensor matmul_out_tensor =
        matmul_struct.create_output_tensors({input_tensor, weight_tensor}, optional_output_tensors)[0];

    /* ReduceScatter setup */
    ttnn::ReduceScatterMinimalAsync reduce_scatter_minimal_async_struct = ttnn::ReduceScatterMinimalAsync(
        devices,
        dim,
        num_links,
        devices.size(),
        memory_config_rs.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        sub_device_id,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);

    std::vector<ttnn::Tensor> full_output = tt::tt_metal::operation::run(
        ttnn::ccl::matmul_reduce_scatter_async_detail::create_matmul_reduce_scatter_async_struct(
            /* Reduce Scatter Params */
            reduce_scatter_minimal_async_struct,
            /* Matmul params */
            matmul_struct,
            /* Fusion params */
            reduce_scatter_core_grid_offset,
            devices),
        {input_tensor, weight_tensor},
        optional_input_tensors,
        optional_output_tensors);
    return std::vector<ttnn::Tensor>{full_output.at(0), full_output.at(2)};
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
