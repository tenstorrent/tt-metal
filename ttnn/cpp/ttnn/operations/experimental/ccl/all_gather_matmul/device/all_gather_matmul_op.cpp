// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "cpp/ttnn/operations/matmul/matmul.hpp"

namespace ttnn {
namespace experimental {

void AllGatherMatmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 4,
        "AllGatherMatmul requires 4 input tensors: [input, weight, all_gather_output, datacopy_output]");

    auto& input_tensor = input_tensors[0];
    auto& all_gather_output_tensor = input_tensors[1];
    auto& weight_tensor = input_tensors[2];

    // All Gather validate
    this->all_gather_struct.validate({input_tensor});

    // Matmul validate.
    this->matmul_struct.validate({all_gather_output_tensor, weight_tensor}, optional_input_tensors, {});

    // All Gather Matmul validate
    TT_FATAL(this->all_gather_struct.dim == 3, "AllGatherMatmul requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        input_tensor.get_padded_shape()[0] == 1 && input_tensor.get_padded_shape()[1] == 1,
        "AllGatherMatmul requires input tensor to have batch size of 1.");
    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::
                        is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                TT_THROW("Unsupported MatmulProgramConfig type for AllGatherMatmul. Needs to be 1D or 2D Multicast.");
            }
        },
        this->matmul_struct.program_config.value());

    const auto& all_gather_output_tensor_shard_spec = all_gather_output_tensor.shard_spec();
    if (all_gather_output_tensor_shard_spec.has_value()) {
        auto const& shard_grid = all_gather_output_tensor_shard_spec->grid.bounding_box();
        auto const& shard_grid_start = shard_grid.start_coord;
        auto const& shard_grid_end = shard_grid.end_coord;
        const uint32_t num_all_gather_output_shards = shard_builder::get_sharding_core_count(all_gather_output_tensor);
        TT_FATAL(
            this->all_gather_struct.ring_size == num_all_gather_output_shards,
            "AllGatherMatmul requires number of tensor slices to equal the number of output shards of the all_gather.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherMatmul::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // All Gather shape
    ttnn::TensorSpec all_gather_output_shape = this->all_gather_struct.compute_output_specs({input_tensors[0]})[0];
    ttnn::TensorSpec datacopy_output_shape = all_gather_output_shape;

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[1], input_tensors[2]}, {})[0];

    return {all_gather_output_shape, matmul_output_specs, datacopy_output_shape};
}

std::vector<Tensor> AllGatherMatmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // All Gather output tensor
    auto& all_gather_output_tensor =
        input_tensors[1];  // this->all_gather_out_tensor =
                           // this->all_gather_struct.create_output_tensors(input_tensors)[0];
    auto& datacopy_output_tensor =
        input_tensors[3];  // this->all_gather_out_tensor =
                           // this->all_gather_struct.create_output_tensors(input_tensors)[0];

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({input_tensors[1], input_tensors[2]})[0];

    return {all_gather_output_tensor, matmul_output_tensor, datacopy_output_tensor};
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherMatmul::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].mesh_device();
    auto [device_index, sender_device_id, receiver_device_id] = ::ttnn::ccl::get_device_index_and_sender_receiver_ids(
        mesh_device->get_device(mesh_coord), this->devices, this->all_gather_struct.topology);
    chip_id_t target_device_id = mesh_device->get_device(mesh_coord)->id();
    // Return the AllGatherMatmul program with callbacks
    return all_gather_matmul_multi_core_with_workers(
        input_tensors[0],   // input_tensor
        output_tensors[0],  // all_gather_output_tensor
        output_tensors[2],  // datacopy_output_tensor
        input_tensors[2],   // weight_tensor
        output_tensors[1],  // matmul_output_tensor

        /* All Gather Params */
        this->all_gather_struct.dim,
        this->all_gather_struct.num_links,
        this->all_gather_struct.ring_size,
        device_index,
        this->all_gather_struct.user_defined_num_workers,
        this->all_gather_struct.user_defined_num_buffers_per_channel,
        target_device_id,
        receiver_device_id,
        sender_device_id,
        this->all_gather_struct.topology,
        this->all_gather_core_grid_offset,

        /* Matmul Params */
        {},  // Bias
        this->matmul_struct.bcast_batch.value(),
        this->matmul_struct.compute_kernel_config.value(),
        this->matmul_struct.program_config.value(),
        this->matmul_struct.untilize_out);
}

}  // namespace experimental

namespace operations {
namespace experimental {
namespace ccl {

std::vector<ttnn::Tensor> all_gather_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const CoreCoord all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::optional<MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "AllGatherMatmul is only supported for Fast Dispatch");

    std::vector<std::optional<const ttnn::Tensor>> optional_input_tensors = {std::nullopt};
    auto mesh_device = input_tensor.mesh_device();
    std::vector<IDevice*> devices = {};
    for (const auto& spec : input_tensor.device_storage().specs) {
        devices.push_back(mesh_device->get_device(spec.first));
    }

    /* AllGather setup */
    ttnn::AllGather all_gather_struct{
        dim,
        num_links,
        devices.size(),
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        memory_config_ag.value_or(input_tensor.memory_config()),
        ttnn::ccl::Topology::Ring,
        /*cluster_axis=*/std::nullopt};

    // Create the all gather output tensor used as input (activation) to the matmul
    ttnn::Tensor all_gather_out_tensor = all_gather_struct.create_output_tensors({input_tensor})[0];
    ttnn::Tensor datacopy_out_tensor = all_gather_struct.create_output_tensors({input_tensor})[0];

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
        ttnn::experimental::AllGatherMatmul{/* All Gather Params */
                                            all_gather_struct,
                                            /* Matmul params */
                                            matmul_struct,
                                            /* Fusion params */
                                            all_gather_core_grid_offset,
                                            std::move(devices)},
        {input_tensor, all_gather_out_tensor, weight_tensor, datacopy_out_tensor},
        optional_input_tensors);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
