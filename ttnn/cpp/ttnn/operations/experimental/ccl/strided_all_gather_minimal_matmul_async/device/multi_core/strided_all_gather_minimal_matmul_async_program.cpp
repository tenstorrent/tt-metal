// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <sstream>
#include <type_traits>

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace experimental::ccl;
using Tensors = std::vector<Tensor>;

tt::tt_metal::operation::ProgramWithCallbacks strided_all_gather_minimal_matmul_async_program(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    IDevice* target_device,
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor> bias,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    operations::experimental::minimal_matmul::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt::tt_metal::Program program{};

    ////////////// Params for fused op signalers //////////////
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, all_gather_output_tensor, dim, ring_index);
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.padded_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::STRIDED_ALL_GATHER);
    matmul_fused_op_signaler->init_all_gather(
        num_transfers,
        ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        is_clockwise_direction,
        tensor_slicer.num_cols *
            weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */
    );

    // Matmul
    std::optional<tt::tt_metal::operation::ProgramWithCallbacks> matmul_program_with_callbacks;
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<Tensors>>
        matmul_override_runtime_arguments_callback;

    matmul_program_with_callbacks = operations::experimental::minimal_matmul::detail::minimal_matmul_factory_helper(
        program,
        all_gather_output_tensor,
        weight_tensor,
        bias,
        fused_activation,
        config,
        matmul_output_tensor,
        compute_kernel_config,
        matmul_fused_op_signaler);
    matmul_override_runtime_arguments_callback = matmul_program_with_callbacks->override_runtime_arguments_callback;

    if (!matmul_program_with_callbacks.has_value()) {
        TT_THROW("Matmul program with callbacks not created");
    }

    // Create the all gather fused op signaler
    std::optional<AllGatherFusedOpSignaler> all_gather_fused_op_signaler = AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    tt::tt_metal::operation::ProgramWithCallbacks program_with_callbacks =
        ttnn::strided_all_gather_async_minimal_default_helper(
            matmul_program_with_callbacks->program,
            input_tensor,
            target_device_coord,
            forward_coord,
            backward_coord,
            all_gather_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            barrier_semaphore,
            sub_device_id,
            all_gather_fused_op_signaler,
            std::nullopt,
            num_workers_per_direction_opt,
            num_buffers_per_channel,
            config.compute_with_storage_grid_size.y,
            config.M_block_size,
            config.K_block_size,
            core_grid_offset);
    const auto all_gather_override_runtime_arguments_callback =
        program_with_callbacks.override_runtime_arguments_callback;

    // Fuse the override runtime arguments callbacks
    auto override_runtime_arguments_callback =
        [all_gather_override_runtime_arguments_callback, matmul_override_runtime_arguments_callback](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            if (matmul_override_runtime_arguments_callback.has_value()) {
                matmul_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {output_tensors[0], input_tensors[1]}, /* all gather output tensor, weight tensor */
                    optional_input_tensors,
                    {output_tensors[1]} /* matmul output tensor */
                );
            }

            if (all_gather_override_runtime_arguments_callback.has_value()) {
                all_gather_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {input_tensors[0]}, /* input tensor */
                    optional_input_tensors,
                    {output_tensors[0]} /* all gather output tensor */
                );
            }
        };

    program_with_callbacks.override_runtime_arguments_callback = override_runtime_arguments_callback;

    return program_with_callbacks;
}

}  // namespace ttnn
