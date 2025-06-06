// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <sstream>
#include <type_traits>

#include "cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace experimental::ccl;
using Tensors = std::vector<Tensor>;

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
tt::tt_metal::operation::ProgramWithCallbacks all_gather_matmul_async_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out

) {
    tt::tt_metal::Program program{};

    ////////////// Params for fused op signalers //////////////
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, all_gather_output_tensor, dim, ring_index);
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.get_padded_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(ttnn::experimental::ccl::MatmulFusedOpSignalerType::ALL_GATHER);
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

    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
                    program,
                    all_gather_output_tensor,
                    weight_tensor,
                    bias,
                    matmul_output_tensor,
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);
                matmul_override_runtime_arguments_callback =
                    matmul_program_with_callbacks->override_runtime_arguments_callback;
            } else {
                TT_THROW("Unsupported MatmulProgramConfig type. Needs to be 1D or 2D Multicast.");
            }
        },
        program_config);

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
        ttnn::all_gather_async_minimal_interleaved_dim3_1_1_any_any_helper(
            matmul_program_with_callbacks->program,
            input_tensor,
            persistent_intermediate_tensor,
            target_device,
            forward_device,
            backward_device,
            all_gather_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            sub_device_id,
            all_gather_fused_op_signaler,
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
                    {output_tensors[1], input_tensors[1]}, /* all gather output tensor, weight tensor */
                    optional_input_tensors,
                    {output_tensors[2]} /* matmul output tensor */
                );
            }

            if (all_gather_override_runtime_arguments_callback.has_value()) {
                all_gather_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {input_tensors[0]}, /* input tensor */
                    optional_input_tensors,
                    {output_tensors[0], output_tensors[1]} /* all gather output tensor */
                );
            }
        };

    program_with_callbacks.override_runtime_arguments_callback = override_runtime_arguments_callback;

    return program_with_callbacks;
}

}  // namespace ttnn
