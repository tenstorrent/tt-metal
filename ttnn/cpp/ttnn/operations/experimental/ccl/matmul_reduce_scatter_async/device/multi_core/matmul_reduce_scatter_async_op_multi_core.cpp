// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
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

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace experimental::ccl;
using Tensors = std::vector<Tensor>;

tt::tt_metal::operation::ProgramWithCallbacks matmul_reduce_scatter_async_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_tensor,
    Tensor& reduce_scatter_output_tensor,
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

    // Create the reduce scatter fused op signaler
    std::optional<ReduceScatterFusedOpSignaler> reduce_scatter_fused_op_signaler = ReduceScatterFusedOpSignaler();
    reduce_scatter_fused_op_signaler->init_fused_op();

    // Reduce Scatter
    tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_program_with_callbacks =
        ttnn::reduce_scatter_minimal_async_helper(
            program,
            matmul_output_tensor,
            persistent_intermediate_tensor,
            target_device,
            forward_device,
            backward_device,
            reduce_scatter_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            sub_device_id,
            reduce_scatter_fused_op_signaler,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            core_grid_offset);
    const auto reduce_scatter_override_runtime_arguments_callback =
        reduce_scatter_program_with_callbacks.override_runtime_arguments_callback;

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::REDUCE_SCATTER);
    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);

    // Matmul
    tt::tt_metal::operation::ProgramWithCallbacks matmul_program_with_callbacks =
        operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
            reduce_scatter_program_with_callbacks.program,
            input_tensor,
            weight_tensor,
            bias,
            matmul_output_tensor,
            bcast_batch,
            compute_kernel_config,
            program_config,
            untilize_out,
            matmul_fused_op_signaler);
    const auto matmul_override_runtime_arguments_callback =
        matmul_program_with_callbacks.override_runtime_arguments_callback;

    // Fuse the override runtime arguments callbacks
    auto override_runtime_arguments_callback =
        [reduce_scatter_override_runtime_arguments_callback, matmul_override_runtime_arguments_callback](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            if (matmul_override_runtime_arguments_callback.has_value()) {
                matmul_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {input_tensors[0], input_tensors[1]}, /* input tensor, weight tensor */
                    optional_input_tensors,
                    {output_tensors[0]} /* matmul output tensor */
                );
            }

            if (reduce_scatter_override_runtime_arguments_callback.has_value()) {
                reduce_scatter_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {output_tensors[0]}, /* matmul output tensor */
                    {},
                    {output_tensors[1], output_tensors[2]} /* all gather output tensor */
                );
            }
        };

    matmul_program_with_callbacks.override_runtime_arguments_callback = override_runtime_arguments_callback;

    return matmul_program_with_callbacks;
}

}  // namespace ttnn
