// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <cstring>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

// Include RS types
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_ring_program_factory.hpp"

using namespace tt::constants;

// Import the RS program artifacts type
using ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts;

namespace ttnn::experimental::prim {

void MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MinimalMatmulStridedReduceScatterAsyncParams& attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs&,
    std::vector<Tensor>&,
    const std::optional<ttnn::MeshCoordinate>&) {
    // Buffer addresses are bindings. Caller semaphores and fused_ternary_scalar are outside attribute_names.
    constexpr uint32_t kReaderSemaphoreArg = 2;
    constexpr uint32_t kReaderDirectionArg = 3;
    constexpr uint32_t kWriterDirectionSemaphoreArg = 4;
    constexpr uint32_t kWriterBatchSemaphoreArg = 5;
    constexpr uint32_t kWriterBarrierSemaphoreArg = 7;
    constexpr uint32_t kWriterDirectionArg = 8;
    constexpr uint32_t kReduceScalarArg = 3;
    constexpr uint32_t kNumDirections = 2;

    TT_FATAL(
        attributes.semaphore.size() > kNumDirections,
        "strided reduce scatter expects one semaphore per direction plus the batch semaphore");
    const auto batch_semaphore = static_cast<uint32_t>(attributes.semaphore.at(kNumDirections).address());
    const auto barrier_semaphore =
        attributes.barrier_semaphore.has_value() ? static_cast<uint32_t>(attributes.barrier_semaphore->address()) : 0u;

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    for (auto& column : reader_runtime_args) {
        for (auto& args : column) {
            if (args.size() <= kReaderDirectionArg) {
                continue;
            }
            args[kReaderSemaphoreArg] =
                static_cast<uint32_t>(attributes.semaphore.at(args[kReaderDirectionArg]).address());
        }
    }

    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for (auto& column : writer_runtime_args) {
        for (auto& args : column) {
            if (args.size() <= kWriterDirectionArg) {
                continue;
            }
            args[kWriterDirectionSemaphoreArg] =
                static_cast<uint32_t>(attributes.semaphore.at(args[kWriterDirectionArg]).address());
            args[kWriterBatchSemaphoreArg] = batch_semaphore;
            if (attributes.barrier_semaphore.has_value()) {
                args[kWriterBarrierSemaphoreArg] = barrier_semaphore;
            }
        }
    }

    if (attributes.fused_ternary_scalar.has_value()) {
        float scalar_f = attributes.fused_ternary_scalar.value();
        uint32_t scalar_u32 = 0;
        std::memcpy(&scalar_u32, &scalar_f, sizeof(uint32_t));
        auto& reduce_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReduceKernelIdx);
        for (auto& column : reduce_runtime_args) {
            for (auto& args : column) {
                if (args.size() > kReduceScalarArg) {
                    args[kReduceScalarArg] = scalar_u32;
                }
            }
        }
    }
}

struct FusedProgram {
    tt::tt_metal::ProgramDescriptor descriptor;
    StridedReduceScatterProgramArtifacts rs_artifacts;
};

FusedProgram minimal_matmul_strided_reduce_scatter_async_program(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    Tensor& rs_intermediate_tensor,
    Tensor& rs_output_tensor,

    /* Reduce Scatter Params */
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
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord reduce_scatter_core_grid_offset,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    std::optional<uint32_t> mm_window_blocks,
    const std::optional<const Tensor>& mm_credit_counters,

    /* Matmul Params */
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    ttnn::experimental::prim::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config,

    /* Fused addcmul params */
    const std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor1 = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor2 = std::nullopt,

    /* Caller-owned per-MM-core progress counter scratch shared across MMRS programs. */
    const std::optional<const Tensor>& mm_progress_counters = std::nullopt,
    /* Fused concat (concat-free): second in0 source (suffix half of K; input_tensor is the prefix). */
    const std::optional<const Tensor>& mm_optional_input_tensor = std::nullopt) {
    tt::tt_metal::ProgramDescriptor program{};

    // Derive matmul geometry parameters for the RS factory.
    // The matmul factory normally transposes its core grid when M > N, but
    // transposition is disabled when fusing with SRS (the RS iteration structure
    // requires mm_N_block_wt <= slice_Wt, which can be violated when transposing).
    // So we always use the non-transposed layout: M parallelized on y, N on x.
    uint32_t mm_cores_y_val = config.compute_with_storage_grid_size.y;
    uint32_t mm_block_ht_val = config.M_block_size;
    uint32_t mm_block_wt_val = config.N_block_size;

    // Compute mm_N_full_block_wt: total N tiles per core (N parallelized on x)
    uint32_t N_tiles = weight_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_cores_x = config.compute_with_storage_grid_size.x;
    uint32_t padded_N_tiles = tt::round_up(N_tiles, num_cores_x);
    uint32_t mm_N_full_block_wt_val = padded_N_tiles / num_cores_x;

    // With a rolling window the matmul output tensor is only mm_window_blocks M blocks tall, so the
    // RS needs the true height told to it separately. Take it from the activations, which are always
    // full height.
    const std::optional<uint32_t> mm_logical_Ht_val =
        mm_window_blocks.has_value() ? std::optional<uint32_t>(input_tensor.padded_shape()[-2] / TILE_HEIGHT)
                                     : std::nullopt;

    // =========================================================================
    // STEP 1: Create the Reduce Scatter program FIRST
    //
    // The RS factory creates a semaphore on the RS reader cores and records
    // their NOC coordinates. This info is captured in srs_fused_op_signaler,
    // which is then passed to the matmul factory in step 2.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler =
        ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler();
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> empty_rs_fused_op_signaler = std::nullopt;

    auto rs_shared_variables = ::ttnn::build_ring_strided_reduce_scatter_async_program_artifacts(
        program,
        matmul_output_tensor,    // RS input = MM output
        rs_intermediate_tensor,  // RS intermediate (scratch)
        target_device_coord,
        forward_coord,
        backward_coord,
        rs_output_tensor,  // RS output
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        empty_rs_fused_op_signaler,  // RS -> next op signaling (not used)
        srs_fused_op_signaler,       // MM -> RS signaling (populated by RS factory)
        num_workers_per_link,
        num_buffers_per_channel,
        reduce_scatter_core_grid_offset,
        mm_cores_y_val,
        mm_block_ht_val,
        mm_block_wt_val,
        mm_N_full_block_wt_val,
        chunk_width_in_mm_blocks,
        mm_window_blocks,
        mm_logical_Ht_val,
        mm_credit_counters,
        // Phase 2: fuse addcmul at the RS final write step (not in MM kernel)
        fused_ternary_scalar,
        addcmul_input_tensor1,
        addcmul_input_tensor2,
        mm_progress_counters);

    // =========================================================================
    // STEP 2: Create the Matmul program SECOND
    //
    // The matmul factory receives the populated srs_fused_op_signaler, which
    // contains the RS reader cores' NOC coordinates and semaphore ID. The
    // matmul kernels use this to signal the RS when output blocks are ready.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_mm_fused_op_signaler;

    std::vector<Tensor> mm_output_tensors = {matmul_output_tensor};
    auto mm_shared_variables = ttnn::experimental::prim::minimal_matmul_factory_helper_common(
        program,
        input_tensor,   // MM input (activations)
        weight_tensor,  // MM weights
        bias,
        fused_activation,
        config,
        mm_output_tensors,  // MM output (= RS input)
        compute_kernel_config,
        empty_mm_fused_op_signaler,  // No AG -> MM signaling
        1,                           // N_chunks = 1
        std::nullopt,                // ternary fused in RS, not MM
        std::nullopt,
        std::nullopt,
        srs_fused_op_signaler,  // MM -> RS signaling (populated from step 1)
        false,                  // fuse_swiglu
        mm_optional_input_tensor);

    (void)mm_shared_variables;
    return {std::move(program), std::move(rs_shared_variables)};
}

tt::tt_metal::WorkloadDescriptor MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MinimalMatmulStridedReduceScatterAsyncParams& attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& mesh_coordinate : tensor_coords.coords()) {
        uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);

        // output_tensor[0] = MM output (= RS input)
        // output_tensor[1] = RS intermediate
        // output_tensor[2] = RS output
        auto built = minimal_matmul_strided_reduce_scatter_async_program(
            tensor_args.input_tensor,   // MM input (activations)
            tensor_args.weight_tensor,  // MM weights
            output_tensor[0],           // MM output = RS input
            output_tensor[1],           // RS intermediate
            output_tensor[2],           // RS output

            /* Reduce Scatter Params */
            mesh_coordinate,
            forward_coord,
            backward_coord,
            attributes.dim,
            attributes.num_links,
            attributes.ring_size,
            device_index,
            attributes.topology,
            attributes.semaphore,
            attributes.barrier_semaphore,
            attributes.using_persistent_buffers,
            attributes.sub_device_id,
            attributes.num_workers_per_link,
            attributes.num_buffers_per_channel,
            attributes.reduce_scatter_core_grid_offset,
            attributes.chunk_width_in_mm_blocks,
            attributes.mm_window_blocks,
            tensor_args.mm_credit_counters,

            /* Matmul Params */
            tensor_args.bias,
            attributes.matmul_struct.fused_activation,
            attributes.matmul_struct.config.value(),
            attributes.matmul_struct.compute_kernel_config,

            /* Fused addcmul params */
            attributes.fused_ternary_scalar,
            tensor_args.addcmul_input_tensor1,
            tensor_args.addcmul_input_tensor2,

            /* Shared MM->RS progress counter scratch */
            tensor_args.mm_progress_counters,
            /* Fused concat: second in0 source */
            tensor_args.mm_optional_input_tensor);

        auto park = [&](const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& mesh_buffer) {
            if (!mesh_buffer) {
                return;
            }
            workload.buffers.push_back(tt::tt_metal::WorkloadBuffer{
                .owner = mesh_buffer,
                .buffer = mesh_buffer->get_device_buffer(mesh_coordinate),
            });
        };
        park(built.rs_artifacts.mm_progress_counters_buffer);
        park(built.rs_artifacts.rs_credit_counters_buffer);
        workload.programs.push_back(tt::tt_metal::WorkloadDescriptor::PerCoordProgram{
            .range = ttnn::MeshCoordinateRange(mesh_coordinate),
            .descriptor = std::move(built.descriptor),
        });
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
