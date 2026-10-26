// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

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

namespace {
// minimal_matmul_factory_helper_common pushes its compute kernel right after its four data-movement kernels.
constexpr uint32_t kMatmulComputeKernelIdx =
    MinimalMatmulStridedReduceScatterAsyncProgramFactory::kMatmulIn1ReceiverKernelIdx + 1;
}  // namespace

void MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>&) {
    // The common runtime args (row count, tensor addresses, semaphores) are rewritten here: this workload
    // never rebuilds. Caller-owned progress and credit counters are per-core args and are refreshed too.
    constexpr uint32_t kReaderArgPrefix = 3;  // direction, worker_id, num_workers
    constexpr uint32_t kWindowedProgressFromEnd = 4;
    constexpr uint32_t kWindowedCreditFromEnd = 2;

    TT_FATAL(output_tensor.size() == 3, "minimal_matmul_strided_reduce_scatter returns 3 tensors");
    TT_FATAL(output_tensor[0].buffer() != nullptr, "matmul output must have a buffer");
    TT_FATAL(output_tensor[1].buffer() != nullptr, "reduce-scatter intermediate must have a buffer");
    TT_FATAL(output_tensor[2].buffer() != nullptr, "reduce-scatter output must have a buffer");
    TT_FATAL(tensor_args.input_tensor.buffer() != nullptr, "matmul input must have a buffer");
    TT_FATAL(tensor_args.weight_tensor.buffer() != nullptr, "matmul weight must have a buffer");

    // output_tensor[0] = MM output = RS input, [1] = RS intermediate, [2] = RS output. The reduce-scattered
    // height is the activations' height (the MM output may be a window).
    const uint32_t slice_Ht = tensor_args.input_tensor.padded_shape()[-2] / TILE_HEIGHT;
    TT_FATAL(
        operation_attributes.semaphore.size() > 2,
        "strided reduce scatter expects one semaphore per direction plus the batch semaphore");
    const StridedReduceScatterProgramArtifacts rs_kernels{
        .reader_kernel_id = kReaderKernelIdx,
        .writer_kernel_id = kWriterKernelIdx,
        .reduce_kernel_id = kReduceKernelIdx};
    ::ttnn::ring_strided_reduce_scatter_async_helper_override_runtime_arguments(
        program,
        rs_kernels,
        operation_attributes.barrier_semaphore,
        operation_attributes.semaphore,
        slice_Ht,
        output_tensor.at(0),
        output_tensor.at(1),
        output_tensor.at(2),
        operation_attributes.fused_ternary_scalar,
        tensor_args.addcmul_input_tensor1,
        tensor_args.addcmul_input_tensor2);

    // Matmul kernels (addcmul is fused in the RS, not here).
    MinimalMatmulProgramFactory::shared_variables_t mm_kernels{
        .in0_sender_kernels_id = kMatmulIn0SenderKernelIdx,
        .in0_receiver_kernels_id = kMatmulIn0ReceiverKernelIdx,
        .in1_sender_kernels_id = kMatmulIn1SenderKernelIdx,
        .in1_receiver_kernels_id = kMatmulIn1ReceiverKernelIdx,
        .compute_kernels_id = kMatmulComputeKernelIdx,
        .read_local_slice_from_input = false,
        .two_input_split = tensor_args.mm_optional_input_tensor.has_value()};
    auto mm_cached_program = MinimalMatmulProgramFactory::cached_program_t::proxy(program, mm_kernels);
    std::vector<Tensor> mm_output_vec = {output_tensor.at(0)};
    MinimalMatmulProgramFactory::override_runtime_arguments(
        mm_cached_program,
        operation_attributes.matmul_struct,
        {tensor_args.input_tensor,
         tensor_args.weight_tensor,
         tensor_args.bias,
         tensor_args.mm_optional_input_tensor,  // fused-concat 2nd in0 source; must re-point on cached reuse
         std::nullopt,
         std::nullopt},
        mm_output_vec);

    const bool windowed = operation_attributes.mm_window_blocks.has_value();
    const bool caller_progress = tensor_args.mm_progress_counters.has_value();
    const bool caller_credit = windowed && tensor_args.mm_credit_counters.has_value();
    if (!caller_progress && !caller_credit) {
        return;
    }
    uint32_t progress_addr = 0;
    if (caller_progress) {
        TT_FATAL(tensor_args.mm_progress_counters->buffer() != nullptr, "mm_progress_counters must have a buffer");
        progress_addr = static_cast<uint32_t>(tensor_args.mm_progress_counters->buffer()->address());
    }
    uint32_t credit_addr = 0;
    if (caller_credit) {
        TT_FATAL(tensor_args.mm_credit_counters->buffer() != nullptr, "mm_credit_counters must have a buffer");
        credit_addr = static_cast<uint32_t>(tensor_args.mm_credit_counters->buffer()->address());
    }
    uint32_t reader_progress_arg = kReaderArgPrefix;
    if (output_tensor[1].is_sharded()) {
        std::vector<uint32_t> shard_args;
        shard_builder::extend_sharding_run_time_args(output_tensor[1], shard_args);
        reader_progress_arg += static_cast<uint32_t>(shard_args.size());
    }
    const uint32_t reader_credit_arg = reader_progress_arg + 1;

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    for (auto& column : reader_runtime_args) {
        for (auto& args : column) {
            if (args.size() == 0) {
                continue;
            }
            if (caller_progress) {
                TT_FATAL(args.size() > reader_progress_arg, "reduce-scatter reader is missing the progress counter");
                args[reader_progress_arg] = progress_addr;
            }
            if (caller_credit) {
                TT_FATAL(args.size() > reader_credit_arg, "reduce-scatter reader is missing the credit counter");
                args[reader_credit_arg] = credit_addr;
            }
        }
    }

    TT_FATAL(
        operation_attributes.matmul_struct.config.has_value(),
        "minimal_matmul_strided_reduce_scatter requires a matmul config");
    const auto grid = operation_attributes.matmul_struct.config->compute_with_storage_grid_size;
    auto& in0_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn0SenderKernelIdx);
    auto& in0_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn0ReceiverKernelIdx);
    auto& in1_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn1SenderKernelIdx);
    auto& in1_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn1ReceiverKernelIdx);
    // The counters close the matmul's per-core args: [..., progress] or [..., progress, window_start, credit, readers].
    const auto refresh_matmul_counters = [&](auto& args) {
        if (caller_progress) {
            TT_FATAL(
                !windowed || args.size() >= kWindowedProgressFromEnd,
                "matmul runtime args are missing the progress counter");
            const uint32_t progress_arg = windowed ? static_cast<uint32_t>(args.size() - kWindowedProgressFromEnd)
                                                   : static_cast<uint32_t>(args.size() - 1);
            args[progress_arg] = progress_addr;
        }
        if (caller_credit) {
            TT_FATAL(args.size() >= kWindowedCreditFromEnd, "matmul runtime args are missing the credit counter");
            args[args.size() - kWindowedCreditFromEnd] = credit_addr;
        }
    };
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            auto& in0_grid = x == 0 ? in0_sender_runtime_args : in0_receiver_runtime_args;
            if (x < in0_grid.size() && y < in0_grid[x].size() && in0_grid[x][y].size() > 0) {
                refresh_matmul_counters(in0_grid[x][y]);
            }
            auto& in1_grid = y == 0 ? in1_sender_runtime_args : in1_receiver_runtime_args;
            if (x < in1_grid.size() && y < in1_grid[x].size() && in1_grid[x][y].size() > 0) {
                refresh_matmul_counters(in1_grid[x][y]);
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

    // The RS geometry comes from the activations' height: the windowed MM output is shorter and the
    // RS output / intermediate may be taller (capacity-sized) than the rows being reduced.
    const std::optional<uint32_t> mm_logical_Ht_val = input_tensor.padded_shape()[-2] / TILE_HEIGHT;

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

    // override_runtime_arguments addresses the kernels by these fixed indices.
    using Factory = MinimalMatmulStridedReduceScatterAsyncProgramFactory;
    TT_FATAL(
        rs_shared_variables.reader_kernel_id == Factory::kReaderKernelIdx &&
            rs_shared_variables.writer_kernel_id == Factory::kWriterKernelIdx &&
            rs_shared_variables.reduce_kernel_id == Factory::kReduceKernelIdx &&
            mm_shared_variables.in0_sender_kernels_id == Factory::kMatmulIn0SenderKernelIdx &&
            mm_shared_variables.in0_receiver_kernels_id == Factory::kMatmulIn0ReceiverKernelIdx &&
            mm_shared_variables.in1_sender_kernels_id == Factory::kMatmulIn1SenderKernelIdx &&
            mm_shared_variables.in1_receiver_kernels_id == Factory::kMatmulIn1ReceiverKernelIdx &&
            mm_shared_variables.compute_kernels_id == kMatmulComputeKernelIdx,
        "minimal_matmul_strided_reduce_scatter: kernel push order does not match the override's fixed indices");
    return {std::move(program), std::move(rs_shared_variables)};
}

tt::tt_metal::WorkloadDescriptor MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& attributes = operation_attributes;
    auto& output_tensor = tensor_return_value;
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
                // Same reference buffer the ring builder bound, including on a remote coordinate.
                .buffer = mesh_buffer->get_reference_buffer(),
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
