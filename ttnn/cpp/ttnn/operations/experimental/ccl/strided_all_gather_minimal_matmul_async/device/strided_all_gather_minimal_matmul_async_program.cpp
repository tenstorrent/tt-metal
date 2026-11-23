// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>
#include <cstdlib>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <sstream>
#include <type_traits>

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_fabric_bound_program_factory.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// The matmul kernels are pushed first, so the all-gather kernels start right after them.
constexpr uint32_t strided_ag_mm_matmul_first_kernel_index = 0;
constexpr uint32_t strided_ag_mm_all_gather_first_kernel_index =
    strided_ag_mm_matmul_first_kernel_index + minimal_matmul_fabric_bound_layout::kNumKernels;

// Row-gather (dim 2): the all-gather is all_gather_async's minimal default (fabric mux, num_workers_per_link workers
// per direction per link), appended right after the matmul kernels, with its ROW_PROGRESS hook reporting per-worker
// slice progress to the matmul's in0 injectors.
constexpr uint32_t row_gather_reader_kernel_index = strided_ag_mm_all_gather_first_kernel_index;
constexpr uint32_t row_gather_writer_kernel_index = strided_ag_mm_all_gather_first_kernel_index + 1;
constexpr uint32_t row_gather_default_workers_per_link = 4;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::ProgramDescriptor StridedAllGatherMinimalMatmulAsyncProgramFactory::create_descriptor(
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "strided_all_gather_minimal_matmul_async builds one program per mesh coordinate; no coordinate was given");
    const auto& mesh_coordinate = mesh_dispatch_coordinate.value();
    const auto& ag_attributes = attributes.strided_all_gather_async_struct;
    const auto& input_tensor = tensor_args.input_tensor;
    Tensor& all_gather_output_tensor = output_tensor[0];
    const bool read_local_slice_from_input = attributes.read_local_slice_from_input;
    const auto& config = attributes.matmul_struct.config.value();

    uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, ag_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, ag_attributes.topology, ag_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, ag_attributes.topology, ag_attributes.cluster_axis);

    // Matmul outputs are all tensors after the all-gather output (one per chunk).
    std::vector<Tensor> matmul_output_tensors(output_tensor.begin() + 1, output_tensor.end());

    tt::tt_metal::ProgramDescriptor desc;

    if (ag_attributes.dim == 2) {
        // Row gather: the matmul splits the gathered M into one band per source device (its in0 injector waits on
        // that slice's progress, M block by M block), and all_gather_async's workers report per-worker progress.
        const auto& sems = ag_attributes.semaphore;
        TT_FATAL(sems.size() >= 2, "row gather (dim=2) needs 2 global semaphores (+ an optional barrier third)");
        const uint32_t workers = ag_attributes.num_workers_per_link.value_or(row_gather_default_workers_per_link);
        const uint32_t K_tiles = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
        const uint32_t slice_pages = all_gather_input_Ht(input_tensor) * K_tiles;

        std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> sig =
            ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler();
        sig->ring_size = ag_attributes.ring_size;
        sig->start_ring_index = ring_index;
        sig->topology = ag_attributes.topology;
        sig->initialized_all_gather = true;
        sig->row_gather = true;
        {
            // Arrival order of the minimal-default ring all-gather (no split forwarding under ROW_PROGRESS): own slice,
            // then reader direction 1 receives chips +1, +2, .. (num_targets_backward of them) and direction 0 chips
            // -1, -2, .. (num_targets_forward), one per step, both directions in parallel.
            const uint32_t S = ag_attributes.ring_size;
            auto [n_fwd, n_bwd] =
                ::ttnn::ccl::get_forward_backward_line_mcast_distance(S, ring_index, ag_attributes.topology, false);
            sig->rg_order.push_back({ring_index, 2u, slice_pages});
            for (uint32_t k = 0; k < std::max(n_fwd, n_bwd); k++) {
                if (k < n_bwd) {
                    sig->rg_order.push_back({(ring_index + k + 1) % S, 1u, (k + 1) * slice_pages});
                }
                if (k < n_fwd) {
                    sig->rg_order.push_back({(ring_index + S - (k + 1) % S) % S, 0u, (k + 1) * slice_pages});
                }
            }
            TT_FATAL(
                sig->rg_order.size() == S, "row gather: arrival order covers {} of {} slices", sig->rg_order.size(), S);
        }
        std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> no_srs;
        minimal_matmul_fabric_bound_factory_helper_common(
            desc,
            all_gather_output_tensor,
            tensor_args.weight_tensor,
            tensor_args.bias,
            attributes.matmul_struct.fused_activation,
            config,
            matmul_output_tensors,
            attributes.matmul_struct.compute_kernel_config,
            sig,
            static_cast<uint32_t>(matmul_output_tensors.size()),
            attributes.matmul_struct.fused_ternary_scalar,
            tensor_args.fused_ternary_input_a,
            tensor_args.fused_ternary_input_b,
            no_srs,
            attributes.matmul_struct.fuse_swiglu);

        std::vector<GlobalSemaphore> ag_sems = {sems.at(0), sems.at(1)};
        std::optional<GlobalSemaphore> barrier =
            sems.size() >= 3 ? std::optional<GlobalSemaphore>(sems.at(2)) : std::nullopt;
        std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> no_ag_signaler;
        // A column offset (x > 0, y == 0) puts the all-gather workers in the columns right of the matmul grid (the
        // transposed layout): the core picker fills rows of the whole grid from the offset, so hand it those
        // columns as a sub-grid instead. A row offset (y > 0) is used as is.
        CoreCoord ag_offset = attributes.all_gather_core_grid_offset;
        std::optional<CoreRangeSet> ag_sub_grid;
        if (ag_offset.x > 0 && ag_offset.y == 0) {
            const auto full = input_tensor.device()->compute_with_storage_grid_size();
            ag_sub_grid = CoreRangeSet(CoreRange({ag_offset.x, 0}, {full.x - 1, full.y - 1}));
            ag_offset = CoreCoord{0, 0};
        }
        const auto art = ttnn::build_all_gather_async_minimal_default_program_descriptor(
            desc,
            input_tensor,
            mesh_coordinate,
            forward_coord,
            backward_coord,
            all_gather_output_tensor,
            2,
            ag_attributes.num_links,
            ag_attributes.ring_size,
            ring_index,
            ag_attributes.topology,
            ag_sems,
            barrier,
            /*using_persistent_buffers=*/false,
            /*sub_device_id=*/std::nullopt,
            no_ag_signaler,
            /*chunks_per_sync=*/std::nullopt,
            workers,
            ag_attributes.num_buffers_per_channel,
            ag_offset,
            /*reverse_order=*/false,
            ag_sub_grid);
        TT_FATAL(
            art.reader_kernel_index == row_gather_reader_kernel_index &&
                art.writer_kernel_index == row_gather_writer_kernel_index,
            "row gather: all-gather reader/writer landed at {}/{}, expected {}/{} (the cache-hit patch depends on it)",
            art.reader_kernel_index,
            art.writer_kernel_index,
            row_gather_reader_kernel_index,
            row_gather_writer_kernel_index);
        for (const size_t k : {art.reader_kernel_index, art.writer_kernel_index}) {
            desc.kernels[k].defines.emplace_back("ROW_PROGRESS", "1");
        }
        // Reader: [direction, ...] + [its direction's counter sem id, injector noc (x, y) per M band]
        for (auto& [core, args] : desc.kernels[art.reader_kernel_index].runtime_args) {
            const uint32_t dir = args.at(0);
            args.push_back(sig->rg_sem_ids.at(dir));
            for (uint32_t d = 0; d < ag_attributes.ring_size; d++) {
                args.push_back(sig->rg_injector_cores_noc.at(d).x);
                args.push_back(sig->rg_injector_cores_noc.at(d).y);
            }
        }
        // Writer: + [own-slice counter sem id, injector noc (x, y) per M band]
        for (auto& [core, args] : desc.kernels[art.writer_kernel_index].runtime_args) {
            args.push_back(sig->rg_sem_ids.at(2));
            for (uint32_t d = 0; d < ag_attributes.ring_size; d++) {
                args.push_back(sig->rg_injector_cores_noc.at(d).x);
                args.push_back(sig->rg_injector_cores_noc.at(d).y);
            }
        }
        return desc;
    }

    // Create a matmul signal info object that gets populated by the matmul kernel
    uint32_t TILE_WIDTH = 32;
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler();
    matmul_fused_op_signaler->init_all_gather(
        ag_attributes.ring_size,
        ring_index,
        input_tensor.padded_shape()[3] / TILE_WIDTH,
        ag_attributes.topology,
        read_local_slice_from_input,
        read_local_slice_from_input ? std::optional<const Tensor>(input_tensor) : std::nullopt);

    // Option W (writer-signals-matmul): the matmul cores keep the legacy 3 semaphores

    // Matmul outputs: one tensor per chunk (chunks == 1 -> single output).
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> empty_srs_fused_op_signaler;
    minimal_matmul_fabric_bound_factory_helper_common(
        desc,
        all_gather_output_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias,
        attributes.matmul_struct.fused_activation,
        config,
        matmul_output_tensors,
        attributes.matmul_struct.compute_kernel_config,
        matmul_fused_op_signaler,
        static_cast<uint32_t>(matmul_output_tensors.size()),  // N_chunks
        attributes.matmul_struct.fused_ternary_scalar,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        empty_srs_fused_op_signaler,
        attributes.matmul_struct.fuse_swiglu);

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    TT_FATAL(
        desc.kernels.size() == strided_ag_mm_all_gather_first_kernel_index,
        "strided_all_gather_minimal_matmul_async: all-gather kernels must start at index {}",
        strided_ag_mm_all_gather_first_kernel_index);
    strided_all_gather_async_minimal_default_helper(
        desc,
        input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        all_gather_output_tensor,
        ag_attributes.dim,
        ag_attributes.num_links,
        ag_attributes.ring_size,
        ring_index,
        ag_attributes.topology,
        ag_attributes.semaphore,
        all_gather_fused_op_signaler,
        read_local_slice_from_input,
        ag_attributes.num_workers_per_link,
        ag_attributes.num_buffers_per_channel,
        matmul_fused_op_signaler->num_fused_op_cores_to_signal,
        config.M_block_size,
        config.K_block_size,
        attributes.all_gather_core_grid_offset,
        attributes.mm_signal_aggregator_mode);

    return desc;
}

void StridedAllGatherMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const Tensor& all_gather_output_tensor = output_tensor.at(0);

    if (attributes.strided_all_gather_async_struct.dim == 2) {
        const auto& sems = attributes.strided_all_gather_async_struct.semaphore;
        const uint32_t in_addr = tensor_args.input_tensor.buffer()->address();
        const uint32_t out_addr = all_gather_output_tensor.buffer()->address();
        for (const uint32_t k : {row_gather_reader_kernel_index, row_gather_writer_kernel_index}) {
            auto& common = tt::tt_metal::GetCommonRuntimeArgs(program, k);
            common[ttnn::ccl::AllGatherCommonArgs::input] = in_addr;
            common[ttnn::ccl::AllGatherCommonArgs::output] = out_addr;
        }
        ttnn::apply_all_gather_async_minimal_default_semaphore_args(
            program,
            ttnn::AllGatherDescriptorArtifacts{
                .reader_kernel_index = row_gather_reader_kernel_index,
                .writer_kernel_index = row_gather_writer_kernel_index},
            sems.size() >= 3 ? std::optional<GlobalSemaphore>(sems.at(2)) : std::nullopt,
            std::vector<GlobalSemaphore>{sems.at(0), sems.at(1)});
        std::vector<Tensor> mm_outs(output_tensor.begin() + 1, output_tensor.end());
        minimal_matmul_fabric_bound_patch_runtime_args(
            program,
            strided_ag_mm_matmul_first_kernel_index,
            all_gather_output_tensor,
            tensor_args.weight_tensor,
            tensor_args.bias,
            std::nullopt,
            tensor_args.fused_ternary_input_a,
            tensor_args.fused_ternary_input_b,
            mm_outs);
        return;
    }

    strided_all_gather_async_patch_runtime_args(
        program,
        strided_ag_mm_all_gather_first_kernel_index,
        attributes.strided_all_gather_async_struct,
        tensor_args.input_tensor,
        all_gather_output_tensor,
        /*fused=*/true);

    // The all-gather output is the matmul's in0; the matmul outputs are all tensors after it (one per chunk).
    std::vector<Tensor> matmul_output_tensors(output_tensor.begin() + 1, output_tensor.end());
    minimal_matmul_fabric_bound_patch_runtime_args(
        program,
        strided_ag_mm_matmul_first_kernel_index,
        all_gather_output_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias,
        attributes.read_local_slice_from_input ? std::optional<const Tensor>(tensor_args.input_tensor) : std::nullopt,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        matmul_output_tensors);
}

}  // namespace ttnn::experimental::prim
