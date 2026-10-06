// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_program_factory.hpp"
#include "moe_compute_device_operation_types.hpp"

#include "ttnn/global_semaphore.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/indestructible.hpp>
#include <umd/device/types/arch.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace {

inline uint32_t non_tile_cb_page_size(tt::DataFormat data_format, uint32_t l1_alignment) {
    return std::max({tt::datum_size(data_format), l1_alignment, CIRCULAR_BUFFER_COMPUTE_WORD_SIZE});
}

uint32_t get_num_pages_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows_st(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

std::tuple<
    std::vector<tt::tt_metal::CoreCoord>,  // T cores
    std::vector<tt::tt_metal::CoreCoord>,  // MM cores
    tt::tt_metal::CoreRangeSet,            // T CoreRangeSet
    tt::tt_metal::CoreRangeSet,            // MM CoreRangeSet
    tt::tt_metal::CoreRangeSet,            // T + MM CoreRangeSet
    tt::tt_metal::CoreRangeSet,            // Combine CoreRangeSet
    tt::tt_metal::CoreRangeSet,            // C + MM CoreRangeSet
    tt::tt_metal::CoreRangeSet,            // All worker cores (T + MM + C)
    std::vector<tt::tt_metal::CoreCoord>,  // Combine vector of CoreCoord
    tt::tt_metal::CoreRange,               // T bounding box
    tt::tt_metal::CoreRange,               // MM bounding box
    std::vector<tt::tt_metal::CoreCoord>,  // every prefill ring's cores, ring-major (ring 0 = MM)
    tt::tt_metal::CoreRangeSet>            // their CoreRangeSet
get_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    uint32_t bh_ring_size,
    const tt::tt_metal::CoreRangeSet& mux_core_range_set,
    uint32_t prefill_rings) {
    const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        mux_core_range_set,
        bh_ring_size,
        prefill_rings);

    return {
        selection.tilize_cores,
        selection.matmul_cores,
        selection.tilize_core_range_set,
        selection.matmul_core_range_set,
        selection.tilize_matmul_core_range_set,
        selection.combine_core_range_set,
        selection.combine_matmul_core_range_set,
        selection.all_worker_cores_range_set,
        selection.combine_cores,
        selection.tilize_bounding_box,
        selection.matmul_bounding_box,
        selection.ring_cores,
        selection.ring_core_range_set};
}

std::string serialize_physical_core_coords(const std::vector<ttnn::CoreCoord>& cores, const ttnn::MeshDevice& device) {
    std::vector<uint32_t> flat_physical_core_coords;
    flat_physical_core_coords.reserve(2 * cores.size());

    for (const auto& c : cores) {
        const auto pc = device.worker_core_from_logical_core(c);
        flat_physical_core_coords.push_back(pc.x);
        flat_physical_core_coords.push_back(pc.y);
    }

    return ttnn::operations::ccl::common::stringify(flat_physical_core_coords);
}
}  // namespace

namespace ttnn::experimental::prim {

// expose a helper function so callers know what cores are available for subsequently running a2a combine.
// Combine cores are selected dynamically based on hidden_size (for tilize core count) and mux_core_range_set
// (for core avoidance). Layout overlap (tilize/matmul/combine bboxes) is checked in get_cores() at program
// build time, so callers that subsequently invoke the op still get the assert coverage.
std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const tt::tt_metal::CoreRangeSet& mux_core_range_set,
    const uint32_t bh_ring_size) {
    const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        mux_core_range_set,
        bh_ring_size);
    return selection.combine_cores;
}

ttnn::CoreCoord get_moe_tilize_drain_core(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const tt::tt_metal::CoreRangeSet& mux_core_range_set,
    const uint32_t bh_ring_size) {
    const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        mux_core_range_set,
        bh_ring_size);
    TT_FATAL(!selection.tilize_cores.empty(), "moe_compute tilize drain core selection returned no tilize cores");
    return selection.tilize_cores.at(0);
}

ttnn::CoreRange get_moe_worker_mcast_bounding_box(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const tt::tt_metal::CoreRangeSet& mux_core_range_set,
    const uint32_t bh_ring_size) {
    const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        mux_core_range_set,
        bh_ring_size);
    return selection.all_worker_cores_range_set.bounding_box();
}

MoEComputeMeshWorkloadFactory::cached_mesh_workload_t MoEComputeMeshWorkloadFactory::create_mesh_workload(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, MoEComputeMeshWorkloadFactory::shared_variables_t> shared_variables;

    constexpr auto combine_core_range_set_return_index = 5;

    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const auto& tilize_input_shape = tensor_args.tilize_input_tensor.tensor_spec().logical_shape();
    const uint32_t hidden_size = tilize_input_shape[-1];

    std::optional<GlobalSemaphore> init_barrier_semaphore;
    std::optional<GlobalSemaphore> final_barrier_semaphore;

    // Only FullCcl needs cross-device barrier semaphores (and the host sync to publish them).
    // SingleDevice's writer compiles out all init/final barrier handling under LOCAL_COMBINE, so
    // there is nothing to allocate; create_at passes a placeholder address of 0 to the combine
    // builder for that path. ComputeOnly skips the combine stage entirely.
    if (args.path == MoEComputePath::FullCcl) {
        // combine_params.has_value() is checked in validate_on_program_cache_miss.
        // mux_core_range_set comes from combine_params (empty for SingleDevice).
        const auto core_ret = get_cores(
            mesh_device,
            args.num_token_parallel_cores,
            args.num_data_parallel_cores,
            hidden_size,
            args.bh_ring_size,
            args.combine_params->mux_core_range_set,
            args.prefill_rings);

        const auto& combine_core_range_set = std::get<combine_core_range_set_return_index>(core_ret);

        init_barrier_semaphore =
            ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_core_range_set, 0);
        final_barrier_semaphore = args.combine_params->optional_cross_device_semaphore.value_or(
            ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_core_range_set, 0));

        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, {});
    }

    for (const auto& coord : mesh_coordinates.coords()) {
        auto cached_program = MoEComputeMeshWorkloadFactory::create_at(
            args,
            coord,
            tensor_args,
            tensor_return_value,
            mesh_coordinates,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return MoEComputeMeshWorkloadFactory::cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoEComputeMeshWorkloadFactory::shared_variables_t>
MoEComputeMeshWorkloadFactory::create_at(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
    const std::optional<GlobalSemaphore>& init_barrier_semaphore,
    const std::optional<GlobalSemaphore>& final_barrier_semaphore) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Alignment
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // Input tensors
    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const ttnn::Tensor& tilize_indices_tensor = tensor_args.tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_input_scores_tensor = tensor_args.tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_mapping_tensor = tensor_args.tilize_expert_mapping_tensor;

    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& tilize_indices_shape = tilize_indices_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_input_scores_shape = tilize_input_scores_tensor.tensor_spec().logical_shape();
    const auto& tilize_mapping_shape = tilize_mapping_tensor.tensor_spec().logical_shape();

    const uint32_t tilize_input_pages = get_num_pages_st(tilize_input_tensor);
    const uint32_t tilize_indices_pages = get_num_pages_st(tilize_indices_tensor);
    const uint32_t tilize_input_scores_pages = get_num_pages_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_pages = get_num_pages_st(tilize_mapping_tensor);

    const uint32_t tilize_input_page_size = get_page_size_st(tilize_input_tensor);
    const uint32_t tilize_indices_page_size = get_page_size_st(tilize_indices_tensor);
    [[maybe_unused]] const uint32_t tilize_input_scores_page_size = get_page_size_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_page_size = get_page_size_st(tilize_mapping_tensor);

    const uint32_t tilize_input_aligned_page_size = get_aligned_page_size_st(tilize_input_tensor);
    const uint32_t tilize_indices_aligned_page_size = get_aligned_page_size_st(tilize_indices_tensor);
    const uint32_t tilize_input_scores_aligned_page_size = get_aligned_page_size_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_aligned_page_size = get_aligned_page_size_st(tilize_mapping_tensor);

    [[maybe_unused]] const ttnn::Tensor& matmul_w0_w1_tensor = tensor_args.matmul_w0_w1_tensor;
    [[maybe_unused]] const ttnn::Tensor& matmul_w2_tensor = tensor_args.matmul_w2_tensor;

    // Output tensors
    TT_FATAL(tensor_return_value.size() >= 5, "Expected at least 5 output tensors, got {}", tensor_return_value.size());
    const ttnn::Tensor& tilize_per_expert_total_tokens_output_tensor = tensor_return_value[0];
    const ttnn::Tensor& tilize_expert_activation_output_tensor = tensor_return_value[1];
    const ttnn::Tensor& tilize_e_t_output_tensor = tensor_return_value[2];
    const ttnn::Tensor& tilize_output_tensor = tensor_return_value[3];
    const ttnn::Tensor& matmul_output_tensor = tensor_return_value[4];

    [[maybe_unused]] const auto& tilize_per_expert_total_tokens_output_shape =
        tilize_per_expert_total_tokens_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_expert_activation_output_shape =
        tilize_expert_activation_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_e_t_output_shape = tilize_e_t_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& output_shape = tilize_output_tensor.tensor_spec().logical_shape();

    [[maybe_unused]] const uint32_t tilize_per_expert_total_tokens_output_pages =
        get_num_pages_st(tilize_per_expert_total_tokens_output_tensor);
    [[maybe_unused]] const uint32_t tilize_expert_activation_output_pages =
        get_num_pages_st(tilize_expert_activation_output_tensor);
    [[maybe_unused]] const uint32_t tilize_e_t_output_pages = get_num_pages_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_pages = get_num_pages_st(tilize_output_tensor);

    const uint32_t tilize_per_expert_total_tokens_output_page_size =
        get_page_size_st(tilize_per_expert_total_tokens_output_tensor);
    const uint32_t tilize_expert_activation_output_page_size = get_page_size_st(tilize_expert_activation_output_tensor);
    const uint32_t tilize_e_t_output_page_size = get_page_size_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_page_size = get_page_size_st(tilize_output_tensor);

    [[maybe_unused]] const uint32_t tilize_per_expert_total_tokens_output_aligned_page_size =
        get_aligned_page_size_st(tilize_per_expert_total_tokens_output_tensor);
    [[maybe_unused]] const uint32_t tilize_expert_activation_output_aligned_page_size =
        get_aligned_page_size_st(tilize_expert_activation_output_tensor);
    [[maybe_unused]] const uint32_t tilize_e_t_output_aligned_page_size =
        get_aligned_page_size_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_aligned_page_size = get_aligned_page_size_st(tilize_output_tensor);

    // Mesh
    auto* mesh_device = tilize_input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t num_devices = mesh_view.num_devices();
    uint32_t linearized_mesh_coord = ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    // Tilize output, matmul input
    const auto tilize_output_dtype = tilize_input_tensor.dtype();
    const auto tilize_output_dataformat = tt::tt_metal::datatype_to_dataformat_converter(tilize_output_dtype);
    const uint32_t tilize_output_page_size = tt::tile_size(tilize_output_dataformat);

    // General info
    uint32_t tokens = get_num_rows_st(tilize_input_tensor);
    uint32_t hidden_size = tilize_input_shape[-1];

    // Logical experts, routed + shared (replicated experts counted as 1)
    uint32_t experts = tilize_mapping_shape[-1];
    // physical experts per device, replicated shared experts are counted per device
    uint32_t experts_per_device = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    uint32_t selected_experts_k = tilize_indices_shape[-1];

    // Output/Combine input core dims, for core selection. These are top-level (lifted) so they
    // remain valid even when combine_params is nullopt (ComputeOnly mode).
    const auto combine_token_parallel_cores = args.num_token_parallel_cores;
    const auto combine_data_parallel_cores = args.num_data_parallel_cores;

    // Derived tile counts from generalized intermediate_size API (PR #43932).
    // Replaces the pre-#43932 per-(hidden_size, config_type) lookup machinery -- formula-driven now.
    const uint32_t intermediate_size = args.intermediate_size;
    const uint32_t hidden_tiles = hidden_size / 32;
    const uint32_t intermediate_tiles = intermediate_size / 32;
    const uint32_t num_shared_experts = args.num_shared_experts_per_device.value_or(0);

    // Cores
    const auto
        [tilize_cores,
         matmul_cores,
         tilize_core_range_set,
         matmul_core_range_set,
         tilize_matmul_core_range_set,
         combine_core_range_set,
         combine_matmul_core_range_set,
         all_worker_cores_range_set,
         combine_cores,
         tilize_bounding_box,
         matmul_bounding_box,
         ring_cores,
         ring_core_range_set] =
            get_cores(
                mesh_device,
                combine_token_parallel_cores,
                combine_data_parallel_cores,
                hidden_size,
                args.bh_ring_size,
                args.combine_params.has_value() ? args.combine_params->mux_core_range_set
                                                : tt::tt_metal::CoreRangeSet{},
                args.prefill_rings);

    const uint32_t tilize_num_cores = tilize_core_range_set.num_cores();
    // The ring size (one core per DRAM bank; the a2a ring, the W2 split, the tilize credits per chunk). With R
    // prefill rings there are R x matmul_num_cores ring cores (ring_cores / ring_core_range_set): the CBs, the
    // semaphores and the kernels go on all of them, the per-ring arithmetic keeps the ring size.
    const uint32_t matmul_num_cores = matmul_core_range_set.num_cores();
    const uint32_t num_rings = args.prefill_rings < 2 ? 1u : args.prefill_rings;
    TT_FATAL(
        ring_cores.size() == static_cast<std::size_t>(num_rings) * matmul_num_cores,
        "moe_compute: {} ring cores for {} ring(s) of {}",
        ring_cores.size(),
        num_rings,
        matmul_num_cores);

    // Which output stage this program carries. FullCcl/SingleDevice build the fused combine kernels on
    // the combine cores; SingleCluster builds none and dm1 writes the final output itself;
    // ComputeOnly builds none and stops at the matmul output.
    const bool combine_built = args.path == MoEComputePath::FullCcl || args.path == MoEComputePath::SingleDevice;
    const bool local_output = args.path == MoEComputePath::SingleCluster;

    // a2a_cb_pages = IN2_TILES_PER_STEP = moe_ring::a2a_exchange_tiles: the largest per-core gate/up column count
    // (at least 2) for a compact shape, else the uniform even stride (at least the W2 A2A matmul width).
    // The ring size is the live DRAM-bank count, so each ring core maps 1:1 to a DRAM bank.
    const uint32_t expected_matmul_n =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default).size();
    TT_FATAL(
        matmul_num_cores == expected_matmul_n,
        "moe_compute: expected matmul_num_cores={}, got {}",
        expected_matmul_n,
        matmul_num_cores);
    const uint32_t a2a_cb_pages = moe_ring::a2a_exchange_tiles(intermediate_tiles, matmul_num_cores);
    // The a2a pipeline (2026-09-26): the ring cores run W0/W1 of their next owned chunk while dm1 exchanges the
    // current chunk's in2 partials, so the in2 buffers (the core's own partial and the landed shards) come in two
    // parity slots and the ready credits hold two chunks; dm1 exchanges the next chunk before it finishes this one's
    // output rows and the feed carries three chunk slots. The streaming decode ring only (explicitly enabled by the
    // caller; the replay ring's whole-slice CB and the multi-ring chunk owners assume one chunk in flight); the named
    // arg is in the kernel hash.
    const uint32_t a2a_pipeline = (args.enable_a2a_pipeline ? 1u : 0u);
    const uint32_t a2a_parity_slots = a2a_pipeline ? 2u : 1u;

    // Per-shape DRAM transaction size of both weight streams (moe_ring::tiles_per_txn_for_shape: 14 tiles, or 20
    // for the 2560/640 expert on the 8-bank ring), passed to the kernels as the "tiles_per_txn" compile arg.
    const uint32_t weight_tiles_per_txn =
        moe_ring::tiles_per_txn_for_shape(hidden_tiles, intermediate_tiles, args.has_bias, matmul_num_cores);
    const uint32_t weight_tiles_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK * weight_tiles_per_txn;
    // The weight CB's blocks: the shape's rule, plus one on the streaming decode ring (2026-09-26: three blocks = 69 KB
    // of DRAM reads in flight per ring core, 10.88 -> 10.68 us per distinct local expert on the 1x4 line; five and six
    // no better). The prefill ring forms keep the rule: their larger feed halves leave no L1 for a fourth block.
    const uint32_t weight_slots = a2a_pipeline ? moe_ring::weight_cb_slots(weight_tiles_per_txn) + 1
                                               : moe_ring::weight_cb_slots(weight_tiles_per_txn);

    const uint32_t tilize_bounding_box_num_cores = tilize_bounding_box.size();
    const uint32_t matmul_bounding_box_num_cores = matmul_bounding_box.size();

    // noc_async_write_multicast / noc_semaphore_set_multicast (non-loopback) exclude the sender.
    // Matmul-targeting mcast is issued by tilize_cores[0] (drain) and, on the first chunk when
    // num_tilize_cores > 1, also by tilize_cores[tilize_num_cores / 2] (secondary mcaster).
    // All tilize kernels share one compile-time num_dests, so every sender must be consistently
    // inside or outside the matmul bounding-box rectangle (not merely bbox intersection).
    bool any_sender_inside = false;
    bool any_sender_outside = false;
    const auto check_matmul_mcast_sender = [&](const tt::tt_metal::CoreCoord& sender) {
        if (matmul_bounding_box.contains(sender)) {
            any_sender_inside = true;
        } else {
            any_sender_outside = true;
        }
    };
    check_matmul_mcast_sender(tilize_cores.at(0));
    if (tilize_num_cores > 1) {
        check_matmul_mcast_sender(tilize_cores.at(tilize_num_cores / 2));
    }
    TT_FATAL(
        !(any_sender_inside && any_sender_outside),
        "moe_compute: tilize matmul mcast senders {} and {} straddle matmul bbox {} (inside vs outside); "
        "core placement must keep all senders on the same side of the rectangle",
        tilize_cores.at(0).str(),
        tilize_num_cores > 1 ? tilize_cores.at(tilize_num_cores / 2).str() : tilize_cores.at(0).str(),
        matmul_bounding_box.str());
    const uint32_t matmul_mcast_num_dests =
        any_sender_inside ? matmul_bounding_box_num_cores - 1 : matmul_bounding_box_num_cores;

    // All worker cores bounding box
    const tt::tt_metal::CoreRange all_worker_cores_bounding_box = all_worker_cores_range_set.bounding_box();
    const uint32_t all_worker_cores_bounding_box_num_cores = all_worker_cores_bounding_box.size();

    // Logical mcast bounding box coordinates
    const tt::tt_metal::CoreCoord tilize_mcast_start_logical = tilize_bounding_box.start_coord;
    const tt::tt_metal::CoreCoord tilize_mcast_end_logical = tilize_bounding_box.end_coord;
    const tt::tt_metal::CoreCoord matmul_mcast_start_logical = matmul_bounding_box.start_coord;
    const tt::tt_metal::CoreCoord matmul_mcast_end_logical = matmul_bounding_box.end_coord;
    const tt::tt_metal::CoreCoord all_worker_cores_mcast_start_logical = all_worker_cores_bounding_box.start_coord;
    const tt::tt_metal::CoreCoord all_worker_cores_mcast_end_logical = all_worker_cores_bounding_box.end_coord;

    // Convert to physical NOC coordinates
    const tt::tt_metal::CoreCoord tilize_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(tilize_mcast_start_logical);
    const tt::tt_metal::CoreCoord tilize_mcast_end_physical =
        mesh_device->worker_core_from_logical_core(tilize_mcast_end_logical);
    const tt::tt_metal::CoreCoord matmul_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_start_logical);
    const tt::tt_metal::CoreCoord matmul_mcast_end_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_end_logical);
    const tt::tt_metal::CoreCoord all_worker_cores_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(all_worker_cores_mcast_start_logical);
    const tt::tt_metal::CoreCoord all_worker_cores_mcast_end_physical =
        mesh_device->worker_core_from_logical_core(all_worker_cores_mcast_end_logical);

    //-------------------------------------------------------------------------
    // Tilize semaphores
    //-------------------------------------------------------------------------

    // Non-drain-sync cores signal to drain-sync core that partial metadata results are ready
    auto tilize_partial_metadata_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    // Non-drain-sync cores signal to drain-sync core that partial chunk has been sent to the matmul cores.
    // Drain-sync core then signals to non-drain-sync core that they can begin sending the next chunk.
    auto tilize_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    // On the secondary multicaster: the drain's count of credited input halves (tilize_writer.cpp == 1 ==)
    auto initial_gather_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Matmul semaphores
    //-------------------------------------------------------------------------

    // Create semaphores for ring synchronization between cores
    // Each core will have a semaphore that its predecessor will signal
    // reuse the same semaphore location for signaling combine cores that per expert data is available
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, ring_core_range_set, INVALID);
    // The ring exchange's credit (dm1): a core's successor counts the chunks whose a2a buffers it has read.
    const uint32_t a2a_free_semaphore_id = tt::tt_metal::CreateSemaphore(program, ring_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize and Matmul semaphores
    //-------------------------------------------------------------------------

    // Tilize drain-sync core signals to tilize non-drain-sync cores that final metadata results are ready
    // Tilize drain-sync core signals to matmul cores that final metadata results are ready
    auto metadata_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // Tilize drain-sync core signals to all matmul cores that a full chunk is ready
    auto matmul_chunk_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // One credit semaphore per chunk half of the ring cores' input buffer: a ring core's dm1 increments half
    // g % chunk_halves on the drain after consuming chunk g; the drain multicasts into a half only when every ring
    // core has credited the chunk that last used it. Ids are passed one by one: consecutive CreateSemaphore calls
    // need not give consecutive ids.
    const uint32_t chunk_halves = moe_ring::rings::feed_halves(args.prefill_rings, a2a_pipeline != 0);
    std::array<uint32_t, moe_ring::rings::MAX_CHUNK_HALVES> half_free_semaphore_ids{};
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        half_free_semaphore_ids[h] = tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);
    }

    // The steady-state gather protocol: the drain counts the other tilize cores' sub-chunks per staging slot (one
    // semaphore per slot), and multicasts the number of chunks it has multicast so far (feed_go), which frees the
    // slot a core is about to gather into (tilize_writer.cpp, == 3b == / == 5b == / == 10 ==).
    const auto feed_go_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);
    std::array<uint32_t, moe_ring::rings::MAX_CHUNK_HALVES> gather_semaphore_ids{};
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        gather_semaphore_ids[h] = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);
    }

    //-------------------------------------------------------------------------
    // Tilize/MatMul and Combine sync semaphore
    //-------------------------------------------------------------------------

    // Tilize drain-sync core signals combine sync core (which then multicasts to the rest)
    // that metadata is ready and task splitting can proceed.
    // Allocate on the combine bounding box (not just selected cores) because the combine
    // reader kernel multicasts to the full rectangle — cores inside the bbox but outside
    // the selected set must also have a valid semaphore address to avoid L1 corruption.
    // Without combine kernels (ComputeOnly, SingleCluster) the kernel-side increment is gated off via
    // the compute_only / local_output CT args, so the semaphore is unused by anyone -- but
    // tilize_reader still calls get_semaphore() on it (a local L1 address lookup), so we still need
    // it to be allocated on at least the matmul core range set.
    const auto tilize_combine_sync_semaphore_id = tt::tt_metal::CreateSemaphore(
        program,
        combine_built ? tt::tt_metal::CoreRangeSet(combine_core_range_set.bounding_box()) : ring_core_range_set,
        INVALID);

    // Matmul dm1 signals combine cores when data is written; combine writer waits on this semaphore.
    // For double buffering, combine cores will also use this semaphore to signal matmul when buffer segments are free.
    // Without combine kernels dm1 still calls get_semaphore() to look up the local L1 address, but the
    // wait/increment/set are all gated off via the compute_only / local_output CT args.
    const auto matmul_combine_sync_semaphore_id = tt::tt_metal::CreateSemaphore(
        program, combine_built ? combine_matmul_core_range_set : ring_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize work split
    //-------------------------------------------------------------------------

    // Split token subregions across tilize cores (similar to sender subtoken splitting)
    // Each tilize core handles a portion of the hidden dimension for each token
    uint32_t tilize_subtoken_bytes_aligned =
        tt::align(tt::div_up(tilize_input_aligned_page_size, tilize_num_cores), l1_alignment);
    uint32_t tilize_subtoken_units_of_work = tt::div_up(tilize_input_aligned_page_size, tilize_subtoken_bytes_aligned);

    auto
        [num_tilize_work_cores,
         all_tilize_work_cores,
         tilize_cores_work_group_1,
         tilize_cores_work_group_2,
         tilize_units_per_core_work_group_1,
         tilize_units_per_core_work_group_2] =
            tt::tt_metal::split_work_to_cores(tilize_core_range_set, tilize_subtoken_units_of_work);

    //-------------------------------------------------------------------------
    // Tilize and Matmul CBs
    //-------------------------------------------------------------------------

    /*
     * Tilize: Used as output CB of tilize operation
     * MM: Used as input CB (where tilized chunks arrive)
     * Combine: Stores output of MM, for input to combine
     */
    uint32_t tilize_output_cb_id = tt::CBIndex::c_0;
    [[maybe_unused]] uint32_t cb_s2c_in_id = tt::CBIndex::c_0;
    uint32_t matmul_writer_cb_id = tt::CBIndex::c_1;

    // after determining the total number of tokens for each expert, this buffer will store the total number of
    // tokens for each expert to pass to the other kernels
    uint32_t per_expert_total_tokens_cb_id = tt::CBIndex::c_2;

    // All cores (not just Tilize and Matmul)
    const tt::tt_metal::CoreRangeSet shard_cores = tilize_output_tensor.memory_config().shard_spec()->grid;
    const uint32_t shared_cb_num_pages = output_pages / shard_cores.num_cores();
    TT_FATAL(
        shared_cb_num_pages == chunk_halves * hidden_tiles,
        "The chunk input buffer's {} pages per core must be {} chunk slots of {} tiles",
        shared_cb_num_pages,
        chunk_halves,
        hidden_tiles);

    auto output_cb = tt::tt_metal::create_cb(
        tilize_output_cb_id,
        program,
        shard_cores,
        output_page_size,
        shared_cb_num_pages,
        tt::tt_metal::datatype_to_dataformat_converter(tilize_output_tensor.dtype()),
        tilize_output_tensor.buffer());
    tt::tt_metal::CBHandle sharded_output_cb_handle = std::get<1>(output_cb);

    // MoE output for combine uses the same buffer as input but create a new CB to manage control flow
    auto matmul_writer_cb = tt::tt_metal::create_cb(
        matmul_writer_cb_id,
        program,
        shard_cores,
        output_page_size,
        shared_cb_num_pages,
        tt::tt_metal::datatype_to_dataformat_converter(tilize_output_tensor.dtype()),
        tilize_output_tensor.buffer());
    tt::tt_metal::CBHandle matmul_writer_cb_handle = std::get<1>(matmul_writer_cb);

    // global tensor backed CB to communicate active tokens per expert
    // TODO(#41827): in ComputeOnly mode this CB still spans `all_worker_cores_range_set` which
    // includes (kernel-less) combine cores. Functionally OK (CB is allocated, no kernel reads it),
    // but the tilize_reader mcast at `tilize_reader.cpp` writes to a wider bounding box than needed.
    // Reduce to `tilize_matmul_core_range_set` and pass `tilize_matmul_*_mcast_*` CT args when
    // path == ComputeOnly. Deferred to a follow-up PR.
    const auto expert_token_cb = tt::tt_metal::create_cb(
        per_expert_total_tokens_cb_id,
        program,
        all_worker_cores_range_set,
        tilize_per_expert_total_tokens_output_page_size,
        1,
        tt::tt_metal::datatype_to_dataformat_converter(tilize_per_expert_total_tokens_output_tensor.dtype()),
        tilize_per_expert_total_tokens_output_tensor.buffer());
    const auto expert_tokens_cb_handle = std::get<1>(expert_token_cb);

    //-------------------------------------------------------------------------
    // Tilize CBs
    //-------------------------------------------------------------------------

    // CB for passing total_chunks from the tilize reader to the tilize compute kernel
    uint32_t total_chunks_cb_id = tt::CBIndex::c_3;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_4;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_5;
    // full scores buffer
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_6;
    // Send preparation buffer [E, T] for untilize, capped by -1 to indicate no more tokens to send for this expert
    uint32_t e_t_cb_id = tt::CBIndex::c_7;
    // tilize input buffer for tokens to be tilized (row-major from reader)
    uint32_t tilize_input_cb_id = tt::CBIndex::c_8;
    // Experts activation buffer [T, 2*E + 1] each row is {token id, expert_0_activated, expert_1_activated,...,
    // expert_0_score, expert_1_score, ...} k+1 if not activated, k value in the indices tensor for that token if
    // activated
    uint32_t expert_activation_cb_id = tt::CBIndex::c_9;
    // BRISC's e_t buffer for parallel metadata processing (BRISC processes tokens/2 to tokens)
    uint32_t brisc_e_t_cb_id = tt::CBIndex::c_10;
    // BRISC's per-expert token counts to communicate to NCRISC after parallel processing
    uint32_t brisc_expert_counts_cb_id = tt::CBIndex::c_11;
    // BRISC's expert activation buffer for parallel processing
    uint32_t brisc_expert_activation_cb_id = tt::CBIndex::c_12;
    // BRISC's activated token count (single uint32_t)
    uint32_t brisc_activated_count_cb_id = tt::CBIndex::c_13;

    uint32_t remote_counts_cb_id = tt::CBIndex::c_14;

    const auto tilize_input_data_format = tt::tt_metal::datatype_to_dataformat_converter(tilize_input_tensor.dtype());
    const auto tilize_indices_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_indices_tensor.dtype());
    const auto tilize_input_scores_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_input_scores_tensor.dtype());
    const auto tilize_mapping_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_mapping_tensor.dtype());
    const auto tilize_e_t_output_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_e_t_output_tensor.dtype());

    uint32_t max_tilize_subtoken_size =
        std::max(tilize_units_per_core_work_group_1, tilize_units_per_core_work_group_2) *
        tilize_subtoken_bytes_aligned;

    constexpr uint32_t tokens_per_chunk = moe_ring::TOKENS_PER_CHUNK;

    // Per-token L1 on the tilize cores: the fused-combine paths keep dense token lists and activation rows (about
    // 4.6 KB per token); the local output path keeps pair lists (c_9 NCRISC, c_10 BRISC), the non-drain cores' lists
    // staged on the drain (c_12) and the packed page (c_7, moe_ring::token_list), which admits a whole prefill slab.
    uint32_t pairs_capacity_bytes = 0;
    if (local_output) {
        // The last tilize core takes the remainder of the token split (the runtime args below); each RISC
        // processes half of its core's range.
        const uint32_t max_tokens_per_tilize_core = tokens - (tilize_num_cores - 1) * (tokens / tilize_num_cores);
        pairs_capacity_bytes = tt::align(
            tt::div_up(max_tokens_per_tilize_core, 2) * selected_experts_k * moe_ring::token_list::PAIR_BYTES,
            l1_alignment);
        const uint32_t page_words =
            moe_ring::token_list::page_words(tokens, selected_experts_k, experts_per_device, tokens_per_chunk);
        TT_FATAL(
            tilize_e_t_output_page_size == page_words * sizeof(uint32_t),
            "moe_compute: the packed token-list page is {} B, the kernels expect {} words",
            tilize_e_t_output_page_size,
            page_words);
        tt::tt_metal::create_cb(
            e_t_cb_id, program, tilize_core_range_set, tilize_e_t_output_page_size, 1, tilize_e_t_output_data_format);
    } else {
        // e_t buffer entry size must be 16B aligned for NOC DMA during BRISC->NCRISC merge
        tt::tt_metal::create_cb(
            e_t_cb_id,
            program,
            tilize_core_range_set,
            tilize_e_t_output_page_size,
            experts_per_device,  // number of experts on the device
            tilize_e_t_output_data_format);
    }

    // Assume indices tensor is sharded in L1
    const auto indices_cb_output = tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_indices_aligned_page_size,
        tilize_indices_pages,  // double buffer buffer packets
        tilize_indices_data_format,
        tilize_indices_tensor.buffer());
    const auto indices_cb_handle = std::get<1>(indices_cb_output);

    // Assume scores tensor is sharded in L1
    const auto scores_cb_output = tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_input_scores_aligned_page_size,
        tilize_input_scores_pages,
        tilize_input_scores_data_format,
        tilize_input_scores_tensor.buffer());
    const auto scores_cb_handle = std::get<1>(scores_cb_output);

    // For each batch's tokens, we need to read the relevant experts from the mapping tensor
    // For in range (tokens) every time tokens/batch increments, read in new mapping tensor page
    tt::tt_metal::create_cb(
        mapping_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_mapping_aligned_page_size,
        tilize_mapping_pages,
        tilize_mapping_data_format);

    // Tilize input buffer: holds subtokens for tokens_per_chunk tokens
    // Each tilize core reads its subtoken portion of incoming tokens
    // One chunk deep: the reader reads the next chunk's rows while the previous one is tilized, and no further
    tt::tt_metal::create_cb(
        tilize_input_cb_id,
        program,
        tilize_core_range_set,
        max_tilize_subtoken_size,
        tokens_per_chunk,
        tilize_input_data_format);

    // BRISC's per-expert token counts to communicate to NCRISC
    // Single page containing counts for all experts
    tt::tt_metal::create_cb(
        brisc_expert_counts_cb_id,
        program,
        tilize_core_range_set,
        sizeof(uint32_t) * experts_per_device,  // all counts in one page
        1,
        tt::DataFormat::UInt32);

    // Each non-drain core's record for the drain: its per-expert counts and an activated-token count, or under
    // local_output its per-expert counts and both pair-list lengths; 16B aligned per core's data.
    const uint32_t counts_per_remote_core = experts_per_device + (local_output ? 2 : 1);
    const uint32_t remote_counts_entry_size = tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment);

    if (local_output) {
        // Pair lists: NCRISC's (c_9) and BRISC's (c_10), one page each.
        for (const auto pairs_cb_id : {expert_activation_cb_id, brisc_e_t_cb_id}) {
            tt::tt_metal::create_cb(
                pairs_cb_id, program, tilize_core_range_set, pairs_capacity_bytes, 1, tt::DataFormat::UInt32);
        }
        // Staging on the drain: the two lists of every non-drain core (one page per core); the remote cores write
        // it directly, like the counts record.
        if (tilize_num_cores > 1) {
            tt::tt_metal::create_cb(
                brisc_expert_activation_cb_id,
                program,
                tilize_core_range_set,
                2 * pairs_capacity_bytes,
                tilize_num_cores - 1,
                tt::DataFormat::UInt32);
        }
    } else {
        tt::tt_metal::create_cb(
            expert_activation_cb_id,
            program,
            tilize_core_range_set,
            tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment),
            tokens,
            tt::DataFormat::UInt32);

        // BRISC's e_t buffer for parallel metadata processing
        // BRISC processes the second half of tokens (tokens/2 to tokens)
        // Single page containing all experts' token lists, each with capacity ceil(tokens/2)
        // Uses same 16B entry alignment as main e_t buffer for NOC DMA compatibility
        tt::tt_metal::create_cb(
            brisc_e_t_cb_id,
            program,
            tilize_core_range_set,
            tt::div_up(tokens, 2) * l1_alignment *
                experts_per_device,  // full buffer, ceil(tokens/2) 16B entries/expert
            1,
            tt::DataFormat::UInt32);

        // BRISC's expert activation buffer - same format as main expert_activation buffer
        // [token_id, k_indices[experts_per_device], scores[experts_per_device]] per activated token
        // Single page containing all activation rows (max ceil(tokens/2))
        uint32_t brisc_activation_row_size = tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment);
        tt::tt_metal::create_cb(
            brisc_expert_activation_cb_id,
            program,
            tilize_core_range_set,
            brisc_activation_row_size * tt::div_up(tokens, 2),  // full buffer in one page
            1,
            tt::DataFormat::UInt32);

        // BRISC's activated token count (single uint32_t to communicate to NCRISC)
        tt::tt_metal::create_cb(
            brisc_activated_count_cb_id, program, tilize_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);
    }

    // CB for receiving the records from non-drain tilize cores (only used on drain core)
    tt::tt_metal::create_cb(
        remote_counts_cb_id,
        program,
        tilize_core_range_set,
        remote_counts_entry_size,
        tilize_num_cores - 1,  // one entry per non-drain core
        tt::DataFormat::UInt32);

    // CB for passing total_chunks from the tilize reader to the tilize compute kernel
    // Single page holding one uint32_t value. Page size floored at l1_alignment /
    // CIRCULAR_BUFFER_COMPUTE_WORD_SIZE so the unpack LLK fifo_* fields (16 B words) are non-zero
    // when tilize_compute pops this CB.
    tt::tt_metal::create_cb(
        total_chunks_cb_id,
        program,
        tilize_core_range_set,
        non_tile_cb_page_size(tt::DataFormat::UInt32, l1_alignment),
        1,  // single page
        tt::DataFormat::UInt32);

    //-------------------------------------------------------------------------
    // Matmul CBs
    //-------------------------------------------------------------------------

    // CBs on matmul (ring) cores.  Tile counts depend on (hidden_size, intermediate_size).
    // Two reference configs for comparison:
    //   DeepSeek  — hidden=7168  intermediate=2048  (Ht=224, Nt=64,  a2a_cb_pages=6)
    //   GPT-OSS   — hidden=2880  intermediate=2880  (Ht=90,  Nt=90,  a2a_cb_pages=8)
    /*
        ----------------------------------------------------------------------------------------------------------
        | Name           | CB Index     | Dtype     | Tile? | Tiles/CB  | DS  | GPT | Remarks                    |
        ----------------------------------------------------------------------------------------------------------
        | cb_s2c_in      | CBIndex::c_0 | Float16_b | true  | (shared)  | 448 | 180 | Shared output buf          |
        | cb_r2c_w0      | CBIndex::c_3 | Bfp4_b    | true  | slots*2*t |  84 |  84 | 3 blocks W0/W1 + W2 (txn=  |
        |                |              |           |       |           |     |     | 14 tiles; 20 -> 3 x 40);   |
        |                |              |           |       |           |     |     | replay ring: one slice     |
        | cb_c2w_rdy     | CBIndex::c_4 | Float32   | false | 1         |   — |   — | Compute->writer ready      |
        | cb_w2c_rdy     | CBIndex::c_5 | Float32   | false | num_cores |   — |   — | Writer->compute ready      |
        | cb_s2c_in2     | CBIndex::c_6 | Float16_b | true  | a2a*cores |  72 |  96 | Ring A2A activation        |
        | cb_w2c_md      | CBIndex::c_7 | UInt32    | false | 2         |   — |   — | Metadata (token counts)    |
        ----------------------------------------------------------------------------------------------------------
        Non-tile CBs use page_size >= non_tile_cb_page_size(data_format, l1_alignment)
        so LLK fifo_* fields (16 B words; circular_buffer_constants.h) are non-zero on compute push/pop.
    */

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    // Sort cores by (descending y, descending x) to create a ring that flows naturally
    std::vector<uint32_t> ring_pos2bank_id(matmul_num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [mesh_device, &matmul_cores](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = mesh_device->worker_core_from_logical_core(matmul_cores[bank_id_a]);
            const auto& pb = mesh_device->worker_core_from_logical_core(matmul_cores[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;  // Descending y
            }
            return pa.x > pb.x;  // Descending x
        });

    // Build a map where key = bank_id, value = {ring position (i), neighbor's bank_id}
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> bank2ring_pos;
    for (uint32_t ring_pos = 0; ring_pos < matmul_num_cores; ++ring_pos) {
        uint32_t this_bank = ring_pos2bank_id[ring_pos];
        uint32_t next_bank = ring_pos2bank_id[(ring_pos + 1) % matmul_num_cores];
        bank2ring_pos[this_bank] = {ring_pos, next_bank};
    }

    // The weight CB (c_3). Today: dm0's slots of one block each (moe_ring::weight_cb_slots: 3 for 14- and 20-tile
    // transactions), whole blocks so the ring buffer wraps on a block boundary. The replay ring (prefill_rings = 1):
    // one whole expert slice per ring position -- its W0/W1 blocks plus its W2 blocks -- so dm0 reads the slice once
    // per expert and re-presents it for every further chunk; the capacity must equal the slice exactly (the pointers
    // wrap to the base), and the slice differs between positions with an even and an odd column count, so the CB is
    // created per group of positions with the same slice.
    const bool replay_slice = args.prefill_rings >= 1;  // several rings forward blocks by their place in the slice
    // The replay CB holds exactly one slice per position; a TP-split shared expert has a shorter one (dm0 sizes the
    // slice from w0_w1_prod_cols, the CB below from the stored columns), so the two must not meet.
    TT_FATAL(
        !replay_slice || num_shared_experts == 0,
        "prefill_rings>=1 replays one whole expert slice per ring core; shared experts (TP-split slices) are not "
        "admitted");
    const auto weight_cb_config = [&](uint32_t tiles) {
        return tt::tt_metal::CircularBufferConfig(
                   tiles * tt::tile_size(tt::DataFormat::Bfp4_b), {{tt::CBIndex::c_3, tt::DataFormat::Bfp4_b}})
            .set_page_size(tt::CBIndex::c_3, tt::tile_size(tt::DataFormat::Bfp4_b));
    };
    if (replay_slice) {
        const uint32_t k_dram_tiles = args.has_bias ? hidden_tiles + 1 : hidden_tiles;
        const uint32_t blocks_per_col = tt::div_up(k_dram_tiles, moe_ring::block_tiles_h(weight_tiles_per_txn));
        const uint32_t blocks_per_half_col =
            tt::div_up(k_dram_tiles, moe_ring::half_block_tiles_h(weight_tiles_per_txn));
        const uint32_t w2_blocks = moe_ring::w2_core_blocks_per_expert(
            hidden_tiles,
            args.has_bias ? intermediate_tiles + 1 : intermediate_tiles,
            matmul_num_cores,
            weight_tiles_per_txn);
        std::map<uint32_t, std::vector<tt::tt_metal::CoreRange>> cores_by_slice_blocks;
        for (uint32_t dram_bank = 0; dram_bank < matmul_num_cores; ++dram_bank) {
            const uint32_t ring_pos = bank2ring_pos[dram_bank].first;
            const uint32_t cols = moe_ring::w0_w1_stored_cols(intermediate_tiles, ring_pos, matmul_num_cores);
            const uint32_t slice_blocks =
                moe_ring::w0_w1_blocks_for_cols(cols, blocks_per_col, blocks_per_half_col) + w2_blocks;
            for (uint32_t r = 0; r < num_rings; ++r) {
                const auto& core = ring_cores[r * matmul_num_cores + dram_bank];
                cores_by_slice_blocks[slice_blocks].emplace_back(core, core);
            }
        }
        for (const auto& [slice_blocks, ranges] : cores_by_slice_blocks) {
            const uint32_t slice_bytes = slice_blocks * weight_tiles_per_block * tt::tile_size(tt::DataFormat::Bfp4_b);
            // tt-metal's own fit check at finalize names no cause: say what did not fit. Half the L1 is the budget
            // the rest of the ring core's CBs and L1 buffers leave at 2048 tokens (E; the fit check is the arbiter).
            TT_FATAL(
                slice_bytes <= mesh_device->l1_size_per_core() / 2,
                "prefill_rings>=1: the expert slice of {} blocks ({} B) on {} ring core(s) does not fit the replay CB "
                "budget (half of the {} B L1 per core)",
                slice_blocks,
                slice_bytes,
                ranges.size(),
                mesh_device->l1_size_per_core());
            log_debug(
                tt::LogOp,
                "moe_compute replay CB: {} blocks = {} B on {} ring core(s)",
                slice_blocks,
                slice_bytes,
                ranges.size());
            tt::tt_metal::CreateCircularBuffer(
                program, tt::tt_metal::CoreRangeSet(ranges), weight_cb_config(slice_blocks * weight_tiles_per_block));
        }
    } else {
        tt::tt_metal::CreateCircularBuffer(
            program, ring_core_range_set, weight_cb_config(weight_slots * weight_tiles_per_block));
    }

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in and cb_c2s_out are handled separately as it is allocated on Tilize, Matmul, and Combine cores
    std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> matmul_cb_specs0 = {
        {"cb_c2w_rdy", tt::CBIndex::c_4, tt::DataFormat::Float32, false, a2a_parity_slots},
        // one ready credit per ring step of the handshake iteration (moe_ring::a2a_handshake_iters): dm1 pushes them
        // as the shards land and the compute pops them as it consumes each shard, neither waiting on the other
        {"cb_w2c_rdy", tt::CBIndex::c_5, tt::DataFormat::Float32, false, matmul_num_cores * a2a_parity_slots},
        {"cb_s2c_in2",
         tt::CBIndex::c_6,
         tt::DataFormat::Float16_b,
         true,
         a2a_cb_pages * matmul_num_cores * a2a_parity_slots},
        {"cb_w2c_md", tt::CBIndex::c_7, tt::DataFormat::UInt32, false, 2},
    };
    if (args.has_bias) {
        // c_8 is free on matmul cores (c_8 is tilize_input_cb_id on tilize cores only).
        matmul_cb_specs0.emplace_back("cb_c2c_ones_tile", tt::CBIndex::c_8, tt::DataFormat::Float16_b, true, 1);
    }

    // Create CBs
    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : matmul_cb_specs0) {
        const uint32_t bytes_per_tile =
            is_tile ? tt::tile_size(data_format) : non_tile_cb_page_size(data_format, l1_alignment);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        tt::tt_metal::CreateCircularBuffer(program, ring_core_range_set, cb_config);
    }

    // LOCAL_OUTPUT: dm1's per-chunk copy of the current expert's packed entries (one 4-B (k slot, token id)
    // entry per token, a whole chunk per read), read from the e_t output page in DRAM: the L1 destination of a
    // DRAM read must be DRAM-aligned, so the CB carries one alignment of slack and dm1 aligns inside it. c_9 is
    // free on the matmul cores (c_8 is the bias ones tile).
    constexpr uint32_t local_output_map_cb_id = tt::CBIndex::c_9;
    if (local_output) {
        // two chunk slots under the a2a pipeline (dm1 reads chunk c+1's map while chunk c's rows read theirs)
        const uint32_t map_bytes =
            tt::align(tokens_per_chunk * moe_ring::token_list::ENTRY_BYTES, dram_alignment) * (a2a_pipeline ? 2u : 1u) +
            dram_alignment;
        tt::tt_metal::CreateCircularBuffer(
            program,
            ring_core_range_set,
            tt::tt_metal::CircularBufferConfig(map_bytes, {{local_output_map_cb_id, tt::DataFormat::UInt32}})
                .set_page_size(local_output_map_cb_id, map_bytes));
    }

    // LOCAL_OUTPUT: the output contract is "every row of [k, T, H] is what this op wrote": the rows
    // of experts this device holds carry their W2 results and every other row is zero, so the
    // per-device partials of a 1xN mesh sum directly, also into a reused caller tensor. Before its
    // first row write dm1 writes its W2 column slice of all k x T rows from this pre-zeroed L1
    // region (one row slice: the widest W2 column slice of the ring), overlapped with the tilize
    // phase. c_10 is free on the matmul cores (c_10..c_14 are tilize-core CBs).
    constexpr uint32_t local_output_zero_cb_id = tt::CBIndex::c_10;
    uint32_t local_output_num_rows = 0;
    if (local_output && args.zero_fill_non_owned_rows) {
        uint32_t max_w2_slice_tiles = 0;
        for (uint32_t core = 0; core < matmul_num_cores; ++core) {
            max_w2_slice_tiles = std::max(
                max_w2_slice_tiles, moe_ring::w2_shard_tiles(hidden_tiles, core, intermediate_tiles, matmul_num_cores));
        }
        const uint32_t zero_bytes =
            max_w2_slice_tiles * tt::constants::TILE_WIDTH * tt::datum_size(tilize_output_dataformat);
        tt::tt_metal::CreateCircularBuffer(
            program,
            ring_core_range_set,
            tt::tt_metal::CircularBufferConfig(zero_bytes, {{local_output_zero_cb_id, tilize_output_dataformat}})
                .set_page_size(local_output_zero_cb_id, zero_bytes));
        TT_FATAL(
            tensor_return_value.size() == 6,
            "path=SingleCluster expects 6 output tensors, got {}",
            tensor_return_value.size());
        const auto& local_output_shape = tensor_return_value[5].logical_shape();
        local_output_num_rows = local_output_shape[0] * local_output_shape[1];  // k x T
    }

    //-------------------------------------------------------------------------
    // Tilize kernels
    //-------------------------------------------------------------------------

    // For NOC 0: start = (min_x, min_y), end = (max_x, max_y)
    // For NOC 1: coordinates are swapped
    // We'll use NOC 0 by default, but pass both orderings and let the kernel handle it
    // Or we can determine the NOC here and swap if needed
    // For simplicity, we pass the NOC 0 ordering (start < end) and the kernel will use NOC 0

    // Store physical NOC coordinates of all tilize cores for cross-core communication
    // Used by drain core to read from non-drain cores, and non-drain to write to drain
    TT_FATAL(
        tilize_cores.size() >= tilize_num_cores,
        "tilize_cores ({}) must cover tilize_num_cores ({})",
        tilize_cores.size(),
        tilize_num_cores);
    std::vector<tt::tt_metal::CoreCoord> tilize_cores_physical(tilize_num_cores);
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        tilize_cores_physical[i] = mesh_device->worker_core_from_logical_core(tilize_cores[i]);
    }

    // tile_width_bytes = TILE_WIDTH * element_size
    // max_tiles_per_chunk = max_tilize_subtoken_size / tile_width_bytes
    uint32_t tile_width_bytes = tt::constants::TILE_WIDTH * tilize_input_tensor.element_size();
    uint32_t max_tiles_per_local_chunk = max_tilize_subtoken_size / tile_width_bytes;

    const uint32_t primary_mcast_gather_group_num_cores = tilize_num_cores / 2;
    const uint32_t secondary_mcast_gather_group_num_cores = tilize_num_cores - primary_mcast_gather_group_num_cores;

    // Drain core is always the first tilize core (index 0)
    TT_FATAL(!tilize_cores_physical.empty(), "tilize_cores_physical must be non-empty");
    tt::tt_metal::CoreCoord tilize_drain_core_physical = tilize_cores_physical[0];

    // combine_cores[0] is read below for the combine_sync NOC coords.
    TT_FATAL(!combine_cores.empty(), "combine_cores must be non-empty");
    std::unordered_map<std::string, uint32_t> tilize_named_compile_time_args = {
        // CBs
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"indices_tensor_cb_id", indices_tensor_cb_id},
        {"scores_tensor_cb_id", scores_tensor_cb_id},
        {"mapping_tensor_cb_id", mapping_tensor_cb_id},
        {"e_t_cb_id", e_t_cb_id},
        {"expert_activation_cb_id", expert_activation_cb_id},
        {"per_expert_total_tokens_cb_id", per_expert_total_tokens_cb_id},
        {"brisc_e_t_cb_id", brisc_e_t_cb_id},
        {"brisc_expert_counts_cb_id", brisc_expert_counts_cb_id},
        {"brisc_expert_activation_cb_id", brisc_expert_activation_cb_id},
        {"brisc_activated_count_cb_id", brisc_activated_count_cb_id},
        {"remote_counts_cb_id", remote_counts_cb_id},

        // Alignment
        {"l1_alignment", l1_alignment},
        {"dram_alignment", dram_alignment},
        {"e_t_entry_size", l1_alignment},

        // Number of pages
        {"input_pages", tilize_input_pages},
        {"indices_pages", tilize_indices_pages},
        {"mapping_pages", tilize_mapping_pages},
        {"scores_pages", tilize_input_scores_pages},
        {"shared_cb_num_pages", shared_cb_num_pages},
        {"chunk_slot_pages", hidden_tiles},

        // Page sizes
        {"input_page_size", tilize_input_page_size},
        {"indices_page_size", tilize_indices_page_size},
        {"mapping_page_size", tilize_mapping_page_size},
        {"per_expert_total_tokens_output_page_size", tilize_per_expert_total_tokens_output_page_size},
        {"expert_activation_output_page_size", tilize_expert_activation_output_page_size},
        {"e_t_output_page_size", tilize_e_t_output_page_size},
        {"tilize_output_page_size", tilize_output_page_size},

        // Aligned page sizes
        {"aligned_input_page_size", tilize_input_aligned_page_size},
        {"aligned_indices_page_size", tilize_indices_aligned_page_size},
        {"aligned_mapping_page_size", tilize_mapping_aligned_page_size},
        {"aligned_scores_page_size", tilize_input_scores_aligned_page_size},

        // General info
        {"tokens", tokens},
        {"hidden_size", hidden_size},
        {"remote_counts_entry_size", remote_counts_entry_size},
        {"experts", experts},
        {"experts_per_device", experts_per_device},
        {"selected_experts_k", selected_experts_k},

        // Chunk info
        {"tokens_per_chunk", tokens_per_chunk},

        // Mesh
        {"num_devices", num_devices},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"linearized_mesh_coord", linearized_mesh_coord},
        // ComputeOnly path has no combine_params, so cluster_axis() is nullopt; default to 1.
        // The CT arg is consumed by tilize_reader/tilize_writer (to derive dispatch_devices /
        // dispatch_index) — harmless on a 1x1 mesh because both axis values resolve to the same
        // dispatch_devices=1. Non-1x1 ComputeOnly callers would need cluster_axis threaded through.
        {"cluster_axis", (uint32_t)args.cluster_axis().value_or(1)},

        // Coordinates for non-drain-sync to drain-sync synchronization
        {"drain_core_noc_x", (uint32_t)tilize_drain_core_physical.x},
        {"drain_core_noc_y", (uint32_t)tilize_drain_core_physical.y},

        // Gather groups
        {"primary_mcast_gather_group_num_cores", primary_mcast_gather_group_num_cores},
        {"secondary_mcast_gather_group_num_cores", secondary_mcast_gather_group_num_cores},

        // T multicast coordinates
        {"num_tilize_cores", tilize_num_cores},

        {"tilize_mcast_start_x", (uint32_t)tilize_mcast_start_physical.x},
        {"tilize_mcast_start_y", (uint32_t)tilize_mcast_start_physical.y},
        {"tilize_mcast_end_x", (uint32_t)tilize_mcast_end_physical.x},
        {"tilize_mcast_end_y", (uint32_t)tilize_mcast_end_physical.y},
        {"tilize_bounding_box_num_cores", tilize_bounding_box_num_cores},

        // MM multicast coordinates
        {"num_matmul_cores", matmul_num_cores},

        {"matmul_mcast_start_x", (uint32_t)matmul_mcast_start_physical.x},
        {"matmul_mcast_start_y", (uint32_t)matmul_mcast_start_physical.y},
        {"matmul_mcast_end_x", (uint32_t)matmul_mcast_end_physical.x},
        {"matmul_mcast_end_y", (uint32_t)matmul_mcast_end_physical.y},
        {"matmul_bounding_box_num_cores", matmul_mcast_num_dests},

        // All worker cores multicast coordinates
        // TODO(#41827): when path == ComputeOnly, this bounding box still includes the
        // unused combine box (5..6, 0..7). Tilize_reader mcasts to kernel-less L1 there.
        // Reduce to tilize_matmul_bounding_box in that mode; deferred to a follow-up PR.
        {"all_worker_cores_mcast_start_x", (uint32_t)all_worker_cores_mcast_start_physical.x},
        {"all_worker_cores_mcast_start_y", (uint32_t)all_worker_cores_mcast_start_physical.y},
        {"all_worker_cores_mcast_end_x", (uint32_t)all_worker_cores_mcast_end_physical.x},
        {"all_worker_cores_mcast_end_y", (uint32_t)all_worker_cores_mcast_end_physical.y},
        {"all_worker_cores_bounding_box_num_cores", all_worker_cores_bounding_box_num_cores},

        // Semaphores
        {"partial_metadata_ready_semaphore_id", tilize_partial_metadata_ready_semaphore_id},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"chunk_halves", chunk_halves},
        {"half_free_semaphore_id_0", half_free_semaphore_ids[0]},
        {"half_free_semaphore_id_1", half_free_semaphore_ids[1]},
        {"half_free_semaphore_id_2", half_free_semaphore_ids[2]},
        {"half_free_semaphore_id_3", half_free_semaphore_ids[3]},
        {"half_free_semaphore_id_4", half_free_semaphore_ids[4]},
        {"tilize_chunk_ready_semaphore_id", tilize_chunk_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"initial_gather_semaphore_id", initial_gather_semaphore_id},
        // the secondary multicaster (the first core of the second gather group; the drain itself with one core)
        {"secondary_mcaster_noc_x",
         (uint32_t)tilize_cores_physical[tilize_num_cores > 1 ? primary_mcast_gather_group_num_cores : 0].x},
        {"secondary_mcaster_noc_y",
         (uint32_t)tilize_cores_physical[tilize_num_cores > 1 ? primary_mcast_gather_group_num_cores : 0].y},
        {"feed_go_semaphore_id", feed_go_semaphore_id},
        {"gather_semaphore_id_0", gather_semaphore_ids[0]},
        {"gather_semaphore_id_1", gather_semaphore_ids[1]},
        {"gather_semaphore_id_2", gather_semaphore_ids[2]},
        {"gather_semaphore_id_3", gather_semaphore_ids[3]},
        {"gather_semaphore_id_4", gather_semaphore_ids[4]},
        // Tilize -> combine: drain core signals combine when metadata is ready.
        // In ComputeOnly mode the increment is gated off in the kernel via the compute_only CT arg.
        // combine_cores is still computed in ComputeOnly mode so combine_cores[0] is valid here.
        {"combine_sync_semaphore_id", tilize_combine_sync_semaphore_id},
        {"combine_sync_noc_x", (uint32_t)mesh_device->worker_core_from_logical_core(combine_cores[0]).x},
        {"combine_sync_noc_y", (uint32_t)mesh_device->worker_core_from_logical_core(combine_cores[0]).y},

        // Bypass selective_reduce_combine path entirely (1=skip combine semaphore inc/wait/set).
        {"compute_only", args.path == MoEComputePath::ComputeOnly ? 1u : 0u},
        // SingleCluster: no combine kernels either; the routing is kept as packed (token, k slot) lists and the
        // drain publishes the packed page before releasing the matmul cores, since dm1 reads it for its output
        // rows. pairs_capacity_bytes sizes one RISC's pair list (a staging slot on the drain).
        {"local_output", local_output ? 1u : 0u},
        {"pairs_capacity_bytes", pairs_capacity_bytes}};

    std::vector<uint32_t> tilize_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(tilize_input_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_indices_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_input_scores_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_mapping_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_per_expert_total_tokens_output_tensor.buffer())
        .append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_expert_activation_output_tensor.buffer())
        .append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_e_t_output_tensor.buffer()).append_to(tilize_compile_time_args);

    // Study knob: TTNN_MOE_COMPUTE_ZONES=[first:count] builds the tilize and ring kernels with per-chunk device
    // profiler zones (moe_ring_common.h MOE_ZONE_IF; recorded for `count` chunks from `first` in feed order, defaults
    // 48:16, inside the profiler's 250-marker L1 budget per RISC). Not part of the program hash: set it for the whole
    // process, never in production.
    std::map<std::string, std::string> tilize_defines;
    if (const char* zones = std::getenv("TTNN_MOE_COMPUTE_ZONES"); zones != nullptr && *zones != '\0') {
        tilize_defines["MOE_ZONES"] = "1";
        const std::string spec(zones);
        if (const auto colon = spec.find(':'); colon != std::string::npos) {
            std::size_t first_end = 0, count_end = 0;
            const unsigned long first = std::stoul(spec.substr(0, colon), &first_end);
            const unsigned long count = std::stoul(spec.substr(colon + 1), &count_end);
            TT_FATAL(
                first_end == colon && colon + 1 + count_end == spec.size() && count > 0,
                "TTNN_MOE_COMPUTE_ZONES must be first:count (two non-negative integers, count > 0), got '{}'",
                spec);
            tilize_defines["MOE_ZONES_FIRST_CHUNK"] = std::to_string(first);
            tilize_defines["MOE_ZONES_CHUNKS"] = std::to_string(count);
        }
        log_warning(
            tt::LogOp, "moe_compute: the tilize and ring kernels carry profiler zones ({}): a study build", spec);
    }
    // Study knob: TTNN_MOE_COMPUTE_STUDY_DELAYS=POINT:iterations[,POINT:iterations] builds every moe_compute kernel
    // with a spin of `iterations` at the named handoff of the feed protocol (moe_ring_common.h MOE_STUDY_DELAY; the
    // points are listed there). Not part of the program hash: whole-process only, never in production.
    if (const char* delays = std::getenv("TTNN_MOE_COMPUTE_STUDY_DELAYS"); delays != nullptr && *delays != '\0') {
        static constexpr std::array<const char*, 18> study_points = {
            "W_BEFORE_MCAST",
            "W_BEFORE_SIGNAL",
            "W_BEFORE_GO",
            "W_BEFORE_POP",
            "W_GATHER_BEFORE_WRITE",
            "W_GATHER_BEFORE_INC",
            "W_SECONDARY_BEFORE_INC",
            "R_BEFORE_READS",
            "O_BEFORE_CREDIT",
            "O_A2A_BEFORE_INC",
            "O_BEFORE_ROWS",
            "D_BEFORE_SLICE",
            "M_BEFORE_W0W1",
            "O_A2A_LAG",
            "O_A2A_LAG_POS",
            "X_GATHER_NO_GO_WAIT",
            "X_DRAIN_NO_FLUSH",
            "X_DRAIN_NO_HALF_WAIT"};
        std::string spec(delays);
        std::size_t begin = 0;
        while (begin <= spec.size()) {
            const std::size_t end = std::min(spec.find(',', begin), spec.size());
            const std::string item = spec.substr(begin, end - begin);
            const std::size_t colon = item.find(':');
            TT_FATAL(
                colon != std::string::npos,
                "TTNN_MOE_COMPUTE_STUDY_DELAYS item must be POINT:iterations, got '{}'",
                item);
            const std::string point = item.substr(0, colon);
            std::size_t value_end = 0;
            const unsigned long iterations = std::stoul(item.substr(colon + 1), &value_end);
            const bool parameter = point.size() > 4 && point.ends_with("_POS");
            TT_FATAL(
                colon + 1 + value_end == item.size() && (iterations > 0 || parameter),
                "TTNN_MOE_COMPUTE_STUDY_DELAYS item must be POINT:iterations (iterations > 0), got '{}'",
                item);
            TT_FATAL(
                std::find(study_points.begin(), study_points.end(), point) != study_points.end(),
                "TTNN_MOE_COMPUTE_STUDY_DELAYS: unknown point '{}' (see moe_ring_common.h)",
                point);
            tilize_defines["MOE_STUDY_DELAY_" + point] = std::to_string(iterations);
            begin = end + 1;
        }
        log_warning(tt::LogOp, "moe_compute: every kernel carries study delays ({}): a study build", spec);
    }
    const std::map<std::string, std::string>& zone_defines = tilize_defines;

    tt::tt_metal::KernelHandle tilize_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp",
        tilize_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = tilize_compile_time_args,
            .defines = tilize_defines,
            .named_compile_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    tt::tt_metal::KernelHandle tilize_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp",
        tilize_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = tilize_compile_time_args,
            .defines = tilize_defines,
            .named_compile_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    // Compute kernel compile-time args for tilization
    std::unordered_map<std::string, uint32_t> compute_tilize_named_compile_time_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"tokens_per_chunk", tokens_per_chunk},
        {"max_tiles_per_local_chunk", max_tiles_per_local_chunk},
        {"shared_cb_num_pages", shared_cb_num_pages},
        {"chunk_slot_pages", hidden_tiles},
    };

    tt::tt_metal::KernelHandle tilize_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_compute.cpp",
        tilize_core_range_set,
        tt::tt_metal::ComputeConfig{
            .defines = tilize_defines, .named_compile_args = compute_tilize_named_compile_time_args});

    std::vector<uint32_t> tilize_runtime_args = {
        tilize_input_tensor.buffer()->address(),                           // 0
        tilize_indices_tensor.buffer()->address(),                         // 1
        tilize_input_scores_tensor.buffer()->address(),                    // 2
        tilize_mapping_tensor.buffer()->address(),                         // 3
        tilize_per_expert_total_tokens_output_tensor.buffer()->address(),  // 4
        tilize_expert_activation_output_tensor.buffer()->address(),        // 5
        tilize_e_t_output_tensor.buffer()->address(),                      // 6
    };

    uint32_t is_drain_tilize_core_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 7: is_drain_tilize_core
    uint32_t is_secondary_mcaster_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 8: is_secondary_mcaster

    // Initial split mcast cores
    uint32_t initial_mcast_gather_core_nox_x_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 9: initial_mcast_gather_core_nox_x
    uint32_t initial_mcast_gather_core_nox_y_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 10: initial_mcast_gather_core_nox_x

    // Add work split runtime args for tilize cores
    uint32_t global_subtoken_offset_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 11: global_subtoken_offset
    uint32_t mcast_group_subtoken_offset_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 12: group_subtoken_offset

    uint32_t mcast_group_subtoken_size_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 13: group_subtoken_size
    uint32_t subtoken_size_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 14: subtoken_size

    // Token range for parallel metadata processing across tilize cores
    uint32_t core_token_start_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 15: core_token_start
    uint32_t core_token_end_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 16: core_token_end
    uint32_t tilize_core_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 17: tilize_core_idx (0 = drain, 1-3 = non-drain)

    // NOC coordinates for all tilize cores (for cross-core communication)
    // Runtime args starting at index 18: [core0_noc_x, core0_noc_y, core1_noc_x, core1_noc_y, ...]
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical[i].x);
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical[i].y);
    }

    // Boundary check for all derived-index writes into tilize_runtime_args below.
    // Highest index used is tilize_core_idx (17), so size must cover [0..17].
    TT_FATAL(
        tilize_runtime_args.size() > tilize_core_idx,
        "tilize_runtime_args size {} must exceed tilize_core_idx {}",
        tilize_runtime_args.size(),
        tilize_core_idx);

    // Calculate number of bytes per mcast_gather_group
    uint32_t primary_mcast_gather_group_subtoken_size = 0;
    uint32_t secondary_mcast_gather_group_subtoken_size = 0;
    uint32_t global_subtoken_offset = 0;
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        uint32_t subtoken_size = 0;
        if (tilize_cores_work_group_1.contains(tilize_cores[i])) {
            subtoken_size = tilize_units_per_core_work_group_1 * tilize_subtoken_bytes_aligned;
        } else if (tilize_cores_work_group_2.contains(tilize_cores[i])) {
            subtoken_size = tilize_units_per_core_work_group_2 * tilize_subtoken_bytes_aligned;
        }

        // Clamp to not exceed the total token size
        if (global_subtoken_offset + subtoken_size > tilize_input_aligned_page_size) {
            subtoken_size = tilize_input_aligned_page_size - global_subtoken_offset;
        }

        if (i < primary_mcast_gather_group_num_cores) {
            primary_mcast_gather_group_subtoken_size += subtoken_size;
        } else {
            secondary_mcast_gather_group_subtoken_size += subtoken_size;
        }

        global_subtoken_offset += subtoken_size;
    }

    // Calculate tokens per tilize core for parallel metadata processing
    uint32_t tokens_per_tilize_core = tokens / tilize_num_cores;

    // Compute kernel runtime args (separate from reader/writer)
    std::vector<uint32_t> tilize_compute_runtime_args = {0};  // [0]: tiles_per_chunk (set per-core below)

    global_subtoken_offset = 0;
    uint32_t group_subtoken_offset = 0;
    TT_FATAL(!tilize_compute_runtime_args.empty(), "tilize_compute_runtime_args must be non-empty");
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        // First tilize core is the drain tilize core (has indices/scores sharded to it)
        tilize_runtime_args[is_drain_tilize_core_idx] = (i == 0) ? 1 : 0;
        tilize_runtime_args[is_secondary_mcaster_idx] =
            (tilize_num_cores > 1 && i == primary_mcast_gather_group_num_cores) ? 1 : 0;

        // Initial split mcast cores
        tt::tt_metal::CoreCoord initial_mcast_gather_core_physical =
            i < primary_mcast_gather_group_num_cores ? tilize_cores_physical[0]
                                                     : tilize_cores_physical[primary_mcast_gather_group_num_cores];
        tilize_runtime_args[initial_mcast_gather_core_nox_x_idx] = (uint32_t)initial_mcast_gather_core_physical.x;
        tilize_runtime_args[initial_mcast_gather_core_nox_y_idx] = (uint32_t)initial_mcast_gather_core_physical.y;

        // Set work split parameters based on which group the core is in
        tilize_runtime_args[global_subtoken_offset_idx] = global_subtoken_offset;
        tilize_runtime_args[mcast_group_subtoken_offset_idx] = group_subtoken_offset;
        tilize_runtime_args[mcast_group_subtoken_size_idx] = i < primary_mcast_gather_group_num_cores
                                                                 ? primary_mcast_gather_group_subtoken_size
                                                                 : secondary_mcast_gather_group_subtoken_size;

        uint32_t subtoken_size = 0;
        if (tilize_cores_work_group_1.contains(tilize_cores[i])) {
            subtoken_size = tilize_units_per_core_work_group_1 * tilize_subtoken_bytes_aligned;
        } else if (tilize_cores_work_group_2.contains(tilize_cores[i])) {
            subtoken_size = tilize_units_per_core_work_group_2 * tilize_subtoken_bytes_aligned;
        }
        if (global_subtoken_offset + subtoken_size > tilize_input_aligned_page_size) {
            // Clamp to not exceed the total token size
            subtoken_size = tilize_input_aligned_page_size - global_subtoken_offset;
        }
        tilize_runtime_args[subtoken_size_idx] = subtoken_size;

        global_subtoken_offset += subtoken_size;
        group_subtoken_offset += subtoken_size;
        if (i == primary_mcast_gather_group_num_cores - 1) {
            group_subtoken_offset = 0;
        }

        // Set token range for this core's metadata processing
        // Each core processes a contiguous range of tokens
        uint32_t core_token_start = i * tokens_per_tilize_core;
        uint32_t core_token_end = (i == tilize_num_cores - 1) ? tokens : (i + 1) * tokens_per_tilize_core;
        tilize_runtime_args[core_token_start_idx] = core_token_start;
        tilize_runtime_args[core_token_end_idx] = core_token_end;
        tilize_runtime_args[tilize_core_idx] = i;

        // Set compute kernel runtime args
        tilize_compute_runtime_args[0] = subtoken_size / tile_width_bytes;

        tt::tt_metal::SetRuntimeArgs(program, tilize_reader_kernel_id, tilize_cores[i], tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, tilize_writer_kernel_id, tilize_cores[i], tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, tilize_compute_kernel_id, tilize_cores[i], tilize_compute_runtime_args);
    }

    //-------------------------------------------------------------------------
    // Matmul kernels
    //-------------------------------------------------------------------------

    // Create compile args for the program
    const auto matmul_tensors =
        std::vector<const ttnn::Tensor*>{&matmul_w0_w1_tensor, &matmul_w2_tensor, &tilize_output_tensor};

    std::vector<uint32_t> matmul_compile_time_args;
    for (const auto& tensor : matmul_tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(matmul_compile_time_args);
    }
    if (local_output) {
        // LOCAL_OUTPUT: dm1 reads the e_t pages and writes the final output through TensorAccessors
        // whose args follow the three matmul tensors (dm0 and compute stop after those). The output
        // accessor is built from the actual slot-5 buffer (the caller's optional_output_tensor when
        // given, else the tensor allocated from combine_params.output_memory_config): interleaved or
        // height-sharded, DRAM or L1, its bank / shard-core coordinates become compile-time args. A
        // program-cache hit reuses them: the hash covers the optional tensor's spec and
        // combine_params, so the distribution is the same and only the address changes
        // (override_runtime_arguments below).
        TT_FATAL(
            tensor_return_value.size() == 6,
            "path=SingleCluster expects 6 output tensors, got {}",
            tensor_return_value.size());
        tt::tt_metal::TensorAccessorArgs(*tilize_e_t_output_tensor.buffer()).append_to(matmul_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*tensor_return_value[5].buffer()).append_to(matmul_compile_time_args);
    }

    // parameters for writing to output shards
    const uint32_t tile_width = tilize_input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tilize_input_tensor.tensor_spec().tile().get_height();
    const uint32_t output_height_shard_dim = args.output_height_shard_dim;
    const uint32_t output_shard_width_tiles = hidden_size / tile_width / combine_data_parallel_cores;
    const uint32_t token_segment_size_bytes =
        output_shard_width_tiles * tile_width * tilize_output_tensor.element_size();
    constexpr uint32_t num_source_buffers = 2;
    const auto& source_shard_shape = tilize_output_tensor.memory_config().shard_spec()->shape;
    // token_expert_row_offset is in token-segment rows (hidden_size / combine_data_parallel_cores
    // wide): dm1 turns it into the byte offset of ring entry 1 inside each combine core's shard
    // (rows * combine_shard_width_tiles * tile_width_size_bytes). The shard itself is
    // [2 x 32, hidden_size], so the offset is 32 * combine_data_parallel_cores segment rows, not 32.
    // Same helper and arguments as selective_reduce_combine's consumer side.
    const auto fused_source_layout = ttnn::experimental::prim::detail::compute_fused_source_buffer_layout(
        source_shard_shape[0],
        source_shard_shape[1],
        hidden_size / combine_data_parallel_cores,
        tilize_output_tensor.buffer()->aligned_size_per_bank(),
        token_segment_size_bytes,
        num_source_buffers);
    const uint32_t token_expert_row_offset = fused_source_layout.rows_per_buffer;

    // NOC_MAX_BURST_SIZE — arch-dependent, used by dm1 to split ring A2A packets
    const uint32_t noc_max_burst_bytes = (mesh_device->arch() == tt::ARCH::BLACKHOLE) ? 16384u : 8192u;

    // activation function
    const ttnn::experimental::prim::detail::MoEActivationFunction activation_type = args.activation_type;

    // num_banks: number of physical DRAM banks the HEIGHT_SHARDED weight tensor lives on.
    // Ring size equals the live bank count: 12 on WH (no DRAM-bank harvesting), 7/8 on BH.
    // Ring cores and banks are 1:1, so no cross-bank walk is needed.
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    // w2_pages_per_ring_core_total: number of tile-pages each ring core "owns" in the FLAT
    // layout of the HEIGHT_SHARDED W2 tensor. The flat layout is core-major (ring_core_0's
    // tiles for all (layer, expert), then ring_core_1's, etc.), so this value equals
    // total_pages / num_cores. The kernel uses it to derive the global page offset for each
    // (ring_core, layer, expert).
    //
    // W0/W1 uses the compact per-expert bank-balanced layout (moe_ring_common.h): every bank
    // holds w0_w1_bank_blocks_per_expert whole blocks per (layer, expert). dm0 derives the
    // geometry from the shape compile args; check here that the tensor was packed that way
    // (a tensor packed with the old per-core stride has a different page count).
    const uint32_t w0_w1_total_pages_buf = static_cast<uint32_t>(matmul_w0_w1_tensor.buffer()->num_pages());
    const uint32_t w2_total_pages_buf = static_cast<uint32_t>(matmul_w2_tensor.buffer()->num_pages());
    {
        const uint32_t w0_w1_layers = matmul_w0_w1_tensor.logical_shape()[1];
        const uint32_t w0_w1_bank_blocks = moe_ring::w0_w1_bank_blocks_per_expert(
            args.has_bias ? hidden_tiles + 1 : hidden_tiles,
            intermediate_tiles,
            matmul_num_cores,
            num_dram_banks,
            weight_tiles_per_txn);
        const uint32_t w0_w1_expected_pages =
            num_dram_banks * w0_w1_layers * experts_per_device * w0_w1_bank_blocks * weight_tiles_per_block;
        TT_FATAL(
            w0_w1_total_pages_buf == w0_w1_expected_pages,
            "moe_compute: w0_w1 tensor has {} tile pages, the compact layout for {} layers x {} experts over {} banks "
            "needs {} ({} blocks of {} tiles per bank per expert); pack it with prepare_w0_w1_tensor_for_moe_compute",
            w0_w1_total_pages_buf,
            w0_w1_layers,
            experts_per_device,
            num_dram_banks,
            w0_w1_expected_pages,
            w0_w1_bank_blocks,
            weight_tiles_per_block);
        const uint32_t w2_layers = matmul_w2_tensor.logical_shape()[1];
        const uint32_t w2_core_blocks = moe_ring::w2_core_blocks_per_expert(
            hidden_tiles,
            args.has_bias ? intermediate_tiles + 1 : intermediate_tiles,
            matmul_num_cores,
            weight_tiles_per_txn);
        const uint32_t w2_expected_pages =
            matmul_num_cores * w2_layers * experts_per_device * w2_core_blocks * weight_tiles_per_block;
        TT_FATAL(
            w2_total_pages_buf == w2_expected_pages,
            "moe_compute: w2 tensor has {} tile pages, the layout for {} layers x {} experts over {} ring cores needs "
            "{} ({} blocks of {} tiles per core per expert); pack it with prepare_w2_tensor_for_moe_compute",
            w2_total_pages_buf,
            w2_layers,
            experts_per_device,
            matmul_num_cores,
            w2_expected_pages,
            w2_core_blocks,
            weight_tiles_per_block);
    }
    TT_FATAL(
        w2_total_pages_buf % matmul_num_cores == 0,
        "moe_compute: w2 total pages ({}) not divisible by num_cores ({})",
        w2_total_pages_buf,
        matmul_num_cores);

    uint32_t shared_expert_tp_factor;
    if (args.path == MoEComputePath::ComputeOnly) {
        // concept of a shared expert doesn't hold in the compute only case.
        shared_expert_tp_factor = 1;
    } else {
        auto* mesh_device = tilize_input_tensor.device();
        const auto& mesh_shape = mesh_device->get_view().shape();
        shared_expert_tp_factor = mesh_shape[1 - args.cluster_axis().value()];
    }
    const uint32_t w2_pages_per_ring_core_total = w2_total_pages_buf / matmul_num_cores;
    std::unordered_map<std::string, uint32_t> matmul_named_compile_time_args = {
        {"num_experts", experts_per_device},
        {"num_shared_experts", num_shared_experts},
        {"shared_expert_tp_factor", shared_expert_tp_factor},
        {"layer_id", args.layer_id},
        {"has_bias", args.has_bias ? 1u : 0u},
        {"num_cores", static_cast<uint32_t>(matmul_num_cores)},
        {"num_banks", num_dram_banks},
        {"a2a_free_semaphore_id", a2a_free_semaphore_id},
        {"prefill_rings", args.prefill_rings},
        {"w2_pages_per_ring_core_total", w2_pages_per_ring_core_total},
        {"activation_function", static_cast<uint32_t>(activation_type)},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"chunk_halves", chunk_halves},
        {"half_free_semaphore_id_0", half_free_semaphore_ids[0]},
        {"half_free_semaphore_id_1", half_free_semaphore_ids[1]},
        {"half_free_semaphore_id_2", half_free_semaphore_ids[2]},
        {"half_free_semaphore_id_3", half_free_semaphore_ids[3]},
        {"half_free_semaphore_id_4", half_free_semaphore_ids[4]},
        {"per_expert_total_tokens_cb_id", per_expert_total_tokens_cb_id},
        {"tokens_per_chunk", tokens_per_chunk},
        {"tilize_drain_core_noc_x", (uint32_t)tilize_drain_core_physical.x},
        {"tilize_drain_core_noc_y", (uint32_t)tilize_drain_core_physical.y},
        // NCT args for sharded output for combine
        {"combine_shard_width_tiles", output_shard_width_tiles},
        {"tile_height", tile_height},
        {"tile_width", tile_width},
        {"tile_width_size_bytes", tile_width * tt::datum_size(tilize_output_dataformat)},
        {"token_expert_row_offset", token_expert_row_offset},
        {"height_shard_dim", output_height_shard_dim},
        {"width_shard_dim", combine_data_parallel_cores},
        {"hidden_tiles", hidden_tiles},
        {"tiles_per_txn", weight_tiles_per_txn},
        {"weight_slots", weight_slots},
        {"intermediate_tiles", intermediate_tiles},
        {"noc_max_burst_bytes", noc_max_burst_bytes},
        // Matmul -> combine: dm1 increments this on combine cores when data is written
        {"matmul_combine_sync_semaphore_id", matmul_combine_sync_semaphore_id},
        // Bypass combine signaling/wait when no combine kernels are built.
        {"compute_only", args.path == MoEComputePath::ComputeOnly ? 1u : 0u},
        // LOCAL_OUTPUT (dm1): write each token row slice straight into the final [k, T, H] output,
        // page k * total_tokens + t from the expert's packed token-list entries.
        {"local_output", local_output ? 1u : 0u},
        {"total_tokens", tokens},
        {"e_t_entry_size", l1_alignment},
        {"dram_alignment", dram_alignment},
        {"local_output_map_cb_id", local_output_map_cb_id},
        // Zero fill of the rows this device does not own: the pre-zeroed row-slice CB and k x T.
        {"zero_fill", local_output && args.zero_fill_non_owned_rows ? 1u : 0u},
        // The replay ring (dm0): the weight CB holds one expert slice, read once per expert, replayed per chunk.
        {"replay_slice", replay_slice ? 1u : 0u},
        {"a2a_pipeline", a2a_pipeline},
        {"local_output_zero_cb_id", local_output_zero_cb_id},
        {"local_output_num_rows", local_output_num_rows},
    };

    // Create kernels for the program
    auto matmul_dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm0.cpp",
        ring_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = matmul_compile_time_args,
            .defines = zone_defines,
            .named_compile_args = matmul_named_compile_time_args});

    std::map<std::string, std::string> dm1_defines = zone_defines;
    dm1_defines["OUTPUT_SHARD_CORE_MAP"] = serialize_physical_core_coords(combine_cores, *mesh_device);

    auto matmul_dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp",
        ring_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = matmul_compile_time_args,
            .defines = dm1_defines,
            .named_compile_args = matmul_named_compile_time_args});

    auto matmul_compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp",
        ring_core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = matmul_compile_time_args,
            .defines = zone_defines,
            .named_compile_args = matmul_named_compile_time_args});

    //-------------------------------------------------------------------------
    // ring ordering
    //-------------------------------------------------------------------------

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> matmul_runtime_args;
    // bank, vchannel, the tensors, the ring semaphore, ring position, successor x/y, ring index, predecessor x/y
    // (9 slots beside the tensors), shard_to_bank, and (SingleCluster) the output and e_t addresses
    matmul_runtime_args.reserve(9 + matmul_tensors.size() + num_dram_banks + 2);
    matmul_runtime_args.push_back(0);  // DRAM Bank ID placeholder
    matmul_runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : matmul_tensors) {
        matmul_runtime_args.push_back(tensor->buffer()->address());
    }
    // Add placeholders for neighbor physical coords and semaphore
    matmul_runtime_args.push_back(ring_semaphore_id);  // Semaphore ID
    matmul_runtime_args.push_back(0);                  // Ring core ID placeholder
    matmul_runtime_args.push_back(0);                  // Neighbor physical x
    matmul_runtime_args.push_back(0);                  // Neighbor physical y
    matmul_runtime_args.push_back(0);                  // Ring index (which prefill ring the core belongs to)
    matmul_runtime_args.push_back(0);                  // Predecessor physical x (the a2a credit's target)
    matmul_runtime_args.push_back(0);                  // Predecessor physical y

    // Append shard_to_bank table: shard_to_bank[i] = chip bank id holding shard i.
    //
    // We query the buffer's actual page mapping (built by ttnn) to recover the precise
    // shard-index → bank-id correspondence. ttnn's `all_cores` list holds CoreCoords of
    // the form `(bank_id, 0)` for DRAM-sharded buffers; the i-th entry is the chip bank
    // holding shard `i`. Using the page mapping directly avoids depending on the C++
    // `ring_pos2bank_id` ordering, which may differ from the Python `sorted_dram_core_coords`.
    {
        const auto& mapping = matmul_w0_w1_tensor.buffer()->get_buffer_page_mapping();
        TT_FATAL(
            mapping->all_cores.size() == num_dram_banks,
            "moe_compute: w0_w1 buffer page mapping has {} cores, expected num_dram_banks={}",
            mapping->all_cores.size(),
            num_dram_banks);
        for (const auto& c : mapping->all_cores) {
            TT_FATAL(
                c.y == 0,
                "moe_compute: DRAM shard core ({}, {}) has y != 0; expected DRAM CoreCoord(bank_id, 0)",
                c.x,
                c.y);
            matmul_runtime_args.push_back(static_cast<uint32_t>(c.x));
        }
    }
    if (local_output) {
        // LOCAL_OUTPUT: final output and e_t addresses trail dm0's bank table (dm0 reads that table
        // by position); override_runtime_arguments refreshes these last two entries.
        matmul_runtime_args.push_back(tensor_return_value[5].buffer()->address());
        matmul_runtime_args.push_back(tilize_e_t_output_tensor.buffer()->address());
    }

    // shard_to_bank translation table: maps shard index → physical chip DRAM bank id.
    //
    // Why we need this: The Python helper builds `dram_core_range_set` from
    // `[CoreCoord(sorted_dram_core_coords[ring_pos], 0) for ring_pos in 0..num_banks-1]`,
    // where `sorted_dram_core_coords[ring_pos]` is the chip bank id whose adjacent worker
    // core is at the ring's `ring_pos`. Critically, `CoreRangeSet(list[CoreRange])` does
    // NOT call `merge_ranges()` (only the `Span<const CoreCoord>` constructor does), so
    // the ring-pos ordering is preserved across `corerange_to_cores`. ttnn HEIGHT_SHARDED
    // places shard `i` on the i-th core in that traversal order — i.e., shard `i` lands
    // on bank `sorted_dram_core_coords[i] == ring_pos2bank_id[i]`, not bank `i`.
    //
    // The kernel's bank-run loop computes `shard_idx = gp / pages_per_bank_total`. To find
    // the actual chip bank, it needs `bank = shard_to_bank[shard_idx]`. We pass this table
    // as runtime args (one entry per bank, after the 12 standard args, indices 0..11). Ring size equals the
    // live bank count, so shard_idx and ring_pos are the same (12 on WH, 7/8 on BH).
    //
    // Sanity-check the divisibility invariants the bank-run kernel relies on. The Python
    // helper `get_weight_mem_configs` enforces matching invariants when constructing the
    // ShardSpec, so this is belt-and-braces.
    {
        const uint32_t w0_w1_total_pages = static_cast<uint32_t>(matmul_w0_w1_tensor.buffer()->num_pages());
        const uint32_t w2_total_pages = static_cast<uint32_t>(matmul_w2_tensor.buffer()->num_pages());
        TT_FATAL(
            w0_w1_total_pages % num_dram_banks == 0,
            "moe_compute: w0_w1 total pages ({}) must be divisible by num_banks ({})",
            w0_w1_total_pages,
            num_dram_banks);
        TT_FATAL(
            w2_total_pages % num_dram_banks == 0,
            "moe_compute: w2 total pages ({}) must be divisible by num_banks ({})",
            w2_total_pages,
            num_dram_banks);
        TT_FATAL(
            w2_total_pages % matmul_num_cores == 0,
            "moe_compute: w2 total pages ({}) must be divisible by num_cores ({})",
            w2_total_pages,
            matmul_num_cores);
    }

    // matmul cores ordered by core ID, this will be used by selective combine to direct semaphore signaling
    std::vector<tt::tt_metal::CoreCoord> ring_pos2core(matmul_num_cores);
    // One NoC virtual channel per ring core, distinct within a row (ring_core_vchannels)
    const std::vector<uint32_t> vchannels =
        ttnn::operations::ccl::common::ring_core_vchannels(ring_cores, matmul_num_cores);
    // Every ring is the same ring of positions: ring r's core at position p passes to ring r's core at the next
    // position. The DRAM bank of a position is the same on every ring (only ring 0 reads it).
    for (uint32_t core_idx = 0; core_idx < ring_cores.size(); ++core_idx) {
        const auto core = ring_cores[core_idx];
        const uint32_t ring = core_idx / matmul_num_cores;
        const uint32_t dram_bank = core_idx % matmul_num_cores;
        const uint32_t vchannel = vchannels[core_idx];

        // Use the optimized ring neighbor mapping
        const auto [ring_pos, next_bank] = bank2ring_pos[dram_bank];
        const auto& next_physical =
            mesh_device->worker_core_from_logical_core(ring_cores[ring * matmul_num_cores + next_bank]);
        const uint32_t prev_bank = ring_pos2bank_id[(ring_pos + matmul_num_cores - 1) % matmul_num_cores];
        const auto& prev_physical =
            mesh_device->worker_core_from_logical_core(ring_cores[ring * matmul_num_cores + prev_bank]);

        if (ring == 0) {
            ring_pos2core[ring_pos] = core;
        }

        matmul_runtime_args[0] = dram_bank;
        matmul_runtime_args[1] = vchannel;
        // matmul_runtime_args[2-4] are already set to tensor addresses
        // matmul_runtime_args[5] is already set to ring_semaphore_id
        matmul_runtime_args[6] = ring_pos;
        matmul_runtime_args[7] = static_cast<uint32_t>(next_physical.x);
        matmul_runtime_args[8] = static_cast<uint32_t>(next_physical.y);
        matmul_runtime_args[9] = ring;
        matmul_runtime_args[10] = static_cast<uint32_t>(prev_physical.x);
        matmul_runtime_args[11] = static_cast<uint32_t>(prev_physical.y);

        tt::tt_metal::SetRuntimeArgs(program, matmul_dm0_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_dm1_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_compute_kernel_handle, core, matmul_runtime_args);

        log_debug(tt::LogOp, "{} -> DRAM {} -> ring pos {}", core.str(), dram_bank, ring_pos);
    }

    //-------------------------------------------------------------------------
    // Combine stage
    //-------------------------------------------------------------------------

    // Shared variables for both Full and ComputeOnly paths; the Full branch fills them in.
    std::vector<tt::tt_metal::KernelHandle> combine_kernel_handles;
    tt::tt_metal::CBHandle combine_data_cb_handle{};
    std::vector<GlobalSemaphore> combine_global_semaphores;
    std::vector<tt::tt_metal::CoreCoord> combine_cores_for_shared = combine_cores;

    if (combine_built) {
        // combine_params validity, num_links, and axis range are all checked in
        // validate_on_program_cache_miss. Barrier semaphores are an internal contract
        // between create_mesh_workload (caller) and create_at (callee), so checked here.
        // FullCcl owns real GlobalSemaphores; SingleDevice has none (writer compiles them out).
        if (args.path == MoEComputePath::FullCcl) {
            TT_FATAL(init_barrier_semaphore.has_value(), "init_barrier_semaphore must be set when path is FullCcl");
            TT_FATAL(final_barrier_semaphore.has_value(), "final_barrier_semaphore must be set when path is FullCcl");
        } else {
            TT_FATAL(args.path == MoEComputePath::SingleDevice, "Unexpected path in combine branch");
            TT_FATAL(!init_barrier_semaphore.has_value(), "init_barrier_semaphore must be nullopt for SingleDevice");
            TT_FATAL(!final_barrier_semaphore.has_value(), "final_barrier_semaphore must be nullopt for SingleDevice");
        }

        TT_FATAL(
            tensor_return_value.size() == 6,
            "path=FullCcl/SingleDevice expects 6 output tensors, got {}",
            tensor_return_value.size());
        ttnn::Tensor& output_tensor = tensor_return_value[5];

        auto combine_params = *args.combine_params;
        combine_params.worker_cores = combine_cores;
        // The combine writes its primary result to `output_tensor` (slot 5, allocated by
        // compute_output_specs). `optional_output_tensor` is a separate user-supplied sink
        // exposed via the op's MoEComputeInputs.optional_output_tensor field.
        ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensor_args{
            .dense_input_tensor = matmul_output_tensor,
            .dense_activations_tensor = tilize_expert_activation_output_tensor,
            .dense_token_maps_tensor = tilize_e_t_output_tensor,
            .dense_token_counts_tensor = tilize_per_expert_total_tokens_output_tensor,
            .optional_output_tensor = tensor_args.optional_output_tensor};

        // Each combine core consumes output pages from a column of compute cores;
        // compute_cores_per_combine_core = matmul_num_cores / combine_data_parallel_cores
        // is variable across shipped models (e.g., 3 for output_width_shard_dim=4 with
        // matmul_num_cores=12; 4 for output_width_shard_dim=3 with matmul_num_cores=12).
        // With ring size = live bank count, matmul_num_cores is 12 on WH and 7/8 on BH.
        const uint32_t compute_cores_per_combine_core = matmul_core_range_set.num_cores() / combine_data_parallel_cores;
        auto selective_reduce_combine_artifacts = build_selective_reduce_combine_program_artifacts(
            program,
            combine_params,
            mesh_coordinate,
            mesh_coordinates.coords(),
            combine_tensor_args,
            output_tensor,
            init_barrier_semaphore,
            final_barrier_semaphore,
            tilize_combine_sync_semaphore_id,
            matmul_combine_sync_semaphore_id,
            compute_cores_per_combine_core,
            ring_pos2core);

        combine_kernel_handles = {
            selective_reduce_combine_artifacts.reader_kernel_id, selective_reduce_combine_artifacts.writer_kernel_id};
        combine_data_cb_handle = selective_reduce_combine_artifacts.data_cb_handle;
        // FullCcl owns the barrier semaphores via shared_variables so addresses stay live for
        // override_runtime_arguments. SingleDevice has no semaphores; leave the vector empty.
        if (args.path == MoEComputePath::FullCcl) {
            combine_global_semaphores = {*init_barrier_semaphore, *final_barrier_semaphore};
        }
    } else {
        // ComputeOnly / SingleCluster: no combine kernels are built. The matmul/tilize kernels'
        // increments to combine semaphores are gated off via the compute_only / local_output CT args.
        combine_cores_for_shared.clear();
    }

    //-------------------------------------------------------------------------
    // Cached program
    //-------------------------------------------------------------------------

    return {
        std::move(program),
        {.tilize_kernel_handles = {tilize_reader_kernel_id, tilize_compute_kernel_id, tilize_writer_kernel_id},
         .tilize_cores = tilize_cores,
         .matmul_kernel_handles = {matmul_dm0_kernel_handle, matmul_dm1_kernel_handle, matmul_compute_kernel_handle},
         .matmul_cores = ring_cores,  // every ring core: override_runtime_arguments refreshes them all
         .indices_cb_handle = indices_cb_handle,
         .scores_cb_handle = scores_cb_handle,
         .sharded_output_cb_handle = sharded_output_cb_handle,
         .matmul_writer_cb_handle = matmul_writer_cb_handle,
         .combine_kernel_handles = std::move(combine_kernel_handles),
         .combine_data_cb_handle = combine_data_cb_handle,
         .expert_tokens_cb_handle = expert_tokens_cb_handle,
         .combine_cores = std::move(combine_cores_for_shared),
         .combine_global_semaphores = std::move(combine_global_semaphores),
         .path = args.path,
         .matmul_runtime_args_size = static_cast<uint32_t>(matmul_runtime_args.size())}};
}

void MoEComputeMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MoEComputeParams& args,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    // output tensors
    TT_FATAL(tensor_return_value.size() >= 5, "Expected at least 5 output tensors, got {}", tensor_return_value.size());
    const ttnn::Tensor& tilize_per_expert_total_tokens_output_tensor = tensor_return_value[0];
    const ttnn::Tensor& tilize_expert_activation_output_tensor = tensor_return_value[1];
    const ttnn::Tensor& tilize_e_t_output_tensor = tensor_return_value[2];
    const ttnn::Tensor& tilize_output_tensor = tensor_return_value[3];
    const ttnn::Tensor& matmul_output_tensor = tensor_return_value[4];

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Update sharded circular buffer address
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.indices_cb_handle, *tensor_args.tilize_expert_indices_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.scores_cb_handle, *tensor_args.tilize_expert_scores_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.sharded_output_cb_handle, *tilize_output_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.matmul_writer_cb_handle, *tilize_output_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.expert_tokens_cb_handle, *tilize_per_expert_total_tokens_output_tensor.buffer());

        //-------------------------------------------------------------------------
        // Tilize
        //-------------------------------------------------------------------------
        // tilize_kernel_handles layout: [0]=reader, [1]=compute, [2]=writer.
        TT_FATAL(
            shared_variables.tilize_kernel_handles.size() >= 3,
            "expected at least 3 tilize kernels (reader, compute, writer), got {}",
            shared_variables.tilize_kernel_handles.size());
        for (const auto& core : shared_variables.tilize_cores) {
            // reader
            auto& tilize_reader_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles[0], core);
            TT_FATAL(
                tilize_reader_runtime_args.size() >= 7,
                "tilize reader runtime args expected size >= 7, got {}",
                tilize_reader_runtime_args.size());
            tilize_reader_runtime_args[0] = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_reader_runtime_args[1] = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_reader_runtime_args[2] = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_reader_runtime_args[3] = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_reader_runtime_args[4] = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_reader_runtime_args[5] = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_reader_runtime_args[6] = tilize_e_t_output_tensor.buffer()->address();

            // writer
            auto& tilize_writer_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles[2], core);
            TT_FATAL(
                tilize_writer_runtime_args.size() >= 7,
                "tilize writer runtime args expected size >= 7, got {}",
                tilize_writer_runtime_args.size());
            tilize_writer_runtime_args[0] = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_writer_runtime_args[1] = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_writer_runtime_args[2] = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_writer_runtime_args[3] = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_writer_runtime_args[4] = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_writer_runtime_args[5] = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_writer_runtime_args[6] = tilize_e_t_output_tensor.buffer()->address();
        }

        //-------------------------------------------------------------------------
        // Matmul
        //-------------------------------------------------------------------------
        for (const auto& core : shared_variables.matmul_cores) {
            for (const auto& kernel_handle : shared_variables.matmul_kernel_handles) {
                auto& matmul_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
                TT_FATAL(
                    matmul_runtime_args.size() >= 5,
                    "matmul runtime args expected size >= 5, got {}",
                    matmul_runtime_args.size());
                matmul_runtime_args[2] = tensor_args.matmul_w0_w1_tensor.buffer()->address();
                matmul_runtime_args[3] = tensor_args.matmul_w2_tensor.buffer()->address();
                matmul_runtime_args[4] = tilize_output_tensor.buffer()->address();
                if (shared_variables.path == MoEComputePath::SingleCluster) {
                    // The final output and e_t addresses are the two trailing entries (see create_at).
                    // tensor_return_value[5] is the caller's optional_output_tensor when one was given
                    // (create_output_tensors returns it), so its address is what dm1 writes to.
                    TT_FATAL(
                        tensor_return_value.size() == 6 &&
                            matmul_runtime_args.size() == shared_variables.matmul_runtime_args_size,
                        "path=SingleCluster expects 6 output tensors and exactly {} matmul runtime args (12 leading, "
                        "the "
                        "bank table, the two trailing addresses), got {} and {}",
                        shared_variables.matmul_runtime_args_size,
                        tensor_return_value.size(),
                        matmul_runtime_args.size());
                    matmul_runtime_args[matmul_runtime_args.size() - 2] = tensor_return_value[5].buffer()->address();
                    matmul_runtime_args[matmul_runtime_args.size() - 1] = tilize_e_t_output_tensor.buffer()->address();
                }
            }
        }

        //-------------------------------------------------------------------------
        // Combine
        //-------------------------------------------------------------------------

        if (shared_variables.path == MoEComputePath::FullCcl || shared_variables.path == MoEComputePath::SingleDevice) {
            // combine_params validity is checked in validate_on_program_cache_miss.
            TT_FATAL(
                shared_variables.combine_kernel_handles.size() == 2,
                "Expected 2 combine kernel handles when path is not ComputeOnly");
            // FullCcl owns 2 barrier semaphores; SingleDevice owns none (writer compiles them out).
            const uint32_t expected_semaphores = shared_variables.path == MoEComputePath::FullCcl ? 2 : 0;
            TT_FATAL(
                shared_variables.combine_global_semaphores.size() == expected_semaphores,
                "Expected {} combine global semaphores for path={}, got {}",
                expected_semaphores,
                static_cast<int>(shared_variables.path),
                shared_variables.combine_global_semaphores.size());
            TT_FATAL(
                tensor_return_value.size() == 6,
                "path=FullCcl/SingleDevice expects 6 output tensors, got {}",
                tensor_return_value.size());

            ttnn::Tensor& output_tensor = tensor_return_value[5];

            auto reader_kernel_id = shared_variables.combine_kernel_handles[0];
            auto writer_kernel_id = shared_variables.combine_kernel_handles[1];
            auto combine_data_cb_handle = shared_variables.combine_data_cb_handle;
            auto cores = shared_variables.combine_cores;
            // 0 for SingleDevice (no semaphores); real address for FullCcl.
            const uint32_t init_semaphore_addr = shared_variables.path == MoEComputePath::FullCcl
                                                     ? shared_variables.combine_global_semaphores[0].address()
                                                     : 0;
            const uint32_t cross_device_semaphore_addr = shared_variables.path == MoEComputePath::FullCcl
                                                             ? shared_variables.combine_global_semaphores[1].address()
                                                             : 0;

            // See create_at for the explanation of optional_output_tensor handling.
            ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensor_args{
                .dense_input_tensor = matmul_output_tensor,
                .dense_activations_tensor = tilize_expert_activation_output_tensor,
                .dense_token_maps_tensor = tilize_e_t_output_tensor,
                .dense_token_counts_tensor = tilize_per_expert_total_tokens_output_tensor,
                .optional_output_tensor = tensor_args.optional_output_tensor};
            selective_reduce_combine_helper_override_runtime_arguments(
                program,
                reader_kernel_id,
                writer_kernel_id,
                combine_data_cb_handle,
                cores,
                combine_tensor_args,
                output_tensor,
                init_semaphore_addr,
                cross_device_semaphore_addr,
                args.combine_params->optional_cross_device_semaphore);
        }
    }
}

}  // namespace ttnn::experimental::prim
