// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_program_factory.hpp"
#include "moe_compute_device_operation_types.hpp"

#include "ttnn/global_semaphore.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <tt_stl/indestructible.hpp>
#include <umd/device/types/arch.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace {

constexpr uint32_t TILE_WIDTH = 32;

uint32_t get_num_pages_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows_st(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

// Per-arch layout for tilize+combine. The matmul cores come from
// mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(...) and live at the
// "DRAM-bank-adjacent" columns: WH packs them at logical x=0,4 (close); BH spreads them
// across logical x=0,7 (far apart). The free corner where tilize+combine fit without
// overlapping the matmul mcast bounding box is therefore arch-specific.
//
// Order in `max_tilize_cores` is LOAD-BEARING: index 0 is the drain-sync core and the
// first half forms the primary mcast group (second half is the secondary mcast group).
// Existing 6U callers hardcode input sharding to (6,9) on WH under the assumption that
// it's the drain core, so the production order is preserved here. The grid-availability
// filter in get_cores() drops out-of-range entries (e.g. y=9 on harvested n150_L) —
// see issue #41827 M1. On a harvested chip that filter leaves {(6,8),(5,8)}; callers
// that hardcode (6,9) must derive the drain core dynamically from the grid (see the
// single-card test for an example).
struct MoEComputeLayout {
    std::vector<CoreCoord> max_tilize_cores;  // [0] = drain core (load-bearing)
    CoreRangeSet max_combine_core_range_set;
};

const MoEComputeLayout& get_layout(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: {
            static tt::stl::Indestructible<MoEComputeLayout> wh{MoEComputeLayout{
                /*max_tilize_cores=*/{CoreCoord(6, 9), CoreCoord(6, 8), CoreCoord(5, 9), CoreCoord(5, 8)},
                /*max_combine_core_range_set=*/CoreRangeSet(CoreRange({5, 0}, {6, 7})),
            }};
            return wh.get();
        }
        case tt::ARCH::BLACKHOLE: {
            // BH has DRAM cols at logical x=0,7 (vs WH's x=0,4) and a presumed 11x10
            // production worker grid. Tilize+combine shift to the right edge (x=9,10)
            // to clear the matmul mcast bounding box ({0,0},{7,9}). See issue #41827 M3
            // PR1 spec §17 for the topology rationale.
            static tt::stl::Indestructible<MoEComputeLayout> bh{MoEComputeLayout{
                /*max_tilize_cores=*/{CoreCoord(10, 9), CoreCoord(10, 8), CoreCoord(9, 9), CoreCoord(9, 8)},
                /*max_combine_core_range_set=*/CoreRangeSet(CoreRange({9, 0}, {10, 7})),
            }};
            return bh.get();
        }
        case tt::ARCH::QUASAR:
        case tt::ARCH::Invalid: break;
    }
    TT_THROW("moe_compute: no layout for arch {}", static_cast<int>(arch));
}

std::tuple<
    std::vector<CoreCoord>,  // T cores
    std::vector<CoreCoord>,  // MM cores
    CoreRangeSet,            // T CoreRangeSet
    CoreRangeSet,            // MM CoreRangeSet
    CoreRangeSet,            // T + MM CoreRangeSet
    CoreRangeSet,            // Combine CoreRangeSet
    CoreRangeSet,            // C + MM CoreRangeSet
    CoreRangeSet,            // All worker cores (T + MM + C)
    std::vector<CoreCoord>,  // Combine vector of CoreCoord
    CoreRange,               // T bounding box
    CoreRange>               // MM bounding box
get_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size) {
    /*
     * - First tilize core is the drain sync
     * - First ((total_tilize_cores + 1) / 2) tilize cores are primary mcast group
     * - Remaining cores are secondary mcast group (with the first of them being the secondary mcaster)
     */

    // Calculate number of tilize cores based on hidden dimension
    const uint32_t hidden_tiles = hidden_size / TILE_WIDTH;

    // Per-arch tilize/combine layout (see issue #41827 M3 PR1; layout struct above).
    const auto& layout = get_layout(mesh_device->arch());
    const auto& max_tilize_cores = layout.max_tilize_cores;
    const auto& max_combine_core_range_set = layout.max_combine_core_range_set;

    // Filter max_tilize_cores by what the device's logical worker grid actually exposes.
    // Harvested SKUs (e.g. n150_L with one disabled compute row) may not expose the y=9
    // entries; combined with the y=8-first ordering of `max_tilize_cores`, this lets the op
    // run on harvested chips while still using all 4 cores on full-grid WH/6U Galaxy.
    const auto grid_size = mesh_device->compute_with_storage_grid_size();
    std::vector<CoreCoord> tilize_cores;
    tilize_cores.reserve(max_tilize_cores.size());
    for (const auto& c : max_tilize_cores) {
        if (c.x < grid_size.x && c.y < grid_size.y) {
            tilize_cores.push_back(c);
        }
    }
    TT_FATAL(!tilize_cores.empty(), "No tilize cores fit the device's logical worker grid");

    // Then prune so the remaining count evenly divides hidden_tiles.
    while (tilize_cores.size() > 1 && hidden_tiles % tilize_cores.size() != 0) {
        tilize_cores.pop_back();
    }

    // matmul cores come from the DRAM-bank-to-worker assignment: WH returns 12, BH returns 8.
    // Both archs run the N=12 ring (templatize knob, see kernels/moe_ring_common.h:
    // DeepSeekRingConfig<N>/GptRingConfig<N>). On BH we pad to 12 by appending 4 extra cores
    // inside the existing matmul mcast bbox ({0,0},{7,9}) — INTERLEAVED weights (see Python
    // `get_weight_mem_configs`) page tiles across all DRAM banks, so the ring core count is
    // independent of bank count. Issue #41827 PR1.
    auto matmul_cores =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    if (mesh_device->arch() == tt::ARCH::BLACKHOLE) {
        // BH returns 8 DRAM-adjacent cores; append 4 extras inside the matmul bbox ({0,0},{7,9}).
        // These extras don't extend the bbox (all within x=0..7, y=3..8), so tilize/combine
        // bboxes (x=9,10) remain non-overlapping.
        matmul_cores.push_back({0, 5});
        matmul_cores.push_back({0, 8});
        matmul_cores.push_back({7, 3});
        matmul_cores.push_back({7, 8});
    }
    TT_FATAL(
        matmul_cores.size() == 12, "moe_compute: expected 12 matmul cores after padding (got {})", matmul_cores.size());

    // CoreRangeSets
    const CoreRangeSet tilize_core_range_set = CoreRangeSet(tilize_cores);
    const CoreRangeSet matmul_core_range_set = CoreRangeSet(matmul_cores);
    const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(matmul_core_range_set);

    // Bounding boxes
    const CoreRange tilize_bounding_box = tilize_core_range_set.bounding_box();
    const CoreRange matmul_bounding_box = matmul_core_range_set.bounding_box();

    // Verify none of the bounding boxes overlap
    TT_FATAL(!tilize_bounding_box.intersects(matmul_bounding_box), "tilize and matmul bounding boxes cannot overlap");

    // Combine cores (16 max), that don't overlap with any of the tilize or matmul bounding boxes
    const auto combine_core_range_set = select_from_corerangeset(
        max_combine_core_range_set,
        /*start_index=*/0,
        (combine_token_parallel_cores * combine_data_parallel_cores) - 1);

    const CoreRange combine_bounding_box = combine_core_range_set.bounding_box();

    // consistent order matters for the list of combine cores so produce them as a sorted vector
    auto combine_cores = corerange_to_cores(combine_core_range_set);
    std::sort(combine_cores.begin(), combine_cores.end(), [](const auto& ca, const auto& cb) {
        if (ca.x != cb.x) {
            return ca.x < cb.x;
        }
        return ca.y < cb.y;
    });

    // MatMul + combine CoreRangeSet, needed for semaphores
    const auto combine_matmul_core_range_set = combine_core_range_set.merge(matmul_core_range_set);

    // All worker cores (tilize + matmul + combine)
    const CoreRangeSet all_worker_cores_range_set = tilize_matmul_core_range_set.merge(combine_core_range_set);

    TT_FATAL(!combine_bounding_box.intersects(tilize_bounding_box), "combine and tilize bounding boxes cannot overlap");
    TT_FATAL(!combine_bounding_box.intersects(matmul_bounding_box), "combine and matmul bounding boxes cannot overlap");

    return {
        tilize_cores,
        matmul_cores,
        tilize_core_range_set,
        matmul_core_range_set,
        tilize_matmul_core_range_set,
        combine_core_range_set,
        combine_matmul_core_range_set,
        all_worker_cores_range_set,
        combine_cores,
        tilize_bounding_box,
        matmul_bounding_box};
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

// expose a helper function so callers know what cores are available for subsequently running a2a combine
std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores) {
    constexpr auto combine_cores_return_index = 8;

    // Use dummy hidden_size since we only need combine cores (which don't depend on hidden_size)
    // This function is only for getting combine cores, not tilize cores
    constexpr uint32_t dummy_hidden_size = 7168;  // DeepSeek default
    const auto get_cores_return =
        get_cores(mesh_device, combine_token_parallel_cores, combine_data_parallel_cores, dummy_hidden_size);

    return std::get<combine_cores_return_index>(get_cores_return);
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

    if (!args.compute_only) {
        TT_FATAL(
            args.combine_params.has_value(),
            "compute_only=false but combine_params is nullopt; combine_params is required when compute_only is false");
        const auto core_ret =
            get_cores(mesh_device, args.num_token_parallel_cores, args.num_data_parallel_cores, hidden_size);

        const auto& combine_core_range_set = std::get<combine_core_range_set_return_index>(core_ret);

        init_barrier_semaphore =
            ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_core_range_set, 0);
        final_barrier_semaphore = args.combine_params->optional_cross_device_semaphore.value_or(
            ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_core_range_set, 0));

        tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
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
    uint32_t experts = tilize_mapping_shape[-1];
    uint32_t selected_experts_k = tilize_indices_shape[-1];

    // NOTE: shared experts are slightly delicate since they show up as an additional entry in the mapping tensor the
    // result is fractional experts per device so div_up is required to get the right value here.
    uint32_t experts_per_device = tt::div_up(experts, num_devices);

    // Output/Combine input core dims, for core selection. These are top-level (lifted) so they
    // remain valid even when combine_params is nullopt (compute_only mode).
    const auto combine_token_parallel_cores = args.num_token_parallel_cores;
    const auto combine_data_parallel_cores = args.num_data_parallel_cores;

    // Determine config type based on hidden size. Bias does not matter for these values.
    // a2a_cb_pages depends on the ring core count (templatized in moe_ring_common.h via N) and
    // is computed after matmul_num_cores is known below.
    uint32_t config_type;
    if (hidden_size == 7168) {
        config_type = static_cast<uint32_t>(detail::MoEConfigType::DEEPSEEK);
    } else if (hidden_size == 2880) {
        config_type = static_cast<uint32_t>(detail::MoEConfigType::GPT);
    } else {
        TT_THROW("Unsupported hidden size {} for moe_compute. Expected 7168 (DeepSeek) or 2880 (GPT)", hidden_size);
    }

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
         matmul_bounding_box] =
            get_cores(mesh_device, combine_token_parallel_cores, combine_data_parallel_cores, hidden_size);

    const uint32_t tilize_num_cores = tilize_core_range_set.num_cores();
    const uint32_t matmul_num_cores = matmul_core_range_set.num_cores();

    // a2a_cb_pages = IN2_TILES_PER_STEP for the active workload at N=12. Both WH and BH instantiate
    // the <12> specialization (BH pads cores; INTERLEAVED weights decouple from bank count).
    TT_FATAL(matmul_num_cores == 12, "moe_compute: expected matmul_num_cores=12, got {}", matmul_num_cores);
    auto get_in2_tiles_per_step = [](detail::MoEConfigType c) -> uint32_t {
        switch (c) {
            case detail::MoEConfigType::DEEPSEEK:
                return moe_ring::DeepSeekRingConfig</*HasBias=*/false, 12>::IN2_TILES_PER_STEP;
            case detail::MoEConfigType::GPT: return moe_ring::GptRingConfig</*HasBias=*/false, 12>::IN2_TILES_PER_STEP;
        }
        TT_THROW("moe_compute: unknown config type {}", static_cast<int>(c));
    };
    const uint32_t a2a_cb_pages = get_in2_tiles_per_step(static_cast<detail::MoEConfigType>(config_type));

    const uint32_t tilize_bounding_box_num_cores = tilize_bounding_box.size();
    const uint32_t matmul_bounding_box_num_cores = matmul_bounding_box.size();

    // All worker cores bounding box
    const CoreRange all_worker_cores_bounding_box = all_worker_cores_range_set.bounding_box();
    const uint32_t all_worker_cores_bounding_box_num_cores = all_worker_cores_bounding_box.size();

    // Logical mcast bounding box coordinates
    const CoreCoord tilize_mcast_start_logical = tilize_bounding_box.start_coord;
    const CoreCoord tilize_mcast_end_logical = tilize_bounding_box.end_coord;
    const CoreCoord matmul_mcast_start_logical = matmul_bounding_box.start_coord;
    const CoreCoord matmul_mcast_end_logical = matmul_bounding_box.end_coord;
    const CoreCoord all_worker_cores_mcast_start_logical = all_worker_cores_bounding_box.start_coord;
    const CoreCoord all_worker_cores_mcast_end_logical = all_worker_cores_bounding_box.end_coord;

    // Convert to physical NOC coordinates
    const CoreCoord tilize_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(tilize_mcast_start_logical);
    const CoreCoord tilize_mcast_end_physical = mesh_device->worker_core_from_logical_core(tilize_mcast_end_logical);
    const CoreCoord matmul_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_start_logical);
    const CoreCoord matmul_mcast_end_physical = mesh_device->worker_core_from_logical_core(matmul_mcast_end_logical);
    const CoreCoord all_worker_cores_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(all_worker_cores_mcast_start_logical);
    const CoreCoord all_worker_cores_mcast_end_physical =
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

    // Since both reader and writer are using NoC1, we have the readers wait until all writers have sent their
    // chunk portion to the matmul cores before reading in another set of tokens to tilize. This frees up NoC1
    // to do the mcast, which is also a requirement for doing linked mcasts.
    auto previous_chunk_sent_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    // For the gather phase scheme used for the first chunk
    auto initial_gather_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Matmul semaphores
    //-------------------------------------------------------------------------

    // Create semaphores for ring synchronization between cores
    // Each core will have a semaphore that its predecessor will signal
    // reuse the same semaphore location for signaling combine cores that per expert data is available
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, matmul_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize and Matmul semaphores
    //-------------------------------------------------------------------------

    // Tilize drain-sync core signals to tilize non-drain-sync cores that final metadata results are ready
    // Tilize drain-sync core signals to matmul cores that final metadata results are ready
    auto metadata_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // Matmul cores signal to tilize drain-sync-core that the input chunk is free to be written to
    // Tilize drain-sync-core propagates this to the tilize non-drain-sync cores
    auto matmul_chunk_available_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // Tilize drain-sync core signals to all matmul cores that a full chunk is ready
    auto matmul_chunk_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize/MatMul and Combine sync semaphore
    //-------------------------------------------------------------------------

    // Tilize drain-sync core signals combine sync core (which then multicasts to the rest)
    // that metadata is ready and task splitting can proceed.
    // Allocate on full rectangle of usable cores so we can multicast without clobbering.
    // In compute_only mode, the kernel-side increment is gated off via the compute_only CT arg, so
    // the semaphore is unused by anyone -- but dm1 still calls get_semaphore() on it (a local L1
    // address lookup), so we still need it to be allocated on at least the matmul core range set.
    const auto tilize_combine_sync_semaphore_id = tt::tt_metal::CreateSemaphore(
        program,
        args.compute_only ? matmul_core_range_set : get_layout(mesh_device->arch()).max_combine_core_range_set,
        INVALID);

    // Matmul dm1 signals combine cores when data is written; combine writer waits on this semaphore.
    // For double buffering, combine cores will also use this semaphore to signal matmul when buffer segments are free.
    // In compute_only mode, dm1 still calls get_semaphore() to look up the local L1 address, but the
    // wait/increment/set are all gated off via the compute_only CT arg.
    const auto matmul_combine_sync_semaphore_id = tt::tt_metal::CreateSemaphore(
        program, args.compute_only ? matmul_core_range_set : combine_matmul_core_range_set, INVALID);

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
    const CoreRangeSet shard_cores = tilize_output_tensor.memory_config().shard_spec()->grid;
    const uint32_t shared_cb_num_pages = output_pages / shard_cores.num_cores();

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
    // TODO(#41827, M3): in compute_only mode this CB still spans `all_worker_cores_range_set` which
    // includes (kernel-less) combine cores. Functionally OK (CB is allocated, no kernel reads it),
    // but the tilize_reader mcast at `tilize_reader.cpp` writes to a wider bounding box than needed.
    // Reduce to `tilize_matmul_core_range_set` and pass `tilize_matmul_*_mcast_*` CT args when
    // `args.compute_only`. Defer until M3 to keep M1 PR scope tight.
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

    // CB for passing total_chunks from writer to compute
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

    constexpr uint32_t tokens_per_chunk = 32;  // Hardcoding for now, can adjust when tiny tiles support added

    // e_t buffer entry size must be 16B aligned for NOC DMA during BRISC->NCRISC merge
    tt::tt_metal::create_cb(
        e_t_cb_id,
        program,
        tilize_core_range_set,
        tilize_e_t_output_page_size,
        experts_per_device,  // number of experts on the device
        tilize_e_t_output_data_format);

    // Assume indices tensor is sharded in L1
    tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_indices_aligned_page_size,
        tilize_indices_pages,  // double buffer buffer packets
        tilize_indices_data_format,
        tilize_indices_tensor.buffer());

    // Assume scores tensor is sharded in L1
    tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_input_scores_aligned_page_size,
        tilize_input_scores_pages,
        tilize_input_scores_data_format,
        tilize_input_scores_tensor.buffer());

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
    // Single buffered so we don't start reading in next set of tokens
    // before previous chunk fully sent to matmul cores
    tt::tt_metal::create_cb(
        tilize_input_cb_id,
        program,
        tilize_core_range_set,
        max_tilize_subtoken_size,
        tokens_per_chunk,
        tilize_input_data_format);

    tt::tt_metal::create_cb(
        expert_activation_cb_id,
        program,
        tilize_core_range_set,
        tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment),
        tokens,
        tt::DataFormat::UInt32);

    // BRISC's e_t buffer for parallel metadata processing
    // BRISC processes the second half of tokens (tokens/2 to tokens)
    // Single page containing all experts' token lists, each with capacity tokens/2
    // Uses same 16B entry alignment as main e_t buffer for NOC DMA compatibility
    tt::tt_metal::create_cb(
        brisc_e_t_cb_id,
        program,
        tilize_core_range_set,
        (tokens / 2) * l1_alignment * experts_per_device,  // full buffer with 16B entries
        1,
        tt::DataFormat::UInt32);

    // BRISC's per-expert token counts to communicate to NCRISC
    // Single page containing counts for all experts
    tt::tt_metal::create_cb(
        brisc_expert_counts_cb_id,
        program,
        tilize_core_range_set,
        sizeof(uint32_t) * experts_per_device,  // all counts in one page
        1,
        tt::DataFormat::UInt32);

    // BRISC's expert activation buffer - same format as main expert_activation buffer
    // [token_id, k_indices[experts_per_device], scores[experts_per_device]] per activated token
    // Single page containing all activation rows (max tokens/2)
    uint32_t brisc_activation_row_size = tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment);
    tt::tt_metal::create_cb(
        brisc_expert_activation_cb_id,
        program,
        tilize_core_range_set,
        brisc_activation_row_size * (tokens / 2),  // full buffer in one page
        1,
        tt::DataFormat::UInt32);

    // BRISC's activated token count (single uint32_t to communicate to NCRISC)
    tt::tt_metal::create_cb(
        brisc_activated_count_cb_id, program, tilize_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);

    // CB for receiving counts from non-drain tilize cores (only used on drain core)
    // Each non-drain core sends: [e_t_count_expert0, e_t_count_expert1, activated_count]
    // Layout: 3 values per core × (tilize_num_cores - 1) cores, 16B aligned per core's data
    uint32_t counts_per_remote_core = experts_per_device + 1;  // e_t counts + activated count
    uint32_t remote_counts_entry_size = tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment);
    tt::tt_metal::create_cb(
        remote_counts_cb_id,
        program,
        tilize_core_range_set,
        remote_counts_entry_size,
        tilize_num_cores - 1,  // one entry per non-drain core
        tt::DataFormat::UInt32);

    // CB for passing total_chunks from writer to compute kernel
    // Single page holding one uint32_t value
    tt::tt_metal::create_cb(
        total_chunks_cb_id,
        program,
        tilize_core_range_set,
        sizeof(uint32_t),
        1,  // single page
        tt::DataFormat::UInt32);

    //-------------------------------------------------------------------------
    // Matmul CBs
    //-------------------------------------------------------------------------

    // CBs used in the MOE operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_s2c_in      | CBIndex::c_0  | Float16_b  | true  |    224*2 |      917504     |
        | cb_r2c_w0      | CBIndex::c_1  | Bfp4_b     | true  |    14*6  |      48384      |
        | cb_c2w_rdy     | CBIndex::c_2  | Float32    | false |    1     |      4          |
        | cb_w2c_rdy     | CBIndex::c_3  | Float32    | false |    1     |      4          |
        | cb_s2c_in2     | CBIndex::c_4  | Float16_b  | true  |    6*12  |      147456     |
        | cb_w2c_md      | CBIndex::c_5  | UInt32     | false |    2     |      8          |
        ------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in and cb_c2s_out are handled separately as it is allocated on Tilize, Matmul, and Combine cores
    std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> matmul_cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_3, tt::DataFormat::Bfp4_b, true, 14 * 6},
        {"cb_c2w_rdy", tt::CBIndex::c_4, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_5, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, a2a_cb_pages * matmul_num_cores},
        {"cb_w2c_md", tt::CBIndex::c_7, tt::DataFormat::UInt32, false, 2},
    };
    if (args.has_bias) {
        // c_8 is free on matmul cores (c_8 is tilize_input_cb_id on tilize cores only).
        matmul_cb_specs0.emplace_back("cb_c2c_ones_tile", tt::CBIndex::c_8, tt::DataFormat::Float16_b, true, 1);
    }

    std::map<std::string, tt::tt_metal::CBHandle> matmul_cb_handles;

    // Create CBs
    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : matmul_cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        matmul_cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, matmul_core_range_set, cb_config);
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
    std::vector<CoreCoord> tilize_cores_physical(tilize_num_cores);
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        tilize_cores_physical.at(i) = mesh_device->worker_core_from_logical_core(tilize_cores.at(i));
    }

    // tile_width_bytes = TILE_WIDTH * element_size
    // max_tiles_per_chunk = max_tilize_subtoken_size / tile_width_bytes
    uint32_t tile_width_bytes = TILE_WIDTH * tilize_input_tensor.element_size();
    uint32_t max_tiles_per_local_chunk = max_tilize_subtoken_size / tile_width_bytes;

    const uint32_t primary_mcast_gather_group_num_cores = tilize_num_cores / 2;
    const uint32_t secondary_mcast_gather_group_num_cores = tilize_num_cores - primary_mcast_gather_group_num_cores;

    // Drain core is always the first tilize core (index 0)
    CoreCoord tilize_drain_core_physical = tilize_cores_physical.at(0);

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
        {"selected_experts_k", selected_experts_k},

        // Chunk info
        {"tokens_per_chunk", tokens_per_chunk},

        // Mesh
        {"num_devices", num_devices},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"linearized_mesh_coord", linearized_mesh_coord},
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
        {"matmul_bounding_box_num_cores", matmul_bounding_box_num_cores},

        // All worker cores multicast coordinates
        // TODO(#41827, M3): when `args.compute_only`, this bounding box still includes the
        // unused combine box (5..6, 0..7). Tilize_reader mcasts to kernel-less L1 there.
        // Reduce to tilize_matmul_bounding_box for compute_only. Defer to M3.
        {"all_worker_cores_mcast_start_x", (uint32_t)all_worker_cores_mcast_start_physical.x},
        {"all_worker_cores_mcast_start_y", (uint32_t)all_worker_cores_mcast_start_physical.y},
        {"all_worker_cores_mcast_end_x", (uint32_t)all_worker_cores_mcast_end_physical.x},
        {"all_worker_cores_mcast_end_y", (uint32_t)all_worker_cores_mcast_end_physical.y},
        {"all_worker_cores_bounding_box_num_cores", all_worker_cores_bounding_box_num_cores},

        // Semaphores
        {"partial_metadata_ready_semaphore_id", tilize_partial_metadata_ready_semaphore_id},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
        {"tilize_chunk_ready_semaphore_id", tilize_chunk_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"previous_chunk_sent_semaphore_id", previous_chunk_sent_semaphore_id},
        {"initial_gather_semaphore_id", initial_gather_semaphore_id},
        // Tilize -> combine: drain core signals combine when metadata is ready.
        // In compute_only mode the increment is gated off in the kernel via the compute_only CT arg.
        // combine_cores is still computed in compute_only mode so .at(0) is valid here.
        {"combine_sync_semaphore_id", tilize_combine_sync_semaphore_id},
        {"combine_sync_noc_x", (uint32_t)mesh_device->worker_core_from_logical_core(combine_cores.at(0)).x},
        {"combine_sync_noc_y", (uint32_t)mesh_device->worker_core_from_logical_core(combine_cores.at(0)).y},

        // Bypass selective_reduce_combine path entirely (1=skip combine semaphore inc/wait/set).
        {"compute_only", args.compute_only ? 1u : 0u}};

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

    tt::tt_metal::KernelHandle tilize_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp",
        tilize_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = tilize_compile_time_args,
            .defines = {},
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
            .defines = {},
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
    };

    tt::tt_metal::KernelHandle tilize_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_compute.cpp",
        tilize_core_range_set,
        tt::tt_metal::ComputeConfig{.named_compile_args = compute_tilize_named_compile_time_args});

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
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).x);
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).y);
    }

    // Calculate number of bytes per mcast_gather_group
    uint32_t primary_mcast_gather_group_subtoken_size = 0;
    uint32_t secondary_mcast_gather_group_subtoken_size = 0;
    uint32_t global_subtoken_offset = 0;
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        uint32_t subtoken_size = 0;
        if (tilize_cores_work_group_1.contains(tilize_cores.at(i))) {
            subtoken_size = tilize_units_per_core_work_group_1 * tilize_subtoken_bytes_aligned;
        } else if (tilize_cores_work_group_2.contains(tilize_cores.at(i))) {
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
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        // First tilize core is the drain tilize core (has indices/scores sharded to it)
        tilize_runtime_args.at(is_drain_tilize_core_idx) = (i == 0) ? 1 : 0;
        tilize_runtime_args.at(is_secondary_mcaster_idx) = (i == primary_mcast_gather_group_num_cores) ? 1 : 0;

        // Initial split mcast cores
        CoreCoord initial_mcast_gather_core_physical =
            i < primary_mcast_gather_group_num_cores ? tilize_cores_physical.at(0)
                                                     : tilize_cores_physical.at(primary_mcast_gather_group_num_cores);
        tilize_runtime_args.at(initial_mcast_gather_core_nox_x_idx) = (uint32_t)initial_mcast_gather_core_physical.x;
        tilize_runtime_args.at(initial_mcast_gather_core_nox_y_idx) = (uint32_t)initial_mcast_gather_core_physical.y;

        // Set work split parameters based on which group the core is in
        tilize_runtime_args.at(global_subtoken_offset_idx) = global_subtoken_offset;
        tilize_runtime_args.at(mcast_group_subtoken_offset_idx) = group_subtoken_offset;
        tilize_runtime_args.at(mcast_group_subtoken_size_idx) = i < primary_mcast_gather_group_num_cores
                                                                    ? primary_mcast_gather_group_subtoken_size
                                                                    : secondary_mcast_gather_group_subtoken_size;

        uint32_t subtoken_size = 0;
        if (tilize_cores_work_group_1.contains(tilize_cores.at(i))) {
            subtoken_size = tilize_units_per_core_work_group_1 * tilize_subtoken_bytes_aligned;
        } else if (tilize_cores_work_group_2.contains(tilize_cores.at(i))) {
            subtoken_size = tilize_units_per_core_work_group_2 * tilize_subtoken_bytes_aligned;
        }
        if (global_subtoken_offset + subtoken_size > tilize_input_aligned_page_size) {
            // Clamp to not exceed the total token size
            subtoken_size = tilize_input_aligned_page_size - global_subtoken_offset;
        }
        tilize_runtime_args.at(subtoken_size_idx) = subtoken_size;

        global_subtoken_offset += subtoken_size;
        group_subtoken_offset += subtoken_size;
        if (i == primary_mcast_gather_group_num_cores - 1) {
            group_subtoken_offset = 0;
        }

        // Set token range for this core's metadata processing
        // Each core processes a contiguous range of tokens
        uint32_t core_token_start = i * tokens_per_tilize_core;
        uint32_t core_token_end = (i == tilize_num_cores - 1) ? tokens : (i + 1) * tokens_per_tilize_core;
        tilize_runtime_args.at(core_token_start_idx) = core_token_start;
        tilize_runtime_args.at(core_token_end_idx) = core_token_end;
        tilize_runtime_args.at(tilize_core_idx) = i;

        // Set compute kernel runtime args
        tilize_compute_runtime_args.at(0) = subtoken_size / tile_width_bytes;

        tt::tt_metal::SetRuntimeArgs(program, tilize_reader_kernel_id, tilize_cores.at(i), tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, tilize_writer_kernel_id, tilize_cores.at(i), tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(
            program, tilize_compute_kernel_id, tilize_cores.at(i), tilize_compute_runtime_args);
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

    // parameters for writing to output shards
    const uint32_t tile_width = tilize_input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tilize_input_tensor.tensor_spec().tile().get_height();
    const uint32_t output_height_shard_dim = args.output_height_shard_dim;

    // this logic is awkward. needs to match selective_reduce_combine_program_factory.
    constexpr auto double_buffer = 2;
    const auto shards = tilize_output_tensor.memory_config().shard_spec()->grid.num_cores();
    const auto token_expert_row_offset = tilize_output_tensor.logical_shape().volume() / shards /
                                         (hidden_size / combine_data_parallel_cores / double_buffer) /
                                         combine_token_parallel_cores;

    // activation function
    const ttnn::experimental::prim::detail::MoEActivationFunction activation_type = args.activation_type;

    const uint32_t output_shard_width_tiles = hidden_size / tile_width / combine_data_parallel_cores;
    std::unordered_map<std::string, uint32_t> matmul_named_compile_time_args = {
        {"num_experts", experts_per_device},
        {"layer_id", args.layer_id},
        {"has_bias", args.has_bias ? 1u : 0u},
        {"num_cores", static_cast<uint32_t>(matmul_num_cores)},
        {"activation_function", static_cast<uint32_t>(activation_type)},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
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
        {"moe_config_type", config_type},
        // Matmul -> combine: dm1 increments this on combine cores when data is written
        {"matmul_combine_sync_semaphore_id", matmul_combine_sync_semaphore_id},
        // Bypass combine signaling/wait when no combine kernels are built.
        {"compute_only", args.compute_only ? 1u : 0u},
    };

    // Create kernels for the program
    auto matmul_dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm0.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = matmul_compile_time_args,
            .named_compile_args = matmul_named_compile_time_args});

    std::map<std::string, std::string> dm1_defines = {
        {"OUTPUT_SHARD_CORE_MAP", serialize_physical_core_coords(combine_cores, *mesh_device)}};

    auto matmul_dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = matmul_compile_time_args,
            .defines = dm1_defines,
            .named_compile_args = matmul_named_compile_time_args});

    auto matmul_compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp",
        matmul_core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = matmul_compile_time_args,
            .named_compile_args = matmul_named_compile_time_args});

    //-------------------------------------------------------------------------
    // ring ordering
    //-------------------------------------------------------------------------

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

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> matmul_runtime_args;
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

    // BH-only extra args: per-core start page_id for INTERLEAVED weight buffers.
    // The prepared weight tensors have the leading num_cores dim, so each ring core's
    // slice is a contiguous range of `pages_per_core` page_ids. On WH (HEIGHT_SHARDED)
    // these are unused — dm0.cpp doesn't even read them in the WH branch.
    const bool is_blackhole = mesh_device->arch() == tt::ARCH::BLACKHOLE;
    uint32_t w0_w1_pages_per_core = 0;
    uint32_t w2_pages_per_core = 0;
    if (is_blackhole) {
        const uint32_t w0_w1_total_pages = static_cast<uint32_t>(matmul_w0_w1_tensor.buffer()->num_pages());
        const uint32_t w2_total_pages = static_cast<uint32_t>(matmul_w2_tensor.buffer()->num_pages());
        TT_FATAL(
            w0_w1_total_pages % matmul_num_cores == 0,
            "BH INTERLEAVED weights: w0_w1 total pages ({}) must be divisible by matmul_num_cores ({})",
            w0_w1_total_pages,
            matmul_num_cores);
        TT_FATAL(
            w2_total_pages % matmul_num_cores == 0,
            "BH INTERLEAVED weights: w2 total pages ({}) must be divisible by matmul_num_cores ({})",
            w2_total_pages,
            matmul_num_cores);
        w0_w1_pages_per_core = w0_w1_total_pages / matmul_num_cores;
        w2_pages_per_core = w2_total_pages / matmul_num_cores;
        matmul_runtime_args.push_back(0);  // w0_w1_core_start_page_id placeholder
        matmul_runtime_args.push_back(0);  // w2_core_start_page_id placeholder
    }

    // matmul cores ordered by core ID, this will be used by selective combine to direct semaphore signaling
    std::vector<CoreCoord> ring_pos2core(matmul_num_cores);
    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (auto core : matmul_cores) {
        uint32_t vchannel = dram_bank & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(matmul_cores.begin(), matmul_cores.begin() + dram_bank, [&](const auto& core_prev) {
            return core_prev.y == core.y;
        });

        // If there is any core with the same row, make sure the VChannel is different
        if (it != matmul_cores.begin() + dram_bank) {
            size_t j = std::distance(matmul_cores.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        // Use the optimized ring neighbor mapping
        const auto [ring_pos, next_bank] = bank2ring_pos[dram_bank];
        const auto& next_physical = mesh_device->worker_core_from_logical_core(matmul_cores[next_bank]);

        ring_pos2core[ring_pos] = core;

        matmul_runtime_args[0] = dram_bank++;
        matmul_runtime_args[1] = vchannel;
        // matmul_runtime_args[2-4] are already set to tensor addresses
        // matmul_runtime_args[5] is already set to ring_semaphore_id
        matmul_runtime_args[6] = ring_pos;
        matmul_runtime_args[7] = static_cast<uint32_t>(next_physical.x);
        matmul_runtime_args[8] = static_cast<uint32_t>(next_physical.y);
        if (is_blackhole) {
            // INTERLEAVED weight buffers: each ring core's slice starts at ring_pos * pages_per_core.
            // dm1.cpp/compute.cpp don't read these args (they only read 9 args); pushing extras to all
            // three SetRuntimeArgs calls is harmless because each kernel only reads what it knows about.
            matmul_runtime_args[9] = ring_pos * w0_w1_pages_per_core;
            matmul_runtime_args[10] = ring_pos * w2_pages_per_core;
        }

        tt::tt_metal::SetRuntimeArgs(program, matmul_dm0_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_dm1_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_compute_kernel_handle, core, matmul_runtime_args);

        log_debug(tt::LogOp, "{} -> DRAM {} -> ring pos {}", core.str(), dram_bank, ring_pos);
    }

    //-------------------------------------------------------------------------
    // Combine stage
    //-------------------------------------------------------------------------

    std::vector<tt::tt_metal::KernelHandle> combine_kernel_handles;
    tt::tt_metal::CBHandle combine_data_cb_handle{};
    std::vector<GlobalSemaphore> combine_global_semaphores;
    std::vector<CoreCoord> combine_cores_for_shared = combine_cores;

    if (!args.compute_only) {
        TT_FATAL(
            args.combine_params.has_value(),
            "compute_only=false but combine_params is nullopt; combine_params is required when compute_only is false");
        TT_FATAL(init_barrier_semaphore.has_value(), "init_barrier_semaphore must be set when compute_only is false");
        TT_FATAL(final_barrier_semaphore.has_value(), "final_barrier_semaphore must be set when compute_only is false");

        // Combine parameters (copy from args and set worker_cores for this mesh)
        TT_FATAL(args.combine_params->num_links > 0, "num_links must be greater than 0");
        TT_FATAL(
            !args.combine_params->axis.has_value() || args.combine_params->axis.value() < 2,
            "cluster_axis must be 0 or 1");

        TT_FATAL(
            tensor_return_value.size() == 6,
            "compute_only=false expects 6 output tensors, got {}",
            tensor_return_value.size());
        ttnn::Tensor& output_tensor = tensor_return_value[5];

        auto combine_params = *args.combine_params;
        combine_params.worker_cores = combine_cores;
        // MoE compute op does not have an optional output tensor in its API; combine writes to the
        // tensor we create (output_tensor) passed explicitly to build_selective_reduce_combine_program_artifacts.
        ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensor_args{
            .dense_input_tensor = matmul_output_tensor,
            .dense_activations_tensor = tilize_expert_activation_output_tensor,
            .dense_token_maps_tensor = tilize_e_t_output_tensor,
            .dense_token_counts_tensor = tilize_per_expert_total_tokens_output_tensor,
            .optional_output_tensor = tensor_args.optional_output_tensor};

        // 3 compute cores write output pages to each combine cores in a column of sharded output
        const uint32_t compute_cores_per_combine_core = matmul_core_range_set.num_cores() / combine_data_parallel_cores;
        auto selective_reduce_combine_artifacts = build_selective_reduce_combine_program_artifacts(
            program,
            combine_params,
            mesh_coordinate,
            mesh_coordinates.coords(),
            combine_tensor_args,
            output_tensor,
            *init_barrier_semaphore,
            *final_barrier_semaphore,
            tilize_combine_sync_semaphore_id,
            matmul_combine_sync_semaphore_id,
            compute_cores_per_combine_core,
            ring_pos2core);

        combine_kernel_handles = {
            selective_reduce_combine_artifacts.reader_kernel_id, selective_reduce_combine_artifacts.writer_kernel_id};
        combine_data_cb_handle = selective_reduce_combine_artifacts.data_cb_handle;
        combine_global_semaphores = {*init_barrier_semaphore, *final_barrier_semaphore};
    } else {
        // compute_only: no combine kernels are built. The matmul/tilize kernels' increments to
        // combine semaphores are gated off via the compute_only CT arg.
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
         .matmul_cores = matmul_cores,
         .sharded_output_cb_handle = sharded_output_cb_handle,
         .matmul_writer_cb_handle = matmul_writer_cb_handle,
         .combine_kernel_handles = std::move(combine_kernel_handles),
         .combine_data_cb_handle = combine_data_cb_handle,
         .expert_tokens_cb_handle = expert_tokens_cb_handle,
         .combine_cores = std::move(combine_cores_for_shared),
         .combine_global_semaphores = std::move(combine_global_semaphores),
         .compute_only = args.compute_only}};
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
            program, shared_variables.sharded_output_cb_handle, *tilize_output_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.matmul_writer_cb_handle, *tilize_output_tensor.buffer());

        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.expert_tokens_cb_handle, *tilize_per_expert_total_tokens_output_tensor.buffer());

        //-------------------------------------------------------------------------
        // Tilize
        //-------------------------------------------------------------------------
        for (const auto& core : shared_variables.tilize_cores) {
            // reader
            auto& tilize_reader_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles.at(0), core);
            tilize_reader_runtime_args.at(0) = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_reader_runtime_args.at(1) = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_reader_runtime_args.at(2) = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_reader_runtime_args.at(3) = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_reader_runtime_args.at(4) = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_reader_runtime_args.at(5) = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_reader_runtime_args.at(6) = tilize_e_t_output_tensor.buffer()->address();

            // writer
            auto& tilize_writer_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles.at(2), core);
            tilize_writer_runtime_args.at(0) = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_writer_runtime_args.at(1) = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_writer_runtime_args.at(2) = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_writer_runtime_args.at(3) = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_writer_runtime_args.at(4) = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_writer_runtime_args.at(5) = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_writer_runtime_args.at(6) = tilize_e_t_output_tensor.buffer()->address();
        }

        //-------------------------------------------------------------------------
        // Matmul
        //-------------------------------------------------------------------------
        for (const auto& core : shared_variables.matmul_cores) {
            for (const auto& kernel_handle : shared_variables.matmul_kernel_handles) {
                auto& matmul_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
                matmul_runtime_args.at(2) = tensor_args.matmul_w0_w1_tensor.buffer()->address();
                matmul_runtime_args.at(3) = tensor_args.matmul_w2_tensor.buffer()->address();
                matmul_runtime_args.at(4) = tilize_output_tensor.buffer()->address();
            }
        }

        //-------------------------------------------------------------------------
        // Combine
        //-------------------------------------------------------------------------

        if (!shared_variables.compute_only) {
            TT_FATAL(
                shared_variables.combine_kernel_handles.size() == 2,
                "Expected 2 combine kernel handles when compute_only=false");
            TT_FATAL(
                shared_variables.combine_global_semaphores.size() == 2,
                "Expected 2 combine global semaphores when compute_only=false");
            TT_FATAL(args.combine_params.has_value(), "combine_params required when compute_only is false");
            TT_FATAL(
                tensor_return_value.size() == 6,
                "compute_only=false expects 6 output tensors, got {}",
                tensor_return_value.size());

            ttnn::Tensor& output_tensor = tensor_return_value[5];

            auto reader_kernel_id = shared_variables.combine_kernel_handles[0];
            auto writer_kernel_id = shared_variables.combine_kernel_handles[1];
            auto combine_data_cb_handle = shared_variables.combine_data_cb_handle;
            auto cores = shared_variables.combine_cores;
            auto init_semaphore = shared_variables.combine_global_semaphores[0];
            auto cross_device_semaphore = shared_variables.combine_global_semaphores[1];

            // MoE compute op does not have an optional output tensor; do not pass it in tensor_args.
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
                init_semaphore,
                cross_device_semaphore,
                args.combine_params->optional_cross_device_semaphore);
        }
    }
}

}  // namespace ttnn::experimental::prim
