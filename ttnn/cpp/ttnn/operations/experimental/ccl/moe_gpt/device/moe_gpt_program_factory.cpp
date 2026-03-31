// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_program_factory.hpp"
#include "moe_gpt_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

uint32_t get_num_pages(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->aligned_page_size(); }

std::string serialize_physical_core_coords(const std::vector<CoreCoord>& cores, tt::tt_metal::IDevice* device) {
    std::vector<uint32_t> flat_physical_core_coords;
    flat_physical_core_coords.reserve(2 * cores.size());

    for (const auto& c : cores) {
        const auto pc = device->worker_core_from_logical_core(c);
        flat_physical_core_coords.push_back(pc.x);
        flat_physical_core_coords.push_back(pc.y);
    }

    return ttnn::operations::ccl::common::stringify(flat_physical_core_coords);
}

}  // namespace

namespace ttnn::operations::experimental::moe_gpt::program {

MoEGPTMeshWorkloadFactory::cached_mesh_workload_t MoEGPTMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, MoEGPTMeshWorkloadFactory::shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            MoEGPTMeshWorkloadFactory::create_at(args, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return MoEGPTMeshWorkloadFactory::cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoEGPTMeshWorkloadFactory::shared_variables_t>
MoEGPTMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet&) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto* device = tensor_args.input_tensor.device();

    // =========================================================================
    // Matmul cores (existing)
    // =========================================================================
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    std::map<std::string, tt::tt_metal::CBHandle> cb_handles_sharded;

    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, true, 14 * 2 * 3},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 8 * 6},
        {"cb_c2c_ones_tile", tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, 1},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // cb_w2c_md (c_5) on matmul cores for metadata bridge (dm1 → compute)
    {
        const uint32_t md_page_size = 16;  // 2 uint32_t values, L1 aligned
        const auto cb_md_config =
            tt::tt_metal::CircularBufferConfig(2 * md_page_size, {{tt::CBIndex::c_5, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_5, md_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_md_config);
    }

    // Infer experts_per_device from w0_w1 weight tensor shape dim 2
    const uint32_t num_experts = tensor_args.w0_w1_tensor.logical_shape()[2];
    const uint32_t total_tokens = tensor_args.input_tensor.logical_shape()[0];

    // c_14 (untilized ROW_MAJOR output buffer) on matmul cores
    {
        constexpr uint32_t SOURCE_WIDTH_TILES = 8;
        constexpr uint32_t c14_page_size = SOURCE_WIDTH_TILES * 32 * 2;  // 512 bytes
        const uint32_t c14_num_pages = 32 * num_experts;                 // 128 pages = 64 KB
        const auto cb_config = tt::tt_metal::CircularBufferConfig(
                                   c14_num_pages * c14_page_size, {{tt::CBIndex::c_14, tt::DataFormat::Float16_b}})
                                   .set_page_size(tt::CBIndex::c_14, c14_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    //=========================================================================
    // Combine cores (fused mode only): COMBINE_W×COMBINE_H grid avoiding matmul cores.
    // Data-parallel (hidden dim split) × token-parallel. Each core packs
    // all experts sequentially for its token slice.
    //=========================================================================
    std::vector<CoreCoord> combine_cores;
    CoreRangeSet combine_core_range_set;
    tt::tt_metal::KernelHandle combine_dm1_handle = 0;
    uint32_t combine_semaphore_id = 0;
    uint32_t output_base_l1_addr = 0;
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles_sharded_combine;

    {
        // Dynamically find a WxH rectangle in the worker grid that avoids matmul cores
        const uint32_t COMBINE_W = operation_attributes.output_width_shard_dim;
        const uint32_t COMBINE_H = operation_attributes.output_height_shard_dim;
        std::set<std::pair<uint32_t, uint32_t>> matmul_core_set_for_combine;
        for (const auto& c : dram_bank2core_coords) {
            matmul_core_set_for_combine.insert({c.x, c.y});
        }
        CoreCoord combine_worker_grid = device->compute_with_storage_grid_size();
        CoreRange combine_core_range({0, 0}, {0, 0});
        bool found_combine_range = false;
        for (uint32_t sy = 0; sy + COMBINE_H <= combine_worker_grid.y && !found_combine_range; sy++) {
            for (uint32_t sx = 0; sx + COMBINE_W <= combine_worker_grid.x && !found_combine_range; sx++) {
                bool valid = true;
                for (uint32_t dy = 0; dy < COMBINE_H && valid; dy++) {
                    for (uint32_t dx = 0; dx < COMBINE_W && valid; dx++) {
                        if (matmul_core_set_for_combine.contains({sx + dx, sy + dy})) {
                            valid = false;
                        }
                    }
                }
                if (valid) {
                    combine_core_range = CoreRange({sx, sy}, {sx + COMBINE_W - 1, sy + COMBINE_H - 1});
                    found_combine_range = true;
                }
            }
        }
        TT_FATAL(
            found_combine_range,
            "Could not find a {}x{} combine core range that avoids matmul cores",
            COMBINE_W,
            COMBINE_H);
        log_info(
            tt::LogOp,
            "moe_gpt: selected combine core range ({},{}) to ({},{})",
            combine_core_range.start_coord.x,
            combine_core_range.start_coord.y,
            combine_core_range.end_coord.x,
            combine_core_range.end_coord.y);

        auto combine_cores_unsorted = corerange_to_cores(CoreRangeSet(combine_core_range));
        std::sort(combine_cores_unsorted.begin(), combine_cores_unsorted.end(), [](const auto& a, const auto& b) {
            return (a.y != b.y) ? a.y < b.y : a.x < b.x;
        });
        combine_cores = combine_cores_unsorted;
        combine_core_range_set = CoreRangeSet(combine_core_range);

        // Combine output CB: c_0 on combine cores, backed by the HEIGHT_SHARDED output tensor
        const auto& tilize_output_tensor = tensor_return_value.at(3);
        const CoreRangeSet shard_cores = tilize_output_tensor.memory_config().shard_spec()->grid;
        const uint32_t output_pages = tilize_output_tensor.buffer()->num_pages();
        const uint32_t output_page_size = tilize_output_tensor.buffer()->page_size();
        const uint32_t shared_cb_num_pages_combine = output_pages / shard_cores.num_cores();
        const auto output_dataformat = tt::tt_metal::datatype_to_dataformat_converter(tilize_output_tensor.dtype());

        auto [combine_cb_id, sharded_output_cb_handle] = tt::tt_metal::create_cb(
            (uint32_t)tt::CBIndex::c_0,
            program,
            combine_core_range_set,
            output_page_size,
            shared_cb_num_pages_combine,
            output_dataformat,
            tilize_output_tensor.buffer());

        cb_handles_sharded_combine["cb_combine_out"] = sharded_output_cb_handle;

        // Combine dm1 kernel
        combine_dm1_handle = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/combine_dm1.cpp",
            combine_core_range_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1});

        // NOTE: Combine semaphore is created later (after ring_semaphore) to avoid
        // semaphore slot 0 which conflicts with dispatch infrastructure on combine cores.

        // Output base L1 address
        output_base_l1_addr = tilize_output_tensor.buffer()->address();
    }

    const auto tensors = std::vector<const Tensor*>{
        &tensor_args.input_tensor, &tensor_args.w0_w1_tensor, &tensor_args.w2_tensor, &tensor_return_value.at(3)};
    // Note: TensorAccessorArgs are passed to matmul kernels as positional compile args.
    // In fused mode, only w0_w1 and w2 accessors are used (for DRAM weight reads).
    // input_tensor (sparse buffer) and output tensor accessors are vestigial but kept for index stability.

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"num_experts", num_experts},
        {"layer_id", 0u},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"enable_dram_output", 0u},
    };

    // Fused mode defines
    std::map<std::string, std::string> matmul_defines;
    std::map<std::string, std::string> dm1_defines;
    matmul_defines["TILIZE_FUSED"] = "1";

    // OUTPUT_SHARD_CORE_MAP define for dm1 (combine core physical coords)
    dm1_defines["OUTPUT_SHARD_CORE_MAP"] = serialize_physical_core_coords(combine_cores, device);

    {
        // Pre-compute tilize core coords for fused compile args.
        // Must match the core selection in the tilize block below.
        constexpr uint32_t EARLY_TILIZE_NUM_CORES = 2;
        std::set<std::pair<uint32_t, uint32_t>> early_matmul_core_set;
        for (const auto& c : dram_bank2core_coords) {
            early_matmul_core_set.insert({c.x, c.y});
        }
        CoreCoord early_worker_grid = device->compute_with_storage_grid_size();
        std::vector<CoreCoord> early_tilize_cores;
        early_tilize_cores.reserve(EARLY_TILIZE_NUM_CORES);
        for (int y = (int)early_worker_grid.y - 1; y >= 0 && early_tilize_cores.size() < EARLY_TILIZE_NUM_CORES; y--) {
            for (int x = (int)early_worker_grid.x - 1; x >= 0 && early_tilize_cores.size() < EARLY_TILIZE_NUM_CORES;
                 x--) {
                if (!early_matmul_core_set.contains({(uint32_t)x, (uint32_t)y})) {
                    early_tilize_cores.push_back(CoreCoord(x, y));
                }
            }
        }
        TT_FATAL(
            early_tilize_cores.size() >= EARLY_TILIZE_NUM_CORES,
            "Could not find {} non-matmul worker cores for early tilize precomputation",
            EARLY_TILIZE_NUM_CORES);
        CoreCoord early_drain_physical = device->worker_core_from_logical_core(early_tilize_cores.at(0));

        // Create fused-mode semaphores on merged tilize+matmul range
        CoreRangeSet early_tilize_set(early_tilize_cores);
        CoreRangeSet early_merged = early_tilize_set.merge(all_cores);
        constexpr uint32_t INVALID_SEM = 0;
        auto fused_metadata_ready_sem = tt::tt_metal::CreateSemaphore(program, early_merged, INVALID_SEM);
        auto fused_chunk_available_sem = tt::tt_metal::CreateSemaphore(program, early_merged, INVALID_SEM);
        auto fused_chunk_ready_sem = tt::tt_metal::CreateSemaphore(program, early_merged, INVALID_SEM);

        // Per-expert token count semaphores (full 32-bit counts, no bit-packing limit)
        auto metadata_count_sem_base = tt::tt_metal::CreateSemaphore(program, early_merged, INVALID_SEM);
        for (uint32_t e = 1; e < num_experts; e++) {
            tt::tt_metal::CreateSemaphore(program, early_merged, INVALID_SEM);
        }

        named_compile_time_args["metadata_ready_semaphore_id"] = fused_metadata_ready_sem;
        named_compile_time_args["metadata_count_semaphore_base_id"] = metadata_count_sem_base;
        named_compile_time_args["matmul_chunk_ready_semaphore_id"] = fused_chunk_ready_sem;
        named_compile_time_args["matmul_chunk_available_semaphore_id"] = fused_chunk_available_sem;
        named_compile_time_args["tokens_per_chunk"] = 32u;
        named_compile_time_args["tilize_drain_core_noc_x"] = (uint32_t)early_drain_physical.x;
        named_compile_time_args["tilize_drain_core_noc_y"] = (uint32_t)early_drain_physical.y;

        // Combine core compile-time args (for dm1 and compute)
        named_compile_time_args["height_shard_dim"] = operation_attributes.output_height_shard_dim;
        named_compile_time_args["width_shard_dim"] = operation_attributes.output_width_shard_dim;
        named_compile_time_args["combine_shard_width_tiles"] =
            operation_attributes.hidden_size / 32 / operation_attributes.output_width_shard_dim;
        named_compile_time_args["tile_width"] = 32u;
        named_compile_time_args["tile_width_size_bytes"] = 64u;
        named_compile_time_args["num_tokens_total"] = total_tokens;
    }

    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .defines = matmul_defines,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .defines = dm1_defines,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .defines = matmul_defines,
            .named_compile_args = named_compile_time_args});

    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Create combine semaphore after ring semaphore so it gets a higher
    // semaphore ID, avoiding conflicts with dispatch infrastructure on combine cores
    combine_semaphore_id = tt::tt_metal::CreateSemaphore(program, combine_core_range_set, 0);

    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &dram_bank2core_coords](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;
            }
            return pa.x > pb.x;
        });

    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> bank2ring_pos;
    for (uint32_t ring_pos = 0; ring_pos < num_cores; ++ring_pos) {
        uint32_t this_bank = ring_pos2bank_id[ring_pos];
        uint32_t next_bank = ring_pos2bank_id[(ring_pos + 1) % num_cores];
        bank2ring_pos[this_bank] = {ring_pos, next_bank};
    }

    constexpr uint32_t tiles_per_core_table[12] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};
    uint32_t k_start_tiles[12] = {0};
    for (uint32_t i = 1; i < num_cores; ++i) {
        k_start_tiles[i] = k_start_tiles[i - 1] + tiles_per_core_table[i - 1];
    }

    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);
    runtime_args.push_back(0);
    for (uint32_t i = 0; i < 4; ++i) {
        runtime_args.push_back(tensors[i]->buffer()->address());
    }
    runtime_args.push_back(ring_semaphore_id);
    runtime_args.push_back(0);
    runtime_args.push_back(0);
    runtime_args.push_back(0);

    // Combine-related runtime args at indices 10-12
    runtime_args.push_back(combine_semaphore_id);  // [10] combine_semaphore_id
    runtime_args.push_back(0);                     // [11] k_start_tile (set per core below)
    runtime_args.push_back(output_base_l1_addr);   // [12] output_base_l1_addr

    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (auto core : dram_bank2core_coords) {
        uint32_t vchannel = dram_bank & 0x3;
        auto it = std::find_if(
            dram_bank2core_coords.begin(), dram_bank2core_coords.begin() + dram_bank, [&](const auto& core_prev) {
                return core_prev.y == core.y;
            });
        if (it != dram_bank2core_coords.begin() + dram_bank) {
            size_t j = std::distance(dram_bank2core_coords.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        const auto [ring_pos, next_bank] = bank2ring_pos[dram_bank];
        const auto& next_physical = device->worker_core_from_logical_core(dram_bank2core_coords[next_bank]);

        runtime_args[0] = dram_bank++;
        runtime_args[1] = vchannel;
        runtime_args[7] = ring_pos;
        runtime_args[8] = static_cast<uint32_t>(next_physical.x);
        runtime_args[9] = static_cast<uint32_t>(next_physical.y);

        runtime_args[11] = k_start_tiles[ring_pos];  // k_start_tile for combine output

        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
    }

    // =========================================================================
    // Set Runtime Args for combine cores (fused mode only)
    // =========================================================================
    {
        const std::vector<uint32_t> combine_rt_args = {combine_semaphore_id};
        for (const auto& core : combine_cores) {
            tt::tt_metal::SetRuntimeArgs(program, combine_dm1_handle, core, combine_rt_args);
        }
    }

    // =========================================================================
    // Tilize cores
    // =========================================================================
    std::vector<tt::tt_metal::KernelHandle> tilize_kernel_handles;
    std::vector<CoreCoord> tilize_cores_vec;

    {
        const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
        const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

        // --- Tilize input tensors ---
        const auto& sparse_buffer = tensor_args.input_tensor;
        const auto& expert_indices = tensor_args.expert_indices;
        const auto& expert_scores = tensor_args.expert_scores;
        const auto& expert_mapping = tensor_args.expert_mapping;

        const auto& sparse_shape = sparse_buffer.tensor_spec().logical_shape();
        const auto& indices_shape = expert_indices.tensor_spec().logical_shape();
        const auto& mapping_shape = expert_mapping.tensor_spec().logical_shape();

        const uint32_t tilize_input_pages = get_num_pages(sparse_buffer);
        const uint32_t tilize_indices_pages = get_num_pages(expert_indices);
        const uint32_t tilize_scores_pages = get_num_pages(expert_scores);
        const uint32_t tilize_mapping_pages = get_num_pages(expert_mapping);

        const uint32_t tilize_input_page_size = get_page_size(sparse_buffer);
        const uint32_t tilize_indices_page_size = get_page_size(expert_indices);
        const uint32_t tilize_scores_page_size = get_page_size(expert_scores);
        const uint32_t tilize_mapping_page_size = get_page_size(expert_mapping);

        const uint32_t tilize_input_aligned_page_size = get_aligned_page_size(sparse_buffer);
        const uint32_t tilize_indices_aligned_page_size = get_aligned_page_size(expert_indices);
        const uint32_t tilize_scores_aligned_page_size = get_aligned_page_size(expert_scores);
        const uint32_t tilize_mapping_aligned_page_size = get_aligned_page_size(expert_mapping);

        // --- Dimensions ---
        uint32_t tokens = sparse_shape[0];
        uint32_t hidden_size = sparse_shape[-1];
        uint32_t experts = mapping_shape[-1];
        uint32_t selected_experts_k = indices_shape[-1];
        uint32_t experts_per_device = num_experts;
        uint32_t num_devices = experts / experts_per_device;

        const auto& mesh_view = device->get_view();
        uint32_t linearized_mesh_coord =
            ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
        auto mesh_shape = mesh_view.shape();
        uint32_t mesh_rows = mesh_shape[0];
        uint32_t mesh_cols = mesh_shape[1];
        uint32_t cluster_axis = operation_attributes.cluster_axis.value_or(0);

        // --- 2 Tilize cores (dynamically selected to avoid matmul cores) ---
        constexpr uint32_t TILIZE_NUM_CORES = 2;

        // Build set of matmul core coordinates for O(1) lookup
        std::set<std::pair<uint32_t, uint32_t>> matmul_core_set;
        for (const auto& c : dram_bank2core_coords) {
            matmul_core_set.insert({c.x, c.y});
        }

        // The dispatch drain core holds the HEIGHT_SHARDED indices/scores shard from
        // all_to_all_dispatch_metadata.  Because dispatch cores cannot host user kernels,
        // we cannot use CB aliasing here.  Instead, both tilize cores do a single bulk NOC
        // read from the dispatch drain core at kernel start.
        const CoreCoord dispatch_drain_logical =
            expert_indices.buffer()->shard_spec().tensor_shard_spec.grid.ranges().begin()->start_coord;

        // Pick tilize cores from the worker grid, avoiding matmul cores.
        // Scan from the bottom-right corner (high y, high x) to stay far from DRAM-mapped cores.
        CoreCoord worker_grid = device->compute_with_storage_grid_size();
        std::vector<CoreCoord> tilize_cores;
        tilize_cores.reserve(TILIZE_NUM_CORES);
        for (int y = (int)worker_grid.y - 1; y >= 0 && tilize_cores.size() < TILIZE_NUM_CORES; y--) {
            for (int x = (int)worker_grid.x - 1; x >= 0 && tilize_cores.size() < TILIZE_NUM_CORES; x--) {
                if (!matmul_core_set.contains({(uint32_t)x, (uint32_t)y})) {
                    tilize_cores.push_back(CoreCoord(x, y));
                }
            }
        }
        TT_FATAL(
            tilize_cores.size() == TILIZE_NUM_CORES,
            "Could not find {} non-matmul worker cores for tilize",
            TILIZE_NUM_CORES);
        tilize_cores_vec = tilize_cores;

        // Verify no tilize core is also a matmul core
        for (const auto& tc : tilize_cores) {
            TT_FATAL(
                !matmul_core_set.contains({tc.x, tc.y}), "Tilize core ({},{}) overlaps with a matmul core", tc.x, tc.y);
        }

        const CoreRangeSet tilize_core_range_set = CoreRangeSet(tilize_cores);
        const CoreRange tilize_bounding_box = tilize_core_range_set.bounding_box();
        const CoreRange matmul_bounding_box = all_cores.bounding_box();

        const uint32_t tilize_bounding_box_num_cores = tilize_bounding_box.size();

        const CoreCoord tilize_mcast_start_physical =
            device->worker_core_from_logical_core(tilize_bounding_box.start_coord);
        const CoreCoord tilize_mcast_end_physical =
            device->worker_core_from_logical_core(tilize_bounding_box.end_coord);
        const CoreCoord matmul_mcast_start_physical =
            device->worker_core_from_logical_core(matmul_bounding_box.start_coord);
        const CoreCoord matmul_mcast_end_physical =
            device->worker_core_from_logical_core(matmul_bounding_box.end_coord);

        // Physical coords for all tilize cores
        std::vector<CoreCoord> tilize_cores_physical;
        tilize_cores_physical.reserve(TILIZE_NUM_CORES);
        for (uint32_t i = 0; i < TILIZE_NUM_CORES; i++) {
            tilize_cores_physical.push_back(device->worker_core_from_logical_core(tilize_cores.at(i)));
        }

        // Physical coords of the dispatch drain core (for NOC reads from both tilize cores)
        const CoreCoord dispatch_drain_physical = device->worker_core_from_logical_core(dispatch_drain_logical);

        // --- Tilize output tensors (metadata written to DRAM by reader) ---
        // These are pre-allocated: per_expert_total_tokens, expert_activation, e_t are embedded
        // in the tilize_output for now. For Phase 1, we only use the main tilize_output tensor.

        // --- Output tensor specs ---
        auto per_expert_total_tokens_row_bytes = tt::align(experts_per_device * sizeof(uint32_t), l1_alignment);
        auto tilize_per_expert_total_tokens_output_page_size = per_expert_total_tokens_row_bytes;

        uint32_t activation_row_elements = (2 * experts_per_device) + 1;
        uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
        uint32_t activation_total_bytes = (tokens + 1) * activation_row_bytes;
        auto tilize_expert_activation_output_page_size = activation_total_bytes;

        // Matches moe_compute format: {E, (T+1)*stride} with 1 page per expert.
        // Each page = (tokens + 1) * align(sizeof(uint32_t), l1_alignment) bytes.
        auto tilize_e_t_output_page_size =
            (tokens + 1) * tt::align(static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);

        // --- Semaphores ---
        constexpr uint32_t INVALID = 0;

        auto tilize_partial_metadata_ready_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);
        auto tilize_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);
        auto previous_chunk_sent_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);
        auto initial_gather_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

        // Tilize+Matmul shared semaphores (created early before matmul kernel creation)
        const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(all_cores);
        uint32_t metadata_ready_semaphore_id = named_compile_time_args.at("metadata_ready_semaphore_id");
        uint32_t matmul_chunk_available_semaphore_id =
            named_compile_time_args.at("matmul_chunk_available_semaphore_id");
        uint32_t matmul_chunk_ready_semaphore_id = named_compile_time_args.at("matmul_chunk_ready_semaphore_id");

        // --- Work split ---
        uint32_t tilize_subtoken_bytes_aligned =
            tt::align(tt::div_up(tilize_input_aligned_page_size, TILIZE_NUM_CORES), l1_alignment);
        uint32_t tilize_subtoken_units_of_work =
            tt::div_up(tilize_input_aligned_page_size, tilize_subtoken_bytes_aligned);

        auto
            [num_tilize_work_cores,
             all_tilize_work_cores,
             tilize_cores_work_group_1,
             tilize_cores_work_group_2,
             tilize_units_per_core_work_group_1,
             tilize_units_per_core_work_group_2] =
                tt::tt_metal::split_work_to_cores(tilize_core_range_set, tilize_subtoken_units_of_work);

        uint32_t max_tilize_subtoken_size =
            std::max(tilize_units_per_core_work_group_1, tilize_units_per_core_work_group_2) *
            tilize_subtoken_bytes_aligned;

        constexpr uint32_t TILE_WIDTH = 32;
        constexpr uint32_t tokens_per_chunk = 32;

        uint32_t tile_width_bytes = TILE_WIDTH * sparse_buffer.element_size();
        uint32_t max_tiles_per_local_chunk = max_tilize_subtoken_size / tile_width_bytes;

        const uint32_t primary_mcast_gather_group_num_cores = (TILIZE_NUM_CORES + 1) / 2;
        const uint32_t secondary_mcast_gather_group_num_cores = TILIZE_NUM_CORES / 2;

        // --- Tilize CBs ---
        uint32_t tilize_input_cb_id = tt::CBIndex::c_7;
        uint32_t tilize_output_cb_id = tt::CBIndex::c_16;
        uint32_t per_expert_total_tokens_cb_id = tt::CBIndex::c_1;
        uint32_t total_chunks_cb_id = tt::CBIndex::c_2;
        uint32_t indices_tensor_cb_id = tt::CBIndex::c_3;
        uint32_t mapping_tensor_cb_id = tt::CBIndex::c_4;
        uint32_t scores_tensor_cb_id = tt::CBIndex::c_5;
        uint32_t e_t_cb_id = tt::CBIndex::c_6;
        uint32_t expert_activation_cb_id = tt::CBIndex::c_8;
        uint32_t brisc_e_t_cb_id = tt::CBIndex::c_19;
        uint32_t brisc_expert_counts_cb_id = tt::CBIndex::c_10;
        uint32_t brisc_expert_activation_cb_id = tt::CBIndex::c_11;
        uint32_t brisc_activated_count_cb_id = tt::CBIndex::c_12;
        uint32_t remote_counts_cb_id = tt::CBIndex::c_13;

        const auto tilize_input_data_format = tt::tt_metal::datatype_to_dataformat_converter(sparse_buffer.dtype());
        const auto tilize_indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_indices.dtype());
        const auto tilize_scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_scores.dtype());
        const auto tilize_mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_mapping.dtype());
        const auto tilize_output_data_format = tt::tt_metal::datatype_to_dataformat_converter(sparse_buffer.dtype());

        // tiles_per_local_chunk for the drain core (used for CB sizing)
        uint32_t shared_cb_num_pages = max_tiles_per_local_chunk;

        tt::tt_metal::create_cb(
            per_expert_total_tokens_cb_id,
            program,
            tilize_core_range_set,
            tilize_per_expert_total_tokens_output_page_size,
            1,
            tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32));

        tt::tt_metal::create_cb(
            total_chunks_cb_id, program, tilize_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);

        tt::tt_metal::create_cb(
            e_t_cb_id,
            program,
            tilize_core_range_set,
            tilize_e_t_output_page_size,
            experts_per_device,
            tt::DataFormat::UInt32);

        // Indices CB: fresh on both tilize cores.  Both cores do a bulk NOC read from the
        // dispatch drain core at kernel start.  Use aligned_page_size so the CB covers the
        // full physical buffer allocation (num_pages * aligned_page_size bytes).
        tt::tt_metal::create_cb(
            indices_tensor_cb_id,
            program,
            tilize_core_range_set,
            tilize_indices_aligned_page_size,
            tilize_indices_pages,
            tilize_indices_data_format);

        // Scores CB: same pattern as indices.
        tt::tt_metal::create_cb(
            scores_tensor_cb_id,
            program,
            tilize_core_range_set,
            tilize_scores_aligned_page_size,
            tilize_scores_pages,
            tilize_scores_data_format);

        tt::tt_metal::create_cb(
            mapping_tensor_cb_id,
            program,
            tilize_core_range_set,
            tilize_mapping_aligned_page_size,
            tilize_mapping_pages,
            tilize_mapping_data_format);

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

        tt::tt_metal::create_cb(
            brisc_e_t_cb_id,
            program,
            tilize_core_range_set,
            (tokens / 2) * l1_alignment * experts_per_device,
            1,
            tt::DataFormat::UInt32);

        tt::tt_metal::create_cb(
            brisc_expert_counts_cb_id,
            program,
            tilize_core_range_set,
            sizeof(uint32_t) * experts_per_device,
            1,
            tt::DataFormat::UInt32);

        uint32_t brisc_activation_row_size = tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment);
        tt::tt_metal::create_cb(
            brisc_expert_activation_cb_id,
            program,
            tilize_core_range_set,
            brisc_activation_row_size * (tokens / 2),
            1,
            tt::DataFormat::UInt32);

        tt::tt_metal::create_cb(
            brisc_activated_count_cb_id, program, tilize_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);

        uint32_t counts_per_remote_core = experts_per_device + 1;
        uint32_t remote_counts_entry_size = tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment);
        tt::tt_metal::create_cb(
            remote_counts_cb_id,
            program,
            tilize_core_range_set,
            remote_counts_entry_size,
            TILIZE_NUM_CORES - 1,
            tt::DataFormat::UInt32);

        // Tilize output CB
        auto tilize_output_tile_size = tt::tile_size(tilize_output_data_format);
        uint32_t tiles_per_global_chunk = hidden_size / TILE_WIDTH;  // 90 for GPT-OSS

        // c_16 is the shared tilize→matmul activation CB.
        // Created on the merged tilize+matmul range so multicast addresses match.
        // Double-buffered: 2 * tiles_per_global_chunk = 2 * 90 = 180 tiles.
        {
            uint32_t fused_cb_num_pages = 2 * tiles_per_global_chunk;
            const auto cb_config = tt::tt_metal::CircularBufferConfig(
                                       fused_cb_num_pages * tilize_output_tile_size,
                                       {{(tt::CBIndex)tilize_output_cb_id, tilize_output_data_format}})
                                       .set_page_size((tt::CBIndex)tilize_output_cb_id, tilize_output_tile_size);
            tt::tt_metal::CreateCircularBuffer(program, tilize_matmul_core_range_set, cb_config);
        }

        // --- Tilize tensor accessors ---
        std::vector<uint32_t> tilize_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(sparse_buffer.buffer()).append_to(tilize_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(expert_indices.buffer()).append_to(tilize_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(expert_scores.buffer()).append_to(tilize_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(expert_mapping.buffer()).append_to(tilize_compile_time_args);
        // Metadata output tensor accessors (outputs 0, 1, 2 written by drain tilize core)
        tt::tt_metal::TensorAccessorArgs(tensor_return_value[0].buffer())
            .append_to(tilize_compile_time_args);  // per_expert_total_tokens
        tt::tt_metal::TensorAccessorArgs(tensor_return_value[1].buffer())
            .append_to(tilize_compile_time_args);  // expert_activation
        tt::tt_metal::TensorAccessorArgs(tensor_return_value[2].buffer()).append_to(tilize_compile_time_args);  // e_t

        // --- Named compile-time args ---
        std::unordered_map<std::string, uint32_t> tilize_named_compile_time_args = {
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

            {"l1_alignment", l1_alignment},
            {"dram_alignment", dram_alignment},
            {"e_t_entry_size", l1_alignment},

            {"input_pages", tilize_input_pages},
            {"indices_pages", tilize_indices_pages},
            {"mapping_pages", tilize_mapping_pages},
            {"scores_pages", tilize_scores_pages},
            {"shared_cb_num_pages", shared_cb_num_pages},

            {"input_page_size", tilize_input_page_size},
            {"indices_page_size", tilize_indices_page_size},
            {"scores_page_size", tilize_scores_page_size},
            {"mapping_page_size", tilize_mapping_page_size},
            {"per_expert_total_tokens_output_page_size", tilize_per_expert_total_tokens_output_page_size},
            {"expert_activation_output_page_size", tilize_expert_activation_output_page_size},
            {"e_t_output_page_size", tilize_e_t_output_page_size},
            {"tilize_output_page_size", (uint32_t)tilize_output_tile_size},

            {"aligned_input_page_size", tilize_input_aligned_page_size},
            {"aligned_indices_page_size", tilize_indices_aligned_page_size},
            {"aligned_mapping_page_size", tilize_mapping_aligned_page_size},
            {"aligned_scores_page_size", tilize_scores_aligned_page_size},

            {"tokens", tokens},
            {"hidden_size", hidden_size},
            {"remote_counts_entry_size", remote_counts_entry_size},
            {"experts", experts},
            {"selected_experts_k", selected_experts_k},

            {"tokens_per_chunk", tokens_per_chunk},

            {"num_devices", num_devices},
            {"mesh_rows", mesh_rows},
            {"mesh_cols", mesh_cols},
            {"linearized_mesh_coord", linearized_mesh_coord},
            {"cluster_axis", cluster_axis},

            {"drain_core_noc_x", (uint32_t)tilize_cores_physical.at(0).x},
            {"drain_core_noc_y", (uint32_t)tilize_cores_physical.at(0).y},

            {"dispatch_drain_noc_x", (uint32_t)dispatch_drain_physical.x},
            {"dispatch_drain_noc_y", (uint32_t)dispatch_drain_physical.y},

            {"primary_mcast_gather_group_num_cores", primary_mcast_gather_group_num_cores},
            {"secondary_mcast_gather_group_num_cores", secondary_mcast_gather_group_num_cores},

            {"num_tilize_cores", TILIZE_NUM_CORES},
            {"tilize_mcast_start_x", (uint32_t)tilize_mcast_start_physical.x},
            {"tilize_mcast_start_y", (uint32_t)tilize_mcast_start_physical.y},
            {"tilize_mcast_end_x", (uint32_t)tilize_mcast_end_physical.x},
            {"tilize_mcast_end_y", (uint32_t)tilize_mcast_end_physical.y},
            {"tilize_bounding_box_num_cores", tilize_bounding_box_num_cores},

            {"num_matmul_cores", (uint32_t)num_cores},
            {"matmul_mcast_start_x", (uint32_t)matmul_mcast_start_physical.x},
            {"matmul_mcast_start_y", (uint32_t)matmul_mcast_start_physical.y},
            {"matmul_mcast_end_x", (uint32_t)matmul_mcast_end_physical.x},
            {"matmul_mcast_end_y", (uint32_t)matmul_mcast_end_physical.y},
            // NOC multicast num_dests must exclude self. Tilize cores are inside the
            // matmul bounding box, so subtract 1 for the sender.
            {"matmul_bounding_box_num_cores", (uint32_t)matmul_bounding_box.size() - 1},

            {"partial_metadata_ready_semaphore_id", tilize_partial_metadata_ready_semaphore_id},
            {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
            {"metadata_count_semaphore_base_id", named_compile_time_args.at("metadata_count_semaphore_base_id")},
            {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
            {"tilize_chunk_ready_semaphore_id", tilize_chunk_ready_semaphore_id},
            {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
            {"previous_chunk_sent_semaphore_id", previous_chunk_sent_semaphore_id},
            {"initial_gather_semaphore_id", initial_gather_semaphore_id},

            {"max_tiles_per_local_chunk", max_tiles_per_local_chunk},
            {"tiles_per_global_chunk", tiles_per_global_chunk},
        };

        // --- Create tilize kernels ---
        std::map<std::string, std::string> tilize_dram_defines;

        auto tilize_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_reader.cpp",
            tilize_core_range_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                .compile_args = tilize_compile_time_args,
                .defines = tilize_dram_defines,
                .named_compile_args = tilize_named_compile_time_args,
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

        auto tilize_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_writer.cpp",
            tilize_core_range_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_1,
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                .compile_args = tilize_compile_time_args,
                .defines = tilize_dram_defines,
                .named_compile_args = tilize_named_compile_time_args,
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

        std::unordered_map<std::string, uint32_t> compute_tilize_named_compile_time_args = {
            {"tilize_input_cb_id", tilize_input_cb_id},
            {"tilize_output_cb_id", tilize_output_cb_id},
            {"total_chunks_cb_id", total_chunks_cb_id},
            {"tokens_per_chunk", tokens_per_chunk},
            {"max_tiles_per_local_chunk", max_tiles_per_local_chunk},
            {"shared_cb_num_pages", shared_cb_num_pages},
        };

        auto tilize_compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_compute.cpp",
            tilize_core_range_set,
            tt::tt_metal::ComputeConfig{.named_compile_args = compute_tilize_named_compile_time_args});

        tilize_kernel_handles = {tilize_reader_kernel_id, tilize_compute_kernel_id, tilize_writer_kernel_id};

        // --- Per-core runtime args ---
        std::vector<uint32_t> tilize_runtime_args = {
            sparse_buffer.buffer()->address(),           // 0
            expert_indices.buffer()->address(),          // 1
            expert_scores.buffer()->address(),           // 2
            expert_mapping.buffer()->address(),          // 3
            tensor_return_value[0].buffer()->address(),  // 4: per_expert_total_tokens
            tensor_return_value[1].buffer()->address(),  // 5: expert_activation
            tensor_return_value[2].buffer()->address(),  // 6: e_t
        };

        uint32_t is_drain_tilize_core_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 7
        uint32_t is_secondary_mcaster_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 8
        uint32_t initial_mcast_gather_core_noc_x_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 9
        uint32_t initial_mcast_gather_core_noc_y_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 10
        uint32_t global_subtoken_offset_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 11
        uint32_t mcast_group_subtoken_offset_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 12
        uint32_t mcast_group_subtoken_size_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 13
        uint32_t subtoken_size_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 14
        uint32_t core_token_start_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 15
        uint32_t core_token_end_idx = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 16
        uint32_t tilize_core_idx_rt = tilize_runtime_args.size();
        tilize_runtime_args.push_back(0);  // 17

        // NOC coordinates for all tilize cores
        for (uint32_t i = 0; i < TILIZE_NUM_CORES; i++) {
            tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).x);
            tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).y);
        }

        // Placeholder for DRAM output address (unused in fused mode)
        tilize_runtime_args.push_back(0);

        // Pre-compute mcast group sizes
        uint32_t primary_mcast_gather_group_subtoken_size = 0;
        uint32_t secondary_mcast_gather_group_subtoken_size = 0;
        uint32_t global_subtoken_offset = 0;
        for (uint32_t i = 0; i < TILIZE_NUM_CORES; i++) {
            uint32_t subtoken_size = 0;
            if (tilize_cores_work_group_1.contains(tilize_cores.at(i))) {
                subtoken_size = tilize_units_per_core_work_group_1 * tilize_subtoken_bytes_aligned;
            } else if (tilize_cores_work_group_2.contains(tilize_cores.at(i))) {
                subtoken_size = tilize_units_per_core_work_group_2 * tilize_subtoken_bytes_aligned;
            }
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

        uint32_t tokens_per_tilize_core = tokens / TILIZE_NUM_CORES;

        std::vector<uint32_t> tilize_compute_runtime_args = {0};  // tiles_per_chunk

        global_subtoken_offset = 0;
        uint32_t group_subtoken_offset = 0;
        for (uint32_t i = 0; i < TILIZE_NUM_CORES; i++) {
            tilize_runtime_args.at(is_drain_tilize_core_idx) = (i == 0) ? 1 : 0;
            tilize_runtime_args.at(is_secondary_mcaster_idx) = (i == primary_mcast_gather_group_num_cores) ? 1 : 0;

            CoreCoord initial_mcast_gather_core_physical =
                i < primary_mcast_gather_group_num_cores
                    ? tilize_cores_physical.at(0)
                    : tilize_cores_physical.at(primary_mcast_gather_group_num_cores);
            tilize_runtime_args.at(initial_mcast_gather_core_noc_x_idx) =
                (uint32_t)initial_mcast_gather_core_physical.x;
            tilize_runtime_args.at(initial_mcast_gather_core_noc_y_idx) =
                (uint32_t)initial_mcast_gather_core_physical.y;

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
                subtoken_size = tilize_input_aligned_page_size - global_subtoken_offset;
            }
            tilize_runtime_args.at(subtoken_size_idx) = subtoken_size;

            global_subtoken_offset += subtoken_size;
            group_subtoken_offset += subtoken_size;
            if (i == primary_mcast_gather_group_num_cores - 1) {
                group_subtoken_offset = 0;
            }

            uint32_t core_token_start = i * tokens_per_tilize_core;
            uint32_t core_token_end = (i == TILIZE_NUM_CORES - 1) ? tokens : (i + 1) * tokens_per_tilize_core;
            tilize_runtime_args.at(core_token_start_idx) = core_token_start;
            tilize_runtime_args.at(core_token_end_idx) = core_token_end;
            tilize_runtime_args.at(tilize_core_idx_rt) = i;

            tilize_compute_runtime_args.at(0) = subtoken_size / tile_width_bytes;

            tt::tt_metal::SetRuntimeArgs(program, tilize_reader_kernel_id, tilize_cores.at(i), tilize_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, tilize_writer_kernel_id, tilize_cores.at(i), tilize_runtime_args);
            tt::tt_metal::SetRuntimeArgs(
                program, tilize_compute_kernel_id, tilize_cores.at(i), tilize_compute_runtime_args);
        }
    }

    // =========================================================================
    // Return
    // =========================================================================
    // Merge sharded CB handles from combine cores
    for (const auto& [k, v] : cb_handles_sharded_combine) {
        cb_handles_sharded[k] = v;
    }

    return ttnn::device_operation::CachedProgram<shared_variables_t>{
        std::move(program),
        MoEGPTSharedVariables{
            .cb_handles_sharded = cb_handles_sharded,
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .tilize_kernel_handles = tilize_kernel_handles,
            .combine_kernel_handle = combine_dm1_handle,
            .worker_cores = dram_bank2core_coords,
            .tilize_cores = tilize_cores_vec,
            .combine_cores = combine_cores}};
}

void MoEGPTMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& output_tensor = tensor_return_value.at(3);

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Update combine output CB
        if (shared_variables.cb_handles_sharded.contains("cb_combine_out")) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(
                program, shared_variables.cb_handles_sharded.at("cb_combine_out"), *output_tensor.buffer());
        }

        // Update runtime args for matmul kernels
        for (const auto& core : shared_variables.worker_cores) {
            for (const auto& kernel_handle : shared_variables.kernel_handles) {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
                runtime_args[3] = tensor_args.w0_w1_tensor.buffer()->address();
                runtime_args[4] = tensor_args.w2_tensor.buffer()->address();
                runtime_args[12] = output_tensor.buffer()->address();
            }
        }

        // Update runtime args for tilize kernels
        if (!shared_variables.tilize_kernel_handles.empty()) {
            for (const auto& core : shared_variables.tilize_cores) {
                for (uint32_t k : {0u, 2u}) {
                    auto& rt =
                        tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles.at(k), core);
                    rt.at(0) = tensor_args.input_tensor.buffer()->address();
                    rt.at(1) = tensor_args.expert_indices.buffer()->address();
                    rt.at(2) = tensor_args.expert_scores.buffer()->address();
                    rt.at(3) = tensor_args.expert_mapping.buffer()->address();
                    rt.at(4) = tensor_return_value.at(0).buffer()->address();  // per_expert_total_tokens
                    rt.at(5) = tensor_return_value.at(1).buffer()->address();  // expert_activation
                    rt.at(6) = tensor_return_value.at(2).buffer()->address();  // e_t
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_gpt::program
