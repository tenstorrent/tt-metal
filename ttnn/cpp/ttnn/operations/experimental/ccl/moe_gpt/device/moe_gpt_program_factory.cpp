// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_program_factory.hpp"
#include "moe_gpt_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

uint32_t get_num_pages(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->aligned_page_size(); }

std::string serialize_physical_core_coords(const std::vector<tt::tt_metal::CoreCoord>& cores, tt::tt_metal::IDevice* device) {
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

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

namespace {

// Lowest id free on every core in the range, matching ProgramImpl::create_semaphore.
uint32_t add_worker_semaphore(
    tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& core_ranges, uint32_t initial_value) {
    const auto cores = corerange_to_cores(core_ranges);
    TT_FATAL(!cores.empty(), "Cannot allocate a semaphore on an empty core range");
    const auto first_free = desc.find_available_semaphore_id(cores.front(), tt::CoreType::WORKER);
    constexpr uint32_t kMaxSemaphores = 16;
    auto used_on_core = [&](uint32_t sem_id, const CoreCoord& core) {
        for (const auto& sem : desc.semaphores) {
            if (sem.core_type == tt::CoreType::WORKER && sem.id == sem_id && sem.core_ranges.contains(core)) {
                return true;
            }
        }
        return false;
    };
    std::optional<uint32_t> sem_id;
    for (uint32_t candidate = first_free.value_or(0); candidate < kMaxSemaphores; ++candidate) {
        bool used = false;
        for (const auto& core : cores) {
            if (used_on_core(candidate, core)) {
                used = true;
                break;
            }
        }
        if (!used) {
            sem_id = candidate;
            break;
        }
    }
    TT_FATAL(
        sem_id.has_value(),
        "Unable to initialize semaphore on CoreRangeSet {}: all {} IDs are in use",
        core_ranges.str(),
        kMaxSemaphores);
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = *sem_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges,
        .initial_value = initial_value,
    });
    return *sem_id;
}

tt::tt_metal::ProgramDescriptor create_program_descriptor_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* device = tensor_args.input_tensor.device();

    auto add_cb = [&](const CoreRangeSet& cores,
                      uint32_t cb_index,
                      tt::DataFormat data_format,
                      uint32_t page_size,
                      uint32_t num_pages,
                      tt::tt_metal::Buffer* buffer = nullptr) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = num_pages * page_size,
            .core_ranges = cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = data_format,
                .page_size = page_size,
            }}},
            .buffer = buffer,
        });
    };

    // =========================================================================
    // Matmul cores (existing)
    // =========================================================================
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, true, 14 * 2 * 3},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 8 * 6},
        {"cb_c2c_ones_tile", tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, 1},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : cb_specs0) {
        (void)name;
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        add_cb(all_cores, static_cast<uint32_t>(index), data_format, bytes_per_tile, tiles_per_cb);
    }

    // cb_w2c_md (c_5) on matmul cores for metadata bridge (dm1 → compute)
    {
        const uint32_t md_page_size = 16;  // 2 uint32_t values, L1 aligned
        add_cb(all_cores, static_cast<uint32_t>(tt::CBIndex::c_5), tt::DataFormat::UInt32, md_page_size, 2);
    }

    // Infer experts_per_device from w0_w1 weight tensor shape dim 2
    const uint32_t num_experts = tensor_args.w0_w1_tensor.logical_shape()[2];
    const uint32_t total_tokens = tensor_args.input_tensor.logical_shape()[0];

    // c_14 (untilized ROW_MAJOR output buffer) on matmul cores
    {
        constexpr uint32_t SOURCE_WIDTH_TILES = 8;
        constexpr uint32_t c14_page_size = SOURCE_WIDTH_TILES * 32 * 2;  // 512 bytes
        const uint32_t c14_num_pages = 32 * num_experts;                 // 128 pages = 64 KB
        add_cb(
            all_cores,
            static_cast<uint32_t>(tt::CBIndex::c_14),
            tt::DataFormat::Float16_b,
            c14_page_size,
            c14_num_pages);
    }

    //=========================================================================
    // Combine cores (fused mode only): COMBINE_W×COMBINE_H grid avoiding matmul cores.
    // Data-parallel (hidden dim split) × token-parallel. Each core packs
    // all experts sequentially for its token slice.
    //=========================================================================
    std::vector<CoreCoord> combine_cores;
    CoreRangeSet combine_core_range_set;
    tt::tt_metal::Buffer* output_buffer = nullptr;
    uint32_t fused_metadata_ready_sem = 0;
    uint32_t fused_chunk_available_sem = 0;
    uint32_t fused_chunk_ready_sem = 0;
    uint32_t metadata_count_sem_base = 0;

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
        const uint32_t output_page_size = static_cast<uint32_t>(tilize_output_tensor.buffer()->page_size());
        const uint32_t shared_cb_num_pages_combine = output_pages / shard_cores.num_cores();
        const auto output_dataformat = tt::tt_metal::datatype_to_dataformat_converter(tilize_output_tensor.dtype());
        output_buffer = tilize_output_tensor.buffer();

        add_cb(
            combine_core_range_set,
            static_cast<uint32_t>(tt::CBIndex::c_0),
            output_dataformat,
            output_page_size,
            shared_cb_num_pages_combine,
            output_buffer);

        // Combine dm1 kernel
        desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
            .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/combine_dm1.cpp",
            .core_ranges = combine_core_range_set,
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1},
        });

        // NOTE: Combine semaphore is created later (after ring_semaphore) to avoid
        // semaphore slot 0 which conflicts with dispatch infrastructure on combine cores.
    }
    const size_t combine_dm1_kernel_id = 0;

    const auto tensors = std::vector<const Tensor*>{
        &tensor_args.input_tensor, &tensor_args.w0_w1_tensor, &tensor_args.w2_tensor, &tensor_return_value.at(3)};
    // Note: TensorAccessorArgs are passed to matmul kernels as positional compile args.
    // In fused mode, only w0_w1 and w2 accessors are used (for DRAM weight reads).
    // input_tensor (sparse buffer) and output tensor accessors are vestigial but kept for index stability.

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    std::vector<std::pair<std::string, uint32_t>> named_compile_time_args = {
        {"num_experts", num_experts},
        {"layer_id", 0u},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"enable_dram_output", 0u},
    };

    // Fused mode defines
    const std::vector<std::pair<std::string, std::string>> matmul_defines = {{"TILIZE_FUSED", "1"}};
    // OUTPUT_SHARD_CORE_MAP define for dm1 (combine core physical coords)
    const std::vector<std::pair<std::string, std::string>> dm1_defines = {
        {"OUTPUT_SHARD_CORE_MAP", serialize_physical_core_coords(combine_cores, device)},
    };

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
        fused_metadata_ready_sem = add_worker_semaphore(desc, early_merged, INVALID_SEM);
        fused_chunk_available_sem = add_worker_semaphore(desc, early_merged, INVALID_SEM);
        fused_chunk_ready_sem = add_worker_semaphore(desc, early_merged, INVALID_SEM);

        // Per-expert token count semaphores (full 32-bit counts, no bit-packing limit)
        metadata_count_sem_base = add_worker_semaphore(desc, early_merged, INVALID_SEM);
        for (uint32_t e = 1; e < num_experts; e++) {
            add_worker_semaphore(desc, early_merged, INVALID_SEM);
        }

        named_compile_time_args.emplace_back("metadata_ready_semaphore_id", fused_metadata_ready_sem);
        named_compile_time_args.emplace_back("metadata_count_semaphore_base_id", metadata_count_sem_base);
        named_compile_time_args.emplace_back("matmul_chunk_ready_semaphore_id", fused_chunk_ready_sem);
        named_compile_time_args.emplace_back("matmul_chunk_available_semaphore_id", fused_chunk_available_sem);
        named_compile_time_args.emplace_back("tokens_per_chunk", 32u);
        named_compile_time_args.emplace_back("tilize_drain_core_noc_x", (uint32_t)early_drain_physical.x);
        named_compile_time_args.emplace_back("tilize_drain_core_noc_y", (uint32_t)early_drain_physical.y);

        // Combine core compile-time args (for dm1 and compute)
        named_compile_time_args.emplace_back("height_shard_dim", operation_attributes.output_height_shard_dim);
        named_compile_time_args.emplace_back("width_shard_dim", operation_attributes.output_width_shard_dim);
        named_compile_time_args.emplace_back(
            "combine_shard_width_tiles",
            operation_attributes.hidden_size / 32 / operation_attributes.output_width_shard_dim);
        named_compile_time_args.emplace_back("tile_width", 32u);
        named_compile_time_args.emplace_back("tile_width_size_bytes", 64u);
        named_compile_time_args.emplace_back("num_tokens_total", total_tokens);
    }

    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/dm0.cpp",
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = matmul_defines,
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    });
    const size_t dm0_kernel_id = 1;

    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/dm1.cpp",
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = dm1_defines,
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    });
    const size_t dm1_kernel_id = 2;

    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/compute.cpp",
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = matmul_defines,
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
                .fp32_dest_acc_en = false,
                .dst_full_sync_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = true,
            },
    });
    const size_t compute_kernel_id = 3;

    const uint32_t ring_semaphore_id = add_worker_semaphore(desc, all_cores, 0);

    // Create combine semaphore after ring semaphore so it gets a higher
    // semaphore ID, avoiding conflicts with dispatch infrastructure on combine cores
    const uint32_t combine_semaphore_id = add_worker_semaphore(desc, combine_core_range_set, 0);

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

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* w0_w1_buffer = tensor_args.w0_w1_tensor.buffer();
    auto* w2_buffer = tensor_args.w2_tensor.buffer();

    std::vector<uint32_t> vchannels;
    vchannels.reserve(dram_bank2core_coords.size());
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
        const uint32_t this_bank = dram_bank++;

        // [2] input and [5] output are addresses the old override left stale; they are still addresses.
        tt::tt_metal::KernelDescriptor::RTArgList runtime_args;
        runtime_args.reserve(13);
        runtime_args.push_back(this_bank);                               // [0] dram_bank
        runtime_args.push_back(vchannel);                                // [1] vchannel
        runtime_args.push_back(input_buffer);                            // [2] input
        runtime_args.push_back(w0_w1_buffer);                            // [3] w0_w1
        runtime_args.push_back(w2_buffer);                               // [4] w2
        runtime_args.push_back(output_buffer);                           // [5] output
        runtime_args.push_back(ring_semaphore_id);                       // [6]
        runtime_args.push_back(ring_pos);                                // [7]
        runtime_args.push_back(static_cast<uint32_t>(next_physical.x));  // [8]
        runtime_args.push_back(static_cast<uint32_t>(next_physical.y));  // [9]
        runtime_args.push_back(combine_semaphore_id);                    // [10] combine_semaphore_id
        runtime_args.push_back(k_start_tiles[ring_pos]);                 // [11] k_start_tile
        runtime_args.push_back(output_buffer);                           // [12] output_base_l1_addr

        desc.kernels[dm0_kernel_id].emplace_runtime_args(core, runtime_args);
        desc.kernels[dm1_kernel_id].emplace_runtime_args(core, runtime_args);
        desc.kernels[compute_kernel_id].emplace_runtime_args(core, runtime_args);
    }

    // =========================================================================
    // Set Runtime Args for combine cores (fused mode only)
    // =========================================================================
    {
        for (const auto& core : combine_cores) {
            desc.kernels[combine_dm1_kernel_id].emplace_runtime_args(core, {combine_semaphore_id});
        }
    }

    // =========================================================================
    // Tilize cores
    // =========================================================================
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
        // all_to_all_dispatch_metadata -- "dispatch" names that op, not a fast-dispatch core.
        // The shard is an ordinary L1 allocation, so it always sits on a compute core, but its
        // coordinate is whichever core that op was given and need not be one of the tilize
        // cores picked below.  So both tilize cores do a single bulk NOC read from it at kernel
        // start rather than aliasing a CB onto it.
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

        auto tilize_partial_metadata_ready_semaphore_id = add_worker_semaphore(desc, tilize_core_range_set, INVALID);
        auto tilize_chunk_ready_semaphore_id = add_worker_semaphore(desc, tilize_core_range_set, INVALID);
        auto previous_chunk_sent_semaphore_id = add_worker_semaphore(desc, tilize_core_range_set, INVALID);
        auto initial_gather_semaphore_id = add_worker_semaphore(desc, tilize_core_range_set, INVALID);

        // Tilize+Matmul shared semaphores (created early before matmul kernel creation)
        const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(all_cores);
        uint32_t metadata_ready_semaphore_id = fused_metadata_ready_sem;
        uint32_t matmul_chunk_available_semaphore_id = fused_chunk_available_sem;
        uint32_t matmul_chunk_ready_semaphore_id = fused_chunk_ready_sem;

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

        // tiles_per_global_chunk: drain core gathers from all tilize cores before multicast
        uint32_t shared_cb_num_pages = hidden_size / TILE_WIDTH;

        add_cb(
            tilize_core_range_set,
            per_expert_total_tokens_cb_id,
            tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32),
            static_cast<uint32_t>(tilize_per_expert_total_tokens_output_page_size),
            1);

        add_cb(
            tilize_core_range_set,
            total_chunks_cb_id,
            tt::DataFormat::UInt32,
            static_cast<uint32_t>(sizeof(uint32_t)),
            1);

        add_cb(
            tilize_core_range_set,
            e_t_cb_id,
            tt::DataFormat::UInt32,
            static_cast<uint32_t>(tilize_e_t_output_page_size),
            experts_per_device);

        // Indices CB: fresh on both tilize cores.  Both cores do a bulk NOC read from the
        // dispatch drain core at kernel start.  Use aligned_page_size so the CB covers the
        // full physical buffer allocation (num_pages * aligned_page_size bytes).
        add_cb(
            tilize_core_range_set,
            indices_tensor_cb_id,
            tilize_indices_data_format,
            tilize_indices_aligned_page_size,
            tilize_indices_pages);

        // Scores CB: same pattern as indices.
        add_cb(
            tilize_core_range_set,
            scores_tensor_cb_id,
            tilize_scores_data_format,
            tilize_scores_aligned_page_size,
            tilize_scores_pages);

        add_cb(
            tilize_core_range_set,
            mapping_tensor_cb_id,
            tilize_mapping_data_format,
            tilize_mapping_aligned_page_size,
            tilize_mapping_pages);

        add_cb(
            tilize_core_range_set,
            tilize_input_cb_id,
            tilize_input_data_format,
            max_tilize_subtoken_size,
            tokens_per_chunk);

        add_cb(
            tilize_core_range_set,
            expert_activation_cb_id,
            tt::DataFormat::UInt32,
            static_cast<uint32_t>(tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment)),
            tokens + 1);

        add_cb(
            tilize_core_range_set,
            brisc_e_t_cb_id,
            tt::DataFormat::UInt32,
            (tokens / 2) * l1_alignment * experts_per_device,
            1);

        add_cb(
            tilize_core_range_set,
            brisc_expert_counts_cb_id,
            tt::DataFormat::UInt32,
            static_cast<uint32_t>(sizeof(uint32_t) * experts_per_device),
            1);

        uint32_t brisc_activation_row_size =
            static_cast<uint32_t>(tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment));
        add_cb(
            tilize_core_range_set,
            brisc_expert_activation_cb_id,
            tt::DataFormat::UInt32,
            brisc_activation_row_size * (tokens / 2),
            1);

        add_cb(
            tilize_core_range_set,
            brisc_activated_count_cb_id,
            tt::DataFormat::UInt32,
            static_cast<uint32_t>(sizeof(uint32_t)),
            1);

        uint32_t counts_per_remote_core = experts_per_device + 1;
        uint32_t remote_counts_entry_size =
            static_cast<uint32_t>(tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment));
        add_cb(
            tilize_core_range_set,
            remote_counts_cb_id,
            tt::DataFormat::UInt32,
            remote_counts_entry_size,
            TILIZE_NUM_CORES - 1);

        // Tilize output CB
        uint32_t tilize_output_tile_size = tt::tile_size(tilize_output_data_format);
        uint32_t tiles_per_global_chunk = hidden_size / TILE_WIDTH;  // 90 for GPT-OSS

        // c_16 is the shared tilize→matmul activation CB.
        // Created on the merged tilize+matmul range so multicast addresses match.
        // Double-buffered: 2 * tiles_per_global_chunk = 2 * 90 = 180 tiles.
        {
            uint32_t fused_cb_num_pages = 2 * tiles_per_global_chunk;
            add_cb(
                tilize_matmul_core_range_set,
                tilize_output_cb_id,
                tilize_output_data_format,
                tilize_output_tile_size,
                fused_cb_num_pages);
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
        std::vector<std::pair<std::string, uint32_t>> tilize_named_compile_time_args = {
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
            {"per_expert_total_tokens_output_page_size",
             static_cast<uint32_t>(tilize_per_expert_total_tokens_output_page_size)},
            {"expert_activation_output_page_size", static_cast<uint32_t>(tilize_expert_activation_output_page_size)},
            {"e_t_output_page_size", static_cast<uint32_t>(tilize_e_t_output_page_size)},
            {"tilize_output_page_size", tilize_output_tile_size},

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
            {"metadata_count_semaphore_base_id", metadata_count_sem_base},
            {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
            {"tilize_chunk_ready_semaphore_id", tilize_chunk_ready_semaphore_id},
            {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
            {"previous_chunk_sent_semaphore_id", previous_chunk_sent_semaphore_id},
            {"initial_gather_semaphore_id", initial_gather_semaphore_id},

            {"max_tiles_per_local_chunk", max_tiles_per_local_chunk},
            {"tiles_per_global_chunk", tiles_per_global_chunk},
        };

        // --- Create tilize kernels ---
        desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
            .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_reader.cpp",
            .core_ranges = tilize_core_range_set,
            .compile_time_args = tilize_compile_time_args,
            .named_compile_time_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2,
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt::tt_metal::NOC::NOC_1,
                    .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                },
        });
        const size_t tilize_reader_kernel_id = 4;

        desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
            .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_writer.cpp",
            .core_ranges = tilize_core_range_set,
            .compile_time_args = tilize_compile_time_args,
            .named_compile_time_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2,
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::NOC_1,
                    .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                },
        });
        const size_t tilize_writer_kernel_id = 5;

        std::vector<std::pair<std::string, uint32_t>> compute_tilize_named_compile_time_args = {
            {"tilize_input_cb_id", tilize_input_cb_id},
            {"tilize_output_cb_id", tilize_output_cb_id},
            {"total_chunks_cb_id", total_chunks_cb_id},
            {"tokens_per_chunk", tokens_per_chunk},
            {"max_tiles_per_local_chunk", max_tiles_per_local_chunk},
            {"shared_cb_num_pages", shared_cb_num_pages},
        };

        desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
            .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_compute.cpp",
            .core_ranges = tilize_core_range_set,
            .named_compile_time_args = std::move(compute_tilize_named_compile_time_args),
            .config = tt::tt_metal::ComputeConfigDescriptor{},
        });
        const size_t tilize_compute_kernel_id = 6;

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

        auto* per_expert_total_tokens_buffer = tensor_return_value[0].buffer();
        auto* expert_activation_buffer = tensor_return_value[1].buffer();
        auto* e_t_buffer = tensor_return_value[2].buffer();

        global_subtoken_offset = 0;
        uint32_t group_subtoken_offset = 0;
        for (uint32_t i = 0; i < TILIZE_NUM_CORES; i++) {
            const uint32_t is_drain_tilize_core = (i == 0) ? 1u : 0u;
            const uint32_t is_secondary_mcaster = (i == primary_mcast_gather_group_num_cores) ? 1u : 0u;

            CoreCoord initial_mcast_gather_core_physical =
                i < primary_mcast_gather_group_num_cores
                    ? tilize_cores_physical.at(0)
                    : tilize_cores_physical.at(primary_mcast_gather_group_num_cores);

            const uint32_t mcast_group_subtoken_size = i < primary_mcast_gather_group_num_cores
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

            const uint32_t this_global_subtoken_offset = global_subtoken_offset;
            const uint32_t this_group_subtoken_offset = group_subtoken_offset;
            global_subtoken_offset += subtoken_size;
            group_subtoken_offset += subtoken_size;
            if (i == primary_mcast_gather_group_num_cores - 1) {
                group_subtoken_offset = 0;
            }

            uint32_t core_token_start = i * tokens_per_tilize_core;
            uint32_t core_token_end = (i == TILIZE_NUM_CORES - 1) ? tokens : (i + 1) * tokens_per_tilize_core;

            tt::tt_metal::KernelDescriptor::RTArgList tilize_runtime_args;
            tilize_runtime_args.reserve(18 + (2 * TILIZE_NUM_CORES) + 1);
            tilize_runtime_args.push_back(sparse_buffer.buffer());          // 0 input
            tilize_runtime_args.push_back(expert_indices.buffer());         // 1
            tilize_runtime_args.push_back(expert_scores.buffer());          // 2
            tilize_runtime_args.push_back(expert_mapping.buffer());         // 3
            tilize_runtime_args.push_back(per_expert_total_tokens_buffer);  // 4 per_expert_total_tokens
            tilize_runtime_args.push_back(expert_activation_buffer);        // 5 expert_activation
            tilize_runtime_args.push_back(e_t_buffer);                      // 6 e_t
            tilize_runtime_args.push_back(is_drain_tilize_core);            // 7
            tilize_runtime_args.push_back(is_secondary_mcaster);            // 8
            tilize_runtime_args.push_back(static_cast<uint32_t>(initial_mcast_gather_core_physical.x));  // 9
            tilize_runtime_args.push_back(static_cast<uint32_t>(initial_mcast_gather_core_physical.y));  // 10
            tilize_runtime_args.push_back(this_global_subtoken_offset);                                  // 11
            tilize_runtime_args.push_back(this_group_subtoken_offset);                                   // 12
            tilize_runtime_args.push_back(mcast_group_subtoken_size);                                    // 13
            tilize_runtime_args.push_back(subtoken_size);                                                // 14
            tilize_runtime_args.push_back(core_token_start);                                             // 15
            tilize_runtime_args.push_back(core_token_end);                                               // 16
            tilize_runtime_args.push_back(i);  // 17 tilize core idx

            // NOC coordinates for all tilize cores
            for (uint32_t t = 0; t < TILIZE_NUM_CORES; t++) {
                tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(t).x);
                tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(t).y);
            }

            // Placeholder for DRAM output address (unused in fused mode)
            tilize_runtime_args.push_back(0u);

            desc.kernels[tilize_reader_kernel_id].emplace_runtime_args(tilize_cores.at(i), tilize_runtime_args);
            desc.kernels[tilize_writer_kernel_id].emplace_runtime_args(tilize_cores.at(i), tilize_runtime_args);
            desc.kernels[tilize_compute_kernel_id].emplace_runtime_args(
                tilize_cores.at(i), {subtoken_size / tile_width_bytes});
        }
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoEGPTMeshWorkloadFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    for (const auto& coord : tensor_coords.coords()) {
        workload_descriptor.programs.push_back(tt::tt_metal::WorkloadDescriptor::PerCoordProgram{
            .range = ttnn::MeshCoordinateRange(coord),
            .descriptor = create_program_descriptor_at(operation_attributes, coord, tensor_args, tensor_return_value),
        });
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::moe_gpt::program
