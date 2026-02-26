// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_fused_program_factory.hpp"
#include "moe_gpt_fused_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace {

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

namespace ttnn::operations::experimental::moe_gpt_fused::program {

//=============================================================================
// Program Factory
//=============================================================================

MoEGPTFusedProgramFactory::cached_program_t MoEGPTFusedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto device = tensor_args.input_tensor.device();

    // Matmul cores: 12 cores aligned to DRAM banks (same as moe_gpt)
    const auto matmul_cores =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    const uint32_t num_cores = matmul_cores.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(matmul_cores);

    const uint32_t experts_per_device = operation_attributes.experts_per_device;
    const uint32_t layer_id = operation_attributes.layer_id;

    //=========================================================================
    // Combine cores: 12 cores in 3x4 grid at CoreRange({1,0},{3,3})
    // x:[1,2,3] = 3 columns (width shards), y:[0,1,2,3] = 4 rows (height shards)
    // ROW_MAJOR orientation: rows = height, cols = width → 4 height × 3 width
    // Verified no overlap with matmul DRAM-aligned cores
    //=========================================================================
    const CoreRange combine_core_range({1, 0}, {3, 3});
    auto combine_cores_unsorted = corerange_to_cores(CoreRangeSet(combine_core_range));
    // Sort by (x, y) to get logical ordering: height shards vary fastest
    std::sort(combine_cores_unsorted.begin(), combine_cores_unsorted.end(), [](const auto& a, const auto& b) {
        return (a.y != b.y) ? a.y < b.y : a.x < b.x;
    });
    auto combine_cores = combine_cores_unsorted;
    CoreRangeSet combine_core_range_set(combine_core_range);

    //=========================================================================
    // Circular Buffers on matmul cores
    //=========================================================================
    // c_0: Triple-buffered weights Bfp4_b (10*2*3 = 60 tiles)
    // c_1: Input Float16_b (90 tiles - input only, no output)
    // c_2: Compute -> DM1 signal (1 datum)
    // c_3: DM1 -> Compute signal (1 datum)
    // c_4: Ring A2A intermediate Float16_b (8*6 = 48 tiles)
    // c_14: Untilized output Float16_b (ROW_MAJOR, 32 pages × 512 bytes)
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles;

    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, true, 10 * 2 * 3},
        {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 90},  // input only
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 8 * 6},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // c_14: Untilized ROW_MAJOR output buffer on matmul cores
    // Page size: SOURCE_WIDTH_TILES * 32 * 2 = 8 * 32 * 2 = 512 bytes (one row of untilized data)
    // Num pages: 32 * E (must hold all experts since dm1 drains after ring A2A completes)
    constexpr uint32_t SOURCE_WIDTH_TILES = 8;
    constexpr uint32_t c14_page_size = SOURCE_WIDTH_TILES * 32 * 2;  // 512 bytes
    const uint32_t c14_num_pages = 32 * experts_per_device;          // 128 pages = 64 KB
    {
        const auto cb_config = tt::tt_metal::CircularBufferConfig(
                                   c14_num_pages * c14_page_size, {{tt::CBIndex::c_14, tt::DataFormat::Float16_b}})
                                   .set_page_size(tt::CBIndex::c_14, c14_page_size);
        cb_handles["cb_c2s_out"] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    //=========================================================================
    // Combine core CB: c_0 backed by output tensor buffer
    //=========================================================================
    auto& output_tensor = tensor_return_value.at(0);

    // Each combine core shard: E * tokens_per_height_shard * combine_shard_width * 2 bytes
    // = 4 * 8 * 30 * 32 * 2 = 61440 bytes = 60 KB
    constexpr uint32_t combine_shard_width_tiles = 30;  // 90/3
    constexpr uint32_t tokens_per_height_shard = 8;     // 32/4
    constexpr uint32_t E = 4;
    constexpr uint32_t output_page_size = combine_shard_width_tiles * 32 * 2;  // 1920 bytes per row
    constexpr uint32_t output_num_pages = E * tokens_per_height_shard;         // 32 pages per shard

    auto [combine_cb_id, sharded_output_cb_handle] = tt::tt_metal::create_cb(
        (uint32_t)tt::CBIndex::c_0,
        program,
        combine_core_range_set,
        output_page_size,
        output_num_pages,
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()),
        output_tensor.buffer());

    cb_handles["cb_combine_out"] = sharded_output_cb_handle;

    //=========================================================================
    // Named compile args
    //=========================================================================
    std::unordered_map<std::string, uint32_t> named_compile_args = {
        {"num_experts", experts_per_device},
        {"layer_id", layer_id},
        {"num_cores", num_cores},
        {"enable_dram_output", 0u},  // no longer writing to DRAM
        // Combine output args for dm1
        {"height_shard_dim", 4u},
        {"width_shard_dim", 3u},
        {"combine_shard_width_tiles", combine_shard_width_tiles},
        {"tile_width", 32u},
        {"tile_width_size_bytes", 64u},  // 32 * 2 bytes bfloat16
    };

    //=========================================================================
    // Create Kernels
    //=========================================================================
    // OUTPUT_SHARD_CORE_MAP define for dm1 (serialized physical coords of combine cores)
    std::map<std::string, std::string> dm1_defines = {
        {"OUTPUT_SHARD_CORE_MAP", serialize_physical_core_coords(combine_cores, device)}};

    auto dm0_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_gpt_fused/device/kernels/matmul_dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .named_compile_args = named_compile_args});

    auto dm1_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_gpt_fused/device/kernels/matmul_dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .defines = dm1_defines,
            .named_compile_args = named_compile_args});

    auto compute_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_gpt_fused/device/kernels/matmul_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .named_compile_args = named_compile_args});

    // Combine dm1 kernel
    auto combine_dm1_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_gpt_fused/device/kernels/combine_dm1.cpp",
        combine_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1});

    //=========================================================================
    // Semaphores
    //=========================================================================
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const uint32_t combine_semaphore_id = tt::tt_metal::CreateSemaphore(program, combine_core_range_set, 0);

    //=========================================================================
    // Ring ordering (same as moe_gpt)
    //=========================================================================
    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &matmul_cores](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(matmul_cores[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(matmul_cores[bank_id_b]);
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

    // Precompute k_start_tile for each ring position
    constexpr uint32_t tiles_per_core_table[12] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};
    uint32_t k_start_tiles[12] = {0};
    for (uint32_t i = 1; i < num_cores; ++i) {
        k_start_tiles[i] = k_start_tiles[i - 1] + tiles_per_core_table[i - 1];
    }

    //=========================================================================
    // Compute output_base_l1_addr for combine cores
    // This is the L1 address of c_0 on combine cores (backed by output tensor).
    // We need to pass it to dm1 so it knows where to write.
    // Use the output tensor's buffer address since c_0 is globally allocated on it.
    //=========================================================================
    const uint32_t output_base_l1_addr = output_tensor.buffer()->address();

    //=========================================================================
    // Set Runtime Args for matmul cores
    //=========================================================================
    // Layout: [0] dram_bank_id, [1] vchannel, [2] w0_w1_addr, [3] w2_addr,
    //         [4] ring_semaphore_id, [5] ring_core_id,
    //         [6] neighbor_x, [7] neighbor_y,
    //         [8] input_dram_addr, [9] combine_semaphore_id, [10] k_start_tile,
    //         [11] output_base_l1_addr

    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (const auto& core : matmul_cores) {
        uint32_t vchannel = dram_bank & 0x3;

        // Avoid same VChannel for cores in the same row
        auto it = std::find_if(matmul_cores.begin(), matmul_cores.begin() + dram_bank, [&](const auto& core_prev) {
            return core_prev.y == core.y;
        });
        if (it != matmul_cores.begin() + dram_bank) {
            size_t j = std::distance(matmul_cores.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        const auto [ring_pos, next_bank] = bank2ring_pos[dram_bank];
        const auto& next_physical = device->worker_core_from_logical_core(matmul_cores[next_bank]);

        std::vector<uint32_t> rt_args;
        rt_args.push_back(dram_bank);                                     // [0] dram_bank_id
        rt_args.push_back(vchannel);                                      // [1] vchannel
        rt_args.push_back(tensor_args.w0_w1_tensor.buffer()->address());  // [2] w0_w1_addr
        rt_args.push_back(tensor_args.w2_tensor.buffer()->address());     // [3] w2_addr
        rt_args.push_back(ring_semaphore_id);                             // [4] ring_semaphore_id
        rt_args.push_back(ring_pos);                                      // [5] ring_core_id
        rt_args.push_back(static_cast<uint32_t>(next_physical.x));        // [6] neighbor_x
        rt_args.push_back(static_cast<uint32_t>(next_physical.y));        // [7] neighbor_y
        rt_args.push_back(tensor_args.input_tensor.buffer()->address());  // [8] input_dram_addr
        rt_args.push_back(combine_semaphore_id);                          // [9] combine_semaphore_id
        rt_args.push_back(k_start_tiles[ring_pos]);                       // [10] k_start_tile
        rt_args.push_back(output_base_l1_addr);                           // [11] output_base_l1_addr

        tt::tt_metal::SetRuntimeArgs(program, dm0_handle, core, rt_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_handle, core, rt_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_handle, core, rt_args);

        dram_bank++;
    }

    //=========================================================================
    // Set Runtime Args for combine cores
    //=========================================================================
    const std::vector<uint32_t> combine_rt_args = {combine_semaphore_id};
    for (const auto& core : combine_cores) {
        tt::tt_metal::SetRuntimeArgs(program, combine_dm1_handle, core, combine_rt_args);
    }

    //=========================================================================
    // Return
    //=========================================================================
    return cached_program_t{
        std::move(program),
        MoEGPTFusedSharedVariables{
            .cb_handles_sharded = {{"cb_combine_out", sharded_output_cb_handle}},
            .kernel_handles = {dm0_handle, dm1_handle, compute_handle, combine_dm1_handle},
            .matmul_cores = matmul_cores,
            .gather_cores = {},
            .combine_cores = combine_cores,
            .l1_input_buffer = nullptr}};
}

void MoEGPTFusedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /* operation_attributes */,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    // Update weight tensor addresses on matmul cores
    for (const auto& core : shared.matmul_cores) {
        // dm0, dm1, compute handles are indices 0, 1, 2
        for (uint32_t i = 0; i < 3; ++i) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared.kernel_handles[i], core);
            runtime_args[2] = tensor_args.w0_w1_tensor.buffer()->address();
            runtime_args[3] = tensor_args.w2_tensor.buffer()->address();
        }
    }

    // Update output tensor buffer address for sharded CB
    auto& output_tensor = tensor_return_value.at(0);
    if (shared.cb_handles_sharded.count("cb_combine_out")) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared.cb_handles_sharded.at("cb_combine_out"), *output_tensor.buffer());
    }

    // Update output_base_l1_addr in dm1 runtime args
    const uint32_t output_base_l1_addr = output_tensor.buffer()->address();
    for (const auto& core : shared.matmul_cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared.kernel_handles[1], core);
        runtime_args[11] = output_base_l1_addr;
    }
}

}  // namespace ttnn::operations::experimental::moe_gpt_fused::program
