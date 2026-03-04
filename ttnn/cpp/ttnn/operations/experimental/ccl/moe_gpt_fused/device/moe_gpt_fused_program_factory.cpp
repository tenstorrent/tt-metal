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
    auto matmul_core_range_set = tt::tt_metal::CoreRangeSet(matmul_cores);

    const uint32_t experts_per_device = operation_attributes.experts_per_device;
    const uint32_t layer_id = operation_attributes.layer_id;

    //=========================================================================
    // Tilize (gather) cores: 3 cores at CoreRange({5,0},{5,2})
    // Each handles 30 tiles of the hidden dimension (90/3 = 30)
    // Core 0 is drain core (gathers from others, signals matmul)
    //=========================================================================
    constexpr uint32_t NUM_GATHER_CORES = 3;
    constexpr uint32_t TILES_PER_GATHER_CORE = 30;  // 90 / 3
    constexpr uint32_t K_TILES = 90;
    constexpr uint32_t TOKENS_PER_CHUNK = 32;

    const CoreRange gather_core_range({5, 0}, {5, 2});
    auto gather_cores = corerange_to_cores(CoreRangeSet(gather_core_range));
    // Sort by y to get consistent ordering: (5,0), (5,1), (5,2)
    std::sort(gather_cores.begin(), gather_cores.end(), [](const auto& a, const auto& b) { return a.y < b.y; });
    CoreRangeSet gather_core_range_set(gather_core_range);

    // Drain core is gather_cores[0] = (5,0)
    const auto drain_core_physical = device->worker_core_from_logical_core(gather_cores[0]);

    //=========================================================================
    // Combine cores: 12 cores in 3x4 grid at CoreRange({1,0},{3,3})
    //=========================================================================
    const CoreRange combine_core_range({1, 0}, {3, 3});
    auto combine_cores_unsorted = corerange_to_cores(CoreRangeSet(combine_core_range));
    std::sort(combine_cores_unsorted.begin(), combine_cores_unsorted.end(), [](const auto& a, const auto& b) {
        return (a.y != b.y) ? a.y < b.y : a.x < b.x;
    });
    auto combine_cores = combine_cores_unsorted;
    CoreRangeSet combine_core_range_set(combine_core_range);

    //=========================================================================
    // Circular Buffers on matmul cores
    //=========================================================================
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles;

    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, true, 10 * 2 * 3},
        {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 90},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 8 * 6},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, matmul_core_range_set, cb_config);
    }

    // c_14: Untilized ROW_MAJOR output buffer on matmul cores
    constexpr uint32_t SOURCE_WIDTH_TILES = 8;
    constexpr uint32_t c14_page_size = SOURCE_WIDTH_TILES * 32 * 2;  // 512 bytes
    const uint32_t c14_num_pages = 32 * experts_per_device;          // 128 pages = 64 KB
    {
        const auto cb_config = tt::tt_metal::CircularBufferConfig(
                                   c14_num_pages * c14_page_size, {{tt::CBIndex::c_14, tt::DataFormat::Float16_b}})
                                   .set_page_size(tt::CBIndex::c_14, c14_page_size);
        cb_handles["cb_c2s_out"] = tt::tt_metal::CreateCircularBuffer(program, matmul_core_range_set, cb_config);
    }

    //=========================================================================
    // Tilize core CBs
    //=========================================================================
    // c_7: Tilize input (ROW_MAJOR), backed by input tensor shard
    // Each core's shard: [32, 960] = 32 pages × 1920 bytes = 60 KB
    constexpr uint32_t tilize_input_page_size = TILES_PER_GATHER_CORE * 32 * 2;  // 30 * 64 = 1920 bytes
    {
        auto [tilize_input_cb_id, tilize_input_cb_handle] = tt::tt_metal::create_cb(
            (uint32_t)tt::CBIndex::c_7,
            program,
            gather_core_range_set,
            tilize_input_page_size,
            TOKENS_PER_CHUNK,  // 32 pages
            tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype()),
            tensor_args.input_tensor.buffer());
        cb_handles["cb_tilize_input"] = tilize_input_cb_handle;
    }

    // c_16: Tilize output (TILE Float16_b), 90 tiles
    // Drain core uses all 90 tiles (30 local + 60 gathered)
    // Non-drain cores only use 30 but get 90 allocated (uniform CB config)
    constexpr uint32_t tilize_output_tile_size = 2048;  // Float16_b tile
    {
        const auto cb_config = tt::tt_metal::CircularBufferConfig(
                                   K_TILES * tilize_output_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                                   .set_page_size(tt::CBIndex::c_16, tilize_output_tile_size);
        cb_handles["cb_tilize_output"] = tt::tt_metal::CreateCircularBuffer(program, gather_core_range_set, cb_config);
    }

    //=========================================================================
    // Combine core CB: c_0 backed by output tensor buffer
    //=========================================================================
    auto& output_tensor = tensor_return_value.at(0);

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
        {"enable_dram_output", 0u},
        {"height_shard_dim", 4u},
        {"width_shard_dim", 3u},
        {"combine_shard_width_tiles", combine_shard_width_tiles},
        {"tile_width", 32u},
        {"tile_width_size_bytes", 64u},
        // TODO(T=128): Change to actual batch size when chunk loop is added
        {"num_tokens_total", TOKENS_PER_CHUNK},
    };

    //=========================================================================
    // Create Kernels
    //=========================================================================
    // OUTPUT_SHARD_CORE_MAP define for dm1
    std::map<std::string, std::string> dm1_defines = {
        {"OUTPUT_SHARD_CORE_MAP", serialize_physical_core_coords(combine_cores, device)}};

    auto dm0_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/matmul_dm0.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .named_compile_args = named_compile_args});

    auto dm1_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/matmul_dm1.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .defines = dm1_defines,
            .named_compile_args = named_compile_args});

    auto compute_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/matmul_compute.cpp",
        matmul_core_range_set,
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
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/combine_dm1.cpp",
        combine_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1});

    // Tilize kernels
    auto gather_reader_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/gather_reader.cpp",
        gather_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_1});

    auto gather_compute_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/gather_compute.cpp",
        gather_core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true});

    auto gather_writer_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels/gather_writer.cpp",
        gather_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0});

    //=========================================================================
    // Semaphores
    //=========================================================================
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, matmul_core_range_set, 0);
    const uint32_t combine_semaphore_id = tt::tt_metal::CreateSemaphore(program, combine_core_range_set, 0);

    // Tilize semaphores
    // gather_semaphore: drain core waits for non-drain cores to send tiles
    const uint32_t gather_semaphore_id = tt::tt_metal::CreateSemaphore(program, gather_core_range_set, 0);
    // tilize_ready_semaphore: tilize writer signals matmul dm0 that input is ready
    // Need this on both gather and matmul cores (gather writer writes, matmul dm0 reads)
    const uint32_t tilize_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, matmul_core_range_set, 0);

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

    constexpr uint32_t tiles_per_core_table[12] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};
    uint32_t k_start_tiles[12] = {0};
    for (uint32_t i = 1; i < num_cores; ++i) {
        k_start_tiles[i] = k_start_tiles[i - 1] + tiles_per_core_table[i - 1];
    }

    //=========================================================================
    // Compute output_base_l1_addr for combine cores
    //=========================================================================
    const uint32_t output_base_l1_addr = output_tensor.buffer()->address();

    //=========================================================================
    // Address exchange semaphore: matmul dm0 (ring_core_id 0) writes its c_1
    // base address to this semaphore on the drain core. The drain gather_writer
    // reads it to know where to push tilized input on all matmul cores.
    //=========================================================================
    const uint32_t addr_exchange_semaphore_id = tt::tt_metal::CreateSemaphore(program, gather_core_range_set, 0);

    //=========================================================================
    // Get physical coords for matmul cores (for gather_writer)
    //=========================================================================
    std::vector<CoreCoord> matmul_cores_physical;
    matmul_cores_physical.reserve(num_cores);
    for (const auto& c : matmul_cores) {
        matmul_cores_physical.push_back(device->worker_core_from_logical_core(c));
    }

    //=========================================================================
    // Set Runtime Args for matmul cores
    //=========================================================================
    // Layout: [0] dram_bank_id, [1] vchannel, [2] w0_w1_addr, [3] w2_addr,
    //         [4] ring_semaphore_id, [5] ring_core_id,
    //         [6] neighbor_x, [7] neighbor_y,
    //         [8] input_dram_addr, [9] combine_semaphore_id, [10] k_start_tile,
    //         [11] output_base_l1_addr,
    //         [12] tilize_ready_semaphore_id, [13] drain_noc_x, [14] drain_noc_y,
    //         [15] drain_tilize_output_l1_addr (0 = placeholder, exchanged at runtime)

    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (const auto& core : matmul_cores) {
        uint32_t vchannel = dram_bank & 0x3;

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
        rt_args.push_back(tensor_args.input_tensor.buffer()->address());  // [8] input_dram_addr (unused)
        rt_args.push_back(combine_semaphore_id);                          // [9] combine_semaphore_id
        rt_args.push_back(k_start_tiles[ring_pos]);                       // [10] k_start_tile
        rt_args.push_back(output_base_l1_addr);                           // [11] output_base_l1_addr
        rt_args.push_back(tilize_ready_semaphore_id);                     // [12] tilize_ready_semaphore_id
        rt_args.push_back(static_cast<uint32_t>(drain_core_physical.x));  // [13] drain_noc_x
        rt_args.push_back(static_cast<uint32_t>(drain_core_physical.y));  // [14] drain_noc_y
        rt_args.push_back(0u);                          // [15] drain_tilize_output_l1_addr (placeholder)
        rt_args.push_back(addr_exchange_semaphore_id);  // [16] addr_exchange_semaphore_id

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
    // Set Runtime Args for tilize (gather) cores
    //=========================================================================
    // gather_writer args: [0] gather_core_id, [1] gather_semaphore_id,
    //                     [2] tilize_ready_semaphore_id, [3] drain_noc_x,
    //                     [4] drain_noc_y, [5] num_matmul_cores,
    //                     [6..] matmul_noc_x[i], matmul_noc_y[i], ...
    //                     [6+2*N] addr_exchange_semaphore_id
    for (uint32_t i = 0; i < NUM_GATHER_CORES; ++i) {
        std::vector<uint32_t> gather_rt_args;
        gather_rt_args.push_back(i);                                             // gather_core_id
        gather_rt_args.push_back(gather_semaphore_id);                           // gather_semaphore_id
        gather_rt_args.push_back(tilize_ready_semaphore_id);                     // tilize_ready_semaphore_id
        gather_rt_args.push_back(static_cast<uint32_t>(drain_core_physical.x));  // drain_noc_x
        gather_rt_args.push_back(static_cast<uint32_t>(drain_core_physical.y));  // drain_noc_y
        gather_rt_args.push_back(num_cores);                                     // num_matmul_cores
        for (uint32_t j = 0; j < num_cores; ++j) {
            gather_rt_args.push_back(static_cast<uint32_t>(matmul_cores_physical[j].x));
            gather_rt_args.push_back(static_cast<uint32_t>(matmul_cores_physical[j].y));
        }
        gather_rt_args.push_back(addr_exchange_semaphore_id);  // addr_exchange_semaphore_id

        tt::tt_metal::SetRuntimeArgs(program, gather_writer_handle, gather_cores[i], gather_rt_args);
    }

    //=========================================================================
    // Return
    //=========================================================================
    return cached_program_t{
        std::move(program),
        MoEGPTFusedSharedVariables{
            .cb_handles_sharded =
                {{"cb_combine_out", sharded_output_cb_handle}, {"cb_tilize_input", cb_handles["cb_tilize_input"]}},
            .kernel_handles =
                {dm0_handle,
                 dm1_handle,
                 compute_handle,
                 combine_dm1_handle,
                 gather_reader_handle,
                 gather_compute_handle,
                 gather_writer_handle},
            .matmul_cores = matmul_cores,
            .gather_cores = gather_cores,
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

    // Update input tensor buffer address for tilize input sharded CB
    if (shared.cb_handles_sharded.count("cb_tilize_input")) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared.cb_handles_sharded.at("cb_tilize_input"), *tensor_args.input_tensor.buffer());
    }

    // Update output_base_l1_addr in dm1 runtime args
    const uint32_t output_base_l1_addr = output_tensor.buffer()->address();
    for (const auto& core : shared.matmul_cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared.kernel_handles[1], core);
        runtime_args[11] = output_base_l1_addr;
    }
}

}  // namespace ttnn::operations::experimental::moe_gpt_fused::program
