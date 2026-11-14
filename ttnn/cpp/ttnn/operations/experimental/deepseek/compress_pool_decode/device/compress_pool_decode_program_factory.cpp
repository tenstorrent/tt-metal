// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compress_pool_decode_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "compress_pool_decode_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::compress_pool_decode {

namespace {

constexpr const char* kCpdReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/compress_pool_decode/device/kernels/dataflow/"
    "reader_compress_pool_decode.cpp";
constexpr const char* kCpdWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/compress_pool_decode/device/kernels/dataflow/"
    "writer_compress_pool_decode.cpp";
constexpr const char* kCpdComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/compress_pool_decode/device/kernels/compute/"
    "compress_pool_decode.cpp";

constexpr uint32_t kCpdCbKv = tt::CBIndex::c_0;
constexpr uint32_t kCpdCbK0 = tt::CBIndex::c_1;
constexpr uint32_t kCpdCbGate = tt::CBIndex::c_2;
constexpr uint32_t kCpdCbG0 = tt::CBIndex::c_3;
constexpr uint32_t kCpdCbGamma = tt::CBIndex::c_4;
constexpr uint32_t kCpdCbCs = tt::CBIndex::c_5;
constexpr uint32_t kCpdCbTmp = tt::CBIndex::c_6;
constexpr uint32_t kCpdCbPooled = tt::CBIndex::c_7;
constexpr uint32_t kCpdCbLocalStat = tt::CBIndex::c_8;
constexpr uint32_t kCpdCbGatherStats = tt::CBIndex::c_9;
constexpr uint32_t kCpdCbScaleHub = tt::CBIndex::c_10;
constexpr uint32_t kCpdCbScale = tt::CBIndex::c_11;
constexpr uint32_t kCpdCbKeyIn = tt::CBIndex::c_12;
constexpr uint32_t kCpdCbKeyW = tt::CBIndex::c_13;
constexpr uint32_t kCpdCbKeyMm = tt::CBIndex::c_14;
constexpr uint32_t kCpdCbKeyGamma = tt::CBIndex::c_15;
constexpr uint32_t kCpdCbOutLatent = tt::CBIndex::c_16;
constexpr uint32_t kCpdCbOutRope = tt::CBIndex::c_17;

// Dataflow-only scratch circular buffers.
constexpr uint32_t kCpdCbWinKvTile = tt::CBIndex::c_18;
constexpr uint32_t kCpdCbWinGateTile = tt::CBIndex::c_19;
constexpr uint32_t kCpdCbKvScratch = tt::CBIndex::c_20;
constexpr uint32_t kCpdCbGateScratch = tt::CBIndex::c_21;
constexpr uint32_t kCpdCbPosReader = tt::CBIndex::c_22;
constexpr uint32_t kCpdCbPosWriter = tt::CBIndex::c_23;
constexpr uint32_t kCpdCbRawScratch = tt::CBIndex::c_24;
constexpr uint32_t kCpdCbOutTileScratch = tt::CBIndex::c_25;
constexpr uint32_t kCpdCbKeyOut = tt::CBIndex::c_26;
constexpr uint32_t kCpdCbKeyTile = tt::CBIndex::c_27;
constexpr uint32_t kCpdCbDenseTile = tt::CBIndex::c_28;
constexpr uint32_t kCpdCbIdx = tt::CBIndex::c_29;

constexpr uint32_t kCpdArriveSemaphoreId = 0;
constexpr uint32_t kCpdReadySemaphoreId = 1;
constexpr uint32_t kCpdKeyLatentSemaphoreId = 2;
constexpr uint32_t kCpdKeyArriveSemaphoreId = 3;
constexpr uint32_t kCpdKeyReadySemaphoreId = 4;

// index_wk K-tiles per streamed in1 page; two pages are in flight.
constexpr uint32_t kCpdKeyWChunkMax = 4;

}  // namespace

CompressPoolDecodeProgramFactory::cached_program_t CompressPoolDecodeProgramFactory::create(
    const CompressPoolDecodeParams& args, const CompressPoolDecodeInputs& t, std::vector<Tensor>& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto* device = t.kv.device();
    const uint32_t W_t = args.head_dim / constants::TILE_WIDTH;
    const uint32_t rope_tiles = args.has_rope ? args.rope_dim / constants::TILE_WIDTH : 0u;
    const bool do_pool = args.do_pool;
    const bool has_window = args.has_window;
    const bool has_prev_slot = args.has_prev_slot;
    const bool has_rope = args.has_rope;

    // Choose W_t cores: align directly with kv's (or gate's) L1 WIDTH_SHARDED grid when it has W_t cores.
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    std::vector<CoreCoord> kv_shard_cores;
    uint32_t kv_shard_w = 0;
    if (args.kv_sharded) {
        const auto& kv_shard = t.kv.memory_config().shard_spec().value();
        kv_shard_w = kv_shard.shape[1];
        kv_shard_cores =
            corerange_to_cores(kv_shard.grid, std::nullopt, kv_shard.orientation == ShardOrientation::ROW_MAJOR);
        if (kv_shard_cores.size() == W_t) {
            all_cores = kv_shard.grid;
            cores = kv_shard_cores;
        }
    }
    std::vector<CoreCoord> gate_shard_cores;
    uint32_t gate_shard_w = 0;
    if (args.gate_sharded) {
        const auto& gate_shard = t.gate->memory_config().shard_spec().value();
        gate_shard_w = gate_shard.shape[1];
        gate_shard_cores =
            corerange_to_cores(gate_shard.grid, std::nullopt, gate_shard.orientation == ShardOrientation::ROW_MAJOR);
        if (cores.empty() && gate_shard_cores.size() == W_t) {
            all_cores = gate_shard.grid;
            cores = gate_shard_cores;
        }
    }
    if (cores.empty()) {
        all_cores = num_cores_to_corerangeset(W_t, device->compute_with_storage_grid_size(), /*row_wise=*/true);
        cores = corerange_to_cores(all_cores, W_t, /*row_wise=*/true);
    }

    // The index key's Dk / 32 output tiles sit on the last cores, which hold the latent's RoPE tiles
    // too (rope_dim <= Dk), so the key's RoPE reuses their cos / sin.
    const bool has_index_key = args.has_index_key;
    const bool has_comp_kv = args.has_comp_kv;
    const bool has_dense = args.has_dense;
    const bool reads_entry = has_index_key || has_comp_kv;
    const uint32_t key_tiles = has_index_key ? args.index_head_dim / constants::TILE_WIDTH : 0u;
    const uint32_t first_key_core = W_t - key_tiles;
    uint32_t key_w_chunk = 1;
    for (uint32_t c = kCpdKeyWChunkMax; c > 1; --c) {
        if (W_t % c == 0) {
            key_w_chunk = c;
            break;
        }
    }
    const std::vector<CoreCoord> key_core_list(cores.begin() + first_key_core, cores.end());
    const CoreRangeSet key_cores = has_index_key ? CoreRangeSet(key_core_list) : CoreRangeSet();

    const tt::DataFormat bf16_df = tt::DataFormat::Float16_b;
    const tt::tt_metal::Tile row_tile({1, constants::TILE_WIDTH});
    const uint32_t row_tile_bytes = row_tile.get_tile_size(bf16_df);  // 64 B
    const uint32_t cache_tile_bytes = tt::tile_size(bf16_df);         // 2048 B
    const uint32_t elem_bytes = t.kv.element_size();                  // 2 B
    const uint32_t win_slot_page_bytes = has_window ? t.win_slot->buffer()->aligned_page_size() : 64u;
    const uint32_t prev_slot_page_bytes = has_prev_slot ? t.prev_slot->buffer()->aligned_page_size() : 64u;
    const uint32_t pos_slot_bytes = std::max<uint32_t>({win_slot_page_bytes, prev_slot_page_bytes, 64u});

    // Index scratch: entry, page_row, dense_entry, then the aligned piece of the page-table row.
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t entry_page_bytes = reads_entry ? t.entry->buffer()->aligned_page_size() : 0u;
    const uint32_t page_row_page_bytes = args.has_page_row ? t.page_row->buffer()->aligned_page_size() : 0u;
    const uint32_t dense_entry_page_bytes = has_dense ? t.dense_entry->buffer()->aligned_page_size() : 0u;
    const uint32_t page_table_chunk_bytes = dram_alignment;
    const uint32_t idx_slot_bytes = tt::round_up(
        std::max<uint32_t>(
            {entry_page_bytes, page_row_page_bytes, dense_entry_page_bytes, page_table_chunk_bytes, 64u}),
        dram_alignment);

    Program program = CreateProgram();

    auto make_row_cb = [&](const CoreRangeSet& on, uint32_t cb_id, uint32_t num_pages) {
        CreateCircularBuffer(
            program,
            on,
            CircularBufferConfig(num_pages * row_tile_bytes, {{cb_id, bf16_df}})
                .set_page_size(cb_id, row_tile_bytes)
                .set_tile_dims(cb_id, row_tile));
    };
    auto make_bytes_cb =
        [&](const CoreRangeSet& on, uint32_t cb_id, uint32_t page_bytes, uint32_t num_pages, tt::DataFormat df) {
            CreateCircularBuffer(
                program,
                on,
                CircularBufferConfig(num_pages * page_bytes, {{cb_id, df}}).set_page_size(cb_id, page_bytes));
        };

    if (do_pool) {
        make_row_cb(all_cores, kCpdCbKv, 1);
        make_row_cb(all_cores, kCpdCbGamma, 1);
        make_row_cb(all_cores, kCpdCbTmp, 1);
        make_row_cb(all_cores, kCpdCbLocalStat, 1);
        make_row_cb(all_cores, kCpdCbGatherStats, W_t);
        make_row_cb(all_cores, kCpdCbScaleHub, 1);
        make_row_cb(all_cores, kCpdCbScale, 1);
        make_row_cb(all_cores, kCpdCbOutLatent, 1);
        make_bytes_cb(all_cores, kCpdCbRawScratch, 64u, 6u, bf16_df);
        make_bytes_cb(all_cores, kCpdCbOutTileScratch, cache_tile_bytes, 1u, bf16_df);
        if (has_window) {
            make_row_cb(all_cores, kCpdCbK0, 1);
            make_row_cb(all_cores, kCpdCbGate, 1);
            make_row_cb(all_cores, kCpdCbG0, 1);
            make_row_cb(all_cores, kCpdCbPooled, 1);
        }
        if (has_rope) {
            // A key RoPE core rotates the latent's tile and then the key's with the same cos / sin pair.
            make_row_cb(all_cores, kCpdCbCs, has_index_key ? 4 : 2);
            make_row_cb(all_cores, kCpdCbOutRope, 1);
        }
    }
    if (has_window) {
        make_bytes_cb(all_cores, kCpdCbWinKvTile, cache_tile_bytes, 1u, bf16_df);
        make_bytes_cb(all_cores, kCpdCbWinGateTile, cache_tile_bytes, 1u, bf16_df);
        make_bytes_cb(all_cores, kCpdCbKvScratch, row_tile_bytes, 1u, bf16_df);
        make_bytes_cb(all_cores, kCpdCbGateScratch, row_tile_bytes, 1u, bf16_df);
        make_bytes_cb(all_cores, kCpdCbPosReader, pos_slot_bytes, 2u, tt::DataFormat::Int32);
        make_bytes_cb(all_cores, kCpdCbPosWriter, pos_slot_bytes, 2u, tt::DataFormat::Int32);
    }
    if (has_index_key) {
        // Every core writes its latent tile into key_in at the same address on each key core.
        make_row_cb(all_cores, kCpdCbKeyIn, W_t);
        make_bytes_cb(key_cores, kCpdCbKeyW, cache_tile_bytes, 2u * key_w_chunk, bf16_df);
        make_row_cb(key_cores, kCpdCbKeyMm, 1);
        make_row_cb(key_cores, kCpdCbKeyGamma, 1);
        make_row_cb(key_cores, kCpdCbKeyOut, 1);
        make_bytes_cb(key_cores, kCpdCbKeyTile, cache_tile_bytes, 1u, bf16_df);
    }
    if (has_dense) {
        make_bytes_cb(all_cores, kCpdCbDenseTile, cache_tile_bytes, 1u, bf16_df);
    }
    if (reads_entry || has_dense) {
        make_bytes_cb(all_cores, kCpdCbIdx, 4u * idx_slot_bytes, 1u, tt::DataFormat::Int32);
    }

    CreateSemaphore(program, all_cores, 0);  // kCpdArriveSemaphoreId = 0
    CreateSemaphore(program, all_cores, 0);  // kCpdReadySemaphoreId  = 1
    if (has_index_key) {
        CreateSemaphore(program, all_cores, 0);  // kCpdKeyLatentSemaphoreId = 2
        CreateSemaphore(program, all_cores, 0);  // kCpdKeyArriveSemaphoreId = 3
        CreateSemaphore(program, all_cores, 0);  // kCpdKeyReadySemaphoreId  = 4
    }

    Buffer* kv_buf = t.kv.buffer();
    Buffer* gate_buf = has_window ? t.gate->buffer() : nullptr;
    Buffer* win_kv_buf = has_window ? t.win_kv->buffer() : nullptr;
    Buffer* win_gate_buf = has_window ? t.win_gate->buffer() : nullptr;
    Buffer* win_slot_buf = has_window ? t.win_slot->buffer() : nullptr;
    Buffer* prev_slot_buf = has_prev_slot ? t.prev_slot->buffer() : nullptr;
    Buffer* norm_weight_buf = t.norm_weight.buffer();
    Buffer* cos_buf = has_rope ? t.cos->buffer() : nullptr;
    Buffer* sin_buf = has_rope ? t.sin->buffer() : nullptr;
    Buffer* out_latent_buf = do_pool ? tensor_return_value[0].buffer() : nullptr;
    Buffer* out_rope_buf = (do_pool && has_rope) ? tensor_return_value[1].buffer() : nullptr;
    Buffer* index_wk_buf = has_index_key ? t.index_wk->buffer() : nullptr;
    Buffer* index_norm_weight_buf = has_index_key ? t.index_norm_weight->buffer() : nullptr;
    Buffer* idx_key_cache_buf = has_index_key ? t.idx_key_cache->buffer() : nullptr;
    Buffer* comp_kv_buf = has_comp_kv ? t.comp_kv->buffer() : nullptr;
    Buffer* page_table_buf = reads_entry ? t.page_table->buffer() : nullptr;
    Buffer* entry_buf = reads_entry ? t.entry->buffer() : nullptr;
    Buffer* page_row_buf = args.has_page_row ? t.page_row->buffer() : nullptr;
    Buffer* dense_kv_buf = has_dense ? t.dense_kv->buffer() : nullptr;
    Buffer* dense_entry_buf = has_dense ? t.dense_entry->buffer() : nullptr;
    Buffer* key_row_buf = args.has_key_row ? t.key_row->buffer() : nullptr;

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(do_pool),
        static_cast<uint32_t>(has_window),
        static_cast<uint32_t>(has_prev_slot),
        static_cast<uint32_t>(has_rope),
        static_cast<uint32_t>(args.kv_sharded),
        static_cast<uint32_t>(args.norm_weight_is_tile),
        W_t,
        row_tile_bytes,
        cache_tile_bytes,
        win_slot_page_bytes,
        prev_slot_page_bytes,
        pos_slot_bytes,
        kCpdCbKv,
        kCpdCbK0,
        kCpdCbGamma,
        kCpdCbCs,
        kCpdCbWinKvTile,
        kCpdCbKvScratch,
        kCpdCbPosReader,
        kCpdCbRawScratch,
        static_cast<uint32_t>(has_index_key),
        static_cast<uint32_t>(args.index_norm_weight_is_tile),
        key_tiles,
        key_w_chunk,
        kCpdCbKeyW,
        kCpdCbKeyGamma,
    };
    TensorAccessorArgs(kv_buf).append_to(reader_ct_args);
    TensorAccessorArgs(win_kv_buf).append_to(reader_ct_args);
    TensorAccessorArgs(win_slot_buf).append_to(reader_ct_args);
    TensorAccessorArgs(prev_slot_buf).append_to(reader_ct_args);
    TensorAccessorArgs(norm_weight_buf).append_to(reader_ct_args);
    TensorAccessorArgs(cos_buf).append_to(reader_ct_args);
    TensorAccessorArgs(sin_buf).append_to(reader_ct_args);
    TensorAccessorArgs(index_wk_buf).append_to(reader_ct_args);
    TensorAccessorArgs(index_norm_weight_buf).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(do_pool),
        static_cast<uint32_t>(has_window),
        static_cast<uint32_t>(has_prev_slot),
        static_cast<uint32_t>(has_rope),
        static_cast<uint32_t>(args.gate_sharded),
        W_t,
        row_tile_bytes,
        cache_tile_bytes,
        win_slot_page_bytes,
        prev_slot_page_bytes,
        pos_slot_bytes,
        kCpdArriveSemaphoreId,
        kCpdReadySemaphoreId,
        kCpdCbGate,
        kCpdCbG0,
        kCpdCbLocalStat,
        kCpdCbGatherStats,
        kCpdCbScaleHub,
        kCpdCbScale,
        kCpdCbOutLatent,
        kCpdCbOutRope,
        kCpdCbWinGateTile,
        kCpdCbGateScratch,
        kCpdCbPosWriter,
        kCpdCbOutTileScratch,
        static_cast<uint32_t>(has_index_key),
        static_cast<uint32_t>(has_comp_kv),
        static_cast<uint32_t>(has_dense),
        static_cast<uint32_t>(args.has_page_row),
        static_cast<uint32_t>(args.has_key_row),
        key_tiles,
        reads_entry ? args.block_rows : 1u,  // the writer divides by it
        idx_slot_bytes,
        entry_page_bytes,
        page_row_page_bytes,
        dense_entry_page_bytes,
        page_table_chunk_bytes,
        kCpdKeyLatentSemaphoreId,
        kCpdKeyArriveSemaphoreId,
        kCpdKeyReadySemaphoreId,
        kCpdCbKeyIn,
        kCpdCbKeyOut,
        kCpdCbKeyTile,
        kCpdCbDenseTile,
        kCpdCbIdx,
    };
    TensorAccessorArgs(gate_buf).append_to(writer_ct_args);
    TensorAccessorArgs(win_gate_buf).append_to(writer_ct_args);
    TensorAccessorArgs(win_slot_buf).append_to(writer_ct_args);
    TensorAccessorArgs(prev_slot_buf).append_to(writer_ct_args);
    TensorAccessorArgs(out_latent_buf).append_to(writer_ct_args);
    TensorAccessorArgs(out_rope_buf).append_to(writer_ct_args);
    TensorAccessorArgs(idx_key_cache_buf).append_to(writer_ct_args);
    TensorAccessorArgs(comp_kv_buf).append_to(writer_ct_args);
    TensorAccessorArgs(page_table_buf).append_to(writer_ct_args);
    TensorAccessorArgs(entry_buf).append_to(writer_ct_args);
    TensorAccessorArgs(page_row_buf).append_to(writer_ct_args);
    TensorAccessorArgs(dense_kv_buf).append_to(writer_ct_args);
    TensorAccessorArgs(dense_entry_buf).append_to(writer_ct_args);
    TensorAccessorArgs(key_row_buf).append_to(writer_ct_args);

    const uint32_t inv_n_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(args.head_dim));
    const uint32_t inv_key_n_bits =
        has_index_key ? std::bit_cast<uint32_t>(1.0f / static_cast<float>(args.index_head_dim)) : 0u;
    const uint32_t eps_bits = std::bit_cast<uint32_t>(args.epsilon);
    const std::vector<uint32_t> compute_ct_args = {
        static_cast<uint32_t>(do_pool),
        static_cast<uint32_t>(has_window),
        static_cast<uint32_t>(has_rope),
        W_t,
        inv_n_bits,
        eps_bits,
        kCpdCbKv,
        kCpdCbK0,
        kCpdCbGate,
        kCpdCbG0,
        kCpdCbGamma,
        kCpdCbCs,
        kCpdCbTmp,
        kCpdCbPooled,
        kCpdCbLocalStat,
        kCpdCbGatherStats,
        kCpdCbScaleHub,
        kCpdCbScale,
        kCpdCbOutLatent,
        kCpdCbOutRope,
        static_cast<uint32_t>(has_index_key),
        key_tiles,
        key_w_chunk,
        inv_key_n_bits,
        kCpdCbKeyIn,
        kCpdCbKeyW,
        kCpdCbKeyMm,
        kCpdCbKeyGamma,
        kCpdCbKeyOut,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kCpdReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_ct_args));
    const KernelHandle writer_kernel_id =
        CreateKernel(program, kCpdWriterKernelPath, all_cores, WriterDataMovementConfig(writer_ct_args));
    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kCpdComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            // The index key's K = Dh reduction accumulates in fp32, as ttnn.linear's HiFi4 config does.
            .fp32_dest_acc_en = has_index_key,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    std::vector<uint32_t> common_writer_coords;
    common_writer_coords.reserve(2 * W_t);
    for (const auto& core : cores) {
        const auto phys = device->worker_core_from_logical_core(core);
        common_writer_coords.push_back(static_cast<uint32_t>(phys.x));
        common_writer_coords.push_back(static_cast<uint32_t>(phys.y));
    }
    SetCommonRuntimeArgs(program, writer_kernel_id, common_writer_coords);

    const auto hub_phys = device->worker_core_from_logical_core(cores[0]);
    const uint32_t kv_addr = kv_buf->address();
    const uint32_t gate_addr = gate_buf ? gate_buf->address() : 0u;
    const uint32_t win_kv_addr = win_kv_buf ? win_kv_buf->address() : 0u;
    const uint32_t win_gate_addr = win_gate_buf ? win_gate_buf->address() : 0u;
    const uint32_t win_slot_addr = win_slot_buf ? win_slot_buf->address() : 0u;
    const uint32_t prev_slot_addr = prev_slot_buf ? prev_slot_buf->address() : 0u;
    const uint32_t norm_weight_addr = norm_weight_buf->address();
    const uint32_t cos_addr = cos_buf ? cos_buf->address() : 0u;
    const uint32_t sin_addr = sin_buf ? sin_buf->address() : 0u;
    const uint32_t out_latent_addr = out_latent_buf ? out_latent_buf->address() : 0u;
    const uint32_t out_rope_addr = out_rope_buf ? out_rope_buf->address() : 0u;
    const uint32_t index_wk_addr = index_wk_buf ? index_wk_buf->address() : 0u;
    const uint32_t index_norm_weight_addr = index_norm_weight_buf ? index_norm_weight_buf->address() : 0u;
    const uint32_t idx_key_cache_addr = idx_key_cache_buf ? idx_key_cache_buf->address() : 0u;
    const uint32_t comp_kv_addr = comp_kv_buf ? comp_kv_buf->address() : 0u;
    const uint32_t page_table_addr = page_table_buf ? page_table_buf->address() : 0u;
    const uint32_t entry_addr = entry_buf ? entry_buf->address() : 0u;
    const uint32_t page_row_addr = page_row_buf ? page_row_buf->address() : 0u;
    const uint32_t dense_kv_addr = dense_kv_buf ? dense_kv_buf->address() : 0u;
    const uint32_t dense_entry_addr = dense_entry_buf ? dense_entry_buf->address() : 0u;
    const uint32_t key_row_addr = key_row_buf ? key_row_buf->address() : 0u;

    for (uint32_t i = 0; i < W_t; ++i) {
        const bool is_rope_core = has_rope && (i + rope_tiles >= W_t);
        const uint32_t rope_tile_idx = is_rope_core ? (i + rope_tiles - W_t) : 0u;
        const bool is_key_core = has_index_key && i >= first_key_core;
        const uint32_t key_idx = is_key_core ? i - first_key_core : 0u;

        uint32_t kv_noc_x = 0;
        uint32_t kv_noc_y = 0;
        uint32_t kv_offset_bytes = i * row_tile_bytes;
        if (args.kv_sharded) {
            const uint32_t shard_idx = (i * constants::TILE_WIDTH) / kv_shard_w;
            kv_offset_bytes = ((i * constants::TILE_WIDTH) % kv_shard_w) * elem_bytes;
            const auto phys = device->worker_core_from_logical_core(kv_shard_cores[shard_idx]);
            kv_noc_x = static_cast<uint32_t>(phys.x);
            kv_noc_y = static_cast<uint32_t>(phys.y);
        }

        uint32_t gate_noc_x = 0;
        uint32_t gate_noc_y = 0;
        uint32_t gate_offset_bytes = i * row_tile_bytes;
        if (args.gate_sharded) {
            const uint32_t shard_idx = (i * constants::TILE_WIDTH) / gate_shard_w;
            gate_offset_bytes = ((i * constants::TILE_WIDTH) % gate_shard_w) * elem_bytes;
            const auto phys = device->worker_core_from_logical_core(gate_shard_cores[shard_idx]);
            gate_noc_x = static_cast<uint32_t>(phys.x);
            gate_noc_y = static_cast<uint32_t>(phys.y);
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            cores[i],
            {i,
             kv_addr,
             kv_noc_x,
             kv_noc_y,
             kv_offset_bytes,
             win_kv_addr,
             win_slot_addr,
             prev_slot_addr,
             norm_weight_addr,
             cos_addr,
             sin_addr,
             static_cast<uint32_t>(is_rope_core),
             rope_tile_idx,
             index_wk_addr,
             index_norm_weight_addr,
             static_cast<uint32_t>(is_key_core),
             key_idx});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            cores[i],
            {i,
             gate_addr,
             gate_noc_x,
             gate_noc_y,
             gate_offset_bytes,
             win_gate_addr,
             win_slot_addr,
             prev_slot_addr,
             out_latent_addr,
             out_rope_addr,
             static_cast<uint32_t>(is_rope_core),
             static_cast<uint32_t>(hub_phys.x),
             static_cast<uint32_t>(hub_phys.y),
             idx_key_cache_addr,
             comp_kv_addr,
             page_table_addr,
             entry_addr,
             page_row_addr,
             dense_kv_addr,
             dense_entry_addr,
             key_row_addr,
             static_cast<uint32_t>(is_key_core),
             key_idx});

        SetRuntimeArgs(
            program,
            compute_kernel_id,
            cores[i],
            {static_cast<uint32_t>(i == 0),
             static_cast<uint32_t>(is_rope_core),
             static_cast<uint32_t>(is_key_core),
             static_cast<uint32_t>(is_key_core && key_idx == 0)});
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .cores = std::move(cores)}};
}

void CompressPoolDecodeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const CompressPoolDecodeParams& args,
    const CompressPoolDecodeInputs& t,
    std::vector<Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t kv_addr = t.kv.buffer()->address();
    const uint32_t gate_addr = args.has_window ? t.gate->buffer()->address() : 0u;
    const uint32_t win_kv_addr = args.has_window ? t.win_kv->buffer()->address() : 0u;
    const uint32_t win_gate_addr = args.has_window ? t.win_gate->buffer()->address() : 0u;
    const uint32_t win_slot_addr = args.has_window ? t.win_slot->buffer()->address() : 0u;
    const uint32_t prev_slot_addr = args.has_prev_slot ? t.prev_slot->buffer()->address() : 0u;
    const uint32_t norm_weight_addr = t.norm_weight.buffer()->address();
    const uint32_t cos_addr = args.has_rope ? t.cos->buffer()->address() : 0u;
    const uint32_t sin_addr = args.has_rope ? t.sin->buffer()->address() : 0u;
    const uint32_t out_latent_addr = args.do_pool ? tensor_return_value[0].buffer()->address() : 0u;
    const uint32_t out_rope_addr = (args.do_pool && args.has_rope) ? tensor_return_value[1].buffer()->address() : 0u;
    const bool reads_entry = args.has_index_key || args.has_comp_kv;
    const uint32_t index_wk_addr = args.has_index_key ? t.index_wk->buffer()->address() : 0u;
    const uint32_t index_norm_weight_addr = args.has_index_key ? t.index_norm_weight->buffer()->address() : 0u;
    const uint32_t idx_key_cache_addr = args.has_index_key ? t.idx_key_cache->buffer()->address() : 0u;
    const uint32_t comp_kv_addr = args.has_comp_kv ? t.comp_kv->buffer()->address() : 0u;
    const uint32_t page_table_addr = reads_entry ? t.page_table->buffer()->address() : 0u;
    const uint32_t entry_addr = reads_entry ? t.entry->buffer()->address() : 0u;
    const uint32_t page_row_addr = args.has_page_row ? t.page_row->buffer()->address() : 0u;
    const uint32_t dense_kv_addr = args.has_dense ? t.dense_kv->buffer()->address() : 0u;
    const uint32_t dense_entry_addr = args.has_dense ? t.dense_entry->buffer()->address() : 0u;
    const uint32_t key_row_addr = args.has_key_row ? t.key_row->buffer()->address() : 0u;

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r_args = reader_args_by_core[core.x][core.y];
        r_args[1] = kv_addr;
        r_args[5] = win_kv_addr;
        r_args[6] = win_slot_addr;
        r_args[7] = prev_slot_addr;
        r_args[8] = norm_weight_addr;
        r_args[9] = cos_addr;
        r_args[10] = sin_addr;
        r_args[13] = index_wk_addr;
        r_args[14] = index_norm_weight_addr;

        auto& w_args = writer_args_by_core[core.x][core.y];
        w_args[1] = gate_addr;
        w_args[5] = win_gate_addr;
        w_args[6] = win_slot_addr;
        w_args[7] = prev_slot_addr;
        w_args[8] = out_latent_addr;
        w_args[9] = out_rope_addr;
        w_args[13] = idx_key_cache_addr;
        w_args[14] = comp_kv_addr;
        w_args[15] = page_table_addr;
        w_args[16] = entry_addr;
        w_args[17] = page_row_addr;
        w_args[18] = dense_kv_addr;
        w_args[19] = dense_entry_addr;
        w_args[20] = key_row_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::compress_pool_decode
