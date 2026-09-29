// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_prefill_device_operation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/fused_experts_prefill/device/kernels";

constexpr uint32_t kTile = 32;

uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }
}  // namespace

// Work split (see the kernels for the per-core detail):
//
//   * 12x10 worker cores -> 15 groups of 4x2 = 8 cores. Group g = (y / 2) * 3 + (x / 4); core-in-group
//     c = (y % 2) * 4 + (x % 4). Expert e is owned by group e % 15, whose 8 cores work on it together.
//   * Routing: EVERY core reads the routing input for all T tokens, selects top-k, weights, and keeps
//     the token list (token, slot, weight) of the experts its group owns, in L1 (cb_meta).
//   * A core gathers its group's tokens' x rows from DRAM (row-major, 1 KB segments), tilizes them, and
//     streams kMBlock tile rows at a time against its resident weights (M innermost).
//   * gate_up (+ SwiGLU, scaled by the token's routing weight): the expert's I/32 tiles are split over the
//     group, it_pc = I/32/8 consecutive tiles per core = gate_up ND shards [c * it_pc, +it_pc).
//   * The 8 cores all-gather their SwiGLU tiles (unicast writes + monotonic semaphores).
//   * down: the expert's H/32 output tiles are split over the group, nt_pc = H/32/8 per core = down ND
//     shards [c * nt_pc/2, +nt_pc/2), each against the full I of the gathered act. The result is untilized
//     and each valid row goes to the partials tensor row (slot * T + token).
ProgramDescriptor FusedExpertsPrefillDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& x = tensor_args.x_tok;
    auto* device = x.device();

    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        grid.x >= kGridX && grid.y >= kGridY,
        "fused_experts_prefill needs a {}x{} worker grid, got {}x{}",
        kGridX,
        kGridY,
        grid.x,
        grid.y);

    const uint32_t num_experts = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const uint32_t hidden = static_cast<uint32_t>(x.logical_shape()[-1]);
    const uint32_t tokens = static_cast<uint32_t>(x.logical_shape()[-2]);
    const uint32_t inter = attributes.intermediate_size;
    const uint32_t top_k = attributes.top_k;

    const uint32_t kt = hidden / kTile;                // K tiles of gate_up == output tile columns
    const uint32_t it = inter / kTile;                 // I tiles == K tiles of down
    const uint32_t it_pc = it / kCoresPerGroup;        // gate_up shards (I tiles) per core
    const uint32_t nt = kt;                            // output tile columns
    const uint32_t nt_pc = nt / kCoresPerGroup;        // output tile columns per core
    const uint32_t dn_shards = nt_pc / 2;              // down shards ([I, 64] = 2 tile columns) per core
    const uint32_t row_bytes = hidden * 2;             // one x_tok / partials row (bf16)
    const uint32_t out_seg_bytes = nt_pc * kTile * 2;  // this core's slice of an output row
    const uint32_t m_chunk = kChunkTiles;
    const uint32_t m_block = kMBlock;

    const uint32_t gu_shard_tiles = kt * 2;
    const uint32_t dn_shard_tiles = it * 2;
    const uint32_t gu_slice_tiles = it_pc * gu_shard_tiles;
    const uint32_t dn_slice_tiles = dn_shards * dn_shard_tiles;
    // Gate_up and down slices alternate in one ring of equal slots, so every slot is sized for the
    // larger of the two; the smaller one uses the front of its slot.
    const uint32_t slot_tiles = std::max(gu_slice_tiles, dn_slice_tiles);

    const tt::DataFormat x_df = tt::DataFormat::Bfp8_b;  // tilized x
    const tt::DataFormat rm_df = tt::DataFormat::Float16_b;
    const tt::DataFormat w_df = datatype_to_dataformat_converter(tensor_args.gate_up_weights[0].dtype());
    const tt::DataFormat act_df = tt::DataFormat::Bfp8_b;
    const tt::DataFormat out_df = tt::DataFormat::Float16_b;
    const uint32_t x_tile_bytes = tt::tile_size(x_df);
    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t w_tile_bytes = tt::tile_size(w_df);
    const uint32_t act_tile_bytes = tt::tile_size(act_df);
    const uint32_t mm_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    TT_FATAL(
        tensor_args.gate_up_weights[0].buffer()->page_size() == w_tile_bytes &&
            tensor_args.down_weights[0].buffer()->page_size() == w_tile_bytes,
        "fused_experts_prefill: weight pages must be single tiles of {} B",
        w_tile_bytes);

    // Routing scratch (one band of 32 tokens: ids tile, score tiles, ranking tiles, then a byte table
    // of the group-owned experts) has its own CB so the weight ring can prefetch during the routing.
    const uint32_t score_tiles = num_experts / kTile + (num_experts % kTile ? 1 : 0);
    // The routing is split over ALL cores (tpc consecutive tokens each) and exchanged: every core keeps the
    // routing of every token as ks words per token (expert id | bf16 weight << 16, ks a multiple of 4 so
    // every core's block starts 16 B aligned) in a route buffer at the front of the scratch.
    constexpr uint32_t kNumCores = kGridX * kGridY;
    const uint32_t tpc = (tokens + kNumCores - 1) / kNumCores;
    const uint32_t ks = align_up(top_k, 4u);
    const uint32_t route_bytes = align_up(tokens * ks * 4, 64u);
    const uint32_t send_bytes = align_up(tpc * ks * 4, 64u);
    const uint32_t scratch_cb_bytes =
        align_up(route_bytes + send_bytes + (1 + 2 * score_tiles) * bf16_tile_bytes + num_experts, bf16_tile_bytes);

    // Routing lists: counts[jl] header, then a T-entry list per owned expert (jl = 0 .. n_owned_max).
    const uint32_t n_owned_max = (num_experts + kNumGroups - 1) / kNumGroups;
    const uint32_t hdr_words = align_up(n_owned_max, 16u);
    const uint32_t meta_bytes = align_up(hdr_words * 4 + n_owned_max * tokens * 4, 64u);

    const uint32_t act_local_tiles = m_chunk * it_pc;
    const uint32_t act_full_tiles = act_local_tiles * kCoresPerGroup;
    TT_FATAL(act_full_tiles == m_chunk * it, "fused_experts_prefill: act gather layout mismatch");

    const uint32_t x_cb_bytes = m_block * kt * x_tile_bytes;
    const uint32_t w_cb_bytes = 2 * slot_tiles * w_tile_bytes;
    const uint32_t mm_cb_bytes = 2 * m_block * mm_tile_bytes;
    const uint32_t act_local_cb_bytes = act_local_tiles * act_tile_bytes;
    const uint32_t act_full_cb_bytes = act_full_tiles * act_tile_bytes;
    const uint32_t outt_cb_bytes = m_block * nt_pc * bf16_tile_bytes;
    // Row-major x staging: one group of `rm_group` segments (16 tiles each) is gathered per barrier; the
    // reader reserves the segments cumulatively so it refills as compute's tilize frees them.
    const uint32_t rm_chunks_per_row = kt / kRmChunkTiles;
    uint32_t rm_group = 1;
    for (uint32_t gsz = 1; gsz <= kRmGroupMax; ++gsz) {
        if (rm_chunks_per_row % gsz == 0) {
            rm_group = gsz;
        }
    }
    const uint32_t rm_cb_bytes = rm_group * kRmChunkTiles * bf16_tile_bytes;
    const uint32_t outrm_cb_bytes = 2 * nt_pc * bf16_tile_bytes;
    const uint32_t rscal_cb_bytes = 2 * m_block * bf16_tile_bytes;
    log_debug(
        tt::LogOp,
        "fused_experts_prefill L1 per core: meta {} x {} w {} mm {} act_local {} act_full {} outt {} rm {} outrm {} "
        "rscal {} scratch {} = {} B",
        meta_bytes,
        x_cb_bytes,
        w_cb_bytes,
        mm_cb_bytes,
        act_local_cb_bytes,
        act_full_cb_bytes,
        outt_cb_bytes,
        rm_cb_bytes,
        outrm_cb_bytes,
        rscal_cb_bytes,
        scratch_cb_bytes,
        meta_bytes + x_cb_bytes + w_cb_bytes + mm_cb_bytes + act_local_cb_bytes + act_full_cb_bytes + outt_cb_bytes +
            rm_cb_bytes + outrm_cb_bytes + rscal_cb_bytes + scratch_cb_bytes);

    const uint32_t limit_bits = std::bit_cast<uint32_t>(attributes.swiglu_limit);
    const uint32_t scaling_bits = std::bit_cast<uint32_t>(attributes.routed_scaling_factor);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(attributes.routing_eps);
    const bool rank_from_scores = tensor_args.ranking_scores.has_value();
    const bool index_is_bf16 =
        tensor_args.routing_indices.has_value() && tensor_args.routing_indices->dtype() == DataType::BFLOAT16;

    // ------------------------------------------------------------------ program
    ProgramDescriptor desc;

    const CoreRange all_range{CoreCoord{0, 0}, CoreCoord{kGridX - 1, kGridY - 1}};
    const CoreRangeSet all_cores{all_range};

    constexpr uint32_t sem_act_ready = 0;
    constexpr uint32_t sem_act_free = 1;
    constexpr uint32_t sem_route = 2;
    for (uint32_t s : {sem_act_ready, sem_act_free, sem_route}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = s,
            .core_type = CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // CBs are allocated identically on all cores: cb_act_full's base address is used by remote writes.
    auto add_cb = [&](uint32_t index, uint32_t total_bytes, tt::DataFormat df, uint32_t page_bytes) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index,
                .data_format = df,
                .page_size = page_bytes,
            }}},
        });
    };
    constexpr uint32_t cb_meta = CBIndex::c_0;
    constexpr uint32_t cb_x = CBIndex::c_1;
    constexpr uint32_t cb_w = CBIndex::c_2;
    constexpr uint32_t cb_mm = CBIndex::c_3;
    constexpr uint32_t cb_act_local = CBIndex::c_4;
    constexpr uint32_t cb_act_full = CBIndex::c_5;
    constexpr uint32_t cb_outt = CBIndex::c_6;
    constexpr uint32_t cb_rm = CBIndex::c_7;
    constexpr uint32_t cb_outrm = CBIndex::c_8;
    constexpr uint32_t cb_rscal = CBIndex::c_9;
    constexpr uint32_t cb_scratch = CBIndex::c_10;
    add_cb(cb_meta, meta_bytes, tt::DataFormat::UInt32, meta_bytes);
    add_cb(cb_x, x_cb_bytes, x_df, x_tile_bytes);
    add_cb(cb_w, w_cb_bytes, w_df, w_tile_bytes);
    add_cb(cb_mm, mm_cb_bytes, tt::DataFormat::Float32, mm_tile_bytes);
    add_cb(cb_act_local, act_local_cb_bytes, act_df, act_tile_bytes);
    add_cb(cb_act_full, act_full_cb_bytes, act_df, act_tile_bytes);
    add_cb(cb_outt, outt_cb_bytes, out_df, bf16_tile_bytes);
    add_cb(cb_rm, rm_cb_bytes, rm_df, bf16_tile_bytes);
    add_cb(cb_outrm, outrm_cb_bytes, out_df, bf16_tile_bytes);
    add_cb(cb_rscal, rscal_cb_bytes, tt::DataFormat::Float16_b, bf16_tile_bytes);
    add_cb(cb_scratch, scratch_cb_bytes, tt::DataFormat::Float16_b, bf16_tile_bytes);

    // ---- kernel args ----
    // Routing tensors that are absent reuse the scores' accessor args (never dereferenced).
    const Tensor& ids_or_scores =
        tensor_args.routing_indices.has_value() ? *tensor_args.routing_indices : tensor_args.routing_scores;
    const Tensor& rank_or_scores =
        tensor_args.ranking_scores.has_value() ? *tensor_args.ranking_scores : tensor_args.routing_scores;

    std::vector<uint32_t> reader_ct = {
        cb_meta,
        cb_rm,
        cb_w,
        cb_rscal,
        tokens,
        num_experts,
        top_k,
        index_is_bf16 ? 1u : 0u,
        rank_from_scores ? 1u : 0u,
        scaling_bits,
        eps_bits,
        kNumGroups,
        kt,
        it_pc,
        dn_shards,
        gu_shard_tiles,
        dn_shard_tiles,
        w_tile_bytes,
        slot_tiles,
        m_chunk,
        m_block,
        hdr_words,
        kRmChunkTiles,
        row_bytes,
        score_tiles,
        bf16_tile_bytes,
        cb_scratch,
        tpc,
        ks,
        sem_route,
        kGridX,
        kGridY,
        rm_group,
    };
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*ids_or_scores.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*tensor_args.routing_scores.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*rank_or_scores.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*tensor_args.gate_up_weights[0].buffer()).append_to(reader_ct);
    TensorAccessorArgs(*tensor_args.down_weights[0].buffer()).append_to(reader_ct);

    std::vector<uint32_t> compute_ct = {
        cb_meta,
        cb_x,
        cb_w,
        cb_mm,
        cb_act_local,
        cb_act_full,
        cb_outt,
        cb_rm,
        cb_outrm,
        cb_rscal,
        m_chunk,
        m_block,
        kt,
        it,
        it_pc,
        nt_pc,
        slot_tiles,
        limit_bits,
        act_local_tiles,
        act_full_tiles,
        kRmChunkTiles,
    };

    std::vector<uint32_t> writer_ct = {
        cb_meta,   cb_act_local, cb_act_full,    cb_outrm,        kNumGroups,   nt_pc,           it_pc,
        m_chunk,   m_block,      act_tile_bytes, sem_act_ready,   sem_act_free, act_local_tiles, act_full_tiles,
        hdr_words, tokens,       out_seg_bytes,  bf16_tile_bytes, row_bytes,
    };
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "/dataflow/reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "/dataflow/writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "/compute/compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi2,
        .fp32_dest_acc_en = true,
    };

    Buffer* x_buffer = x.buffer();
    Buffer* ids_buffer = ids_or_scores.buffer();
    Buffer* scores_buffer = tensor_args.routing_scores.buffer();
    Buffer* rank_buffer = rank_or_scores.buffer();
    Buffer* out_buffer = output_tensor.buffer();

    for (uint32_t y = 0; y < kGridY; ++y) {
        for (uint32_t cx = 0; cx < kGridX; ++cx) {
            const CoreCoord core{cx, y};
            const uint32_t gx = cx / kGroupCoresX;
            const uint32_t gy = y / kGroupCoresY;
            const uint32_t g = gy * (kGridX / kGroupCoresX) + gx;
            const uint32_t c = (y % kGroupCoresY) * kGroupCoresX + (cx % kGroupCoresX);
            // Experts g, g + 15, g + 30, ... (< num_experts).
            const uint32_t n_owned = g < num_experts ? (num_experts - g + kNumGroups - 1) / kNumGroups : 0;

            // Reader: buffer bindings + raw weight addresses of this group's experts.
            std::vector<std::variant<uint32_t, Buffer*>> reader_rt = {
                x_buffer, ids_buffer, scores_buffer, rank_buffer, c, g, n_owned, y * kGridX + cx};
            // NoC coordinates of the grid's columns / rows (the worker grid is a product of the two).
            for (uint32_t dx = 0; dx < kGridX; ++dx) {
                reader_rt.push_back(static_cast<uint32_t>(device->worker_core_from_logical_core({dx, 0}).x));
            }
            for (uint32_t dy = 0; dy < kGridY; ++dy) {
                reader_rt.push_back(static_cast<uint32_t>(device->worker_core_from_logical_core({0, dy}).y));
            }
            for (uint32_t j = 0; j < n_owned; ++j) {
                reader_rt.push_back(
                    static_cast<uint32_t>(tensor_args.gate_up_weights[g + j * kNumGroups].buffer()->address()));
            }
            for (uint32_t j = 0; j < n_owned; ++j) {
                reader_rt.push_back(
                    static_cast<uint32_t>(tensor_args.down_weights[g + j * kNumGroups].buffer()->address()));
            }
            reader_desc.emplace_runtime_args(core, reader_rt);

            // Writer: output binding + NoC coordinates of the group's 8 cores (indexed by c').
            std::vector<std::variant<uint32_t, Buffer*>> writer_rt = {out_buffer, c, g, n_owned};
            for (uint32_t cp = 0; cp < kCoresPerGroup; ++cp) {
                const CoreCoord peer{gx * kGroupCoresX + (cp % kGroupCoresX), gy * kGroupCoresY + (cp / kGroupCoresX)};
                const auto peer_noc = device->worker_core_from_logical_core(peer);
                writer_rt.push_back(static_cast<uint32_t>(peer_noc.x));
                writer_rt.push_back(static_cast<uint32_t>(peer_noc.y));
            }
            writer_desc.emplace_runtime_args(core, writer_rt);

            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{n_owned});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill
