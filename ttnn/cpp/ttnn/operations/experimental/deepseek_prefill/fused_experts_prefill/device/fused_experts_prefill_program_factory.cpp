// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_prefill_device_operation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <string>
#include <vector>

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
//   * 12x10 worker cores -> groups of gx_cores x gy_cores (4x2: 15 groups of 8, or 3x2: 20 groups of 6; the
//     shape whose busiest core reads the fewest weight tiles wins, see below). Group
//     g = (y / gy_cores) * (12 / gx_cores) + (x / gx_cores); core-in-group c = (y % gy_cores) * gx_cores +
//     (x % gx_cores). Expert e is owned by group e % num_groups, whose cores work on it together.
//   * Routing: EVERY core reads the routing input for all T tokens, selects top-k, weights, and keeps
//     the token list (token, slot, weight) of the experts its group owns, in L1 (cb_meta).
//   * A core gathers its group's tokens' x rows from DRAM (row-major, 1 KB segments), tilizes them, and
//     streams kMBlock tile rows at a time against its resident weights (M innermost).
//   * gate_up (+ SwiGLU, scaled by the token's routing weight): the expert's I/32 tiles are split over the
//     group in consecutive runs (the first I/32 % cpg cores get one tile more): core c owns gate_up ND shards
//     [first_c, first_c + n_c), n_c <= it_max = ceil(I/32/cpg). The weight ring's slot holds spp such shards.
//     With spp = it_max the whole run stays resident while every M-block of the chunk passes over it;
//     with a smaller spp (an expert too big for L1) the run is streamed through the ring once per M-block.
//   * The group's cores all-gather their SwiGLU tiles (unicast writes + monotonic semaphores), each core's block
//     laid out [row][jj] with a stride of it_max tiles (cores with n_c < it_max leave the tail unused).
//   * down: the expert's H/64 down ND shards (2 output tile columns each) are split over the group in
//     consecutive runs (the first H/64 % cpg cores get one more): core c owns shards [dn_first_c, +dn_cnt_c),
//     each against the full I of the gathered act. The result is untilized and each valid row goes to the
//     partials tensor row (slot * T + token). The kernels are compiled once per distinct dn_cnt_c.
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

    const uint32_t kt = hidden / kTile;  // K tiles of gate_up == output tile columns
    const uint32_t it = inter / kTile;   // I tiles == K tiles of down
    const uint32_t dn_total = kt / 2;    // down shards ([I, 64] = 2 output tile columns)
    const uint32_t gu_shard_tiles = kt * 2;
    const uint32_t dn_shard_tiles = it * 2;

    // Group shape: the experts of a group run one after the other and the gate_up and down phases of an expert
    // are separated by the group's all-gather, so a call takes about (experts per group) * (gate_up shards +
    // down shards of the busiest core) weight reads. Ties keep the first shape.
    const auto shape_cost = [&](const GroupShape& shape) {
        const uint32_t cpg = shape.x * shape.y;
        const uint32_t groups = (kGridX / shape.x) * (kGridY / shape.y);
        const uint64_t per_expert = static_cast<uint64_t>((it + cpg - 1) / cpg) * gu_shard_tiles +
                                    static_cast<uint64_t>((dn_total + cpg - 1) / cpg) * dn_shard_tiles;
        return ((num_experts + groups - 1) / groups) * per_expert;
    };
    GroupShape group = kGroupShapes[0];
    for (const auto& shape : kGroupShapes) {
        if (shape_cost(shape) < shape_cost(group)) {
            group = shape;
        }
    }
    const uint32_t cores_per_group = group.x * group.y;
    const uint32_t groups_x = kGridX / group.x;
    const uint32_t num_groups = groups_x * (kGridY / group.y);

    const uint32_t it_base = it / cores_per_group;  // gate_up shards (I tiles) per core, at least
    const uint32_t it_rem = it % cores_per_group;   // cores [0, it_rem) own one more
    const uint32_t it_max = it_base + (it_rem ? 1 : 0);
    const uint32_t dn_base = dn_total / cores_per_group;  // down shards per core, at least
    const uint32_t dn_rem = dn_total % cores_per_group;   // cores [0, dn_rem) own one more
    const uint32_t dn_max = dn_base + (dn_rem ? 1 : 0);
    const uint32_t nt_max = 2 * dn_max;     // most output tile columns of a core (the cb_outt / cb_outrm stride)
    const uint32_t row_bytes = hidden * 2;  // one x_tok / partials row (bf16)
    const uint32_t m_block = kMBlock;

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
    const uint32_t n_owned_max = (num_experts + num_groups - 1) / num_groups;
    const uint32_t hdr_words = align_up(n_owned_max, 16u);
    const uint32_t meta_bytes = align_up(hdr_words * 4 + n_owned_max * tokens * 4, 64u);

    const uint32_t x_cb_bytes = m_block * kt * x_tile_bytes;
    const uint32_t mm_cb_bytes = 2 * m_block * mm_tile_bytes;
    const uint32_t outt_cb_bytes = m_block * nt_max * bf16_tile_bytes;
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
    const uint32_t outrm_cb_bytes = 2 * nt_max * bf16_tile_bytes;
    const uint32_t rscal_cb_bytes = 2 * m_block * bf16_tile_bytes;

    // L1 per core of a (spp, m_chunk) configuration: the weight ring (two slots of the larger of spp gate_up
    // shards and the down slice) and the SwiGLU activations (m_chunk tile rows of the group's I) are the
    // terms that depend on the I of the expert.
    const auto slot_tiles_of = [&](uint32_t spp, uint32_t dpp) {
        return std::max(spp * gu_shard_tiles, dpp * dn_shard_tiles);
    };
    const auto l1_bytes = [&](uint32_t spp, uint32_t dpp, uint32_t m_chunk) {
        const uint32_t act_local_bytes = m_chunk * it_max * act_tile_bytes;
        return meta_bytes + x_cb_bytes + 2 * slot_tiles_of(spp, dpp) * w_tile_bytes + mm_cb_bytes + act_local_bytes +
               cores_per_group * act_local_bytes + outt_cb_bytes + rm_cb_bytes + outrm_cb_bytes + rscal_cb_bytes +
               scratch_cb_bytes;
    };
    // The resident run at the full chunk size first (every M-block of a chunk reads each weight once), then the
    // run streamed one shard at a time with the largest chunk that fits (an expert with more rows re-reads its
    // weights per chunk).
    // Circular buffers end where the lowest L1 buffer starts (L1 buffers are allocated top-down).
    const auto lowest_occupied_l1 = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_region_end =
        lowest_occupied_l1.has_value() ? lowest_occupied_l1.value() : device->l1_size_per_core();
    const uint32_t l1_region_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_budget = l1_region_end > l1_region_base ? l1_region_end - l1_region_base : 0;
    // (gate_up shards per slot, down shards per slot, M tile rows per chunk); a core's down run is resident (one
    // slot for every M-block) when dpp >= its dn_cnt.
    const std::array<std::array<uint32_t, 3>, 7> candidates = {{
        {it_max, dn_max, kChunkTiles},
        {1, dn_max, kChunkTiles},
        {1, 1, kChunkTiles},
        {1, dn_max, kChunkTiles / 2},
        {1, 1, kChunkTiles / 2},
        {1, dn_max, m_block},
        {1, 1, m_block},
    }};
    uint32_t spp = 0;
    uint32_t dpp = 0;
    uint32_t m_chunk = 0;
    for (const auto& [cand_spp, cand_dpp, cand_m] : candidates) {
        if (l1_bytes(cand_spp, cand_dpp, cand_m) <= l1_budget) {
            spp = cand_spp;
            dpp = cand_dpp;
            m_chunk = cand_m;
            break;
        }
    }
    TT_FATAL(
        spp != 0,
        "fused_experts_prefill: I = {} at H = {} needs {} B of L1 per core even streaming one shard at a time with "
        "{}-row "
        "chunks, but only {} B are free",
        inter,
        hidden,
        l1_bytes(1, 1, m_block),
        m_block,
        l1_budget);
    const bool resident = spp >= it_max;
    const uint32_t slot_tiles = slot_tiles_of(spp, dpp);
    const uint32_t act_local_tiles = m_chunk * it_max;
    const uint32_t act_full_tiles = act_local_tiles * cores_per_group;

    const uint32_t w_cb_bytes = 2 * slot_tiles * w_tile_bytes;
    const uint32_t act_local_cb_bytes = act_local_tiles * act_tile_bytes;
    const uint32_t act_full_cb_bytes = act_full_tiles * act_tile_bytes;
    log_debug(
        tt::LogOp,
        "fused_experts_prefill: {}x{} groups, gate_up {} (spp {}), down {} (dpp {}), m_chunk {}, L1 per core {} of {} "
        "B",
        group.x,
        group.y,
        resident ? "resident" : "streamed",
        spp,
        dpp >= dn_max ? "resident" : "streamed",
        dpp,
        m_chunk,
        l1_bytes(spp, dpp, m_chunk),
        l1_budget);

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

    // One reader / compute / writer per distinct down run length (dn_max, then dn_base when it differs).
    struct Variant {
        KernelDescriptor reader;
        KernelDescriptor compute;
        KernelDescriptor writer;
    };
    std::vector<Variant> variants;
    const auto variant_cores = [&](uint32_t dn_cnt) {
        std::vector<CoreRange> ranges;
        for (uint32_t y = 0; y < kGridY; ++y) {
            for (uint32_t cx = 0; cx < kGridX; ++cx) {
                const uint32_t c = (y % group.y) * group.x + (cx % group.x);
                if (dn_base + (c < dn_rem ? 1 : 0) == dn_cnt) {
                    ranges.emplace_back(CoreCoord{cx, y});
                }
            }
        }
        return CoreRangeSet(std::move(ranges)).merge_ranges();
    };
    for (const uint32_t dn_cnt : dn_rem ? std::vector<uint32_t>{dn_max, dn_base} : std::vector<uint32_t>{dn_max}) {
        const uint32_t nt_pc = 2 * dn_cnt;
        const bool dn_resident = dpp >= dn_cnt;
        const CoreRangeSet cores = variant_cores(dn_cnt);

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
            num_groups,
            kt,
            spp,
            dn_cnt,
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
            resident ? 1u : 0u,
            dpp,
            dn_resident ? 1u : 0u,
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
            it_max,
            nt_pc,
            slot_tiles,
            limit_bits,
            act_local_tiles,
            act_full_tiles,
            kRmChunkTiles,
            spp,
            resident ? 1u : 0u,
            it_base,
            it_rem,
            dpp,
            dn_resident ? 1u : 0u,
            nt_max,
            cores_per_group,
        };

        std::vector<uint32_t> writer_ct = {
            cb_meta,   cb_act_local, cb_act_full,       cb_outrm,        num_groups,   nt_max,          it_max,
            m_chunk,   m_block,      act_tile_bytes,    sem_act_ready,   sem_act_free, act_local_tiles, act_full_tiles,
            hdr_words, tokens,       nt_pc * kTile * 2, bf16_tile_bytes, row_bytes,    cores_per_group,
        };
        TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

        Variant v;
        v.reader.kernel_source = std::string(kKernelDir) + "/dataflow/reader.cpp";
        v.reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        v.reader.core_ranges = cores;
        v.reader.compile_time_args = reader_ct;
        v.reader.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
        };

        v.writer.kernel_source = std::string(kKernelDir) + "/dataflow/writer.cpp";
        v.writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        v.writer.core_ranges = cores;
        v.writer.compile_time_args = writer_ct;
        v.writer.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_1,
        };

        v.compute.kernel_source = std::string(kKernelDir) + "/compute/compute.cpp";
        v.compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
        v.compute.core_ranges = cores;
        v.compute.compile_time_args = compute_ct;
        v.compute.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
        };
        variants.push_back(std::move(v));
    }

    Buffer* x_buffer = x.buffer();
    Buffer* ids_buffer = ids_or_scores.buffer();
    Buffer* scores_buffer = tensor_args.routing_scores.buffer();
    Buffer* rank_buffer = rank_or_scores.buffer();
    Buffer* out_buffer = output_tensor.buffer();

    for (uint32_t y = 0; y < kGridY; ++y) {
        for (uint32_t cx = 0; cx < kGridX; ++cx) {
            const CoreCoord core{cx, y};
            const uint32_t gx = cx / group.x;
            const uint32_t gy = y / group.y;
            const uint32_t g = gy * groups_x + gx;
            const uint32_t c = (y % group.y) * group.x + (cx % group.x);
            // This core's runs of gate_up shards (I tiles) and down shards: the first it_rem / dn_rem cores own
            // one more.
            const uint32_t gu_cnt = it_base + (c < it_rem ? 1 : 0);
            const uint32_t gu_first = c * it_base + std::min(c, it_rem);
            const uint32_t dn_cnt = dn_base + (c < dn_rem ? 1 : 0);
            const uint32_t dn_first = c * dn_base + std::min(c, dn_rem);
            Variant& v = variants[dn_cnt == dn_max ? 0 : 1];

            // Experts g, g + num_groups, g + 2 * num_groups, ... (< num_experts).
            const uint32_t n_owned = g < num_experts ? (num_experts - g + num_groups - 1) / num_groups : 0;

            // Reader: buffer bindings + raw weight addresses of this group's experts.
            std::vector<std::variant<uint32_t, Buffer*>> reader_rt = {
                x_buffer,
                ids_buffer,
                scores_buffer,
                rank_buffer,
                dn_first,
                g,
                n_owned,
                y * kGridX + cx,
                gu_cnt,
                gu_first};
            // NoC coordinates of the grid's columns / rows (the worker grid is a product of the two).
            for (uint32_t dx = 0; dx < kGridX; ++dx) {
                reader_rt.push_back(static_cast<uint32_t>(device->worker_core_from_logical_core({dx, 0}).x));
            }
            for (uint32_t dy = 0; dy < kGridY; ++dy) {
                reader_rt.push_back(static_cast<uint32_t>(device->worker_core_from_logical_core({0, dy}).y));
            }
            for (uint32_t j = 0; j < n_owned; ++j) {
                reader_rt.push_back(
                    static_cast<uint32_t>(tensor_args.gate_up_weights[g + j * num_groups].buffer()->address()));
            }
            for (uint32_t j = 0; j < n_owned; ++j) {
                reader_rt.push_back(
                    static_cast<uint32_t>(tensor_args.down_weights[g + j * num_groups].buffer()->address()));
            }
            v.reader.emplace_runtime_args(core, reader_rt);

            // Writer: output binding, this core's column offset in a partials row, and the NoC coordinates of the
            // group's cores (indexed by c').
            std::vector<std::variant<uint32_t, Buffer*>> writer_rt = {
                out_buffer, c, g, n_owned, dn_first * 2 * kTile * 2};
            for (uint32_t cp = 0; cp < cores_per_group; ++cp) {
                const CoreCoord peer{gx * group.x + (cp % group.x), gy * group.y + (cp / group.x)};
                const auto peer_noc = device->worker_core_from_logical_core(peer);
                writer_rt.push_back(static_cast<uint32_t>(peer_noc.x));
                writer_rt.push_back(static_cast<uint32_t>(peer_noc.y));
            }
            v.writer.emplace_runtime_args(core, writer_rt);

            v.compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{n_owned, gu_cnt});
        }
    }

    for (auto& v : variants) {
        desc.kernels.push_back(std::move(v.reader));
        desc.kernels.push_back(std::move(v.compute));
        desc.kernels.push_back(std::move(v.writer));
    }
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill
