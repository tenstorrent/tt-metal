// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) program factory. Cores are grouped per head: the group's first core is a
// fetch-only LEADER that streams the head's KV blocks from DRAM once per pass into its L1 slots;
// the WORKERS hold resident query rows as CONTIGUOUS runs (real VSA index sets are spatially
// correlated and every row lists the exempt prefix, so contiguity maximizes how many resident rows
// each arriving block visits -- the multi-row batching that amortizes per-visit overhead), pull
// blocks from the leader's L1 over the NoC, and run the fused multi-row online-softmax engine. DRAM reads the head's KV exactly n_passes times per group instead
// of once per (row, listed block) -- the redundancy that made v1/v2 DRAM-bound.

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace ttnn::prim {

namespace {
using RtArgs = std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>;

constexpr uint32_t kRMax = 10;         // resident rows per pass (<= 32 for the bitmasks); see kStreamDepth
constexpr uint32_t kRowGroup = 8;      // rows fused per compute phase group
constexpr uint32_t kStreamDepth =
    16;  // 18 -> 16 funds the fp32 running-sum buffer (+80 KB) in the model's L1;  // KV blocks in flight (two window
         // halves of 9). Rows-for-depth trade measured on REAL selections (2.4 blocks/visit): 15/12 -> 10/18 is -4%
         // standalone, -2% median in the 15 s block; 10/20 is -6% but overflows the model's live L1 by 54 KB
constexpr uint32_t kLogDepth = 16;     // leader arrival-log ring (> leader depth + sentinel slack)
constexpr uint32_t kMaxWorkers = 16;   // leader runtime-arg array bound
constexpr uint32_t kRowChunk = 4;      // contiguous rows per placement chunk (matches the kernels)
constexpr uint32_t kMaxPeers = 16;     // distributed kernel: peers per head group (ackbox / ready board words)

// Row dealing by static cost. `units` are indivisible row groups (a 4-row chunk of sparse rows keeps
// the adjacent-row locality the streaming engine batches on; a dense row is a unit of its own since it
// costs ~9 sparse rows). Units go longest-first onto the least-loaded (pass, consumer) bin that still
// has room for them (<= rmax rows per bin); every consumer runs the same number of passes in lockstep
// with the sequence sweep, so it is the PER-PASS cost that must balance. n_passes is the smallest
// count for which all units fit. Returns bins[pass][consumer] = ascending row ids.
struct Dealt {
    std::vector<std::vector<std::vector<uint32_t>>> bins;
    uint32_t n_passes = 0;
};
Dealt deal_units_by_cost(
    const std::vector<std::vector<uint32_t>>& units,
    const std::vector<uint64_t>& unit_cost,
    uint32_t n_consumers,
    uint32_t rmax) {
    uint32_t n_rows = 0;
    for (const auto& u : units) {
        n_rows += static_cast<uint32_t>(u.size());
    }
    std::vector<uint32_t> order(units.size());
    for (uint32_t i = 0; i < units.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return unit_cost[a] > unit_cost[b]; });
    for (uint32_t n_passes = std::max<uint32_t>(1, (n_rows + n_consumers * rmax - 1) / (n_consumers * rmax));;
         ++n_passes) {
        const uint32_t n_bins = n_passes * n_consumers;
        Dealt d;
        d.n_passes = n_passes;
        d.bins.assign(n_passes, std::vector<std::vector<uint32_t>>(n_consumers));
        std::vector<uint64_t> load(n_bins, 0);
        bool ok = true;
        for (uint32_t ui : order) {
            const auto& u = units[ui];
            uint32_t best = n_bins;
            for (uint32_t b = 0; b < n_bins; ++b) {
                auto& bin = d.bins[b / n_consumers][b % n_consumers];
                if (bin.size() + u.size() > rmax) {
                    continue;
                }
                // least loaded; ties -> lowest bin (pass-major), which fills pass 0 across the
                // consumers first and hands consumer c the chunks c, c+n, c+2n of a pass -- the
                // chunk-cyclic locality the streaming engine's multi-row batching relies on
                if (best == n_bins || load[b] < load[best]) {
                    best = b;
                }
            }
            if (best == n_bins) {
                ok = false;
                break;
            }
            auto& bin = d.bins[best / n_consumers][best % n_consumers];
            bin.insert(bin.end(), u.begin(), u.end());
            load[best] += unit_cost[ui];
        }
        if (!ok) {
            TT_FATAL(
                n_passes < 1024,
                "vsa_sdpa dealing: units do not fit (rows {}, consumers {}, rmax {})",
                n_rows,
                n_consumers,
                rmax);
            continue;
        }
        for (auto& pass : d.bins) {
            for (auto& bin : pass) {
                std::sort(bin.begin(), bin.end());
            }
        }
        return d;
    }
}

struct StreamSchedule {
    uint32_t head = 0;
    bool is_leader = false;
    uint32_t group_first = 0;  // core id of the group's leader
    uint32_t n_workers = 0;
    uint32_t worker_index = 0;  // valid when !is_leader
};

StreamSchedule core_schedule(uint32_t core, uint32_t num_cores, uint32_t heads) {
    uint32_t head = static_cast<uint32_t>((static_cast<uint64_t>(core) * heads) / num_cores);
    while ((static_cast<uint64_t>(head + 1) * num_cores) / heads <= core) {
        ++head;
    }
    while ((static_cast<uint64_t>(head) * num_cores) / heads > core) {
        --head;
    }
    const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(head) * num_cores) / heads);
    const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(head + 1) * num_cores) / heads);
    return StreamSchedule{head, core == c0, c0, c1 - c0 - 1, core - c0 - 1};
}

uint32_t active_workers(uint32_t group_workers, uint32_t n_q_tiles) {
    // The leader tracks per-worker coords in a fixed array, and a head never benefits from more
    // workers than it has rows; surplus cores idle (n_passes = 0, row_count = 0).
    uint32_t n = group_workers;
    if (n > kMaxWorkers) {
        n = kMaxWorkers;
    }
    if (n > n_q_tiles) {
        n = n_q_tiles;
    }
    return n == 0 ? 1 : n;
}
}  // namespace

VsaSdpaOperation::program_factory_t VsaSdpaOperation::select_program_factory(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& /*t*/) {
    if (attrs.streaming) {
        return VsaSdpaStreamProgramFactory{};
    }
    return VsaSdpaProgramFactory{};
}

tt::tt_metal::ProgramDescriptor VsaSdpaOperation::VsaSdpaStreamProgramFactory::create_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output) {
    // Function-scoped to avoid unity-build collisions with the v1 factory's enum.
    enum StreamCB : uint32_t {
        cb_q_res = 0,      // resident Q rows [R_MAX, Sqt*DHt] (RAM-mode)
        cb_k_stream,       // stream slots [depth, Skt*DHt] (RAM-mode; leader staging on leaders)
        cb_v_stream,       // stream slots [depth, Skt*vDHt]
        cb_o_res,          // resident O accumulators [R_MAX, Sqt*vDHt]
        cb_max_res,        // resident running max, ping-pong [R_MAX, 2, Sqt]
        cb_sum_res,        // resident running sum, ping-pong [R_MAX, 2, Sqt]
        cb_corr,           // per-group rescale factors [G, Sqt]
        cb_qk,             // per-group scores/probs scratch [G, Sqt*Skt]
        cb_scale,          // reduce identity scaler
        cb_col_identity,   // ones-in-col0 for the flush sum reduction
        cb_recip_scratch,  // reciprocal scratch
        cb_neginf,         // persistent -inf tile
        cb_vmask,          // ragged partial-column mask tiles (FIFO)
        cb_ctrl,           // reader -> compute visit/flush messages
        cb_kreq,           // reader -> writer K fetch/pull requests
        cb_kack,           // writer -> reader K completion
        cb_free,           // compute -> reader stream-slot credits
        cb_qdone,          // writer -> compute pass-Q-resident token
        cb_out,            // normalized row outputs (compute -> writer)
        cb_idxrow,         // reader scratch: one index row
        cb_counts,         // counts row, resident
        cb_bitmap,         // reader scratch: R_MAX x ceil(KVB/32) membership words
        cb_log,            // leader arrival log ring (leader stages, workers receive)
        cb_ackbox,         // per-worker progress words (leader-resident; workers post their count)
        cb_count
    };
    constexpr uint32_t sem_arrivals = 0;

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t W = t.indices.logical_shape()[3];
    const uint32_t block_size = attrs.block_size;

    constexpr uint32_t q_chunk_tokens = 64;
    const uint32_t DHt = d / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t Sqt = q_chunk_tokens / tt::constants::TILE_HEIGHT;
    const uint32_t Skt = block_size / tt::constants::TILE_WIDTH;
    const uint32_t n_q_tiles = S / q_chunk_tokens;
    const uint32_t n_kv_blocks = T / block_size;
    const uint32_t k_tiles_per_block = Skt * DHt;
    const uint32_t v_tiles_per_block = Skt * vDHt;
    const uint32_t q_tiles_per_row = Sqt * DHt;
    const uint32_t out_tiles_per_row = Sqt * vDHt;
    const uint32_t k_head_stride = (T / tt::constants::TILE_HEIGHT) * DHt;
    const uint32_t v_head_stride = (T / tt::constants::TILE_HEIGHT) * vDHt;
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);
    const uint32_t idx_row_bytes = W * t.indices.element_size();
    const uint32_t counts_row_bytes = t.block_counts.logical_shape()[3] * t.block_counts.element_size();
    // Residency vs. window trade (lever 1): fewer resident rows free L1 for a deeper stream ring
    // (bigger windows -> more blocks per visit -> amortized softmax rounds) at the cost of more
    // passes. Tuning knobs for the sweep; defaults are the committed values.
    uint32_t rmax = kRMax;
    uint32_t stream_depth = kStreamDepth;
    if (attrs.distributed) {
        // 2 x 4 owned (double-buffered slice) + 12 gather slots; the 5 fewer resident rows pay for the
        // 8 extra slots in L1 (a third pass at 15 s)
        rmax = 10;
        stream_depth = 22;
    }
    if (const char* e = std::getenv("TT_VSA_RMAX"); e != nullptr && e[0] != '\0') {
        rmax = static_cast<uint32_t>(std::atoi(e));
    }
    if (const char* e = std::getenv("TT_VSA_DEPTH"); e != nullptr && e[0] != '\0') {
        stream_depth = static_cast<uint32_t>(std::atoi(e));
    }
    TT_FATAL(rmax >= 1 && rmax <= 16 && stream_depth >= 8 && stream_depth % 2 == 0, "bad rmax/depth");
    const uint32_t log_depth = stream_depth + 8;  // arrival-log ring must exceed the slot ring + sentinel slack
    const bool dist = attrs.distributed;
    // widest visit (blocks) = qk region width; the streaming kernel's windows are half the ring, the
    // distributed kernel's visits are capped at 6 regardless of ring depth
    const uint32_t chunk_slots = dist ? 6u : stream_depth / 2;
    const uint32_t bitmap_bytes = ((rmax * ((n_kv_blocks + 31) / 32) * 4 + 15) / 16) * 16;
    // A VISIT message carries up to half_slots entries; the page must hold whichever is larger.
    const uint32_t ctrl_words = 4 + ((rmax > chunk_slots) ? rmax : chunk_slots);
    const uint32_t ctrl_page_bytes = (ctrl_words * 4 + 15) / 16 * 16;

    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t q_tile_bytes = tt::tile_size(q_df);
    const uint32_t k_tile_bytes = tt::tile_size(k_df);
    const uint32_t v_tile_bytes = tt::tile_size(v_df);
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    auto* device = t.q.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(num_cores >= 2 * H, "vsa_sdpa streaming needs >= 2 cores per head (H {}, cores {})", H, num_cores);
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    // The reader/writer kernels compile their LEADER and WORKER halves separately (both bodies in
    // one binary overflow the Tensix kernel-config buffer), so each is instantiated twice on
    // disjoint core sets with the role fixed by a define.
    std::set<tt::tt_metal::CoreRange> leader_ranges, worker_ranges;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord c = {i % grid.x, i / grid.x};
        if (core_schedule(i, num_cores, H).is_leader) {
            leader_ranges.insert(tt::tt_metal::CoreRange(c, c));
        } else {
            worker_ranges.insert(tt::tt_metal::CoreRange(c, c));
        }
    }
    const auto leader_grid = tt::tt_metal::CoreRangeSet(leader_ranges);
    const auto worker_grid = tt::tt_metal::CoreRangeSet(worker_ranges);

    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = sem_arrivals, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});

    // ---- CBs (fixed order = StreamCB enum; identical layout on every core is load-bearing:
    // workers address leader slots as their own CB base + slot offset) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
        return idx;
    };
    cb(q_tile_bytes, rmax * q_tiles_per_row, q_df);           // cb_q_res
    cb(k_tile_bytes, stream_depth * k_tiles_per_block, k_df);  // cb_k_stream
    cb(v_tile_bytes, stream_depth * v_tiles_per_block, v_df);  // cb_v_stream
    cb(tile_bytes, rmax * out_tiles_per_row, bf);             // cb_o_res
    cb(tile_bytes, rmax * 2 * Sqt, bf);                       // cb_max_res
    // cb_sum_res: legacy slot (the row sum is accumulated on the writer core, see vsa_sum_service.hpp)
    cb(tile_bytes, 1, bf);  // cb_sum_res (placeholder)
    // corr slots are indexed by the visit's position in its compute chunk, and a chunk can hold up
    // to half_slots single-block visits.
    cb(tile_bytes, rmax * Sqt, bf);  // cb_corr: per-row-slot candidate max tile (corr after a move)
    cb(tile_bytes, 2 * chunk_slots * Skt * Sqt, bf);           // cb_qk (two ping-pong chunk regions)
    cb(tile_bytes, 1, bf);                                     // cb_scale
    cb(tile_bytes, 1, bf);                                     // cb_col_identity
    cb(tile_bytes, 1, bf);                                     // cb_recip_scratch
    cb(tile_bytes, 1, bf);                                     // cb_neginf
    cb(tile_bytes, stream_depth, bf);                          // cb_vmask (slot-indexed RAM)
    // distributed: messages carry up to 16 visits and credits (n_pulled + 1 per message) can queue
    // ahead of the reader's reclaim, so both rings are deepened (16 B pages: a few hundred bytes)
    cb(ctrl_page_bytes, dist ? 32 : 8, bf);                    // cb_ctrl
    cb(16, dist ? 32 : stream_depth, bf);                      // cb_kreq
    cb(16, dist ? 32 : stream_depth, bf);                      // cb_kack
    cb(16, dist ? 128 : stream_depth + 2, bf);                 // cb_free
    cb(16, 2, bf);                                             // cb_qdone
    cb(out_tile_bytes, 2 * out_tiles_per_row, out_df);         // cb_out
    cb(idx_row_bytes, 1, bf);                                  // cb_idxrow
    cb(counts_row_bytes, 1, bf);                               // cb_counts
    cb(bitmap_bytes, 1, bf);                                   // cb_bitmap
    cb(16, dist ? 256 : log_depth, bf);                        // cb_log (dist: ready board + pending-message ring)
    cb(4 * 2 * kMaxWorkers, 1, bf);                            // cb_ackbox: [0,16) progress, [16,32) READY flags
    // one page of the dense-row mask tensor (its whole row) when given; a 32 B placeholder otherwise
    const uint32_t dense_bytes = t.dense_row_mask.has_value()
                                     ? t.dense_row_mask->logical_shape()[3] * t.dense_row_mask->element_size()
                                     : 32u;
    const uint32_t cb_dense = cb(dense_bytes, 1, bf);          // cb_dense: raw-selection dense-row bitmask (one page)
    // stream order row (leader only): the whole permutation, one page; 32 B placeholder when absent
    const uint32_t order_bytes = t.stream_order.has_value() ? ((n_kv_blocks * 4 + 31) / 32) * 32 : 32u;
    const uint32_t cb_order = cb(order_bytes, 1, bf);  // cb_order
    // exact row-sum service (compute -> writer): headers, partial/corr tiles, sums back, int64 accumulators
    const uint32_t cb_shdr = cb(16, 64, bf);                               // cb_shdr
    const uint32_t cb_stiles = cb(tile_bytes, 16, bf);                     // cb_stiles
    const uint32_t cb_sumback = cb(tile_bytes, 2 * Sqt, bf);               // cb_sumback
    const uint32_t cb_sacc = cb(((rmax * 64 * 8 + 31) / 32) * 32, 1, bf);  // cb_sacc: int64[rmax][64]

    // ---- compile-time args ----
    std::vector<uint32_t> reader_ct = {
        W,
        n_kv_blocks,
        n_q_tiles,
        block_size,
        rmax,
        stream_depth,
        log_depth,
        k_tiles_per_block,
        v_tiles_per_block,
        v_head_stride,
        idx_row_bytes,
        counts_row_bytes,
        k_tile_bytes,
        v_tile_bytes};
    for (uint32_t id :
         {cb_k_stream, cb_v_stream, cb_idxrow, cb_counts, cb_bitmap, cb_log, cb_ctrl, cb_kreq, cb_kack, cb_free,
          cb_vmask, cb_ackbox}) {
        reader_ct.push_back(id);
    }
    reader_ct.push_back(sem_arrivals);
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.v.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.block_counts.buffer()).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {
        n_q_tiles,
        rmax,
        q_tiles_per_row,
        out_tiles_per_row,
        k_tiles_per_block,
        k_head_stride,
        q_tile_bytes,
        k_tile_bytes,
        out_tile_bytes};
    for (uint32_t id : {cb_q_res, cb_k_stream, cb_scale, cb_col_identity, cb_neginf, cb_kreq, cb_kack, cb_qdone, cb_out}) {
        writer_ct.push_back(id);
    }
    for (uint32_t v : {cb_shdr, cb_stiles, cb_sumback, cb_sacc, Sqt}) {  // row-sum service (args 18..22)
        writer_ct.push_back(v);
    }
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(writer_ct, writer_crt);

    std::vector<uint32_t> compute_ct = {
        DHt,
        vDHt,
        Skt,
        Sqt,
        rmax,
        kRowGroup,
        block_size,
        scale_packed,
        cb_q_res,
        cb_k_stream,
        cb_v_stream,
        cb_o_res,
        cb_max_res,
        cb_sum_res,
        cb_corr,
        cb_qk,
        cb_scale,
        cb_col_identity,
        cb_recip_scratch,
        cb_neginf,
        cb_vmask,
        cb_ctrl,
        cb_free,
        cb_qdone,
        cb_out,
        stream_depth,
        chunk_slots,
        cb_shdr,
        cb_stiles,
        cb_sumback};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    // Perf triage only: TT_VSA_PROBE=1 consumes visits without math (delivery floor);
    // =2 runs QK+PV without the softmax phases (math floor); =3 is 1 plus the worker K/V pulls
    // skipped (protocol-only floor). Output is garbage in every probe mode.
    std::map<std::string, std::string> probe_defines;
    if (const char* probe = std::getenv("TT_VSA_PROBE"); probe != nullptr && probe[0] != '\0' && probe[0] != '0') {
        probe_defines["VSA_PROBE"] = probe;
    }
    static_assert((kRowChunk & (kRowChunk - 1)) == 0, "kernel chunk math shifts by log2(kRowChunk)");
    if (const char* t = std::getenv("TT_VSA_LAZY_T"); t != nullptr && t[0] != '\0') {
        probe_defines["VSA_LAZY_T_OVERRIDE"] = std::string(t);  // lazy-rescale threshold in logits (default 2)
    }
    if (const char* e = std::getenv("VSA_NO_SUMS"); e != nullptr && e[0] == '1') {
        probe_defines["VSA_NO_SUMS"] = "1";  // timing-only: no exact row-sum traffic (garbage output)
    }
    probe_defines["VSA_ROW_CHUNK_LOG2"] = std::to_string(std::bit_width(kRowChunk) - 1);

    if (dist) {
        const char* sl = std::getenv("TT_VSA_SLICE");  // owned blocks per peer per window
        probe_defines["VSA_SLICE"] = (sl != nullptr && sl[0] != '\0') ? std::string(sl) : std::string("4");
    }
    auto leader_defines = probe_defines;
    leader_defines["VSA_IS_LEADER"] = "1";

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        kdir + (dist ? "dataflow/vsa_sdpa_dist_reader.cpp" : "dataflow/vsa_sdpa_stream_reader.cpp");
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = dist ? core_grid : worker_grid;  // one role in dist mode: one instance
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    // TT_VSA_OS=1: size-optimized dataflow binaries (fits the watcher's instrumented build)
    const bool size_opt = std::getenv("TT_VSA_OS") != nullptr && std::getenv("TT_VSA_OS")[0] == '1';
    if (size_opt) {
        reader_desc.opt_level = tt::tt_metal::KernelBuildOptLevel::Os;
    }
    reader_desc.defines = tt::tt_metal::KernelDescriptor::Defines(probe_defines.begin(), probe_defines.end());

    tt::tt_metal::KernelDescriptor reader_leader_desc = reader_desc;
    reader_leader_desc.core_ranges = leader_grid;
    reader_leader_desc.defines = tt::tt_metal::KernelDescriptor::Defines(leader_defines.begin(), leader_defines.end());

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        kdir + (dist ? "dataflow/vsa_sdpa_dist_writer.cpp" : "dataflow/vsa_sdpa_stream_writer.cpp");
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = dist ? core_grid : worker_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    if (size_opt) {
        writer_desc.opt_level = tt::tt_metal::KernelBuildOptLevel::Os;
    }
    writer_desc.defines = tt::tt_metal::KernelDescriptor::Defines(probe_defines.begin(), probe_defines.end());

    tt::tt_metal::KernelDescriptor writer_leader_desc = writer_desc;
    writer_leader_desc.core_ranges = leader_grid;
    writer_leader_desc.defines = tt::tt_metal::KernelDescriptor::Defines(leader_defines.begin(), leader_defines.end());

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    TT_FATAL(Sqt * vDHt <= dst_size, "vsa_sdpa streaming needs the PV subblock ({}) to fit DEST ({})",
             Sqt * vDHt, dst_size);

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/vsa_sdpa_stream_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .math_approx_mode = math_approx};
    std::map<std::string, std::string> cdefs{
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    if (!probe_defines.empty()) {
        cdefs.insert(probe_defines.begin(), probe_defines.end());
    }
    compute_desc.defines = tt::tt_metal::KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* k_buf = t.k.buffer();
    auto* v_buf = t.v.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* counts_buf = t.block_counts.buffer();
    auto* out_buf = output.buffer();
    // ---- distributed dealing (per head group, once) ----
    // cost model: a dense row visits every real block; a sparse row its list plus the exempt ids
    std::vector<std::vector<std::vector<uint32_t>>> dist_bins;  // [pass][peer] -> rows
    uint32_t dist_n_peers = 0, dist_n_passes = 0;
    if (dist) {
        const auto sched0 = core_schedule(0, num_cores, H);
        dist_n_peers = std::min<uint32_t>(sched0.n_workers + 1, kMaxPeers);
        // every group has the same size to within one core; use the smallest so peers exist everywhere
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto sc = core_schedule(i, num_cores, H);
            if (sc.is_leader) {
                dist_n_peers = std::min<uint32_t>(dist_n_peers, sc.n_workers + 1);
            }
        }
        TT_FATAL(dist_n_peers >= 1, "vsa_sdpa distributed: empty group");
        const uint32_t sparse_cost =
            (attrs.list_len ? attrs.list_len : W) + static_cast<uint32_t>(attrs.exempt_ids.size());
        std::vector<std::vector<uint32_t>> units(n_q_tiles);
        std::vector<uint64_t> cost(n_q_tiles, sparse_cost);
        for (uint32_t r = 0; r < n_q_tiles; ++r) {
            units[r] = {r};
        }
        // a dense row visits ~9x the blocks of a sparse row in 6-block visits; a sparse row's visits
        // average ~3.5 blocks under 32-block windows -> ~5x the cost
        for (uint32_t r : attrs.dense_row_hint) {
            cost[r] = 5ull * sparse_cost;
        }
        Dealt d = deal_units_by_cost(units, cost, dist_n_peers, rmax);
        dist_n_passes = d.n_passes;
        dist_bins = std::move(d.bins);
    }

    // Streaming kernel: rows are dealt to consumers as 4-row CHUNKS (adjacent rows share most of their
    // index sets, feeding the engine's multi-row batching) by static cost, longest chunk first onto the
    // least-loaded consumer -- the leader's slot gate runs the group at its slowest core's pace, so an
    // exempt (dense) row must not stack onto a core that already holds one. Keyed by consumer count.
    std::map<uint32_t, Dealt> stream_deal;  // n_consumers -> bins[pass][consumer]
    const auto stream_deal_for = [&](uint32_t n_consumers) -> const Dealt& {
        auto it = stream_deal.find(n_consumers);
        if (it != stream_deal.end()) {
            return it->second;
        }
        const uint32_t sparse_cost =
            (attrs.list_len ? attrs.list_len : W) + static_cast<uint32_t>(attrs.exempt_ids.size());
        std::vector<bool> dense(n_q_tiles, false);
        for (uint32_t r : attrs.dense_row_hint) {
            dense[r] = true;
        }
        // Layout: the exact chunk-cyclic dealing for the sparse rows (consumer c gets chunks c, c+n,
        // c+2n ... with passes filled in order -- its per-pass chunks are 4n rows apart, the locality
        // the engine's multi-row batching relies on; a cost-balanced LPT layout measured ~1 ms slower
        // on the median shard). The dense rows are then placed one at a time into the least-loaded
        // (pass, consumer) bin with room: measured ~7x a sparse row, so two of them in one bin
        // (interleaved placement puts rows 0 and 75 on consumer 0's pass 0 under pure cyclic dealing)
        // stall the whole group behind that core.
        Dealt d;
        const uint32_t n_sparse = n_q_tiles - static_cast<uint32_t>(std::count(dense.begin(), dense.end(), true));
        std::vector<std::vector<uint32_t>> per_consumer(n_consumers);
        {
            uint32_t chunk = 0, in_chunk = 0;
            for (uint32_t r = 0; r < n_q_tiles; ++r) {
                if (dense[r]) {
                    continue;
                }
                per_consumer[chunk % n_consumers].push_back(r);
                if (++in_chunk == kRowChunk) {
                    in_chunk = 0;
                    ++chunk;
                }
            }
        }
        uint32_t max_rows = 0;
        for (const auto& v : per_consumer) {
            max_rows = std::max<uint32_t>(max_rows, static_cast<uint32_t>(v.size()));
        }
        d.n_passes = std::max<uint32_t>(1, (max_rows + rmax - 1) / rmax);
        while (d.n_passes * n_consumers * rmax < n_q_tiles) {
            ++d.n_passes;  // the dense rows need room too
        }
        d.bins.assign(d.n_passes, std::vector<std::vector<uint32_t>>(n_consumers));
        std::vector<uint64_t> load(d.n_passes * n_consumers, 0);
        for (uint32_t k = 0; k < n_consumers; ++k) {
            for (uint32_t i = 0; i < per_consumer[k].size(); ++i) {
                d.bins[i / rmax][k].push_back(per_consumer[k][i]);  // passes filled in order (first fit)
                load[(i / rmax) * n_consumers + k] += sparse_cost;
            }
        }
        (void)n_sparse;
        for (uint32_t r = 0; r < n_q_tiles; ++r) {
            if (!dense[r]) {
                continue;
            }
            uint32_t best = UINT32_MAX;
            for (uint32_t b = 0; b < load.size(); ++b) {
                if (d.bins[b / n_consumers][b % n_consumers].size() >= rmax) {
                    continue;
                }
                if (best == UINT32_MAX || load[b] < load[best]) {
                    best = b;
                }
            }
            TT_FATAL(best != UINT32_MAX, "vsa_sdpa dealing: no room for dense row {}", r);
            d.bins[best / n_consumers][best % n_consumers].push_back(r);
            load[best] += 7ull * sparse_cost;
        }
        for (auto& pass : d.bins) {
            for (auto& bin : pass) {
                std::sort(bin.begin(), bin.end());
            }
        }
        return stream_deal.emplace(n_consumers, std::move(d)).first->second;
    };

    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        const auto sched = core_schedule(i, num_cores, H);
        if (dist) {
            const uint32_t my_peer = sched.is_leader ? 0u : sched.worker_index + 1;
            const bool idle = my_peer >= dist_n_peers;
            std::vector<uint32_t> pass_rows, rows;
            if (!idle) {
                for (uint32_t p = 0; p < dist_n_passes; ++p) {
                    const auto& bin = dist_bins[p][my_peer];
                    pass_rows.push_back(static_cast<uint32_t>(bin.size()));
                    rows.insert(rows.end(), bin.begin(), bin.end());
                }
            }
            const uint32_t row_count = static_cast<uint32_t>(rows.size());
            const uint32_t n_passes = idle ? 0u : dist_n_passes;

            RtArgs reader_rt;
            reader_rt.push_back(v_buf);
            reader_rt.push_back(idx_buf);
            reader_rt.push_back(counts_buf);
            reader_rt.push_back(sched.head);
            reader_rt.push_back(my_peer);
            reader_rt.push_back(dist_n_peers);
            reader_rt.push_back(n_passes);
            reader_rt.push_back(row_count);
            reader_rt.push_back(attrs.list_len);
            reader_rt.push_back(t.dense_row_mask.has_value() ? t.dense_row_mask->buffer()->address() : 0u);  // arg 9
            reader_rt.push_back(cb_dense);
            reader_rt.push_back(dense_bytes);
            reader_rt.push_back(attrs.coarse_slots_shift);
            reader_rt.push_back(
                attrs.coarse_slots_shift ? (1u << attrs.coarse_slots_shift) - attrs.coarse_real_per_shard : 0u);
            reader_rt.push_back(static_cast<uint32_t>(attrs.exempt_ids.size()));
            for (uint32_t b : attrs.exempt_ids) {
                reader_rt.push_back(b);
            }
            // peers: the group's first dist_n_peers cores (consecutive logical ids), NoC coords
            for (uint32_t p = 0; p < dist_n_peers; ++p) {
                const uint32_t pc = sched.group_first + p;
                const auto phys = device->worker_core_from_logical_core({pc % grid.x, pc / grid.x});
                reader_rt.push_back(static_cast<uint32_t>(phys.x));
                reader_rt.push_back(static_cast<uint32_t>(phys.y));
            }
            for (uint32_t pr : pass_rows) {
                reader_rt.push_back(pr);
            }
            for (uint32_t r : rows) {
                reader_rt.push_back(r);
            }
            reader_desc.emplace_runtime_args(core, reader_rt);

            RtArgs writer_rt;
            writer_rt.push_back(out_buf);
            writer_rt.push_back(k_buf);
            writer_rt.push_back(q_buf);
            writer_rt.push_back(sched.head);
            writer_rt.push_back(n_passes);
            writer_rt.push_back(row_count);
            for (uint32_t pr : pass_rows) {
                writer_rt.push_back(pr);
            }
            for (uint32_t r : rows) {
                writer_rt.push_back(r);
            }
            writer_desc.emplace_runtime_args(core, writer_rt);

            RtArgs compute_rt;
            compute_rt.push_back(row_count);
            compute_rt.push_back(n_passes);
            for (uint32_t pr : pass_rows) {
                compute_rt.push_back(pr);
            }
            compute_desc.emplace_runtime_args(core, compute_rt);
            continue;
        }
        const uint32_t n_active = active_workers(sched.n_workers, n_q_tiles);
        const bool is_idle = !sched.is_leader && sched.worker_index >= n_active;

        const tt::tt_metal::CoreCoord leader_logical = {sched.group_first % grid.x, sched.group_first / grid.x};
        const tt::tt_metal::CoreCoord leader_phys = device->worker_core_from_logical_core(leader_logical);
        // The leader computes too (its K/V are local and its TRISCs are otherwise idle), so rows
        // are dealt over n_active workers PLUS the leader; the leader is consumer n_active.
        // Consumer 0 always holds the most rows under chunk-cyclic dealing. Small shapes are
        // excluded: with few rows per consumer the leader's extra serial work outweighs its
        // compute contribution (measured on the 5s/10s shards).
        const bool leader_computes = n_q_tiles >= 20 * (n_active + 1);
        const uint32_t n_consumers = n_active + (leader_computes ? 1u : 0u);
        const Dealt& dealt = stream_deal_for(n_consumers);
        const uint32_t n_passes = is_idle ? 0 : dealt.n_passes;
        // Chunked round-robin placement: 4-row contiguous chunks dealt cyclically across the
        // group's workers. Adjacent rows share most of their index sets (spatial correlation plus
        // the exempt prefix), so intra-chunk contiguity feeds the engine's multi-row batching,
        // while dealing chunks cyclically spreads the fully-dense exempt-query rows -- a purely
        // contiguous split piles them all onto worker 0 (measured 3x worst-shard regression).
        const uint32_t consumer_index = sched.is_leader ? n_active : sched.worker_index;
        const bool consumes = !is_idle && (leader_computes || !sched.is_leader);
        std::vector<uint32_t> my_rows, my_pass_rows;
        for (uint32_t p = 0; p < n_passes; ++p) {
            const auto& bin = consumes ? dealt.bins[p][consumer_index] : std::vector<uint32_t>{};
            my_pass_rows.push_back(static_cast<uint32_t>(bin.size()));
            my_rows.insert(my_rows.end(), bin.begin(), bin.end());
        }
        const uint32_t row_count = static_cast<uint32_t>(my_rows.size());
        const uint32_t row_start = consumer_index * kRowChunk;  // legacy arg (kernels use the row list)

        RtArgs reader_rt;
        reader_rt.reserve(13 + 2 * kMaxWorkers);
        reader_rt.push_back(v_buf);
        reader_rt.push_back(idx_buf);
        reader_rt.push_back(counts_buf);
        reader_rt.push_back(sched.head);
        reader_rt.push_back(sched.is_leader ? 1u : 0u);
        reader_rt.push_back(n_passes);
        reader_rt.push_back(static_cast<uint32_t>(leader_phys.x));
        reader_rt.push_back(static_cast<uint32_t>(leader_phys.y));
        reader_rt.push_back(n_active);
        reader_rt.push_back(sched.is_leader ? 0u : sched.worker_index);
        reader_rt.push_back(row_start);
        reader_rt.push_back(n_consumers * kRowChunk);  // chunk-cyclic big stride
        reader_rt.push_back(row_count);
        // Raw-selection mode: list length, dense-row bitmask address (0 = none) and its CB, then the
        // exempt block ids every row also attends (count first). Address patched on cache hits.
        reader_rt.push_back(attrs.list_len);
        reader_rt.push_back(t.dense_row_mask.has_value() ? t.dense_row_mask->buffer()->address() : 0u);
        reader_rt.push_back(cb_dense);
        reader_rt.push_back(dense_bytes);
        reader_rt.push_back(attrs.coarse_slots_shift);
        reader_rt.push_back(attrs.coarse_slots_shift ? (1u << attrs.coarse_slots_shift) - attrs.coarse_real_per_shard : 0u);
        reader_rt.push_back(static_cast<uint32_t>(attrs.exempt_ids.size()));
        for (uint32_t b : attrs.exempt_ids) {
            reader_rt.push_back(b);
        }
        // stream order: tensor address (0 = ascending block ids) and its CB; patched on cache hits
        reader_rt.push_back(t.stream_order.has_value() ? t.stream_order->buffer()->address() : 0u);
        reader_rt.push_back(cb_order);
        reader_rt.push_back(order_bytes);
        if (sched.is_leader) {
            // The group's workers are consecutive logical core ids in a row-major grid: they span
            // at most ceil(n/grid.x)+1 row segments, each a height-1 multicast rectangle. The
            // leader publishes log entries with ONE multicast write per segment.
            struct Strip {
                uint32_t sx, sy, ex, ey, n;
            };
            std::vector<Strip> strips;
            uint32_t w = 0;
            while (w < n_active) {
                const uint32_t wc = sched.group_first + 1 + w;
                const uint32_t row = wc / grid.x;
                uint32_t n_seg = 1;
                while (w + n_seg < n_active && (wc + n_seg) / grid.x == row) {
                    ++n_seg;
                }
                const auto p0 = device->worker_core_from_logical_core({wc % grid.x, row});
                const auto p1 = device->worker_core_from_logical_core({(wc + n_seg - 1) % grid.x, row});
                strips.push_back(
                    {static_cast<uint32_t>(p0.x),
                     static_cast<uint32_t>(p0.y),
                     static_cast<uint32_t>(p1.x),
                     static_cast<uint32_t>(p1.y),
                     n_seg});
                w += n_seg;
            }
            reader_rt.push_back(static_cast<uint32_t>(strips.size()));
            for (const auto& st : strips) {
                reader_rt.push_back(st.sx);
                reader_rt.push_back(st.sy);
                reader_rt.push_back(st.ex);
                reader_rt.push_back(st.ey);
                reader_rt.push_back(st.n);
            }
            // per-worker NoC coords: the log is published by unicast writes (strips kept for the layout)
            for (uint32_t w = 0; w < n_active; ++w) {
                const uint32_t wc = sched.group_first + 1 + w;
                const auto phys = device->worker_core_from_logical_core({wc % grid.x, wc / grid.x});
                reader_rt.push_back(static_cast<uint32_t>(phys.x));
                reader_rt.push_back(static_cast<uint32_t>(phys.y));
            }
        }
        for (uint32_t pr : my_pass_rows) {
            reader_rt.push_back(pr);
        }
        for (uint32_t r : my_rows) {
            reader_rt.push_back(r);
        }
        if (sched.is_leader) {
            reader_leader_desc.emplace_runtime_args(core, reader_rt);
        } else {
            reader_desc.emplace_runtime_args(core, reader_rt);
        }

        RtArgs writer_rt(12);
        writer_rt[0] = out_buf;
        writer_rt[1] = k_buf;
        writer_rt[2] = q_buf;
        writer_rt[3] = sched.head;
        writer_rt[4] = sched.is_leader ? 1u : 0u;
        writer_rt[5] = n_passes;
        writer_rt[6] = static_cast<uint32_t>(leader_phys.x);
        writer_rt[7] = static_cast<uint32_t>(leader_phys.y);
        writer_rt[8] = row_start;
        writer_rt[9] = n_consumers * kRowChunk;  // chunk-cyclic big stride
        writer_rt[10] = row_count;
        writer_rt[11] = 0u;
        for (uint32_t pr : my_pass_rows) {
            writer_rt.push_back(pr);
        }
        for (uint32_t r : my_rows) {
            writer_rt.push_back(r);
        }
        if (sched.is_leader) {
            writer_leader_desc.emplace_runtime_args(core, writer_rt);
        } else {
            writer_desc.emplace_runtime_args(core, writer_rt);
        }

        RtArgs compute_rt;
        compute_rt.push_back(row_count);
        compute_rt.push_back(n_passes);
        for (uint32_t pr : my_pass_rows) {
            compute_rt.push_back(pr);
        }
        compute_desc.emplace_runtime_args(core, compute_rt);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (!dist) {  // role-split instances (leader bodies) only exist for the streaming kernel
        desc.kernels.push_back(std::move(reader_leader_desc));
        desc.kernels.push_back(std::move(writer_leader_desc));
    }
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void VsaSdpaOperation::VsaSdpaStreamProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value) {
    // The reader and writer are each instantiated per ROLE on disjoint core sets, so a core's
    // buffer args live in whichever instance is placed on it. Resolve instances by kernel NAME
    // (collect_kernel_meta) and by which instance actually holds args for the core, rather than
    // trusting kernel indices -- an index assumption here silently corrupted the leaders' V/idx
    // addresses on every program-cache hit (traced model hang).
    const auto metas = tt::tt_metal::detail::collect_kernel_meta(program, nullptr);
    std::vector<uint32_t> reader_ids, writer_ids;
    for (uint32_t k = 0; k < metas.size(); ++k) {
        const std::string_view name = metas[k].name;
        if (name.find("vsa_sdpa_stream_reader") != std::string_view::npos ||
            name.find("vsa_sdpa_dist_reader") != std::string_view::npos) {
            reader_ids.push_back(k);
        } else if (
            name.find("vsa_sdpa_stream_writer") != std::string_view::npos ||
            name.find("vsa_sdpa_dist_writer") != std::string_view::npos) {
            writer_ids.push_back(k);
        }
    }
    TT_FATAL(
        reader_ids.size() == (attrs.distributed ? 1u : 2u) && writer_ids.size() == reader_ids.size(),
        "vsa_sdpa stream: unexpected kernel set");
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t v_addr = t.v.buffer()->address();
    const uint32_t idx_addr = t.indices.buffer()->address();
    const uint32_t counts_addr = t.block_counts.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();
    const uint32_t k_addr = t.k.buffer()->address();
    const uint32_t q_addr = t.q.buffer()->address();
    const uint32_t kDenseMaskArg = attrs.distributed ? 9u : 14u;  // reader: dense-mask address arg
    const uint32_t dense_addr = t.dense_row_mask.has_value() ? t.dense_row_mask->buffer()->address() : 0u;
    const uint32_t order_addr = t.stream_order.has_value() ? t.stream_order->buffer()->address() : 0u;
    const uint32_t n_exempt = static_cast<uint32_t>(attrs.exempt_ids.size());
    const uint32_t kOrderArg = attrs.distributed ? 0u : 20u + n_exempt;  // streaming reader only
    const auto patch = [&](const std::vector<uint32_t>& ids,
                           const tt::tt_metal::CoreCoord& core,
                           uint32_t a0,
                           uint32_t a1,
                           uint32_t a2,
                           bool is_reader = false) {
        for (uint32_t id : ids) {
            auto& args = tt::tt_metal::GetRuntimeArgs(program, id, core);
            if (args.size() >= 3) {  // the instance placed on this core
                args[0] = a0;
                args[1] = a1;
                args[2] = a2;
                if (is_reader) {
                    args[kDenseMaskArg] = dense_addr;
                    if (kOrderArg != 0) {
                        args[kOrderArg] = order_addr;
                    }
                }
                return;
            }
        }
        TT_FATAL(false, "vsa_sdpa stream: no kernel instance holds runtime args on core {}", core.str());
    };
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        patch(reader_ids, core, v_addr, idx_addr, counts_addr, /*is_reader=*/true);
        patch(writer_ids, core, out_addr, k_addr, q_addr);
    }
}

}  // namespace ttnn::prim
