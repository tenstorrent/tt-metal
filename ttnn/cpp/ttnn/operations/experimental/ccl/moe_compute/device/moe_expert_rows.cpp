// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <umd/device/types/arch.hpp>

#include "kernels/moe_ring_common.h"
#include "ttnn/operation.hpp"
#include "moe_compute_device_operation.hpp"
#include "moe_compute_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kTileBytesBf16 = 2048;
constexpr uint32_t kJobRows = 32;
constexpr std::array<uint32_t, 3> kBlockSlots = {8, 6, 4};  // weight blocks per reader core, tried in order
constexpr uint32_t kBlockBytes = 16 * 1024;                 // weight bytes per read block
constexpr uint32_t kTwoReadersCoresPerBank = 8;             // worker cores per DRAM bank from which k = 2
constexpr double kRingHopCycles = 350.0;                    // cost model: the ring's pass of one job, per ring position
constexpr double kCostMargin = 0.10;           // cost model: the margin the expert rows path must predict over the ring
constexpr uint64_t kRingSmallCbBytes = 4096;   // the ring tilize cores' count / flag CBs
constexpr double kFeedExpertCycles = 1500.0;   // cost model: the combine feed's handshake per active expert
constexpr double kFeedRowCyclesLocal = 200.0;  // cost model: the combine feed per row, FullLocal
constexpr double kFeedRowCyclesCcl = 400.0;    // cost model: the combine feed per row, FullCcl (fabric sends)
constexpr uint64_t kRingFitMargin = 32768;     // L1 the ring-fit estimate leaves for what it does not model
constexpr uint32_t kOutGroupTiles = moe_ring::W2_TILES_PER_A2A_ITER_W;  // 4
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/expert_rows/";

// (x slots, a2 slots) tried in order; a2 >= 3 (rows_writer.cpp)
constexpr std::array<std::pair<uint32_t, uint32_t>, 3> kBuffering = {{{2, 4}, {2, 3}, {1, 3}}};

uint32_t r64(uint32_t v) { return tt::align(v, 64u); }

// moe_compute's prepared layout of one ring position (prepare_w0_w1 / prepare_w2, moe_ring_common.h)
struct ShardLayout {
    uint32_t cols = 0, col_start = 0;       // real intermediate columns
    uint32_t out_tiles = 0, out_start = 0;  // W2 output tiles
    uint32_t groups() const { return tt::div_up(cols, 2u); }
    uint32_t out_groups() const { return tt::div_up(out_tiles, kOutGroupTiles); }
};

// The prepared weights as the expert rows reader addresses them: bank r holds ring position r's whole W0/W1 column
// pairs (w01_groups per (layer, expert), K padded to k_pad tiles) and its W2 output groups (w2_groups, N padded to
// n_pad tiles). moe_compute stores this per-core stride layout when every ring core stores the same even column count
// with 14-tile transactions (moe_ring_common.h); its compact layouts (odd or unequal column counts, 20-tile
// transactions) lay the cores' slices across bank boundaries with half-width block-columns, which per_core_stride
// marks false.
struct PreparedLayout {
    uint32_t k_pad = 0, n_pad = 0;
    uint32_t w01_groups = 0, w2_groups = 0;
    bool per_core_stride = false;
};

PreparedLayout prepared_layout(uint32_t Ht, uint32_t Nt, bool has_bias, uint32_t ring) {
    const uint32_t tiles_per_txn = moe_ring::tiles_per_txn_for_shape(Ht, Nt, has_bias, ring);
    const uint32_t block_h = moe_ring::block_tiles_h(tiles_per_txn);
    const uint32_t cols = moe_ring::w0_w1_stored_cols(Nt, 0, ring);
    PreparedLayout l;
    l.k_pad = tt::round_up(Ht + (has_bias ? 1 : 0), block_h);
    l.n_pad = tt::round_up(Nt + (has_bias ? 1 : 0), block_h);
    l.w01_groups = cols / 2;
    l.w2_groups = moe_ring::w2_num_a2a_iters(Ht, ring);
    l.per_core_stride = tiles_per_txn == moe_ring::DEFAULT_TILES_PER_TXN && cols % 2 == 0;
    for (uint32_t r = 1; r < ring; ++r) {
        l.per_core_stride = l.per_core_stride && moe_ring::w0_w1_stored_cols(Nt, r, ring) == cols;
    }
    return l;
}

std::vector<ShardLayout> shard_layouts(uint32_t Ht, uint32_t Nt, uint32_t ring) {
    std::vector<ShardLayout> layouts(ring);
    uint32_t col = 0, out = 0;
    for (uint32_t r = 0; r < ring; ++r) {
        layouts[r].cols = moe_ring::shard_tiles(Nt, r, ring);
        layouts[r].col_start = col;
        col += layouts[r].cols;
        layouts[r].out_tiles = moe_ring::w2_shard_tiles(Ht, r, Nt, ring);
        layouts[r].out_start = out;
        out += layouts[r].out_tiles;
    }
    return layouts;
}

MoEExpertRowsRoutingLayout routing_layout(const MoEExpertRowsShape& s, uint32_t np, uint32_t spc) {
    MoEExpertRowsRoutingLayout o;
    const uint32_t T = s.tokens, K = s.top_k, E = s.local_experts, NID = s.global_experts;
    const uint32_t tcap = tt::div_up(T, np);
    const uint32_t me = K;  // entries per token at most: one per k (repeated ids are listed per k)
    const uint32_t estr = tt::align(4 + 8 * tcap * me, 16u);
    uint32_t cur = 0;
    auto take = [&](uint32_t& field, uint32_t bytes) {
        field = cur;
        cur = r64(cur + bytes);
    };
    take(o.own, NID * 2);
    take(o.maps, spc * NID * 2);
    take(o.ids, tcap * s.index_page_bytes);
    take(o.scores, tcap * s.index_page_bytes);
    take(o.ctl, 16);
    take(o.slots, E * 2);
    take(o.counts, E * 2);
    take(o.offsets, E * 2);
    take(o.table, 64);
    take(o.table_counts, E * 2);
    take(o.table_offsets, E * 2);
    take(o.rows, T * K * 2);
    take(o.entry_slots, T * me * 2);
    take(o.entry_tokens, T * me * 2);
    take(o.entry_scores, T * me * 2);
    take(o.entry_k, T * me * 2);
    take(o.jobs, 3 * T * K * 2);
    take(o.areas, np * estr);
    o.size = cur;
    return o;
}

uint32_t z_bytes(uint32_t x_tiles_max) {
    return kJobRows * x_tiles_max * 64 + x_tiles_max * kTileBytesBf16 + kJobRows * 256;
}

uint32_t cb_bytes(
    const MoEExpertRowsShape& s,
    uint32_t xs,
    uint32_t a2,
    uint32_t a_tiles,
    uint32_t x_tiles_max,
    uint32_t block_tiles,
    uint32_t block_slots,
    uint32_t routing_bytes) {
    const uint32_t Ht = s.hidden_size / tt::constants::TILE_WIDTH;
    const uint32_t Nt = s.intermediate_size / tt::constants::TILE_WIDTH;
    return xs * Ht * kTileBytesBf16                                             // cb_x
           + block_slots * block_tiles * tt::tile_size(tt::DataFormat::Bfp4_b)  // cb_w
           + 2 * a_tiles * kTileBytesBf16                                       // cb_a
           + a2 * Nt * kTileBytesBf16                                           // cb_a2
           + 2 * kOutGroupTiles * kTileBytesBf16                                // cb_rows
           + 64 + routing_bytes                                                 // cb_ctl, cb_rt
           + z_bytes(x_tiles_max)                                               // cb_z
           + (s.has_bias ? kTileBytesBf16 : 0);                                 // cb_ones
}

// Bank b's cores: its NOC-0-optimal DRAM reader core first, then the nearest free cores (round-robin over banks).
std::vector<std::vector<CoreCoord>> place_cores(
    const std::vector<CoreCoord>& optimal, const std::vector<uint32_t>& counts, CoreCoord grid) {
    std::set<std::pair<uint32_t, uint32_t>> taken;
    std::vector<std::vector<CoreCoord>> out(optimal.size());
    for (uint32_t b = 0; b < optimal.size(); ++b) {
        if (counts[b]) {
            out[b].push_back(optimal[b]);
            taken.insert({optimal[b].x, optimal[b].y});
        }
    }
    const uint32_t most = *std::max_element(counts.begin(), counts.end());
    for (uint32_t step = 1; step < most; ++step) {
        for (uint32_t b = 0; b < optimal.size(); ++b) {
            if (out[b].size() >= counts[b]) {
                continue;
            }
            const int ox = optimal[b].x, oy = optimal[b].y;
            double best = std::numeric_limits<double>::max();
            CoreCoord pick{0, 0};
            for (uint32_t x = 0; x < grid.x; ++x) {
                for (uint32_t y = 0; y < grid.y; ++y) {
                    if (taken.contains({x, y})) {
                        continue;
                    }
                    const int dx = static_cast<int>(x) - ox, dy = static_cast<int>(y) - oy;
                    double cost = std::abs(dx) + std::abs(dy) + (dy != 0 ? 0.1 : 0.0) + (dx < 0 ? 0.3 : 0.0) +
                                  (dy < 0 ? 0.05 : 0.0) + (ox > 0 && dx < 0 ? 6.0 : 0.0);
                    if (cost < best) {
                        best = cost;
                        pick = CoreCoord(x, y);
                    }
                }
            }
            TT_FATAL(best < std::numeric_limits<double>::max(), "moe_compute expert rows: worker grid is full");
            out[b].push_back(pick);
            taken.insert({pick.x, pick.y});
        }
    }
    return out;
}

std::vector<uint32_t> shard_banks_of(const ttnn::Tensor& w) {
    const auto& mapping = w.buffer()->get_buffer_page_mapping();
    std::vector<uint32_t> banks;
    banks.reserve(mapping->all_cores.size());
    for (const auto& c : mapping->all_cores) {
        banks.push_back(static_cast<uint32_t>(c.x));
    }
    return banks;
}

// Virtual coordinate of every grid column / row (the kernels rebuild a core's NoC address from its linear index).
struct GridCoords {
    std::vector<uint32_t> xs, ys;
};

GridCoords grid_coords(ttnn::MeshDevice* mesh_device, CoreCoord grid) {
    GridCoords g;
    for (uint32_t x = 0; x < grid.x; ++x) {
        g.xs.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(x, 0)).x);
    }
    for (uint32_t y = 0; y < grid.y; ++y) {
        g.ys.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(0, y)).y);
    }
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            const CoreCoord v = mesh_device->worker_core_from_logical_core(CoreCoord(x, y));
            TT_FATAL(
                v.x == g.xs[x] && v.y == g.ys[y],
                "moe_compute expert rows: worker grid is not a product of rows and columns");
        }
    }
    return g;
}

std::vector<uint32_t> core_mask(const std::vector<CoreCoord>& cores, CoreCoord grid) {
    std::vector<uint32_t> words(tt::div_up(grid.x * grid.y, 32u), 0);
    for (const auto& c : cores) {
        const uint32_t li = c.x * grid.y + c.y;
        words[li / 32] |= 1u << (li % 32);
    }
    return words;
}

uint32_t to_u32(float v) { return std::bit_cast<uint32_t>(v); }

// mapping rows of the devices whose tokens this device receives (moe_compute tilize_reader: the devices of the
// dispatch axis through this device, in axis order)
std::vector<uint32_t> source_rows(
    const ttnn::MeshDevice& mesh_device, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const auto& view = mesh_device.get_view();
    const uint32_t rows = view.num_rows(), cols = view.num_cols();
    std::vector<uint32_t> out;
    if (axis == 1) {
        for (uint32_t j = 0; j < cols; ++j) {
            out.push_back(coord[0] * cols + j);
        }
    } else {
        for (uint32_t i = 0; i < rows; ++i) {
            out.push_back(i * cols + coord[1]);
        }
    }
    return out;
}

uint32_t aligned_page(const ttnn::Tensor& t) { return t.buffer()->aligned_page_size(); }

// The weight-stream reader / compute pairing of the expert-row program per architecture; the routing, exchange, row
// writer and metadata kernels are shared. Wormhole and Blackhole use the same pair: their plans differ only in numbers
// (readers per bank, packets per read block) that plan_moe_expert_rows derives from the device.
struct StreamKernels {
    const char* reader;
    const char* compute;
};

std::optional<StreamKernels> stream_kernels(tt::ARCH arch) {
    if (arch == tt::ARCH::BLACKHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        return StreamKernels{.reader = "rows_reader.cpp", .compute = "rows_compute.cpp"};
    }
    return std::nullopt;
}

// Peak DRAM bandwidth (bytes / s): ttnn's OpPerformanceModel figures (ttnn/core/operation.cpp), which are GiB / s.
double dram_peak_bytes_per_s(tt::ARCH arch) {
    return (arch == tt::ARCH::BLACKHOLE ? 512.0 : 258.0) * static_cast<double>(1ull << 30);
}

// Expected jobs (at most 32 rows of one expert) of one call: E_loc experts, each routed by each of T tokens with
// probability top_k / global_experts (uniform routing).
double expected_jobs(const MoEExpertRowsShape& s) {
    const double p = std::min(1.0, static_cast<double>(s.top_k) / s.global_experts);
    const uint32_t T = s.tokens;
    if (p >= 1.0) {
        return s.local_experts * std::ceil(T / static_cast<double>(kJobRows));
    }
    // sum over j >= 1 of P(rows > 32 (j - 1)), rows ~ Binomial(T, p)
    double pmf = std::pow(1.0 - p, T), cdf = 0.0, jobs = 0.0;
    for (uint32_t n = 0; n <= T; ++n) {
        cdf += pmf;
        if (n % kJobRows == 0) {
            jobs += std::max(0.0, 1.0 - cdf);  // P(rows > n)
        }
        pmf *= (static_cast<double>(T - n) / (n + 1)) * (p / (1.0 - p));
    }
    return s.local_experts * jobs;
}

// Program 2's routing-table offsets (rows_reader.cpp's table block for this shape) and place.cpp's scratch layout.
struct PlaceLayout {
    uint32_t t_counts = 0, t_offsets = 0, t_rows = 0, t_es = 0, t_et = 0, t_ec = 0, t_ek = 0, t_bytes = 0;
    uint32_t s_counts = 0, s_offsets = 0, s_et = 0, s_act = 0, s_crow = 0, s_table = 0, scratch = 0;
    uint32_t count_words = 0, e_t_page = 0, activation_row = 0;
};

PlaceLayout place_layout(uint32_t T, uint32_t K, uint32_t E) {
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t me = K;  // token-ordered entries: one per k
    PlaceLayout o;
    uint32_t cur = 0;
    auto next = [&](uint32_t bytes) {
        const uint32_t at = cur;
        cur = r64(cur + bytes);
        return at;
    };
    next(64);
    o.t_counts = next(E * 2);
    o.t_offsets = next(E * 2);
    o.t_rows = next(T * K * 2);
    o.t_es = next(T * me * 2);
    o.t_et = next(T * me * 2);
    o.t_ec = next(T * me * 2);
    o.t_ek = next(T * me * 2);
    o.t_bytes = cur;
    o.count_words = tt::align(E * 4, l1_align) / 4;
    o.e_t_page = (T + 1) * tt::align(4u, l1_align);
    o.activation_row = tt::align((2 * E + 1) * 4, l1_align);
    cur = 0;
    o.s_counts = next(E * 4);
    o.s_offsets = next(E * 4);
    o.s_et = next(o.e_t_page);
    o.s_act = next(8 * o.activation_row);
    o.s_crow = next(o.count_words * 4);
    o.s_table = next(o.t_bytes);
    o.scratch = cur;
    return o;
}

// feed.cpp's scratch: the table head, then two staging buffers of `batch` row slices (one ring position's W2 output
// columns of a row).
struct FeedLayout {
    uint32_t slice = 0, batch = 0, stage = 0, bytes = 0;
};

FeedLayout feed_layout(const PlaceLayout& p, const std::vector<ShardLayout>& layouts, uint32_t tile_width_bytes) {
    constexpr uint32_t kStageBytes = 16 * 1024;  // per staging buffer
    FeedLayout f;
    for (const auto& l : layouts) {
        f.slice = std::max(f.slice, l.out_tiles * tile_width_bytes);
    }
    f.slice = r64(f.slice);
    f.batch = std::clamp(kStageBytes / f.slice, 1u, kJobRows);
    f.stage = r64(p.t_rows);
    f.bytes = f.stage + 2 * f.batch * f.slice;
    return f;
}

}  // namespace

std::optional<MoEExpertRowsPlan> plan_moe_expert_rows(
    ttnn::MeshDevice* mesh_device,
    const MoEExpertRowsShape& s,
    const std::vector<uint32_t>& shard_banks,
    uint32_t cb_budget,
    std::string& refusal) {
    const uint32_t Ht = s.hidden_size / tt::constants::TILE_WIDTH;
    const uint32_t Nt = s.intermediate_size / tt::constants::TILE_WIDTH;
    const std::vector<CoreCoord> optimal =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    const uint32_t ring = shard_banks.size();
    if (ring != optimal.size()) {
        refusal = "the prepared weights have " + std::to_string(ring) + " shards for " +
                  std::to_string(optimal.size()) + " DRAM banks";
        return std::nullopt;
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    // a read block is kBlockBytes of whole 4-tile K rows in as many NoC packets as that takes; one packet per block
    // when the larger block does not fit
    const uint32_t wtile = tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t packet_tiles = burst / wtile;
    std::vector<uint32_t> packet_counts = {std::max(1u, kBlockBytes / burst)};
    if (packet_counts[0] > 1) {
        packet_counts.push_back(1);
    }
    if (packet_tiles < 4) {
        refusal = "a NoC packet holds less than one K row of four weight tiles";
        return std::nullopt;
    }
    const auto layout = shard_layouts(Ht, Nt, ring);
    uint32_t most_groups = 0;
    for (const auto& l : layout) {
        if (l.cols == 0) {
            refusal = "a DRAM bank holds no intermediate column";
            return std::nullopt;
        }
        if (l.out_tiles == 0) {
            refusal = "a DRAM bank holds no W2 output tile";
            return std::nullopt;
        }
        most_groups = std::max(most_groups, l.groups());
    }
    if (!prepared_layout(Ht, Nt, s.has_bias, ring).per_core_stride) {
        refusal = "the prepared weights are in a compact layout, which the expert rows reader does not address";
        return std::nullopt;
    }
    const uint32_t tps = s.tokens / s.num_sources;
    const uint32_t p1_run = 4 * (Ht + (s.has_bias ? 1 : 0));
    const uint32_t p2_run = 4 * (Nt + (s.has_bias ? 1 : 0));

    // reader cores per bank: 1 while the grid has fewer than kTwoReadersCoresPerBank worker cores per DRAM bank,
    // else 2; a larger split (up to the busiest bank's column groups) only when fewer block slots do not fit
    std::vector<uint32_t> ks;
    for (const uint32_t k : {grid_cores / ring < kTwoReadersCoresPerBank ? 1u : 2u, 2u, most_groups}) {
        if (k <= most_groups && std::find(ks.begin(), ks.end(), k) == ks.end()) {
            ks.push_back(k);
        }
    }
    for (const uint32_t block_packets : packet_counts) {
        for (const uint32_t k : ks) {
            const uint32_t block_tiles = std::min(block_packets * packet_tiles, kBlockBytes / wtile) / 4 * 4;
            std::vector<MoEExpertRowsCore> members;
            for (uint32_t r = 0; r < ring; ++r) {
                const ShardLayout& l = layout[r];
                const uint32_t kb = std::min(k, l.groups());
                std::vector<uint32_t> per(kb), load(kb), outs(kb, 0);
                for (uint32_t m = 0; m < kb; ++m) {
                    per[m] = l.groups() / kb + (m < l.groups() % kb ? 1 : 0);
                    load[m] = per[m] * p1_run;
                }
                for (uint32_t q = 0; q < l.out_groups(); ++q) {
                    const uint32_t m = std::min_element(load.begin(), load.end()) - load.begin();
                    ++outs[m];
                    load[m] += p2_run;
                }
                uint32_t g0 = 0, q0 = 0;
                for (uint32_t m = 0; m < kb; ++m) {
                    MoEExpertRowsCore c;
                    c.ring_pos = r;
                    c.bank = shard_banks[r];
                    c.g0 = g0;
                    c.ng = per[m];
                    c.c0 = l.col_start + 2 * g0;
                    c.na = std::min(l.cols - 2 * g0, 2 * per[m]);
                    c.q0 = q0;
                    c.nq = outs[m];
                    c.n0 = l.out_start + kOutGroupTiles * q0;
                    c.nout = outs[m] ? std::min(l.out_tiles - kOutGroupTiles * q0, kOutGroupTiles * outs[m]) : 0;
                    members.push_back(c);
                    g0 += per[m];
                    q0 += outs[m];
                }
            }
            const uint32_t nc = members.size();
            if (nc > grid_cores) {
                continue;
            }
            uint32_t a_tiles = 0, x_tiles_max = 0;
            for (uint32_t i = 0; i < nc; ++i) {
                members[i].x0 = i * Ht / nc;
                members[i].xn = (i + 1) * Ht / nc - members[i].x0;
                a_tiles = std::max(a_tiles, 2 * members[i].ng);
                x_tiles_max = std::max(x_tiles_max, members[i].xn);
            }
            for (const uint32_t block_slots : kBlockSlots) {
                for (const auto& [xs, a2] : kBuffering) {
                    for (uint32_t G = grid_cores / nc; G >= 1; --G) {
                        const uint32_t np = G * nc;
                        const uint32_t spc = tt::div_up(tt::div_up(s.tokens, np), tps) + 1;
                        const auto rt = routing_layout(s, np, spc);
                        const uint32_t bytes =
                            cb_bytes(s, xs, a2, a_tiles, x_tiles_max, block_tiles, block_slots, rt.size);
                        if (bytes > cb_budget) {
                            continue;
                        }
                        MoEExpertRowsPlan p;
                        p.readers_per_bank = k;
                        p.groups = G;
                        p.cores_per_group = nc;
                        p.x_slots = xs;
                        p.a2_slots = a2;
                        p.a_tiles = a_tiles;
                        p.x_tiles_max = x_tiles_max;
                        p.block_tiles = block_tiles;
                        p.block_packets = block_packets;
                        p.block_slots = block_slots;
                        p.sources_per_core = spc;
                        p.cb_bytes = bytes;
                        p.routing = rt;
                        // G copies of the member split, cores near their bank's optimal reader core, group-major
                        std::vector<uint32_t> counts(ring, 0);
                        for (const auto& m : members) {
                            counts[m.ring_pos] += G;
                        }
                        std::vector<uint32_t> bank_counts(optimal.size(), 0);
                        for (uint32_t r = 0; r < ring; ++r) {
                            bank_counts[shard_banks[r]] = counts[r];
                        }
                        const auto placed = place_cores(optimal, bank_counts, grid);
                        std::vector<uint32_t> member_index(ring, 0);
                        std::vector<uint32_t> first_member(ring, 0);
                        for (uint32_t i = nc; i-- > 0;) {
                            first_member[members[i].ring_pos] = i;
                        }
                        p.blocks_in_flight = *std::max_element(bank_counts.begin(), bank_counts.end()) >= 3 ? 1 : 2;
                        for (uint32_t gi = 0; gi < G; ++gi) {
                            for (const auto& m : members) {
                                MoEExpertRowsCore c = m;
                                const uint32_t i = (&m - members.data() - first_member[m.ring_pos]) * G + gi;
                                c.group = gi;
                                c.core = placed[m.bank][i];
                                p.cores.push_back(c);
                            }
                        }
                        return p;
                    }
                }
            }
        }
    }
    refusal = "no core split fits the circular-buffer budget of " + std::to_string(cb_budget) + " B per core";
    return std::nullopt;
}

//-----------------------------------------------------------------------------------------------------------------
// Selection between the ring program and the expert rows programs
//-----------------------------------------------------------------------------------------------------------------
std::optional<MoEExpertRowsParams> select_moe_compute_expert_rows(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args) {
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    // test and benchmark hook: TT_METAL_MOE_COMPUTE_KERNEL=ring forces the ring; =expert_rows forces the expert rows
    // path (overrides the cost model and the Wormhole gate, and refuses the call with the reason when the expert rows
    // path cannot run it)
    const char* forced_kernel = std::getenv("TT_METAL_MOE_COMPUTE_KERNEL");
    const bool forced_expert_rows = forced_kernel != nullptr && std::string(forced_kernel) == "expert_rows";
    auto ring_because = [&](const std::string& why) -> std::optional<MoEExpertRowsParams> {
        TT_FATAL(
            !forced_expert_rows,
            "moe_compute: TT_METAL_MOE_COMPUTE_KERNEL=expert_rows, but the expert rows path cannot run: {}",
            why);
        log_debug(tt::LogOp, "moe_compute: ring kernel ({})", why);
        return std::nullopt;
    };
    if (forced_kernel != nullptr && std::string(forced_kernel) == "ring") {
        return ring_because("TT_METAL_MOE_COMPUTE_KERNEL=ring");
    }
    if (!stream_kernels(mesh_device->arch()).has_value()) {
        return ring_because("no expert rows weight-stream variant for this architecture");
    }
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE && !forced_expert_rows) {
        return ring_because(
            "Wormhole runs the expert rows path only under the test hook until its Galaxy A/B and CI pass");
    }
    const auto& x = tensor_args.tilize_input_tensor;
    const auto& idx = tensor_args.tilize_expert_indices_tensor;
    const auto& sc = tensor_args.tilize_expert_scores_tensor;
    const auto& map = tensor_args.tilize_expert_mapping_tensor;
    const auto& w01 = tensor_args.matmul_w0_w1_tensor;
    const auto& w2 = tensor_args.matmul_w2_tensor;
    if (x.logical_shape().rank() < 3 || idx.logical_shape().rank() < 2 || map.logical_shape().rank() != 2 ||
        w01.logical_shape().rank() != 6 || w2.logical_shape().rank() != 6 || x.layout() != Layout::ROW_MAJOR ||
        idx.layout() != Layout::ROW_MAJOR || sc.layout() != Layout::ROW_MAJOR || map.layout() != Layout::ROW_MAJOR ||
        w01.dtype() != DataType::BFLOAT4_B || w2.dtype() != DataType::BFLOAT4_B ||
        w01.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED ||
        w2.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return ring_because("inputs outside the expert rows layout rules");
    }
    if (args.num_shared_experts_per_device.value_or(0) > 0 && args.combine_params.has_value()) {
        return ring_because("shared experts split over the mesh");
    }
    MoEExpertRowsShape shape;
    shape.hidden_size = x.logical_shape()[-1];
    shape.intermediate_size = args.intermediate_size;
    shape.local_experts = w01.logical_shape()[2];
    shape.top_k = idx.logical_shape()[-1];
    shape.tokens = x.logical_shape()[0] * x.logical_shape()[1];
    shape.global_experts = map.logical_shape()[-1];
    const uint32_t axis = args.cluster_axis().value_or(1);
    const auto& view = mesh_device->get_view();
    shape.num_sources = axis == 1 ? view.num_cols() : view.num_rows();
    shape.index_page_bytes = aligned_page(idx);
    shape.has_bias = args.has_bias;
    // rows kept per local expert: its e_t page (one entry per token) and its double-buffer half (out 3: per core
    // 2 halves of ntp-way split rows of H / dp values, read with the combine's per-shard row stride)
    const auto specs = MoEComputeDeviceOperation::compute_output_specs(args, tensor_args);
    {
        const uint32_t H = shape.hidden_size, dp = args.num_data_parallel_cores, ntp = args.num_token_parallel_cores;
        const auto& dense = specs[3];
        const uint32_t per_core = dense.logical_shape().volume() / dense.memory_config().shard_spec()->grid.num_cores();
        const uint32_t combine_stride = per_core / (H / dp / 2) / ntp;  // selective_reduce_combine's row offset
        const uint32_t fits = per_core / 2 / (H / dp);                  // rows of one half that fit one core
        shape.row_cap = std::min(shape.tokens, ntp * std::min(combine_stride, fits));
    }
    if (aligned_page(sc) != shape.index_page_bytes || shape.tokens % shape.num_sources != 0 ||
        shape.tokens * shape.top_k > 65535) {
        return ring_because("routing tensors outside the expert rows rules");
    }
    // circular buffers end below the lowest live L1 buffer (and below L1_SMALL); bucketed, it is a cache key
    const auto& alloc = mesh_device->allocator();
    const uint64_t base = alloc->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t top = std::min<uint64_t>(
        mesh_device->lowest_occupied_compute_l1_address().value_or(std::numeric_limits<uint64_t>::max()),
        ttnn::ccl::l1_small_floor_address(*mesh_device));
    constexpr uint32_t kBudgetBucket = 16 * 1024;
    const uint32_t budget = top > base ? static_cast<uint32_t>((top - base) / kBudgetBucket * kBudgetBucket) : 0;
    std::string refusal;
    const auto plan = plan_moe_expert_rows(mesh_device, shape, shard_banks_of(w01), budget, refusal);
    if (!plan.has_value()) {
        return ring_because(refusal);
    }
    // program 2 runs after moe_compute's outputs are allocated: its CBs end below them (per-bank upper bound)
    uint64_t outputs_per_core = 0;
    const uint32_t l1_banks = alloc->get_num_banks(tt::tt_metal::BufferType::L1);
    for (uint32_t i = 0; i < specs.size(); ++i) {
        if (i != 4 && specs[i].memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {  // 4 aliases 3
            outputs_per_core +=
                specs[i].compute_consumed_memory_bytes_per_bank(tt::tt_metal::hal::get_l1_alignment(), l1_banks);
        }
    }
    const PlaceLayout pl = place_layout(shape.tokens, shape.top_k, shape.local_experts);
    uint64_t program2_cb_bytes = pl.scratch;
    if (args.path != MoEComputePath::ComputeOnly) {
        program2_cb_bytes += feed_layout(
                                 pl,
                                 shard_layouts(
                                     shape.hidden_size / tt::constants::TILE_WIDTH,
                                     shape.intermediate_size / tt::constants::TILE_WIDTH,
                                     shard_banks_of(w01).size()),
                                 tt::constants::TILE_WIDTH * 2)
                                 .bytes;
    }
    if (program2_cb_bytes + outputs_per_core > budget) {
        return ring_because(fmt::format(
            "program 2 CBs ({} B) and moe_compute's L1 outputs ({} B per core) exceed the {} B CB budget",
            program2_cb_bytes,
            outputs_per_core,
            budget));
    }
    // Can the ring program run this call at all? Its tilize cores hold token-sized CBs (e_t lists, activation rows,
    // the second RISC's halves of both, the mapping, a 32-token input chunk) below the same outputs; when they do not
    // fit, the ring program refuses the call and the caller would have to split it, so the expert rows path runs it.
    const auto ring_cores = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        args.num_token_parallel_cores,
        args.num_data_parallel_cores,
        shape.hidden_size,
        args.combine_params.has_value() ? args.combine_params->mux_core_range_set : CoreRangeSet{},
        args.bh_ring_size);
    const uint64_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    const uint64_t T = shape.tokens, E = shape.local_experts;
    const uint64_t arow = tt::align((2 * E + 1) * 4, l1_align);
    const uint64_t tilize_cores = std::max<size_t>(1, ring_cores.tilize_cores.size());
    const uint64_t x_page = x.buffer()->aligned_page_size();
    const uint64_t ring_tilize_cb_bytes = E * specs[2].compute_page_size_bytes()  // e_t lists
                                          + T * arow                              // activation rows
                                          + tt::div_up(T, 2) * l1_align * E       // second RISC's e_t
                                          + tt::div_up(T, 2) * arow               // second RISC's rows
                                          + map.buffer()->num_pages() * map.buffer()->aligned_page_size()     // mapping
                                          + kJobRows * tt::align(tt::div_up(x_page, tilize_cores), l1_align)  // input
                                          + kRingSmallCbBytes;
    const bool ring_fits = ring_tilize_cb_bytes + outputs_per_core + kRingFitMargin <= budget;
    // Cost model. Both programs stream each touched expert's weights once per job of at most 32 of its rows, the same
    // bytes; a bank's weights come at 1 / banks of the DRAM peak, so the busiest bank sets the stream time S of a job.
    // The ring computes each job on one core per bank and pays a ring pass per job; the expert rows path spreads the
    // work over readers_per_bank cores per bank and runs G expert groups in parallel. With the combine (FullLocal,
    // FullCcl) the ring overlaps it with its compute, while the expert rows path feeds it one expert at a time after
    // its rows exist, so that path also pays the feed: per active expert and per row.
    //   t_ring = J (max(S, ring tiles * c) + ring * hop)
    //   t_expert_rows = max(J S, ceil(J / G) busiest * c) [+ active experts * feed_expert + rows * feed_row]
    // c: seconds per tile matmul (16 cycles per fidelity phase at the device clock); hop, feed_expert, feed_row: cycle
    // counts at the clock; J: expected jobs from the rows per expert (top_k / experts of each token). The expert rows
    // path runs only with a clear predicted margin (kCostMargin); within it the ring is kept.
    const uint32_t Ht = shape.hidden_size / tt::constants::TILE_WIDTH;
    const uint32_t Nt = shape.intermediate_size / tt::constants::TILE_WIDTH;
    const uint32_t ring = shard_banks_of(w01).size();
    const uint32_t kd = Ht + (shape.has_bias ? 1 : 0), nd = Nt + (shape.has_bias ? 1 : 0);
    const double clock_hz = mesh_device->get_clock_rate_mhz() * 1e6;
    const double tile_s =
        16.0 * tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(args.math_fidelity) / clock_hz;
    std::map<uint32_t, double> bank_tiles;
    double busiest = 0;
    for (const auto& c : plan->cores) {
        const double tiles = 4.0 * (c.ng * kd + c.nq * nd);
        busiest = std::max(busiest, tiles);
        if (c.group == 0) {
            bank_tiles[c.bank] += tiles;
        }
    }
    double busiest_bank = 0;
    for (const auto& [bank, tiles] : bank_tiles) {
        busiest_bank = std::max(busiest_bank, tiles);
    }
    const double stream_s =
        busiest_bank * ring * tt::tile_size(tt::DataFormat::Bfp4_b) / dram_peak_bytes_per_s(mesh_device->arch());
    uint32_t ring_cols = 0;  // gate/up columns of the ring's busiest core (moe_ring_common.h)
    for (uint32_t r = 0; r < ring; ++r) {
        ring_cols = std::max(ring_cols, moe_ring::w0_w1_stored_cols(Nt, r, ring));
    }
    const double ring_tiles = 2.0 * kd * ring_cols + 4.0 * moe_ring::w2_num_a2a_iters(Ht, ring) * nd;
    const double jobs = expected_jobs(shape);
    const double t_ring = jobs * (std::max(stream_s, ring_tiles * tile_s) + ring * kRingHopCycles / clock_hz);
    double t_expert_rows = std::max(jobs * stream_s, std::ceil(jobs / plan->groups) * busiest * tile_s);
    if (args.path != MoEComputePath::ComputeOnly) {
        const double p = std::min(1.0, static_cast<double>(shape.top_k) / shape.global_experts);
        const double active_experts = shape.local_experts * (1.0 - std::pow(1.0 - p, shape.tokens));
        const double rows =
            static_cast<double>(shape.tokens) * shape.top_k * shape.local_experts / shape.global_experts;
        const double row_cycles = args.path == MoEComputePath::FullCcl ? kFeedRowCyclesCcl : kFeedRowCyclesLocal;
        t_expert_rows += (active_experts * kFeedExpertCycles + rows * row_cycles) / clock_hz;
    }
    if (ring_fits && t_expert_rows * (1.0 + kCostMargin) >= t_ring && !forced_expert_rows) {
        return ring_because(fmt::format(
            "cost model: ring {:.1f} us, expert rows {:.1f} us for {:.1f} expected jobs, no clear margin",
            t_ring * 1e6,
            t_expert_rows * 1e6,
            jobs));
    }
    log_debug(
        tt::LogOp,
        "moe_compute: expert rows kernel (cost model: expert rows {:.1f} us, ring {:.1f} us for {:.1f} expected jobs, "
        "ring program {} L1; {} readers per bank, {} expert groups of {} cores, {} B CBs)",
        t_expert_rows * 1e6,
        t_ring * 1e6,
        jobs,
        ring_fits ? "fits" : "does not fit",
        plan->readers_per_bank,
        plan->groups,
        plan->cores_per_group,
        plan->cb_bytes);
    return MoEExpertRowsParams{
        .shape = shape,
        .layer_id = args.layer_id,
        .cluster_axis = axis,
        .cb_budget = budget,
        .activation_type = args.activation_type,
        .activation_limit = args.activation_limit,
        .math_fidelity = args.math_fidelity,
        .fp32_dest_acc_en = args.fp32_dest_acc_en};
}

//-----------------------------------------------------------------------------------------------------------------
// Program 1: expert rows
//-----------------------------------------------------------------------------------------------------------------
MoEExpertRowsDeviceOperation::program_factory_t MoEExpertRowsDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEExpertRowsFactory{};
}

ttsl::hash::hash_t MoEExpertRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // the layer only offsets the weight addresses (runtime arguments): one program serves every layer
    operation_attributes_t keyed = args;
    keyed.layer_id = 0;
    return ttsl::hash::hash_objects_with_default_seed(keyed, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& s = args.shape;
    TT_FATAL(
        s.tokens % s.num_sources == 0, "moe_compute expert rows: {} tokens over {} sources", s.tokens, s.num_sources);
    TT_FATAL(s.tokens * s.top_k <= 65535, "moe_compute expert rows: tokens * top_k must fit 16 bits");
    TT_FATAL(
        tensor_args.w0_w1_tensor.dtype() == DataType::BFLOAT4_B && tensor_args.w2_tensor.dtype() == DataType::BFLOAT4_B,
        "moe_compute expert rows: weights must be bfloat4_b");
    TT_FATAL(
        aligned_page(tensor_args.expert_indices_tensor) == s.index_page_bytes &&
            aligned_page(tensor_args.expert_scores_tensor) == s.index_page_bytes,
        "moe_compute expert rows: indices and scores must share one page size");
    TT_FATAL(
        shard_banks_of(tensor_args.w0_w1_tensor) == shard_banks_of(tensor_args.w2_tensor),
        "moe_compute expert rows: W0/W1 and W2 shards must sit on the same banks");
    TT_FATAL(
        args.layer_id < tensor_args.w0_w1_tensor.logical_shape()[1],
        "moe_compute expert rows: layer_id {} is out of range for {} layers of weights",
        args.layer_id,
        tensor_args.w0_w1_tensor.logical_shape()[1]);
}

MoEExpertRowsDeviceOperation::spec_return_value_t MoEExpertRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& s = args.shape;
    auto* mesh_device = tensor_args.input_tensor.device();
    std::string refusal;
    const auto plan =
        plan_moe_expert_rows(mesh_device, s, shard_banks_of(tensor_args.w0_w1_tensor), args.cb_budget, refusal);
    TT_FATAL(plan.has_value(), "moe_compute expert rows: {}", refusal);
    const uint32_t rows = std::min(s.tokens * s.top_k, s.local_experts * s.row_cap);
    const auto rows_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({rows, s.hidden_size}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    const auto table_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, plan->routing.table_bytes() / 2}),
        tt::tt_metal::TensorLayout(
            DataType::UINT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    return {rows_spec, table_spec};
}

MoEExpertRowsDeviceOperation::tensor_return_value_t MoEExpertRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

ttnn::device_operation::CachedProgram<MoEExpertRowsFactory::shared_variables_t> MoEExpertRowsFactory::create_at(
    const MoEExpertRowsParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using tt::tt_metal::CBHandle;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto* mesh_device = tensor_args.input_tensor.device();
    const auto& s = args.shape;
    const uint32_t Ht = s.hidden_size / tt::constants::TILE_WIDTH;
    const uint32_t Nt = s.intermediate_size / tt::constants::TILE_WIDTH;
    const auto shard_banks = shard_banks_of(tensor_args.w0_w1_tensor);
    const uint32_t ring = shard_banks.size();
    std::string refusal;
    const auto plan_opt = plan_moe_expert_rows(mesh_device, s, shard_banks, args.cb_budget, refusal);
    TT_FATAL(plan_opt.has_value(), "moe_compute expert rows: {}", refusal);
    const auto kernels_opt = stream_kernels(mesh_device->arch());
    TT_FATAL(kernels_opt.has_value(), "moe_compute expert rows: no weight-stream variant for this architecture");
    const StreamKernels& kernels = *kernels_opt;
    const MoEExpertRowsPlan& plan = *plan_opt;
    const auto& rt = plan.routing;
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords vc = grid_coords(mesh_device, grid);
    const auto& view = mesh_device->get_view();
    const uint32_t device_id = mesh_coordinate[0] * view.num_cols() + mesh_coordinate[1];
    const auto sources = source_rows(*mesh_device, mesh_coordinate, args.cluster_axis);
    TT_FATAL(
        sources.size() == s.num_sources,
        "moe_compute expert rows: {} sources, expected {}",
        sources.size(),
        s.num_sources);

    const ttnn::Tensor& rows_tensor = tensor_return_value[0];
    const ttnn::Tensor& table_tensor = tensor_return_value[1];

    // prepared layout strides (tiles)
    const uint32_t wtile = tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint32_t kd = Ht + (s.has_bias ? 1 : 0), nd = Nt + (s.has_bias ? 1 : 0);
    const PreparedLayout layout = prepared_layout(Ht, Nt, s.has_bias, ring);
    const uint32_t kp = layout.k_pad, npad = layout.n_pad;
    const uint32_t p1_groups = layout.w01_groups, p2_groups = layout.w2_groups;
    const auto& w01_shape = tensor_args.w0_w1_tensor.logical_shape();
    const auto& w2_shape = tensor_args.w2_tensor.logical_shape();
    TT_FATAL(
        w01_shape[-3] == p1_groups && w01_shape[-2] == kp * tt::constants::TILE_HEIGHT && w2_shape[-3] == p2_groups &&
            w2_shape[-2] == npad * tt::constants::TILE_HEIGHT,
        "moe_compute expert rows: weights do not have the prepared layout of H {} / N {} on {} banks",
        s.hidden_size,
        s.intermediate_size,
        ring);
    const uint32_t layer_w01 = args.layer_id * s.local_experts * p1_groups * kp * 4;
    const uint32_t layer_w2 = args.layer_id * s.local_experts * p2_groups * npad * 4;

    std::vector<CoreCoord> all_cores;
    for (const auto& c : plan.cores) {
        all_cores.push_back(c.core);
    }
    const CoreRangeSet all_set = CoreRangeSet(ttsl::Span<const CoreCoord>(all_cores));
    const std::vector<uint32_t> all_mask = core_mask(all_cores, grid);

    // circular buffers
    enum Cb : uint32_t { CB_X, CB_W, CB_A, CB_A2, CB_ROWS, CB_CTL, CB_RT, CB_Z, CB_ONES };
    auto make_cb = [&](uint32_t index, uint32_t bytes, tt::DataFormat fmt, uint32_t page) {
        return tt::tt_metal::CreateCircularBuffer(
            program, all_set, tt::tt_metal::CircularBufferConfig(bytes, {{index, fmt}}).set_page_size(index, page));
    };
    make_cb(CB_X, plan.x_slots * Ht * kTileBytesBf16, tt::DataFormat::Float16_b, kTileBytesBf16);
    make_cb(CB_W, plan.block_slots * plan.block_tiles * wtile, tt::DataFormat::Bfp4_b, wtile);
    make_cb(CB_A, 2 * plan.a_tiles * kTileBytesBf16, tt::DataFormat::Float16_b, kTileBytesBf16);
    make_cb(CB_A2, plan.a2_slots * Nt * kTileBytesBf16, tt::DataFormat::Float16_b, kTileBytesBf16);
    make_cb(CB_ROWS, 2 * kOutGroupTiles * kTileBytesBf16, tt::DataFormat::Float16_b, kTileBytesBf16);
    make_cb(CB_CTL, 64, tt::DataFormat::UInt32, 64);
    make_cb(CB_RT, rt.size, tt::DataFormat::UInt32, rt.size);
    make_cb(CB_Z, z_bytes(plan.x_tiles_max), tt::DataFormat::UInt32, z_bytes(plan.x_tiles_max));
    if (s.has_bias) {
        make_cb(CB_ONES, kTileBytesBf16, tt::DataFormat::Float16_b, kTileBytesBf16);
    }

    // semaphores: routing (entries at the root, table ready), a2 slots, x slots
    const uint32_t sem_entries = tt::tt_metal::CreateSemaphore(program, all_set, 0);
    const uint32_t sem_table = tt::tt_metal::CreateSemaphore(program, all_set, 0);
    const uint32_t sem_a2 = tt::tt_metal::CreateSemaphore(program, all_set, 0);
    for (uint32_t i = 1; i < plan.a2_slots; ++i) {
        tt::tt_metal::CreateSemaphore(program, all_set, 0);
    }
    const uint32_t sem_x = tt::tt_metal::CreateSemaphore(program, all_set, 0);
    for (uint32_t i = 1; i < plan.x_slots; ++i) {
        tt::tt_metal::CreateSemaphore(program, all_set, 0);
    }

    const std::unordered_map<std::string, uint32_t> reader_ct = {
        {"cb_w", CB_W},
        {"cb_rt", CB_RT},
        {"cb_ctl", CB_CTL},
        {"block_tiles", plan.block_tiles},
        {"tile_bytes", wtile},
        {"block_slots", plan.block_slots},
        {"block_packets", plan.block_packets},
        {"blocks_in_flight", plan.blocks_in_flight},
        {"tokens", s.tokens},
        {"local_experts", s.local_experts},
        {"top_k", s.top_k},
        {"global_experts", s.global_experts},
        {"index_page_bytes", s.index_page_bytes},
        {"mapping_page_bytes", aligned_page(tensor_args.expert_mapping_tensor)},
        {"tokens_per_source", s.tokens / s.num_sources},
        {"sources_per_core", plan.sources_per_core},
        {"expert_groups", plan.groups},
        {"job_rows", kJobRows},
        {"row_cap", s.row_cap},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"sem_entries", sem_entries},
        {"sem_table", sem_table},
        {"rt_own", rt.own},
        {"rt_maps", rt.maps},
        {"rt_ids", rt.ids},
        {"rt_scores", rt.scores},
        {"rt_ctl", rt.ctl},
        {"rt_slots", rt.slots},
        {"rt_counts", rt.counts},
        {"rt_offsets", rt.offsets},
        {"rt_table", rt.table},
        {"rt_table_counts", rt.table_counts},
        {"rt_table_offsets", rt.table_offsets},
        {"rt_rows", rt.rows},
        {"rt_entry_slots", rt.entry_slots},
        {"rt_entry_tokens", rt.entry_tokens},
        {"rt_entry_scores", rt.entry_scores},
        {"rt_entry_k", rt.entry_k},
        {"rt_table_bytes", rt.table_bytes()},
        {"rt_jobs", rt.jobs},
        {"rt_areas", rt.areas},
        {"w0_w1_run_tiles", 4 * kd},
        {"w0_w1_group_tiles", 4 * kp},
        {"w0_w1_groups_per_core", p1_groups},
        {"w2_run_tiles", 4 * nd},
        {"w2_group_tiles", 4 * npad},
        {"w2_groups_per_core", p2_groups},
    };
    std::vector<uint32_t> reader_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_indices_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_scores_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_mapping_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table_tensor.buffer()).append_to(reader_ct_pos);

    const std::unordered_map<std::string, uint32_t> writer_ct = {
        {"cb_a", CB_A},
        {"cb_a2", CB_A2},
        {"cb_rt", CB_RT},
        {"cb_x", CB_X},
        {"cb_z", CB_Z},
        {"cb_rows", CB_ROWS},
        {"group_cores", plan.cores_per_group},
        {"intermediate_tiles", Nt},
        {"hidden_tiles", Ht},
        {"a2_slots", plan.a2_slots},
        {"x_slots", plan.x_slots},
        {"a_tiles", plan.a_tiles},
        {"rt_ctl", rt.ctl},
        {"rt_rows", rt.rows},
        {"rt_jobs", rt.jobs},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"x_tiles_max", plan.x_tiles_max},
        {"row_bytes", s.hidden_size * 2},
        {"expert_groups", plan.groups},
        {"sem_a2", sem_a2},
        {"sem_x", sem_x},
    };
    std::vector<uint32_t> writer_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.input_tensor.buffer()).append_to(writer_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*rows_tensor.buffer()).append_to(writer_ct_pos);

    const std::unordered_map<std::string, uint32_t> compute_ct = {
        {"block_tiles", plan.block_tiles},
        {"cb_w", CB_W},
        {"hidden_tiles", Ht},
        {"intermediate_tiles", Nt},
        {"ring_cores", ring},
        {"has_bias", s.has_bias ? 1u : 0u},
        {"cb_x", CB_X},
        {"cb_a", CB_A},
        {"cb_a2", CB_A2},
        {"cb_rows", CB_ROWS},
        {"cb_ctl", CB_CTL},
        {"cb_ones", CB_ONES},
        {"a_tiles", plan.a_tiles},
        {"activation_function", static_cast<uint32_t>(args.activation_type)},
        {"activation_limit_bits", to_u32(args.activation_limit)},
    };

    // readers on NOC 0, writers on NOC 1: one kernel each over the whole core set (one binary per program, written
    // to the cores' rectangles; a per-core NoC split doubled the launch cost, MEASURED 17 against 5 us)
    const auto reader = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + kernels.reader,
        all_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = reader_ct_pos,
            .named_compile_args = reader_ct});
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "rows_writer.cpp",
        all_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = writer_ct_pos,
            .named_compile_args = writer_ct});
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + kernels.compute,
        all_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = args.math_fidelity,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .dst_full_sync_en = false,
            .math_approx_mode = true,
            .named_compile_args = compute_ct});

    // runtime args: per core, plus common args shared by every core of a kernel
    std::vector<uint32_t> reader_common = {static_cast<uint32_t>(sources.size())};
    reader_common.insert(reader_common.end(), sources.begin(), sources.end());
    reader_common.insert(reader_common.end(), vc.xs.begin(), vc.xs.end());
    reader_common.insert(reader_common.end(), vc.ys.begin(), vc.ys.end());
    reader_common.insert(reader_common.end(), all_mask.begin(), all_mask.end());
    std::vector<uint32_t> writer_common = vc.xs;
    writer_common.insert(writer_common.end(), vc.ys.begin(), vc.ys.end());
    tt::tt_metal::SetCommonRuntimeArgs(program, reader, reader_common);
    tt::tt_metal::SetCommonRuntimeArgs(program, writer, writer_common);
    const CoreCoord root = mesh_device->worker_core_from_logical_core(plan.cores.front().core);
    const uint32_t np = plan.cores.size();
    for (uint32_t ip = 0; ip < np; ++ip) {
        const MoEExpertRowsCore& c = plan.cores[ip];
        std::vector<CoreCoord> group_cores;
        for (const auto& o : plan.cores) {
            if (o.group == c.group) {
                group_cores.push_back(o.core);
            }
        }
        std::vector<uint32_t> rd = {
            c.bank,
            tensor_args.expert_indices_tensor.buffer()->address(),
            tensor_args.expert_scores_tensor.buffer()->address(),
            tensor_args.expert_mapping_tensor.buffer()->address(),
            table_tensor.buffer()->address(),
            static_cast<uint32_t>(tensor_args.w0_w1_tensor.buffer()->address() + layer_w01 * wtile),
            static_cast<uint32_t>(tensor_args.w2_tensor.buffer()->address() + layer_w2 * wtile),
            c.g0,
            c.ng,
            c.q0,
            c.nq,
            c.group,
            ip,
            np,
            static_cast<uint32_t>(root.x),
            static_cast<uint32_t>(root.y),
            device_id};
        tt::tt_metal::SetRuntimeArgs(program, reader, c.core, rd);

        std::vector<uint32_t> wr = {
            tensor_args.input_tensor.buffer()->address(),
            rows_tensor.buffer()->address(),
            c.x0,
            c.xn,
            c.na,
            c.c0,
            c.nq,
            c.n0,
            c.nout,
            c.group};
        const auto gm = core_mask(group_cores, grid);
        wr.insert(wr.end(), gm.begin(), gm.end());
        tt::tt_metal::SetRuntimeArgs(program, writer, c.core, wr);

        tt::tt_metal::SetRuntimeArgs(program, compute, c.core, {c.ng, c.nq, c.ring_pos});
    }
    return {
        std::move(program), shared_variables_t{.reader_kernel = reader, .writer_kernel = writer, .cores = all_cores}};
}

void MoEExpertRowsFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MoEExpertRowsParams& args,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& s = args.shape;
    const uint32_t Ht = s.hidden_size / tt::constants::TILE_WIDTH;
    const uint32_t Nt = s.intermediate_size / tt::constants::TILE_WIDTH;
    const uint32_t ring = shard_banks_of(tensor_args.w0_w1_tensor).size();
    const uint32_t wtile = tt::tile_size(tt::DataFormat::Bfp4_b);
    const PreparedLayout layout = prepared_layout(Ht, Nt, s.has_bias, ring);
    const uint32_t layer_w01 = args.layer_id * s.local_experts * layout.w01_groups * layout.k_pad * 4;
    const uint32_t layer_w2 = args.layer_id * s.local_experts * layout.w2_groups * layout.n_pad * 4;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        for (uint32_t i = 0; i < shared.cores.size(); ++i) {
            const CoreCoord& core = shared.cores[i];
            auto& rd = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel, core);
            rd[1] = tensor_args.expert_indices_tensor.buffer()->address();
            rd[2] = tensor_args.expert_scores_tensor.buffer()->address();
            rd[3] = tensor_args.expert_mapping_tensor.buffer()->address();
            rd[4] = tensor_return_value[1].buffer()->address();
            rd[5] = tensor_args.w0_w1_tensor.buffer()->address() + layer_w01 * wtile;
            rd[6] = tensor_args.w2_tensor.buffer()->address() + layer_w2 * wtile;
            auto& wr = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel, core);
            wr[0] = tensor_args.input_tensor.buffer()->address();
            wr[1] = tensor_return_value[0].buffer()->address();
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------
// Program 2: moe_compute's outputs from the rows and the routing table. ComputeOnly: place.cpp on the combine cores
// writes the metadata and the double buffer. FullLocal / FullCcl: place.cpp writes the metadata from the feeder cores
// (one per ring position) and signals the combine; feed.cpp on the same cores streams each expert's rows into the
// double buffer with the ring program's handshake; the combine cores run selective_reduce_combine's own kernels
// (with FullCcl, its fabric path and mux cores).
//-----------------------------------------------------------------------------------------------------------------
MoEComputePlaceFactory::cached_mesh_workload_t MoEComputePlaceFactory::create_mesh_workload(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    std::optional<GlobalSemaphore> init_barrier_semaphore;
    std::optional<GlobalSemaphore> final_barrier_semaphore;
    if (args.path == MoEComputePath::FullCcl) {
        // the combine's barrier counters, above the mux's ceiling where an L1_SMALL region exists (as the standalone
        // selective_reduce_combine allocates them)
        auto* mesh_device = tensor_args.tilize_input_tensor.device();
        const CoreRangeSet combine_set = CoreRangeSet(ttsl::Span<const CoreCoord>(args.combine_params->worker_cores));
        const auto buffer_type = ttnn::ccl::prefer_l1_small_buffer_type(*mesh_device);
        init_barrier_semaphore =
            ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_set, 0, buffer_type);
        // only when the caller passes none (value_or would create one either way)
        final_barrier_semaphore =
            args.combine_params->optional_cross_device_semaphore.has_value()
                ? *args.combine_params->optional_cross_device_semaphore
                : ttnn::global_semaphore::create_global_semaphore(mesh_device, combine_set, 0, buffer_type);
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, {});
    }
    const std::vector<ttnn::MeshCoordinate> all_coordinates = mesh_coordinates.coords();
    for (const auto& coord : all_coordinates) {
        auto cached_program = create_at(
            args,
            coord,
            all_coordinates,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoEComputePlaceFactory::shared_variables_t> MoEComputePlaceFactory::create_at(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_barrier_semaphore,
    const std::optional<GlobalSemaphore>& final_barrier_semaphore) {
    const bool combine = args.path != MoEComputePath::ComputeOnly;
    const bool ccl = args.path == MoEComputePath::FullCcl;
    TT_FATAL(
        tensor_args.expert_rows_tensor.has_value() && tensor_args.expert_rows_table_tensor.has_value(),
        "moe_compute expert rows place: needs the expert-row program's outputs");
    TT_FATAL(
        !ccl || (init_barrier_semaphore.has_value() && final_barrier_semaphore.has_value()),
        "moe_compute expert rows place: FullCcl needs the combine's barrier semaphores");
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const ttnn::Tensor& rows = *tensor_args.expert_rows_tensor;
    const ttnn::Tensor& table = *tensor_args.expert_rows_table_tensor;
    const uint32_t H = tensor_args.tilize_input_tensor.logical_shape()[-1];
    const uint32_t T =
        tensor_args.tilize_input_tensor.logical_shape()[0] * tensor_args.tilize_input_tensor.logical_shape()[1];
    const uint32_t K = tensor_args.tilize_expert_indices_tensor.logical_shape()[-1];
    const uint32_t E = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    const uint32_t ntp = args.num_token_parallel_cores, dp = args.num_data_parallel_cores;
    const PlaceLayout pl = place_layout(T, K, E);
    TT_FATAL(
        table.buffer()->size() >= pl.t_bytes,
        "moe_compute expert rows place: routing table is {} B, expected {}",
        table.buffer()->size(),
        pl.t_bytes);

    const ttnn::Tensor& counts_out = tensor_return_value[0];
    const ttnn::Tensor& activation_out = tensor_return_value[1];
    const ttnn::Tensor& e_t_out = tensor_return_value[2];
    const ttnn::Tensor& dense_out = tensor_return_value[3];
    const uint32_t shards = dense_out.memory_config().shard_spec()->grid.num_cores();
    const uint32_t tero = dense_out.logical_shape().volume() / shards / (H / dp / 2) / ntp;
    const uint32_t seg = H / dp * 2;
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords vc = grid_coords(mesh_device, grid);
    auto virt = [&](const CoreCoord& c) { return mesh_device->worker_core_from_logical_core(c); };

    shared_variables_t shared;
    uint32_t done_sem = 0, meta_sem = 0;
    CoreCoord lead, sync;
    if (!combine) {
        shared.cores = get_moe_combine_cores(mesh_device, ntp, dp, H, CoreRangeSet{}, args.bh_ring_size);
    } else {
        TT_FATAL(
            args.combine_params.has_value() && args.combine_params->local_combine == !ccl &&
                tensor_return_value.size() == 6,
            "moe_compute expert rows place: FullLocal / FullCcl need their combine parameters and 6 outputs");
        const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
            mesh_device, ntp, dp, H, args.combine_params->mux_core_range_set, args.bh_ring_size);
        shared.cores = selection.matmul_cores;
        const uint32_t ring = shared.cores.size();
        const auto& combine_params = *args.combine_params;
        TT_FATAL(
            combine_params.worker_cores.size() >= ntp * dp,
            "moe_compute expert rows place: {} combine cores, need {}",
            combine_params.worker_cores.size(),
            ntp * dp);
        shared.combine_cores.assign(
            combine_params.worker_cores.begin(), combine_params.worker_cores.begin() + ntp * dp);
        const CoreRangeSet feed_set = CoreRangeSet(ttsl::Span<const CoreCoord>(shared.cores));
        const CoreRangeSet combine_set = CoreRangeSet(ttsl::Span<const CoreCoord>(shared.combine_cores));
        // the combine reader multicasts its metadata semaphore over its cores' bounding box
        meta_sem = tt::tt_metal::CreateSemaphore(program, CoreRangeSet(combine_set.bounding_box()).merge(feed_set), 0);
        const uint32_t sync_sem = tt::tt_metal::CreateSemaphore(program, combine_set.merge(feed_set), 0);
        done_sem = tt::tt_metal::CreateSemaphore(program, feed_set, 0);
        lead = virt(shared.cores[0]);
        sync = virt(shared.combine_cores[0]);

        const ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensors{
            .dense_input_tensor = tensor_return_value[4],
            .dense_activations_tensor = activation_out,
            .dense_token_maps_tensor = e_t_out,
            .dense_token_counts_tensor = counts_out,
            .optional_output_tensor = tensor_args.optional_output_tensor};
        // Each combine column (width shard) is fed by the feeders whose W2 width slice overlaps it; with a ring size
        // that is not a multiple of the column count a slice straddles columns (as in the ring program's dm1)
        const uint32_t Ht = H / tt::constants::TILE_WIDTH;
        const auto layouts = shard_layouts(Ht, args.intermediate_size / tt::constants::TILE_WIDTH, ring);
        std::vector<std::vector<CoreCoord>> feeders_by_column(dp);
        for (uint32_t r = 0; r < ring; ++r) {
            const uint32_t begin = moe_ring::w2_combine_col_begin(layouts[r].out_start, Ht / dp);
            const uint32_t end = moe_ring::w2_combine_col_end(layouts[r].out_start, layouts[r].out_tiles, Ht / dp);
            for (uint32_t c = begin; c < end; ++c) {
                feeders_by_column.at(c).push_back(shared.cores[r]);
            }
        }
        const auto artifacts = build_selective_reduce_combine_program_artifacts(
            program,
            combine_params,
            mesh_coordinate,
            all_mesh_coordinates,
            combine_tensors,
            tensor_return_value[5],
            init_barrier_semaphore,
            final_barrier_semaphore,
            meta_sem,
            sync_sem,
            feeders_by_column);
        if (ccl) {
            shared.combine_semaphores = {*init_barrier_semaphore, *final_barrier_semaphore};
        }
        shared.combine_reader = artifacts.reader_kernel_id;
        shared.combine_writer = artifacts.writer_kernel_id;
        shared.combine_data_cb = artifacts.data_cb_handle;

        // feed.cpp: feeder r streams the W2 output columns ring position r owns in the ring program
        const uint32_t twb = tt::constants::TILE_WIDTH * rows.element_size();
        const FeedLayout fl = feed_layout(pl, layouts, twb);
        constexpr uint32_t kFeedCb = 1;
        tt::tt_metal::CreateCircularBuffer(
            program,
            feed_set,
            tt::tt_metal::CircularBufferConfig(fl.bytes, {{kFeedCb, tt::DataFormat::UInt32}})
                .set_page_size(kFeedCb, fl.bytes));
        const std::unordered_map<std::string, uint32_t> feed_ct = {
            {"cb_scratch", kFeedCb},
            {"local_experts", E},
            {"height_shards", ntp},
            {"width_shards", dp},
            {"shard_width_tiles", Ht / dp},
            {"tile_width_bytes", twb},
            {"half_bytes", tero * seg},
            {"row_bytes", H * 2},
            {"batch_rows", fl.batch},
            {"slice_stride", fl.slice},
            {"table_counts", pl.t_counts},
            {"table_offsets", pl.t_offsets},
            {"table_rows", pl.t_rows},
            {"table_bytes", pl.t_bytes},
            {"s_stage", fl.stage},
            {"combine_sync_semaphore_id", sync_sem},
        };
        std::vector<uint32_t> feed_pos;
        tt::tt_metal::TensorAccessorArgs(*rows.buffer()).append_to(feed_pos);
        tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(feed_pos);
        shared.feed_kernel = tt::tt_metal::CreateKernel(
            program,
            std::string(kKernelDir) + "feed.cpp",
            feed_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
                .compile_args = feed_pos,
                .named_compile_args = feed_ct});
        for (uint32_t r = 0; r < ring; ++r) {
            const uint32_t first_shard = layouts[r].out_start / (Ht / dp);
            const uint32_t last_shard = (layouts[r].out_start + layouts[r].out_tiles - 1) / (Ht / dp);
            TT_FATAL(
                last_shard - first_shard < 3, "moe_compute expert rows feed: a slice spans more than 3 width shards");
            std::vector<uint32_t> a = {
                rows.buffer()->address(),
                table.buffer()->address(),
                dense_out.buffer()->address(),
                layouts[r].out_start,
                layouts[r].out_tiles};
            for (const auto& c : shared.combine_cores) {
                const CoreCoord v = virt(c);
                a.push_back(v.x);
                a.push_back(v.y);
            }
            tt::tt_metal::SetRuntimeArgs(program, *shared.feed_kernel, shared.cores[r], a);
        }
    }

    const CoreRangeSet set = CoreRangeSet(ttsl::Span<const CoreCoord>(shared.cores));
    tt::tt_metal::CreateCircularBuffer(
        program,
        set,
        tt::tt_metal::CircularBufferConfig(pl.scratch, {{0, tt::DataFormat::UInt32}}).set_page_size(0, pl.scratch));

    const std::unordered_map<std::string, uint32_t> ct = {
        {"cb_scratch", 0},
        {"tokens", T},
        {"top_k", K},
        {"local_experts", E},
        {"height_shards", ntp},
        {"segment_bytes", seg},
        {"half_bytes", tero * seg},
        {"count_words", pl.count_words},
        {"e_t_page_bytes", pl.e_t_page},
        {"activation_row_bytes", pl.activation_row},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"program_cores", static_cast<uint32_t>(shared.cores.size())},
        {"row_bytes", H * 2},
        {"place_dense", combine ? 0u : 1u},
        {"done_semaphore_id", done_sem},
        {"metadata_semaphore_id", meta_sem},
        {"table_counts", pl.t_counts},
        {"table_offsets", pl.t_offsets},
        {"table_rows", pl.t_rows},
        {"table_entry_slots", pl.t_es},
        {"table_entry_tokens", pl.t_et},
        {"table_entry_scores", pl.t_ec},
        {"table_entry_k", pl.t_ek},
        {"table_bytes", pl.t_bytes},
        {"s_counts", pl.s_counts},
        {"s_offsets", pl.s_offsets},
        {"s_e_t", pl.s_et},
        {"s_activation", pl.s_act},
        {"s_count_row", pl.s_crow},
        {"s_table", pl.s_table},
    };
    std::vector<uint32_t> ct_pos;
    tt::tt_metal::TensorAccessorArgs(*rows.buffer()).append_to(ct_pos);
    tt::tt_metal::TensorAccessorArgs(*e_t_out.buffer()).append_to(ct_pos);
    tt::tt_metal::TensorAccessorArgs(*activation_out.buffer()).append_to(ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(ct_pos);
    shared.place_kernel = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "place.cpp",
        set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = ct_pos,
            .named_compile_args = ct});
    for (uint32_t i = 0; i < shared.cores.size(); ++i) {
        std::vector<uint32_t> a = {
            rows.buffer()->address(),
            e_t_out.buffer()->address(),
            activation_out.buffer()->address(),
            counts_out.buffer()->address(),
            dense_out.buffer()->address(),
            table.buffer()->address(),
            combine ? 0 : i / dp,
            combine ? 0 : i % dp,
            i,
            static_cast<uint32_t>(lead.x),
            static_cast<uint32_t>(lead.y),
            static_cast<uint32_t>(sync.x),
            static_cast<uint32_t>(sync.y)};
        a.insert(a.end(), vc.xs.begin(), vc.xs.end());
        a.insert(a.end(), vc.ys.begin(), vc.ys.end());
        tt::tt_metal::SetRuntimeArgs(program, shared.place_kernel, shared.cores[i], a);
    }
    return {std::move(program), std::move(shared)};
}

void MoEComputePlaceFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MoEComputeParams& args,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        for (const auto& core : shared.cores) {
            auto& a = tt::tt_metal::GetRuntimeArgs(program, shared.place_kernel, core);
            a[0] = tensor_args.expert_rows_tensor->buffer()->address();
            a[1] = tensor_return_value[2].buffer()->address();
            a[2] = tensor_return_value[1].buffer()->address();
            a[3] = tensor_return_value[0].buffer()->address();
            a[4] = tensor_return_value[3].buffer()->address();
            a[5] = tensor_args.expert_rows_table_tensor->buffer()->address();
            if (shared.feed_kernel.has_value()) {
                auto& f = tt::tt_metal::GetRuntimeArgs(program, *shared.feed_kernel, core);
                f[0] = tensor_args.expert_rows_tensor->buffer()->address();
                f[1] = tensor_args.expert_rows_table_tensor->buffer()->address();
                f[2] = tensor_return_value[3].buffer()->address();
            }
        }
        if (shared.feed_kernel.has_value()) {
            const ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensors{
                .dense_input_tensor = tensor_return_value[4],
                .dense_activations_tensor = tensor_return_value[1],
                .dense_token_maps_tensor = tensor_return_value[2],
                .dense_token_counts_tensor = tensor_return_value[0],
                .optional_output_tensor = tensor_args.optional_output_tensor};
            selective_reduce_combine_helper_override_runtime_arguments(
                program,
                shared.combine_reader,
                shared.combine_writer,
                shared.combine_data_cb,
                shared.combine_cores,
                combine_tensors,
                tensor_return_value[5],
                shared.combine_semaphores.size() == 2 ? shared.combine_semaphores[0].address() : 0,
                shared.combine_semaphores.size() == 2 ? shared.combine_semaphores[1].address() : 0,
                args.combine_params->optional_cross_device_semaphore);
        }
    }
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_expert_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::experimental::prim::MoEExpertRowsParams& params) {
    using OperationType = ttnn::experimental::prim::MoEExpertRowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_scores_tensor = expert_scores_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor});
}

}  // namespace ttnn::prim
