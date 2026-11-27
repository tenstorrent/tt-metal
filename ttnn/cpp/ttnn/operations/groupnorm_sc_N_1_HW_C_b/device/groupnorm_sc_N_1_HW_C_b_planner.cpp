// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sc_N_1_HW_C_b_planner.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/groupnorm_sc_N_1_HW_C_b_args.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b::planner {

using namespace tt::tt_metal;

namespace {

constexpr int64_t kTileDim = tt::constants::TILE_WIDTH;  // channels / rows per tile edge
constexpr int64_t kChunkTilesTarget = 32;                // streaming quantum target (tiles per chunk)
constexpr int64_t kInputStreamDepth = 2;                 // D_IN: TILE input stream depth (chunks)
constexpr int64_t kOutputChunks = 2;                     // D_OUT: output chunks at minimum
constexpr int64_t kMaxGroupLanes = gn_sc_args::MAX_GROUP_LANES;
constexpr int64_t kFaceRow0ReadBytes = gn_sc_args::FACE_ROW0_READ_BYTES;
constexpr int64_t kPilotSlotBytes = gn_sc_args::PILOT_SLOT_BYTES;
constexpr int64_t kStageAlign = gn_sc_args::STAGE_ALIGN;
constexpr int64_t kRmPilotAlign = gn_sc_args::RM_PILOT_ALIGN;

int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int64_t bf16_tile_bytes() { return tt::tt_metal::tile_size(DataType::BFLOAT16); }
int64_t f32_tile_bytes() { return tt::tt_metal::tile_size(DataType::FLOAT32); }

int64_t elem_bytes(DataType dtype) {
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32,
        "groupnorm_sc_N_1_HW_C_b: no stick element size for dtype {}",
        dtype);
    return tt::datum_size(datatype_to_dataformat_converter(dtype));
}

// Python Formats properties.
int64_t in_tile(const Formats& f) { return tt::tt_metal::tile_size(f.in_dtype); }
bool in_bfp8(const Formats& f) { return f.in_dtype == DataType::BFLOAT8_B; }
int64_t in_elem(const Formats& f) { return elem_bytes(f.in_dtype); }
bool aff_bfp8(const Formats& f) { return f.aff_dtype == DataType::BFLOAT8_B; }
DataType amap_dtype(const Formats& f) { return aff_bfp8(f) ? DataType::BFLOAT16 : f.aff_dtype; }
int64_t amap_tile(const Formats& f) { return tt::tt_metal::tile_size(amap_dtype(f)); }
int64_t aff_elem(const Formats& f) { return elem_bytes(amap_dtype(f)); }
int64_t pilot_tile_stage(const Formats& f) { return (in_bfp8(f) ? 3 : 2) * kFaceRow0ReadBytes; }
int64_t aff_tile_stage(const Formats& f) { return (aff_bfp8(f) ? 3 : 2) * kFaceRow0ReadBytes; }

int64_t ring_pages_of(int64_t T, int64_t K_alloc, bool stream, bool is_rm) {
    return (K_alloc * T) + (static_cast<int64_t>(stream) * (is_rm ? 1 : kInputStreamDepth) * T);
}

int64_t output_ring_bytes(int64_t T, const Formats& f) {
    return (kOutputChunks * T * in_tile(f)) + (2 * gn_sc_args::OUT_SEQ_PAGES * gn_sc_args::OUT_SEQ_PAGE_BYTES);
}

int64_t affine_stage_bytes(int64_t W, int64_t map_period, const Formats& f) {
    if (f.aff_tile) {
        return W * aff_tile_stage(f);
    }
    return tt::round_up(map_period * aff_elem(f), kStageAlign) + kStageAlign;
}

int64_t staging_of(
    int64_t W, int64_t map_period, bool is_rm, int64_t n_affine, const Formats& f, bool sharded, int64_t pilot_cols) {
    const int64_t stick = is_rm ? map_period * in_elem(f) : 0;
    int64_t pilot = 0;
    if (sharded && is_rm) {
        pilot = std::max(stick, pilot_cols * (tt::round_up(stick, kRmPilotAlign) + kRmPilotAlign));
    } else if (sharded) {
        pilot = kMaxGroupLanes * kPilotSlotBytes;
    } else {
        pilot = is_rm ? stick : W * pilot_tile_stage(f);
    }
    const int64_t affine = n_affine * affine_stage_bytes(W, map_period, f);
    return tt::round_up(std::max(pilot, affine), kStageAlign);
}

struct Footprint {
    int64_t W = 0, T = 0, R = 0, K_alloc = 0;
    bool stream = false, is_rm = false, has_gamma = false, has_beta = false;
    Formats fmt;
    std::optional<int64_t> map_period;
    bool sharded = false;
    std::optional<int64_t> ring_pages;
    int64_t stage_pages = 0;
    int64_t NU = 1, NSEG = 1, pilot_cols = 1;
};

// Python footprint_bytes.
int64_t footprint_bytes(const Footprint& p) {
    const int64_t hg = p.has_gamma ? 1 : 0;
    const int64_t hb = p.has_beta ? 1 : 0;
    const int64_t map_period = p.map_period.has_value() ? *p.map_period : p.W * kTileDim;
    const bool tile_shard = p.sharded && !p.is_rm;
    int64_t ring = 0;
    if (tile_shard) {
        ring = 0;
    } else if (p.ring_pages.has_value()) {
        ring = *p.ring_pages;
    } else {
        ring = ring_pages_of(p.T, p.K_alloc, p.stream, p.is_rm);
    }
    const int64_t in_pages =
        (p.is_rm && !p.sharded ? static_cast<int64_t>(kRmStickChunks) * p.T : 0) + ring + p.stage_pages;
    const int64_t bf16_pages = 1 + (2 * p.W);
    const int64_t amap_pages = p.W * (hg + hb);
    const int64_t out_ring = p.sharded ? 0 : output_ring_bytes(p.T, p.fmt);
    const int64_t f32_bytes =
        out_ring + f32_tile_bytes() * ((2 * p.W) + (2 * p.R * p.NU) + (2 * p.NSEG) + (4 * p.NU) + 2);
    const int64_t staging = staging_of(p.W, map_period, p.is_rm, hg + hb, p.fmt, p.sharded, p.pilot_cols);
    return (in_tile(p.fmt) * in_pages) + (bf16_tile_bytes() * bf16_pages) + (amap_tile(p.fmt) * amap_pages) +
           f32_bytes + staging;
}

int64_t largest_divisor_at_most(int64_t n, int64_t cap) {
    for (int64_t d = std::min(n, cap); d > 0; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

[[noreturn]] void refuse_interleaved(uint32_t C, uint32_t G) {
    throw PlanNotImplemented(fmt::format(
        "groupnorm_sc_N_1_HW_C_b: no feasible column-slice count for C={}, G={} (G_core <= 32 and L1 fit); "
        "cross_column_group_combine regime not built",
        C,
        G));
}

[[noreturn]] void refuse_sharded(const std::string& why) {
    throw PlanNotImplemented(fmt::format("groupnorm_sc_N_1_HW_C_b: block-sharded input: {}", why));
}

struct GxCandidate {
    int64_t gx = 0, W = 0, G_core = 0, U = 0, cpi = 0, rows_core_max = 0, h = 0, T = 0, R = 0, f_fixed = 0;
    int64_t max_core_tiles = 0, cores_used = 0;
};

struct InterleavedGeometry {
    int64_t N, Ht, Wt, S, span_tiles, gps, P, chunk_target;
    bool is_rm, has_gamma, has_beta;
    Formats fmt;
    int64_t l1_budget;
};

std::optional<GxCandidate> evaluate_gx(int64_t gx, const InterleavedGeometry& g) {
    GxCandidate c;
    c.gx = gx;
    c.W = std::min((g.S / gx) * g.span_tiles, g.Wt);
    c.G_core = (g.S / gx) * g.gps;
    if (c.G_core > kMaxGroupLanes) {
        return std::nullopt;
    }
    c.U = g.N * gx;
    c.cpi = c.U < g.P ? std::min(g.P / c.U, g.Ht) : 1;
    c.rows_core_max = tt::div_up(g.Ht, c.cpi);
    int64_t h = std::max<int64_t>(1, std::min(g.chunk_target / c.W, c.rows_core_max));
    const int64_t n_chunks = std::max(tt::div_up(c.rows_core_max, h), std::min<int64_t>(c.rows_core_max, 2));
    c.h = tt::div_up(c.rows_core_max, n_chunks);
    c.T = c.h * c.W;
    c.R = tt::div_up(c.cpi, kTileDim);
    Footprint fp;
    fp.W = c.W;
    fp.T = c.T;
    fp.R = c.R;
    fp.K_alloc = 0;
    fp.stream = true;
    fp.is_rm = g.is_rm;
    fp.has_gamma = g.has_gamma;
    fp.has_beta = g.has_beta;
    fp.fmt = g.fmt;
    c.f_fixed = footprint_bytes(fp);
    if (c.f_fixed > g.l1_budget) {
        return std::nullopt;
    }
    if (c.U < g.P) {
        c.max_core_tiles = c.rows_core_max * c.W;
        c.cores_used = c.U * c.cpi;
    } else {
        c.max_core_tiles = tt::div_up(c.U, g.P) * g.Ht * c.W;
        c.cores_used = g.P;
    }
    return c;
}

std::vector<CoreWork> distribute(int64_t U, int64_t P, int64_t cpi, int64_t Ht, int64_t grid_x) {
    std::vector<CoreWork> work;
    auto set_core = [grid_x](uint32_t& x, uint32_t& y, int64_t c) {
        x = static_cast<uint32_t>(c % grid_x);
        y = static_cast<uint32_t>(c / grid_x);
    };
    if (U < P) {
        const int64_t q = Ht / cpi;
        const int64_t rem = Ht % cpi;
        work.reserve(static_cast<size_t>(U * cpi));
        for (int64_t c = 0; c < U * cpi; ++c) {
            const int64_t unit = c / cpi;
            const int64_t slot = c % cpi;
            CoreWork w;
            set_core(w.core_x, w.core_y, c);
            w.unit_start = static_cast<uint32_t>(unit);
            w.num_blocks = 1;
            w.row_start = static_cast<uint32_t>((slot * q) + std::min(slot, rem));
            w.rows_core = static_cast<uint32_t>(q + (slot < rem ? 1 : 0));
            w.is_root = slot == 0 ? 1 : 0;
            w.slot = static_cast<uint32_t>(slot);
            set_core(w.root_x, w.root_y, unit * cpi);
            work.push_back(std::move(w));
        }
    } else {
        const int64_t base = U / P;
        const int64_t extra = U % P;
        work.reserve(static_cast<size_t>(P));
        for (int64_t c = 0; c < P; ++c) {
            CoreWork w;
            set_core(w.core_x, w.core_y, c);
            w.unit_start = static_cast<uint32_t>((c * base) + std::min(c, extra));
            w.num_blocks = static_cast<uint32_t>(base + (c < extra ? 1 : 0));
            w.row_start = 0;
            w.rows_core = static_cast<uint32_t>(Ht);
            w.is_root = 1;
            w.slot = 0;
            set_core(w.root_x, w.root_y, c);
            work.push_back(std::move(w));
        }
    }
    return work;
}

Plan plan_interleaved(const PlanInputs& in) {
    const int64_t N = in.N, HW = in.HW, C = in.C, G = in.G;
    TT_FATAL(G >= 1 && C % G == 0, "groupnorm_sc_N_1_HW_C_b: num_groups={} must divide C={}", G, C);
    const int64_t Ht = tt::div_up(HW, kTileDim);
    const int64_t Wt = tt::div_up(C, kTileDim);
    const int64_t Cg = C / G;
    const int64_t L = std::lcm(Cg, kTileDim);
    const int64_t span_tiles = L / kTileDim;
    const int64_t gps = L / Cg;
    const int64_t S = tt::div_up(C, L);

    const int64_t P = static_cast<int64_t>(in.grid_x) * in.grid_y;
    const int64_t chunk_target = kChunkTilesTarget;
    const int64_t l1_budget = in.l1_budget;

    const InterleavedGeometry geom{
        N, Ht, Wt, S, span_tiles, gps, P, chunk_target, in.is_rm, in.has_gamma, in.has_beta, in.fmt, l1_budget};
    std::optional<GxCandidate> best;
    std::tuple<int64_t, int64_t, int64_t> best_key{};
    for (int64_t gx = 1; gx <= S; ++gx) {
        if (S % gx != 0) {
            continue;
        }
        const auto cand = evaluate_gx(gx, geom);
        if (!cand.has_value()) {
            continue;
        }
        // Minimize max per-core tiles; ties: more cores, then larger gx (TILE) / smaller gx (RM).
        const std::tuple<int64_t, int64_t, int64_t> key{cand->max_core_tiles, -cand->cores_used, in.is_rm ? gx : -gx};
        if (!best.has_value() || key < best_key) {
            best = cand;
            best_key = key;
        }
    }
    if (!best.has_value()) {
        refuse_interleaved(in.C, in.G);
    }

    const int64_t W = best->W, T = best->T, R = best->R, h = best->h;
    const int64_t B_max = tt::div_up(best->rows_core_max, h);
    const int64_t chunk_bytes = in_tile(in.fmt) * T;
    int64_t K_max = floor_div(l1_budget - best->f_fixed, chunk_bytes);
    Footprint fp;
    fp.W = W;
    fp.T = T;
    fp.R = R;
    fp.K_alloc = 0;
    fp.stream = false;
    fp.is_rm = in.is_rm;
    fp.has_gamma = in.has_gamma;
    fp.has_beta = in.has_beta;
    fp.fmt = in.fmt;
    const int64_t f_resident = footprint_bytes(fp);
    if (floor_div(l1_budget - f_resident, chunk_bytes) >= B_max) {
        K_max = std::max(K_max, B_max);
    }
    const int64_t K_alloc = std::min(B_max, K_max);
    const bool stream = B_max > K_alloc;
    fp.K_alloc = K_alloc;
    fp.stream = stream;
    int64_t footprint = footprint_bytes(fp);
    const int64_t out_writer_chunks = l1_budget - footprint >= chunk_bytes ? 2 : 1;
    footprint += (out_writer_chunks - 1) * chunk_bytes;

    const int64_t vc_last = C - ((best->gx - 1) * W * kTileDim);
    Plan plan;
    plan.N = in.N;
    plan.HW = in.HW;
    plan.C = in.C;
    plan.G = in.G;
    plan.Ht = static_cast<uint32_t>(Ht);
    plan.Wt = static_cast<uint32_t>(Wt);
    plan.Cg = static_cast<uint32_t>(Cg);
    plan.span_tiles = static_cast<uint32_t>(span_tiles);
    plan.gps = static_cast<uint32_t>(gps);
    plan.S = static_cast<uint32_t>(S);
    plan.gx = static_cast<uint32_t>(best->gx);
    plan.W = static_cast<uint32_t>(W);
    plan.G_core = static_cast<uint32_t>(best->G_core);
    plan.U = static_cast<uint32_t>(best->U);
    plan.P = static_cast<uint32_t>(P);
    plan.cpi = static_cast<uint32_t>(best->cpi);
    plan.h = static_cast<uint32_t>(h);
    plan.T = static_cast<uint32_t>(T);
    plan.R = static_cast<uint32_t>(R);
    plan.rows_core_max = static_cast<uint32_t>(best->rows_core_max);
    plan.B_max = static_cast<uint32_t>(B_max);
    plan.K_alloc = static_cast<uint32_t>(K_alloc);
    plan.stream = stream;
    plan.is_rm = in.is_rm;
    plan.has_gamma = in.has_gamma;
    plan.has_beta = in.has_beta;
    plan.grid_x = in.grid_x;
    plan.grid_y = in.grid_y;
    plan.map_period = static_cast<uint32_t>(W * kTileDim);
    plan.vc_last = static_cast<uint32_t>(vc_last);
    plan.col_mask = vc_last < W * kTileDim;
    plan.out_writer_chunks = static_cast<uint32_t>(out_writer_chunks);
    plan.fmt = in.fmt;
    return plan;
}

struct Segment {
    int64_t img, a, b;
};

// The block-sharded geometry (Python _plan_sharded up to its band check), with its refusals in the Python's order.
// Scalars only: a row's or a column's values are member functions, so the scalar solve builds no per-core data.
struct ShardGeometry {
    int64_t N = 0, HW = 0, C = 0, Cg = 0, NHW = 0;
    int64_t sh = 0, sw = 0, x0 = 0, y0 = 0, gx = 0, gy = 0;
    int64_t k = 1, W = 0, tr = 0;
    bool split_cols = false;
    int64_t cl_cols = 0, lanes_max = 0, bh = 0, unit_cores = 0, R = 0;
    int64_t rows_nominal = 0;

    int64_t vrows(int64_t y) const { return std::min(sh, NHW - (y * sh)); }
    int64_t vcols(int64_t x) const { return std::min(sw, C - (x * sw)); }
    // (first image, image count) of the band holding shard row y.
    std::pair<int64_t, int64_t> band_images(int64_t y) const {
        const int64_t yb = y - (y % bh);
        const int64_t first = (yb * sh) / HW;
        const int64_t last = (std::min((yb + bh) * sh, NHW) - 1) / HW;
        return {first, last - first + 1};
    }
    // Calls fn(segment) for each image's part of shard row y, in order (Python segments).
    template <typename Fn>
    void for_each_segment(int64_t y, Fn&& fn) const {
        const int64_t r0 = y * sh, r1 = (y * sh) + vrows(y);
        for (int64_t n = r0 / HW; n * HW < r1; ++n) {
            fn(Segment{n, std::max(n * HW, r0) - r0, std::min((n + 1) * HW, r1) - r0});
        }
    }
    int64_t num_segments(int64_t y) const {
        int64_t n = 0;
        for_each_segment(y, [&n](const Segment& /*segment*/) { ++n; });
        return n;
    }
};

ShardGeometry shard_geometry(const PlanInputs& in) {
    ShardGeometry g;
    g.N = in.N;
    g.HW = in.HW;
    g.C = in.C;
    const int64_t G = in.G;
    TT_FATAL(G >= 1 && g.C % G == 0, "groupnorm_sc_N_1_HW_C_b: num_groups={} must divide C={}", G, g.C);
    g.Cg = g.C / G;
    g.sh = in.sh;
    g.sw = in.sw;
    g.x0 = in.x0;
    g.y0 = in.y0;
    g.gx = in.gx;
    g.gy = in.gy;
    const int64_t sh = g.sh, sw = g.sw, gx = g.gx, gy = g.gy, HW = g.HW;

    if (!in.shard_in_l1) {
        refuse_sharded("only L1 shards are supported");
    }
    if (!in.shard_row_major || static_cast<int64_t>(in.shard_grid_num_cores) != gx * gy) {
        refuse_sharded("the shard grid must be one ROW_MAJOR rectangle");
    }
    g.NHW = g.N * HW;
    int64_t min_rows = std::numeric_limits<int64_t>::max();
    int64_t min_cols = std::numeric_limits<int64_t>::max();
    for (int64_t y = 0; y < gy; ++y) {
        min_rows = std::min(min_rows, g.vrows(y));
    }
    for (int64_t x = 0; x < gx; ++x) {
        min_cols = std::min(min_cols, g.vcols(x));
    }
    if (min_rows <= 0 || min_cols <= 0) {
        refuse_sharded(fmt::format("shard {}x{} on a {}x{} grid leaves a core without data", sh, sw, gx, gy));
    }
    if (in.is_rm) {
        const int64_t L = std::lcm(sw, kTileDim);
        g.k = L / sw;
        g.W = L / kTileDim;
    } else {
        if (HW % kTileDim != 0) {
            refuse_sharded(fmt::format("TILE images of {} rows are not whole tile rows", HW));
        }
        g.k = 1;
        g.W = sw / kTileDim;
    }
    g.tr = kTileDim * g.k;

    g.split_cols = sw % g.Cg == 0;
    g.cl_cols = g.split_cols ? 1 : gx;
    g.lanes_max = g.split_cols ? sw / g.Cg : G;
    if (g.lanes_max > kMaxGroupLanes) {
        refuse_sharded(fmt::format("{} groups per statistic unit exceed one tile row of group lanes", g.lanes_max));
    }
    std::vector<int64_t> band_starts{0};
    for (int64_t y = 1; y < gy; ++y) {
        if ((y * sh) % HW == 0) {
            band_starts.push_back(y);
        }
    }
    g.bh = band_starts.size() > 1 ? band_starts[1] : gy;
    for (size_t i = 0; i < band_starts.size(); ++i) {
        const int64_t end = i + 1 < band_starts.size() ? band_starts[i + 1] : gy;
        if (end - band_starts[i] != g.bh) {
            refuse_sharded(fmt::format("image bands of unequal height (shard height {}, HW={})", sh, HW));
        }
    }
    g.unit_cores = g.bh * g.cl_cols;
    g.R = tt::div_up(g.unit_cores, kTileDim);
    g.rows_nominal = tt::div_up(sh, g.tr);
    return g;
}

// (shard stick, position in its tile row, sticks) of one staged row.
using StagedRow = std::array<uint32_t, gn_sc_args::STAGED_WORDS>;

// One (segment, tile row) of a shard row (Python _rm_core_ops' items): whole, or staged.
struct RmItem {
    int64_t si = 0, t = 0;
    bool staged = false;
    StagedRow trip{};
};

template <typename Visit>
void for_each_rm_item(const ShardGeometry& g, int64_t y, Visit&& visit) {
    const int64_t tr = g.tr;
    int64_t si = 0;
    g.for_each_segment(y, [&](const Segment& s) {
        for (int64_t t = s.a / tr; t * tr < s.b; ++t) {
            const int64_t lo = t * tr, hi = (t * tr) + tr;
            const bool whole = s.a <= lo && hi <= s.b;
            RmItem it{.si = si, .t = t, .staged = !whole};
            if (it.staged) {
                const int64_t st = std::max(s.a, lo);
                const int64_t end = std::min(s.b, hi);
                it.trip = {static_cast<uint32_t>(st), static_cast<uint32_t>(st - lo), static_cast<uint32_t>(end - st)};
            }
            visit(it);
        }
        ++si;
    });
}

// Python _rm_core_ops for shard row y and chunk height h: emit_op(word) per op word, emit_staged(trip) per staged
// row, in order. The items stream through with one item of lookahead, so counting them stores nothing.
template <typename EmitOp, typename EmitStaged>
void rm_core_ops(const ShardGeometry& g, int64_t y, int64_t h, EmitOp&& emit_op, EmitStaged&& emit_staged) {
    int64_t max_full = -1;
    for_each_rm_item(g, y, [&max_full](const RmItem& it) {
        if (!it.staged) {
            max_full = std::max(max_full, it.t);
        }
    });
    int64_t run = 0;
    auto step = [&](const RmItem& it, const RmItem* nxt) {
        const bool seg_last = nxt == nullptr || nxt->si != it.si;
        if (!it.staged) {
            ++run;
            if (run == h || seg_last || nxt->staged) {
                emit_op(static_cast<uint32_t>(run) | (seg_last ? gn_sc_args::OP_SEG_END : 0u));
                run = 0;
            }
            return;
        }
        emit_op(1u | gn_sc_args::OP_STAGED | (seg_last ? gn_sc_args::OP_SEG_END : 0u));
        emit_staged(it.trip);
        if ((nxt == nullptr || nxt->t != it.t) && it.t < max_full) {
            emit_op(gn_sc_args::OP_SKIP);
        }
    };
    std::optional<RmItem> pending;
    for_each_rm_item(g, y, [&](const RmItem& it) {
        if (pending.has_value()) {
            step(*pending, &it);
        }
        pending = it;
    });
    if (pending.has_value()) {
        step(*pending, nullptr);
    }
}

Plan plan_sharded(const PlanInputs& in) {
    const ShardGeometry g = shard_geometry(in);
    const int64_t N = g.N, HW = g.HW, C = g.C, Cg = g.Cg;
    const int64_t sw = g.sw, gx = g.gx, gy = g.gy;
    const int64_t W = g.W, tr = g.tr, R = g.R;
    const bool is_rm = in.is_rm;
    const Formats& fmt = in.fmt;

    int64_t NU = 0;
    int64_t NSEG = 0;
    for (int64_t y = 0; y < gy; ++y) {
        NU = std::max(NU, g.band_images(y).second);
        NSEG = std::max(NSEG, g.num_segments(y));
    }
    bool simple = NSEG == 1;
    for (int64_t y = 0; y < gy; ++y) {
        simple = simple && g.vrows(y) % tr == 0;
    }

    // The Python subtracted the out-of-place output shard, which it allocated after reading the budget. Here the
    // output is allocated before plan_inputs reads the budget, so the budget already excludes it.
    const int64_t l1_budget = in.l1_budget;

    const int64_t rows_nominal = g.rows_nominal;
    auto fp_common = [&](int64_t T) {
        Footprint fp;
        fp.W = W;
        fp.T = T;
        fp.R = R;
        fp.is_rm = is_rm;
        fp.has_gamma = in.has_gamma;
        fp.has_beta = in.has_beta;
        fp.map_period = sw;
        fp.sharded = true;
        fp.NU = NU;
        fp.NSEG = NSEG;
        fp.pilot_cols = g.cl_cols;
        fp.fmt = fmt;
        return fp;
    };
    const bool col_mask = is_rm && g.vcols(gx - 1) < sw;

    // RM: each shard row's ring rows and staged rows, counted from its op words without storing them.
    int64_t h_ops = 0, fp_ops = 0, stage_pages = 0, max_ring_rows = 0;
    if (is_rm) {
        h_ops = rows_nominal;
        int64_t n_staged = 0;
        for (int64_t y = 0; y < gy; ++y) {
            int64_t rr = 0;
            int64_t staged_rows = 0;
            auto count_rows = [&rr](uint32_t op) {
                if ((op & gn_sc_args::OP_SKIP) == 0) {
                    rr += op & gn_sc_args::OP_ROWS;
                }
            };
            auto count_staged = [&staged_rows](const StagedRow& /*trip*/) { ++staged_rows; };
            rm_core_ops(g, y, h_ops, count_rows, count_staged);
            max_ring_rows = std::max(max_ring_rows, rr);
            n_staged = std::max(n_staged, staged_rows);
        }
        stage_pages = n_staged != 0 ? (n_staged + 1) * W : 0;
        Footprint fp = fp_common(h_ops * W);
        fp.K_alloc = 0;
        fp.stream = false;
        fp.ring_pages = max_ring_rows * W;
        fp.stage_pages = stage_pages;
        fp_ops = footprint_bytes(fp);
    }

    int64_t rows_core = 0, h = 0, T = 0, B = 0, K_alloc = 0, footprint = 0;
    bool ops_path = false;
    if (!is_rm) {
        rows_core = rows_nominal;
        h = rows_core;
        T = h * W;
        B = rows_core / h;
        K_alloc = B;
        stage_pages = 0;
        Footprint fp = fp_common(T);
        fp.K_alloc = K_alloc;
        fp.stream = false;
        footprint = footprint_bytes(fp);
        ops_path = true;
    } else if (simple && fp_ops > l1_budget) {
        rows_core = rows_nominal;
        stage_pages = 0;
        h = largest_divisor_at_most(rows_core, std::max<int64_t>(1, kChunkTilesTarget / W));
        T = h * W;
        B = rows_core / h;
        const int64_t chunk_bytes = in_tile(fmt) * T;
        Footprint fp = fp_common(T);
        fp.K_alloc = 0;
        fp.stream = true;
        const int64_t f_fixed = footprint_bytes(fp);
        const int64_t K_part = floor_div(l1_budget - f_fixed, chunk_bytes);
        if (K_part < 0) {
            refuse_sharded(
                fmt::format("the static CBs ({} B) exceed the L1 left beside the shards ({} B)", f_fixed, l1_budget));
        }
        K_alloc = std::min(K_part, B);
        fp.K_alloc = K_alloc;
        fp.stream = B > K_alloc;
        footprint = footprint_bytes(fp);
        ops_path = false;
    } else {
        if (fp_ops > l1_budget) {
            refuse_sharded(fmt::format(
                "the fully resident ragged RM pieces ({} B) exceed the L1 left beside the shards ({} B)",
                fp_ops,
                l1_budget));
        }
        rows_core = rows_nominal;
        h = h_ops;
        T = h * W;
        B = tt::div_up(rows_core, h);
        K_alloc = B;
        footprint = fp_ops;
        ops_path = true;
    }
    if (footprint > l1_budget) {
        refuse_sharded(
            fmt::format("the static CBs ({} B) exceed the L1 left beside the shards ({} B)", footprint, l1_budget));
    }

    Plan plan;
    plan.N = in.N;
    plan.HW = in.HW;
    plan.C = in.C;
    plan.G = in.G;
    plan.Ht = static_cast<uint32_t>(HW / kTileDim);
    plan.Wt = static_cast<uint32_t>(C / kTileDim);
    plan.Cg = static_cast<uint32_t>(Cg);
    plan.span_tiles = 0;
    plan.gps = 0;
    plan.S = static_cast<uint32_t>(gx);
    plan.gx = static_cast<uint32_t>(gx);
    plan.W = static_cast<uint32_t>(W);
    plan.G_core = static_cast<uint32_t>(g.lanes_max);
    plan.U = static_cast<uint32_t>(N * gx);
    plan.P = static_cast<uint32_t>(gx * gy);
    plan.cpi = static_cast<uint32_t>(g.unit_cores);
    plan.h = static_cast<uint32_t>(h);
    plan.T = static_cast<uint32_t>(T);
    plan.R = static_cast<uint32_t>(R);
    plan.rows_core_max = static_cast<uint32_t>(rows_core);
    plan.B_max = static_cast<uint32_t>(B);
    plan.K_alloc = static_cast<uint32_t>(K_alloc);
    plan.stream = B > K_alloc;
    plan.is_rm = is_rm;
    plan.has_gamma = in.has_gamma;
    plan.has_beta = in.has_beta;
    plan.grid_x = in.grid_x;
    plan.grid_y = in.grid_y;
    plan.sharded = true;
    plan.in_place = in.in_place;
    plan.map_period = static_cast<uint32_t>(sw);
    plan.shard_x0 = in.x0;
    plan.shard_y0 = in.y0;
    plan.shard_gx = in.gx;
    plan.shard_gy = in.gy;
    plan.NU = static_cast<uint32_t>(NU);
    plan.NSEG = static_cast<uint32_t>(NSEG);
    plan.ops_path = ops_path;
    plan.stage_pages = static_cast<uint32_t>(ops_path ? stage_pages : 0);
    plan.col_mask = col_mask;
    plan.mcast_row_major = !g.split_cols;
    plan.ring_pages = static_cast<uint32_t>(is_rm && ops_path ? max_ring_rows * W : 0);
    plan.pilot_cols = static_cast<uint32_t>(g.cl_cols);
    plan.fmt = fmt;
    return plan;
}

// The sharded per-core work (the tail of Python _plan_sharded): grid rows, then columns.
std::vector<CoreWork> sharded_work(const PlanInputs& in) {
    const ShardGeometry g = shard_geometry(in);
    const int64_t HW = g.HW, Cg = g.Cg, sh = g.sh, sw = g.sw;
    const int64_t x0 = g.x0, y0 = g.y0, gx = g.gx, gy = g.gy, W = g.W;
    const int64_t cl_cols = g.cl_cols;
    const bool split_cols = g.split_cols;
    const bool is_rm = in.is_rm;
    const int64_t h_ops = g.rows_nominal;

    std::vector<CoreWork> work;
    work.reserve(static_cast<size_t>(gx * gy));
    for (int64_t y = 0; y < gy; ++y) {
        const int64_t yb = y - (y % g.bh);
        const auto [n_first, n_img] = g.band_images(y);
        const int64_t img0 = (y * sh) / HW;  // the image of the row's first segment
        const int64_t n_seg = g.num_segments(y);
        std::vector<uint32_t> ops;
        std::vector<StagedRow> staged;
        int64_t rm_credit_pages = 0;
        if (is_rm) {
            auto add_op = [&ops](uint32_t op) { ops.push_back(op); };
            auto add_staged = [&staged](const StagedRow& trip) { staged.push_back(trip); };
            rm_core_ops(g, y, h_ops, add_op, add_staged);
            int64_t full_rows = 0, skips = 0;
            for (const uint32_t op : ops) {
                if ((op & (gn_sc_args::OP_STAGED | gn_sc_args::OP_SKIP)) == 0) {
                    full_rows += op & gn_sc_args::OP_ROWS;
                }
                if ((op & gn_sc_args::OP_SKIP) != 0) {
                    ++skips;
                }
            }
            rm_credit_pages = (full_rows + skips) * W;
        } else {
            g.for_each_segment(y, [&ops](const Segment& s) {
                ops.push_back(static_cast<uint32_t>((s.b - s.a) / kTileDim) | gn_sc_args::OP_SEG_END);
            });
        }
        for (int64_t x = 0; x < gx; ++x) {
            const int64_t cx0 = split_cols ? x : 0;
            const int64_t xs = x * sw;
            const int64_t g_lo = (cx0 * sw) / Cg;
            int64_t last_ch = cx0 * sw;
            for (int64_t c = cx0; c < cx0 + cl_cols; ++c) {
                last_ch += g.vcols(c);
            }
            const int64_t slot = split_cols ? (y - yb) : ((y - yb) * gx) + x;
            CoreWork w;
            w.core_x = static_cast<uint32_t>(x0 + x);
            w.core_y = static_cast<uint32_t>(y0 + y);
            w.unit_start = static_cast<uint32_t>((img0 * gx) + x);
            w.num_blocks = 1;
            w.row_start = 0;
            w.rows_core = static_cast<uint32_t>(g.vrows(y) / g.tr);
            w.is_root = slot == 0 ? 1 : 0;
            w.slot = static_cast<uint32_t>(slot);
            w.root_x = static_cast<uint32_t>(x0 + cx0);
            w.root_y = static_cast<uint32_t>(y0 + yb);
            w.xs = static_cast<uint32_t>(xs);
            w.vc = static_cast<uint32_t>(g.vcols(x));
            w.l0 = static_cast<uint32_t>((xs / Cg) - g_lo);
            w.off0 = static_cast<uint32_t>(xs % Cg);
            w.g_unit = static_cast<uint32_t>(tt::div_up(last_ch, Cg) - g_lo);
            w.n_img = static_cast<uint32_t>(n_img);
            w.n_seg = static_cast<uint32_t>(n_seg);
            w.seg_img0 = static_cast<uint32_t>(img0 - n_first);
            w.ops = ops;
            w.staged = staged;
            w.rm_credit_pages = static_cast<uint32_t>(rm_credit_pages);
            for (int64_t c = cx0; c < cx0 + cl_cols; ++c) {
                w.pilot_cols.push_back(static_cast<uint32_t>(x0 + c));
            }
            w.w_valid = static_cast<uint32_t>(is_rm ? W : g.vcols(x) / kTileDim);
            work.push_back(std::move(w));
        }
    }
    return work;
}

struct PlanInputsHash {
    size_t operator()(const PlanInputs& p) const {
        size_t h = 0;
        auto mix = [&h](auto v) { h ^= std::hash<decltype(v)>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(p.N);
        mix(p.HW);
        mix(p.C);
        mix(p.G);
        mix(p.is_rm);
        mix(p.has_gamma);
        mix(p.has_beta);
        mix(p.in_place);
        mix(static_cast<int>(p.fmt.in_dtype));
        mix(static_cast<int>(p.fmt.aff_dtype));
        mix(p.fmt.aff_tile);
        mix(p.grid_x);
        mix(p.grid_y);
        mix(p.l1_budget);
        mix(p.sharded);
        mix(p.sh);
        mix(p.sw);
        mix(p.x0);
        mix(p.y0);
        mix(p.gx);
        mix(p.gy);
        return h;
    }
};

}  // namespace

uint32_t l1_cb_budget(tt::tt_metal::IDevice* device) {
    // Python: the memory view's per-bank size (the allocator's L1 bank, measured from its base; the L1_SMALL region
    // above it is not counted), capped by the lowest live L1 buffer.
    const auto& allocator = device->allocator();
    uint64_t budget = allocator->get_statistics(BufferType::L1).total_allocatable_size_bytes;
    const auto lowest = device->lowest_occupied_compute_l1_address();
    if (lowest.has_value()) {
        budget = std::min<uint64_t>(budget, *lowest - allocator->get_base_allocator_addr(HalMemType::L1));
    }
    return static_cast<uint32_t>(budget);
}

PlanInputs plan_inputs(const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& tensors) {
    const Tensor& input = tensors.input;
    const auto& shape = input.logical_shape();
    PlanInputs in;
    in.N = shape[0];
    in.HW = shape[2];
    in.C = shape[3];
    in.G = attrs.num_groups;
    in.is_rm = input.layout() == Layout::ROW_MAJOR;
    in.has_gamma = tensors.gamma.has_value();
    in.has_beta = tensors.beta.has_value();
    in.in_place = attrs.in_place;
    const std::optional<Tensor>& ref = tensors.gamma.has_value() ? tensors.gamma : tensors.beta;
    in.fmt.in_dtype = input.dtype();
    in.fmt.aff_dtype = ref.has_value() ? ref->dtype() : DataType::BFLOAT16;
    in.fmt.aff_tile = ref.has_value() && ref->layout() == Layout::TILE;
    auto* device = input.device();
    const auto grid = device->compute_with_storage_grid_size();
    in.grid_x = static_cast<uint32_t>(grid.x);
    in.grid_y = static_cast<uint32_t>(grid.y);
    in.l1_budget = l1_cb_budget(device);
    if (input.is_sharded()) {
        const auto& mem = input.memory_config();
        TT_FATAL(mem.shard_spec().has_value(), "groupnorm_sc_N_1_HW_C_b: a sharded input needs a shard spec");
        const ShardSpec& spec = *mem.shard_spec();
        in.sharded = true;
        in.shard_in_l1 = mem.buffer_type() == BufferType::L1;
        in.shard_row_major = spec.orientation == ShardOrientation::ROW_MAJOR;
        in.shard_grid_num_cores = spec.grid.num_cores();
        in.sh = spec.shape[0];
        in.sw = spec.shape[1];
        const auto bbox = spec.grid.bounding_box();
        in.x0 = static_cast<uint32_t>(bbox.start_coord.x);
        in.y0 = static_cast<uint32_t>(bbox.start_coord.y);
        in.gx = static_cast<uint32_t>(bbox.end_coord.x - bbox.start_coord.x + 1);
        in.gy = static_cast<uint32_t>(bbox.end_coord.y - bbox.start_coord.y + 1);
    }
    return in;
}

const Plan& solve_plan(const PlanInputs& inputs) {
    static ttsl::Indestructible<std::mutex> mutex;
    static ttsl::Indestructible<std::unordered_map<PlanInputs, std::unique_ptr<Plan>, PlanInputsHash>> memo;
    const std::lock_guard<std::mutex> lock(mutex.get());
    auto& plans = memo.get();
    const auto it = plans.find(inputs);
    if (it != plans.end()) {
        return *it->second;
    }
    auto plan = std::make_unique<Plan>(inputs.sharded ? plan_sharded(inputs) : plan_interleaved(inputs));
    const Plan& ref = *plan;
    plans.emplace(inputs, std::move(plan));
    return ref;
}

std::vector<CoreWork> build_work(const PlanInputs& inputs, const Plan& plan) {
    if (inputs.sharded) {
        return sharded_work(inputs);
    }
    return distribute(plan.U, plan.P, plan.cpi, plan.Ht, plan.grid_x);
}

ttsl::hash::hash_t plan_hash(const Plan& p) {
    return ttsl::hash::hash_objects(
        0,
        std::array<uint32_t, 22>{
            p.N, p.HW,     p.C, p.G, p.Ht,  p.Wt, p.Cg, p.span_tiles, p.gps,           p.S,     p.gx,
            p.W, p.G_core, p.U, p.P, p.cpi, p.h,  p.T,  p.R,          p.rows_core_max, p.B_max, p.K_alloc},
        std::array<bool, 9>{
            p.stream,
            p.is_rm,
            p.has_gamma,
            p.has_beta,
            p.sharded,
            p.in_place,
            p.ops_path,
            p.col_mask,
            p.mcast_row_major},
        std::array<uint32_t, 14>{
            p.grid_x,
            p.grid_y,
            p.out_writer_chunks,
            p.map_period,
            p.vc_last,
            p.shard_x0,
            p.shard_y0,
            p.shard_gx,
            p.shard_gy,
            p.NU,
            p.NSEG,
            p.stage_pages,
            p.ring_pages,
            p.pilot_cols},
        p.fmt.in_dtype,
        p.fmt.aff_dtype,
        p.fmt.aff_tile);
}

uint32_t input_ring_pages(uint32_t T, uint32_t K_alloc, bool stream, bool is_rm) {
    return static_cast<uint32_t>(ring_pages_of(T, K_alloc, stream, is_rm));
}

uint32_t staging_bytes(const Plan& plan) {
    const int64_t n_affine = (plan.has_gamma ? 1 : 0) + (plan.has_beta ? 1 : 0);
    return static_cast<uint32_t>(
        staging_of(plan.W, plan.map_period, plan.is_rm, n_affine, plan.fmt, plan.sharded, plan.pilot_cols));
}

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b::planner
