// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_model_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tt_stl/assert.hpp>

// The features match the ones the model was trained on bit for bit only if the compiler evaluates every expression as
// written: no fused multiply-add contraction (build with -ffp-contract=off on GCC) and no -ffast-math. Expressions
// below keep the operation order the model was trained with on purpose; do not "simplify" them.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace ttnn::operations::matmul::model {

std::uint32_t tile_size_bytes(DataType t) {
    switch (t) {
        case DataType::kBfloat16: return kTileDim * kTileDim * 2;
        case DataType::kBfloat8B: return kTileDim * kTileDim + kTileDim * kTileDim / 16;
        case DataType::kBfloat4B: return kTileDim * kTileDim / 2 + kTileDim * kTileDim / 16;
        case DataType::kFloat32: return kTileDim * kTileDim * 4;
    }
    TT_THROW("Unreachable");
}

const std::array<std::string_view, kNumFeatures> kFeatureNames = {
    "x_fam",
    "x_kb",
    "x_nK",
    "x_log_nK",
    "x_obh",
    "x_obw",
    "x_sbh",
    "x_sbw",
    "x_nsb",
    "x_nob",
    "x_cores",
    "x_rx0",
    "x_rx1",
    "x_rd0",
    "x_rd1",
    "x_src_a",
    "x_src_b",
    "x_dst_o",
    "x_tb_a",
    "x_tb_b",
    "x_tb_p",
    "x_ph",
    "x_reloads",
    "x_dbuf",
    "x_kpad",
    "x_bk_a",
    "x_bk_b",
    "x_step_kb",
    "x_read_frac",
    "x_write_frac",
    "x_read_over_comp",
    "x_log_base_us",
    "x_batch",
    "x_Mt",
    "x_Kt",
    "x_Nt",
    "y_t_fetch0",
    "y_t_mcast0",
    "y_t_fetch1",
    "y_t_mcast1",
    "y_t_chip",
    "y_t_link",
    "y_t_comp",
    "y_t_write",
    "y_r_fetch0",
    "y_r_mcast0",
    "y_r_fetch1",
    "y_r_mcast1",
    "y_r_chip",
    "y_r_link",
    "y_r_comp",
    "y_bind",
    "y_link_over_read2",
    "y_dom_link",
    "y_top2_ratio",
    "z_l1c",
    "z_l1_over_comp",
};

std::string_view to_string(ScoreStatus s) {
    switch (s) {
        case ScoreStatus::kOk: return "ok";
        case ScoreStatus::kInvalidProblem: return "missing/invalid M,K,N,batch";
        case ScoreStatus::kGridExceedsDevice: return "grid exceeds the device worker grid";
        case ScoreStatus::kNonPositiveBlock: return "nonpositive block size";
        case ScoreStatus::kInvalidSubblock: return "nonpositive subblock size";
        case ScoreStatus::kNeedsMoreCores: return "config needs more cores than its grid";
        case ScoreStatus::kTransposedExceedsDevice:
            return "2d transpose_mcast=1 whose placement exceeds the device grid";
        case ScoreStatus::kNonFinite: return "model returned non-finite";
    }
    return "?";
}

void ModelCache::clear() {
    link_.clear();
    l1_.clear();
}

std::size_t ModelCache::size() const { return link_.size() + l1_.size(); }

const ModelCache::Lanes* ModelCache::find_link(std::uint64_t key) const {
    auto it = link_.find(key);
    return it == link_.end() ? nullptr : &it->second;
}

const ModelCache::Lanes& ModelCache::insert_link(std::uint64_t key, Lanes lanes) {
    return link_.insert_or_assign(key, std::move(lanes)).first->second;
}

const double* ModelCache::find_l1(std::uint64_t k0, std::uint64_t k1) const {
    auto it = l1_.find({k0, k1});
    return it == l1_.end() ? nullptr : &it->second;
}

void ModelCache::insert_l1(std::uint64_t k0, std::uint64_t k1, double cycles) {
    l1_.insert_or_assign({k0, k1}, cycles);
}

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// max that keeps the first argument unless a later one compares greater (NaN handling depends on the order).
inline double first_max(double a, double b) { return b > a ? b : a; }
inline double first_max(double a, double b, double c) { return first_max(first_max(a, b), c); }
// max / min that propagate NaN.
inline double nan_max(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? kNaN : (a >= b ? a : b); }
inline double nan_min(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? kNaN : (a <= b ? a : b); }

// Placement code: 0 DRAM interleaved, 1 L1 interleaved, 2 sharded
int src_code(MemLayout m) {
    switch (m) {
        case MemLayout::kDramInterleaved: return 0;
        case MemLayout::kL1Interleaved: return 1;
        case MemLayout::kL1HeightSharded:
        case MemLayout::kL1WidthSharded:
        case MemLayout::kL1BlockSharded: return 2;
    }
    TT_THROW("Unreachable");
}

double tiles_of(std::uint32_t v) { return std::ceil(static_cast<double>(v) / kTileDim); }

// ---------------------------------------------------------------- geometry (one row)
struct Geometry {
    Family fam{};
    double Mt{}, Kt{}, Nt{}, B{}, kb{}, obh{}, obw{}, sbh{}, sbw{};
    double cores{}, rx0{}, rx1{}, rd0{}, rd1{}, rrounds{}, rbsf{}, rtail{}, nob{}, nK{}, bloop{}, nsb{};
    bool dbuf{};
    double tb_a{}, tb_b{}, tb_o{}, tb_p{}, reloads{}, ph{}, kpad{};
    bool bias{}, sfpu{};
    int src_a{}, src_b{}, dst_o{};
    double R2{}, C2{};  // output block grid (rows, cols), batch rows stacked when fused
    double link = 0.0, bk_a = 0.0, bk_b = 0.0;
};

Geometry geometry(const MatmulProblem& p, const MatmulCandidate& c) {
    Geometry g{};
    g.fam = c.family;
    g.Mt = tiles_of(p.M);
    g.Kt = tiles_of(p.K);
    g.Nt = tiles_of(p.N);
    g.B = static_cast<double>(p.batch);
    const bool fuse = c.fuse_batch;
    const double pcM = c.per_core_M, pcN = c.per_core_N;
    double obh = c.out_block_h == 0 ? pcM : static_cast<double>(c.out_block_h);
    double obw = c.out_block_w == 0 ? pcN : static_cast<double>(c.out_block_w);
    g.kb = c.in0_block_w;
    g.sbh = c.out_subblock_h;
    g.sbw = c.out_subblock_w;
    const double Mrows = fuse ? g.B * g.Mt : g.Mt;
    const double bloop = fuse ? 1.0 : g.B;
    const bool is2d = c.family == Family::k2D, in0 = c.family == Family::k1DMcastIn0,
               in1 = c.family == Family::k1DMcastIn1, reuse = c.family == Family::kReuse;
    const double R2 = std::ceil(Mrows / pcM), C2 = std::ceil(g.Nt / pcN);
    const double P0 = std::ceil(g.Nt / pcN), P1 = std::ceil(Mrows / pcM);
    g.R2 = R2;
    g.C2 = C2;
    const double grid = static_cast<double>(c.grid_x) * static_cast<double>(c.grid_y);
    const double bsf = pcM > g.Mt ? std::floor(pcM / g.Mt) : 1.0;
    const double rblocks = std::ceil(g.B * g.Mt / pcM) * std::ceil(g.Nt / pcN);
    const double rcores = nan_min(rblocks, grid);
    g.cores = is2d ? R2 * C2 : in0 ? P0 : in1 ? P1 : rcores;
    g.rx0 = is2d ? C2 - 1 : in0 ? P0 - 1 : 0.0;
    g.rx1 = is2d ? R2 - 1 : in1 ? P1 - 1 : 0.0;
    g.rd0 = is2d ? R2 : in0 ? 1.0 : in1 ? P1 : rcores;
    g.rd1 = is2d ? C2 : in0 ? P0 : in1 ? 1.0 : rcores;
    if (reuse) {
        obh = nan_min(pcM, g.Mt);
        obw = pcN;
    }
    const double rounds = std::ceil(rblocks / rcores);
    g.rrounds = rounds;
    g.rbsf = bsf;
    g.rtail = reuse ? (rblocks - (rounds - 1) * rcores) / rcores : 1.0;
    g.nob = reuse ? rounds * bsf : bloop * std::ceil(pcM / obh) * std::ceil(pcN / obw);
    g.nK = std::ceil(g.Kt / g.kb);
    g.dbuf = reuse ? true : bloop * g.nK > 1;
    g.bloop = reuse ? 1.0 : bloop;
    g.obh = obh;
    g.obw = obw;
    g.nsb = std::ceil(obh / g.sbh) * std::ceil(obw / g.sbw);
    g.tb_a = tile_size_bytes(p.a_dtype);
    g.tb_b = tile_size_bytes(p.b_dtype);
    g.tb_o = tile_size_bytes(p.has_out_dtype ? p.out_dtype : p.a_dtype);
    const bool acc32 = p.fp32_dest_acc, l1acc = p.packer_l1_acc;
    g.bias = p.bias;
    g.tb_p = acc32 ? 4096.0 : (l1acc ? 2048.0 : g.tb_o);
    const double nK = g.nK;
    const bool l1on = l1acc && (g.bias ? nK > 1 : nK > 2);
    g.reloads = l1on ? (g.bias ? 0.0 : 1.0) : nan_max(nK - 1, 0.0);
    g.sfpu = p.activation == Activation::kSfpu;
    g.ph = fidelity_phases(p.fidelity);
    g.src_a = src_code(p.a_mem);
    g.src_b = src_code(p.b_mem);
    g.dst_o = src_code(p.out_mem);
    const double kr = std::fmod(static_cast<double>(p.K), static_cast<double>(kTileDim));
    g.kpad = kr > 0 ? kTileDim * (kTileDim - kr) : 0.0;
    return g;
}

// ---------------------------------------------------------------- NoC link loads per K step
// Link array: (noc, axis, x, y) -> index ((noc * 2 + axis) * X + x) * Y + y.
class LinkGrid {
public:
    explicit LinkGrid(const DeviceParams& d) :
        X_(static_cast<int>(d.noc_size_x)), Y_(static_cast<int>(d.noc_size_y)), v_(4 * X_ * Y_, 0.0) {}
    double& at(int noc, int axis, int x, int y) { return v_[((noc * 2 + axis) * X_ + x) * Y_ + y]; }
    int X() const { return X_; }
    int Y() const { return Y_; }
    const std::vector<double>& values() const { return v_; }

private:
    int X_, Y_;
    std::vector<double> v_;
};

struct Xy {
    int x, y;
};

inline int inc(int v, int n) { return v + 1 == n ? 0 : v + 1; }  // (v + 1) % n for 0 <= v < n
inline int dec(int v, int n) { return v == 0 ? n - 1 : v - 1; }  // (v - 1) mod n for 0 <= v < n

// Unicast route: NOC0 goes +x then +y, NOC1 goes -y then -x, both wrapping
void route(LinkGrid& L, int noc, Xy s, Xy t, double b) {
    int x = s.x, y = s.y;
    if (noc == 0) {
        while (x != t.x) {
            L.at(0, 0, x, y) += b;
            x = inc(x, L.X());
        }
        while (y != t.y) {
            L.at(0, 1, x, y) += b;
            y = inc(y, L.Y());
        }
    } else {
        while (y != t.y) {
            L.at(1, 1, x, y) += b;
            y = dec(y, L.Y());
        }
        while (x != t.x) {
            L.at(1, 0, x, y) += b;
            x = dec(x, L.X());
        }
    }
}

// Multicast over the physical rectangle [x0..x1] x [y0..y1]
void mcast_route(LinkGrid& L, int noc, Xy s, int x0, int x1, int y0, int y1, double b) {
    if (noc == 0) {
        route(L, 0, s, {x0, y0}, b);
        int x = x0;
        while (x != x1) {
            L.at(0, 0, x, y0) += b;
            x = inc(x, L.X());
        }
        for (x = x0; x < x1 + 1; ++x) {
            int y = y0;
            while (y != y1) {
                L.at(0, 1, x, y) += b;
                y = inc(y, L.Y());
            }
        }
    } else {
        route(L, 1, s, {x1, y1}, b);
        int y = y1;
        while (y != y0) {
            L.at(1, 1, x1, y) += b;
            y = dec(y, L.Y());
        }
        for (y = y0; y < y1 + 1; ++y) {
            int x = x1;
            while (x != x0) {
                L.at(1, 0, x, y) += b;
                x = dec(x, L.X());
            }
        }
    }
}

Xy phys(const DeviceParams& d, int i, int j) {
    return {static_cast<int>(d.worker_cols[i]), static_cast<int>(d.worker_rows[j])};
}

// One byte read by `reader`, spread evenly over the interleaved banks (DRAM) or over every worker (L1)
void read_route(LinkGrid& L, const DeviceParams& d, int noc, Xy reader, int src) {
    if (src == 0) {
        const double share = 1.0 / static_cast<double>(d.num_banks);
        for (std::uint32_t b = 0; b < d.num_banks; ++b) {
            const auto& bk = d.banks[b];
            const auto& ep = d.channel_endpoints[bk[0]][noc == 0 ? bk[1] : bk[2]];
            route(L, noc, {static_cast<int>(ep[0]), static_cast<int>(ep[1])}, reader, share);
        }
    } else if (src == 1) {
        const int gx = static_cast<int>(d.worker_grid_x), gy = static_cast<int>(d.worker_grid_y);
        const double share = 1.0 / static_cast<double>(gx * gy);
        for (int i = 0; i < gx; ++i) {
            for (int j = 0; j < gy; ++j) {
                route(L, noc, phys(d, i, j), reader, share);
            }
        }
    }
}

// Link loads per byte of one in0 / in1 block (transposed 2D multicast routed as transposed)
ModelCache::Lanes unit_loads(
    const DeviceParams& d, Family fam, bool transposed, int R, int C, int P, int gx, int gy, int src_a, int src_b) {
    LinkGrid U0(d), U1(d);
    auto ph = [&](int i, int j) { return phys(d, i, j); };
    if (fam == Family::k2D && !transposed) {  // in0: left column reads + mcasts along its row (NOC1); in1: top row
        for (int j = 0; j < R; ++j) {
            Xy s = ph(0, j);
            read_route(U0, d, 1, s, src_a);
            if (C > 1) {
                mcast_route(U0, 1, s, ph(1, j).x, ph(C - 1, j).x, s.y, s.y, 1.0);
            }
        }
        for (int i = 0; i < C; ++i) {
            Xy s = ph(i, 0);
            read_route(U1, d, 0, s, src_b);
            if (R > 1) {
                mcast_route(U1, 0, s, s.x, s.x, ph(i, 1).y, ph(i, R - 1).y, 1.0);
            }
        }
    } else if (fam == Family::k2D) {  // transposed: in0 top row reads (NOC1) + mcasts down its column (NOC1, -y)
        for (int i = 0; i < R; ++i) {
            Xy s = ph(i, 0);
            read_route(U0, d, 1, s, src_a);
            if (C > 1) {
                mcast_route(U0, 1, s, s.x, s.x, ph(i, 1).y, ph(i, C - 1).y, 1.0);
            }
        }
        for (int j = 0; j < C; ++j) {  // in1: left column reads (NOC0) + mcasts along its row (NOC0, +x)
            Xy s = ph(0, j);
            read_route(U1, d, 0, s, src_b);
            if (R > 1) {
                mcast_route(U1, 0, s, ph(1, j).x, ph(R - 1, j).x, s.y, s.y, 1.0);
            }
        }
    } else if (fam == Family::k1DMcastIn0 || fam == Family::k1DMcastIn1) {
        // one sender at (0,0) mcasts to the whole grid; every core reads the other operand
        const int rows = static_cast<int>(std::ceil(static_cast<double>(P) / gx));
        const bool m0 = fam == Family::k1DMcastIn0;
        LinkGrid& snd = m0 ? U0 : U1;
        LinkGrid& own = m0 ? U1 : U0;
        const int ns = m0 ? 1 : 0, no = m0 ? 0 : 1, ss = m0 ? src_a : src_b, so = m0 ? src_b : src_a;
        Xy s = ph(0, 0);
        read_route(snd, d, ns, s, ss);
        if (P > 1) {
            mcast_route(snd, ns, s, ph(0, 0).x, ph(gx - 1, 0).x, ph(0, 0).y, ph(0, std::max(rows, gy) - 1).y, 1.0);
        }
        for (int k = 0; k < P; ++k) {
            read_route(own, d, no, ph(k % gx, k / gx), so);
        }
    } else {  // Reuse: every core reads both (in0 on NOC0, in1 on NOC1)
        for (int k = 0; k < P; ++k) {
            Xy c = ph(k % gx, k / gx);
            read_route(U0, d, 0, c, src_a);
            read_route(U1, d, 1, c, src_b);
        }
    }
    ModelCache::Lanes lanes;
    const auto& a = U0.values();
    const auto& b = U1.values();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] > 0 || b[i] > 0) {
            lanes.emplace_back(a[i], b[i]);
        }
    }
    return lanes;
}

double link_bytes(const DeviceParams& d, const Geometry& g, const MatmulCandidate& c, ModelCache* cache) {
    const bool transposed = g.fam == Family::k2D && c.transpose_mcast;
    const int R = g.fam == Family::k2D ? static_cast<int>(g.R2) : 0;
    const int C = g.fam == Family::k2D ? static_cast<int>(g.C2) : 0;
    const int P = static_cast<int>(g.cores);
    const int gx = static_cast<int>(c.grid_x), gy = static_cast<int>(c.grid_y);
    // key: family (2 bits) | transposed | src_a, src_b (2 bits each) | gx, gy (6 bits) | R, C (8 bits) | P (12 bits)
    const bool keyable = R < 256 && C < 256 && P < 4096 && gx < 64 && gy < 64 && R >= 0 && C >= 0 && P >= 0;
    const std::uint64_t key = (static_cast<std::uint64_t>(g.fam) << 0) | (std::uint64_t(transposed) << 2) |
                              (std::uint64_t(g.src_a) << 3) | (std::uint64_t(g.src_b) << 5) | (std::uint64_t(gx) << 7) |
                              (std::uint64_t(gy) << 13) | (std::uint64_t(R) << 19) | (std::uint64_t(C) << 27) |
                              (std::uint64_t(P) << 35);
    const ModelCache::Lanes* lanes = (cache && keyable) ? cache->find_link(key) : nullptr;
    ModelCache::Lanes local;
    if (lanes == nullptr) {
        local = unit_loads(d, g.fam, transposed, R, C, P, gx, gy, g.src_a, g.src_b);
        lanes = (cache && keyable) ? &cache->insert_link(key, std::move(local)) : &local;
    }
    if (lanes->empty()) {
        return 0.0;
    }
    const double b0 = g.obh * g.kb * g.tb_a;
    const double b1 = g.kb * g.obw * g.tb_b;
    double best = -kInf;
    for (const auto& [u0, u1] : *lanes) {
        const double v = b0 * u0 + b1 * u1;
        best = nan_max(best, v);
    }
    return best;
}

// ---------------------------------------------------------------- DRAM banks touched per K step
// Row (or column) indices one K step touches, readers per_core apart, capped at 48 distinct
std::vector<std::int64_t> touched_rows(
    std::int64_t readers, std::int64_t per_core, std::int64_t block, std::int64_t total) {
    std::vector<std::int64_t> idx;
    const std::int64_t nr = std::min(readers, total), nbk = std::min(block, total);
    for (std::int64_t r = 0; r < nr; ++r) {
        for (std::int64_t i = 0; i < nbk; ++i) {
            const std::int64_t v = r * per_core + i;
            if (v < total && std::find(idx.begin(), idx.end(), v) == idx.end()) {
                idx.push_back(v);
            }
            if (idx.size() >= 48) {
                break;
            }
        }
        if (idx.size() >= 48) {
            break;
        }
    }
    return idx;
}

double banks_touched(
    std::int64_t nb,
    std::int64_t stride_r,
    const std::vector<std::int64_t>& rows,
    std::int64_t stride_k,
    std::int64_t kb) {
    std::array<bool, DeviceParams::kMaxBanks> hit{};
    const std::int64_t nk = std::min(kb, nb);
    for (std::int64_t r : rows) {
        for (std::int64_t k = 0; k < nk; ++k) {
            const std::int64_t v = r * stride_r + k * stride_k;
            hit[((v % nb) + nb) % nb] = true;
        }
    }
    return static_cast<double>(std::count(hit.begin(), hit.begin() + nb, true));
}

void bank_loads(const DeviceParams& d, const MatmulProblem& p, const MatmulCandidate& c, Geometry& g) {
    const auto n = static_cast<std::int64_t>(d.num_banks);
    const auto Mt = static_cast<std::int64_t>(g.Mt), Kt = static_cast<std::int64_t>(g.Kt),
               Nt = static_cast<std::int64_t>(g.Nt);
    auto iv = [](double v, std::int64_t dflt) {
        return (std::isfinite(v) && v > 0) ? static_cast<std::int64_t>(v) : dflt;
    };
    const std::int64_t kb = iv(g.kb, Kt);
    const std::int64_t Mrows = c.fuse_batch ? static_cast<std::int64_t>(g.Mt * g.B) : Mt;
    const auto rows = touched_rows(iv(g.rd0, 1), iv(c.per_core_M, 1), iv(g.obh, Mrows), Mrows);
    const auto cols = touched_rows(iv(g.rd1, 1), iv(c.per_core_N, 1), iv(g.obw, Nt), Nt);
    g.bk_a = p.transpose_a ? banks_touched(n, 1, rows, Mt, kb) : banks_touched(n, Kt, rows, 1, kb);
    g.bk_b = p.transpose_b ? banks_touched(n, Kt, cols, 1, kb) : banks_touched(n, 1, cols, Nt, kb);
}

// ---------------------------------------------------------------- the latency equation (one row)
// Terms: link bank sat burst spill mcast pack call write epilogue issue pad msync mcrate corecc reusesync dramw
// shardhop wfam reusetail pack32. No other terms are modeled.
struct Prediction {
    double t, read, comp, write, nK, nob;
    double fetch0, mcast0, fetch1, mcast1, chip, lk;
};

Prediction predict(const DeviceParams& d, const Geometry& g) {
    const FittedConstants& p = d.c;
    const double clk = d.clock_hz;
    const double noc = d.noc_bytes_per_cycle * p.noc_eff;
    const double dram = d.dram_GBs * 1e9 * p.dram_eff / clk;
    const double l1bw = d.dram_GBs * 1e9 * p.l1_frac / clk;
    const double kb = g.kb, obh = g.obh, obw = g.obw, sbh = g.sbh, sbw = g.sbw, nsb = g.nsb, nK = g.nK;
    const double b0 = obh * kb * g.tb_a, b1 = kb * obw * g.tb_b;

    const double ea = nan_min(1.0, g.bk_a * p.bank_frac);
    const double eb = nan_min(1.0, g.bk_b * p.bank_frac);
    const double cg0 = 1 + b0 / (p.burst_KB * 1e3);
    const double cg1 = 1 + b1 / (p.burst_KB * 1e3);

    auto fetch = [&](double nbytes, int src, double readers, double eff, double cg, double tile) {
        const double r = nan_max(readers, 1.0);
        const double floor = nbytes / tile * p.issue;
        const double noc_rd = r > 1 ? d.noc_bytes_per_cycle * p.noc_eff_cc : noc;
        const double cgd = 1.0, xs = 0.0;
        const double dr =
            nan_max(p.lat_dram + nbytes * cgd / noc_rd, xs * p.lat_dram + nbytes * r * cgd / (dram * eff));
        const double l1 = nan_max(p.lat_l1 + nbytes / noc, xs * p.lat_l1 + nbytes * r * cg / l1bw);
        return src == 0 ? nan_max(dr, floor) : src == 1 ? nan_max(l1, floor) : 0.0;
    };
    auto mcast = [&](double nbytes, double rx) {
        const double rate = d.noc_bytes_per_cycle * nan_max(p.mc_eff0 - p.mc_eff_rx * rx, 0.05);
        return rx > 0 ? p.lat_mcast + rx * p.ack_rx + nbytes / rate : 0.0;
    };

    Prediction o{};
    o.fetch0 = fetch(b0, g.src_a, g.rd0, ea, cg0, g.tb_a);
    o.mcast0 = mcast(b0, g.rx0);
    o.fetch1 = fetch(b1, g.src_b, g.rd1, eb, cg1, g.tb_b);
    o.mcast1 = mcast(b1, g.rx1);
    const double step0 = o.fetch0 + o.mcast0;
    const double step1 = o.fetch1 + o.mcast1;
    const double da = g.rd0 * b0 * static_cast<double>(g.src_a == 0),
                 db = g.rd1 * b1 * static_cast<double>(g.src_b == 0);
    const double la = g.rd0 * b0 * static_cast<double>(g.src_a == 1),
                 lb = g.rd1 * b1 * static_cast<double>(g.src_b == 1);
    double chip = nan_max((da + db) / dram, (la * cg0 + lb * cg1) / l1bw);
    chip = nan_max(chip, nan_max(da / (dram * ea), db / (dram * eb)));
    double read = nan_max(nan_max(step0, step1), chip);
    double shared = chip;
    const double lk = g.link * (g.fam == Family::kReuse ? p.reuse_link : 1.0);
    read = nan_max(read, lk / (d.noc_bytes_per_cycle * p.link_eff));
    shared = nan_max(shared, lk / (d.noc_bytes_per_cycle * p.link_eff));
    o.chip = chip;
    o.lk = lk;

    // compute per K block
    const double math = kb * sbh * sbw * 16.0 * g.ph;
    double pack = sbh * sbw * g.tb_p / p.pack_Bpc;
    if (g.tb_p >= 4096) {
        pack = sbh * sbw * g.tb_p / p.pack_Bpc32;
    }
    const double fixed = kb * p.call;
    const double tiles = obh * obw;
    const double reload = g.reloads * (nsb * p.rl_init + (g.tb_p >= 4096 ? tiles * p.u2d : 0.0));
    const double stage = nan_max(math, pack) + 0.0;
    double comp = nsb * (stage + fixed) + reload / nan_max(nK, 1.0);
    double epi = g.bias ? tiles * g.tb_o / p.pack_Bpc : 0.0;
    epi = epi + (g.sfpu ? tiles * p.sfpu_tile : 0.0);

    // output write
    const double wbytes = tiles * g.tb_o;
    const double cw = 1.0;
    double dram_w = d.dram_GBs * 1e9 * p.dram_w_eff / clk;
    const double wf = g.fam == Family::kReuse        ? p.wf_reuse
                      : g.fam == Family::k1DMcastIn0 ? p.wf_in0
                      : g.fam == Family::k1DMcastIn1 ? p.wf_in1
                                                     : 1.0;
    dram_w = dram_w * wf;
    const double rate_w = nan_min(noc, dram_w / (nan_max(g.cores, 1.0) * cw));
    const double write = g.dst_o == 0   ? nsb * p.lat_write + wbytes / rate_w
                         : g.dst_o == 1 ? nsb * p.lat_l1 + wbytes / noc
                                        : 0.0;

    const double sync = g.rx0 * p.ack_rx + g.rx1 * p.ack_rx;  // msync
    comp = comp + sync;
    comp = comp + ((g.src_a == 2 && g.rx0 > 0) ? p.lat_shard : 0.0);  // shardhop
    const double ks = 0.0;
    const double piped = read + (nK - 1) * (nan_max(read, comp) + ks) + comp;
    const double serial = nK * (read + comp);
    double block = (g.dbuf ? piped : serial) + epi + write;
    const double lat0 = g.src_a == 0 ? p.lat_dram : g.src_a == 1 ? p.lat_l1 : 0.0;
    const double pad = g.kpad > 0 ? obh * (lat0 + g.kpad * p.pad_elem) : 0.0;
    block = block + pad;
    double core = g.nob * block;
    if (g.fam == Family::kReuse) {  // reusetail: the last round runs on the tail's share of the cores
        const double ft = g.rtail;
        const double f0 = fetch(b0, g.src_a, g.rd0 * ft, ea, cg0, g.tb_a);
        const double f1 = fetch(b1, g.src_b, g.rd1 * ft, eb, cg1, g.tb_b);
        const double read_t = nan_max(nan_max(f0, f1), shared * ft);
        const double saved = (read - read_t) + nan_max(nK - 1, 0.0) * (nan_max(read, comp) - nan_max(read_t, comp));
        core = core - g.rbsf * saved;
    }
    o.t = p.launch_us * 1e3 + core / clk * 1e9;
    o.read = read;
    o.comp = comp;
    o.write = write;
    o.nK = nK;
    o.nob = g.nob;
    return o;
}

// ---------------------------------------------------------------- L1 per-core compute model
const L1ShapeCosts* l1_shape(const L1CostTables& t, std::uint32_t h, std::uint32_t w, Fidelity f) {
    for (const auto& s : t.shapes) {
        if (s.h == h && s.w == w && s.fidelity == f) {
            return &s;
        }
    }
    return nullptr;
}

// Linear interpolation of vs over sorted ks at x, held constant beyond the ends (NaN when the table has no points)
double interp(
    double x,
    std::uint32_t n,
    const std::array<std::uint32_t, L1ShapeCosts::kMaxPoints>& ks,
    const std::array<double, L1ShapeCosts::kMaxPoints>& vs) {
    if (n == 0) {
        return kNaN;
    }
    if (n == 1 || x < ks[0]) {
        return (x > ks[0]) ? vs[n - 1] : vs[0];
    }
    if (x >= ks[n - 1]) {
        return vs[n - 1];
    }
    std::uint32_t j = 0;
    while (j + 1 < n && static_cast<double>(ks[j + 1]) <= x) {
        ++j;
    }
    if (x == static_cast<double>(ks[j])) {
        return vs[j];
    }
    const double slope = (vs[j + 1] - vs[j]) / (static_cast<double>(ks[j + 1]) - static_cast<double>(ks[j]));
    double r = slope * (x - static_cast<double>(ks[j])) + vs[j];
    if (std::isnan(r)) {
        r = slope * (x - static_cast<double>(ks[j + 1])) + vs[j + 1];
        if (std::isnan(r) && vs[j] == vs[j + 1]) {
            r = vs[j];
        }
    }
    return r;
}

// The subblock-level compute / pack recurrence over nb K blocks of S subblocks. Pe(q) is the pack end of
// subblock-step q; only Pe(q - 1), Pe(q - 2) and Pe(q - S) are read, so a power-of-two ring of > max(S, 2) slots holds
// them.
double l1_run(
    double kw,
    std::int64_t nb,
    std::int64_t S,
    double call,
    double o_c,
    double pack,
    double xp,
    double xr,
    double f1,
    double f2,
    double oblk,
    std::vector<double>& ring) {
    const std::int64_t n = nb * S;
    std::size_t m = 4;
    while (m <= static_cast<std::size_t>(std::max<std::int64_t>(S, 2))) {
        m <<= 1;
    }
    if (ring.size() < m) {
        ring.resize(m);
    }
    const std::size_t mask = m - 1;
    double* P = ring.data();
    double Ce = 0.0;
    const double work = kw * call;
    for (std::int64_t k = 0; k < nb; ++k) {
        for (std::int64_t j = 0; j < S; ++j) {
            const std::int64_t q = k * S + j;
            const double dst = q >= 2 ? P[static_cast<std::size_t>(q - 2) & mask] : -kInf;
            if (k == 0) {
                Ce = first_max(Ce, dst) + work + o_c;
            } else {
                double first = 0.0;
                if (k == 1 && j == 0) {
                    first = S > 1 ? f2 : f1;
                }
                const double ob = (j == 0 ? oblk : 0.0) + first;
                const double ready =
                    first_max(Ce + ob + xr, P[static_cast<std::size_t>(q - S) & mask] + ob + xp, dst + xr);
                Ce = ready + work + o_c;
            }
            const double prev = q >= 1 ? P[static_cast<std::size_t>(q - 1) & mask] : -kInf;
            P[static_cast<std::size_t>(q) & mask] = first_max(prev, Ce) + pack;
        }
    }
    return P[static_cast<std::size_t>(n - 1) & mask];
}

double l1_cycles(const DeviceParams& d, const MatmulProblem& p, const MatmulCandidate& c, ModelCache* cache) {
    if (p.fidelity == Fidelity::kHiFi3) {  // no HiFi3 tables
        return kNaN;
    }
    const std::uint32_t h = c.out_subblock_h, w = c.out_subblock_w, kwi = c.in0_block_w;
    const std::uint32_t pcM = c.per_core_M, pcN = c.per_core_N;
    if (h == 0 || w == 0 || kwi == 0 || pcM % h != 0 || pcN % w != 0) {
        return kNaN;
    }
    const L1ShapeCosts* s = l1_shape(d.l1, h, w, p.fidelity);
    if (s == nullptr) {
        return kNaN;
    }
    const auto nb = static_cast<std::int64_t>(std::ceil(std::ceil(static_cast<double>(p.K) / 32) / kwi));
    const auto S = static_cast<std::int64_t>(pcM / h) * static_cast<std::int64_t>(pcN / w);
    // memo key: (h, w, fidelity, kw) and (nb, pcM, pcN)
    const std::uint64_t k0 = (std::uint64_t(h) << 40) | (std::uint64_t(w) << 32) |
                             (std::uint64_t(static_cast<std::uint8_t>(p.fidelity)) << 24) | kwi;
    const std::uint64_t k1 = (std::uint64_t(nb) << 40) ^ (std::uint64_t(pcM) << 20) ^ std::uint64_t(pcN);
    const bool keyable = kwi < (1u << 24) && nb < (std::int64_t(1) << 24) && pcM < (1u << 20) && pcN < (1u << 20);
    if (cache && keyable) {
        if (const double* v = cache->find_l1(k0, k1)) {
            return *v;
        }
    }
    const double kw = kwi;
    const double pack = d.l1.pack_p0 + d.l1.pack_p1 * static_cast<double>(h * w);
    // interp_kw(F2, key, kw): measured at kw in {1, 4}
    bool has_f2 = true;
    double f2 = 0.0;
    if (!s->has_f2_kw1 || !s->has_f2_kw4) {
        has_f2 = s->has_f2_kw4 || s->has_f2_kw1;
        f2 = s->has_f2_kw4 ? s->f2_kw4 : s->f2_kw1;
    } else if (kw <= 1) {
        f2 = s->f2_kw1;
    } else if (kw >= 4) {
        f2 = s->f2_kw4;
    } else {
        f2 = s->f2_kw1 + (s->f2_kw4 - s->f2_kw1) * (kw - 1) / 3;
    }
    double f1;
    if (s->has_f1) {
        f1 = s->f1;
    } else {
        f1 = s->has_f2_kw4 ? s->f2_kw4
                           : ((has_f2 && f2 != 0.0) ? f2 : 0.0);  // the kw=4 cost, else the kw=1 cost, else 0
    }
    if (!has_f2) {
        f2 = f1;
    }
    const double xp = interp(kw, s->n_xp, s->xp_kw, s->xp), xr = interp(kw, s->n_xr, s->xr_kw, s->xr);
    const double xp1 = interp(1.0, s->n_xp, s->xp_kw, s->xp), xr1 = interp(1.0, s->n_xr, s->xr_kw, s->xr);
    thread_local std::vector<double> ring;
    const double o_c = d.l1.overhead_per_subblock, oblk = d.l1.block_overhead;
    const double a = l1_run(kw, nb, S, s->calls, o_c, pack, xp, xr, f1, f2, oblk, ring);
    const double b = l1_run(1.0, 1, 1, s->calls, o_c, pack, xp1, xr1, f1, f2, oblk, ring);
    const double cycles = s->kref + a - b;
    if (cache && keyable) {
        cache->insert_l1(k0, k1, cycles);
    }
    return cycles;
}

// ---------------------------------------------------------------- scorability
ScoreStatus check(const DeviceParams& d, const MatmulProblem& p, const MatmulCandidate& c) {
    if (p.M == 0 || p.K == 0 || p.N == 0 || p.batch == 0) {
        return ScoreStatus::kInvalidProblem;
    }
    const double GX = d.worker_grid_x, GY = d.worker_grid_y;
    const double gx = c.grid_x, gy = c.grid_y;
    if (gx > GX || gy > GY) {
        return ScoreStatus::kGridExceedsDevice;
    }
    if (c.per_core_M < 1 || c.per_core_N < 1 || c.in0_block_w < 1) {
        return ScoreStatus::kNonPositiveBlock;
    }
    const double Mt = tiles_of(p.M), Nt = tiles_of(p.N);
    const double Mrows = c.fuse_batch ? static_cast<double>(p.batch) * Mt : Mt;
    const double R2 = std::ceil(Mrows / c.per_core_M), C2 = std::ceil(Nt / c.per_core_N);
    const double P = c.family == Family::k1DMcastIn0 ? C2 : c.family == Family::k1DMcastIn1 ? R2 : 0.0;
    const bool is2d = c.family == Family::k2D;
    const bool tm = is2d && c.transpose_mcast;
    if ((is2d && !tm && (R2 > gy || C2 > gx)) || (tm && (R2 > gx || C2 > gy)) || P > gx * gy) {
        return ScoreStatus::kNeedsMoreCores;
    }
    if (tm && (R2 > GX || C2 > GY)) {
        return ScoreStatus::kTransposedExceedsDevice;
    }
    if (c.out_subblock_h == 0 || c.out_subblock_w == 0) {
        return ScoreStatus::kInvalidSubblock;
    }
    return ScoreStatus::kOk;
}

double fam_code(Family f) {
    switch (f) {
        case Family::k2D: return 0.0;
        case Family::k1DMcastIn0: return 1.0;
        case Family::k1DMcastIn1: return 2.0;
        case Family::kReuse: return 3.0;
    }
    return 5.0;
}

}  // namespace

ModelInput compute_model_input(
    const MatmulProblem& problem, const MatmulCandidate& candidate, const DeviceParams& device, ModelCache* cache) {
    ModelInput out;
    out.features.fill(kNaN);
    out.base_ns = kNaN;
    out.terms = {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN};

    // z_l1c is computed for every row (as in training); z_l1_over_comp needs the equation's compute term
    out.l1_cycles = l1_cycles(device, problem, candidate, cache);
    out.features[55] = std::log(out.l1_cycles);

    out.status = check(device, problem, candidate);
    if (out.status != ScoreStatus::kOk) {
        return out;
    }
    Geometry g = geometry(problem, candidate);
    g.link = link_bytes(device, g, candidate, cache);
    bank_loads(device, problem, candidate, g);
    const Prediction pr = predict(device, g);
    if (!std::isfinite(pr.t)) {
        out.status = ScoreStatus::kNonFinite;
        return out;
    }
    out.base_ns = pr.t;

    // The 36 mechanism features
    auto& f = out.features;
    const double read = pr.read, comp = pr.comp, write = pr.write, nK = pr.nK;
    const double loop = read + nan_max(nK - 1, 0.0) * nan_max(read, comp) + comp;
    const double tot = loop + write + 1.0;
    f[0] = fam_code(g.fam);
    f[1] = g.kb;
    f[2] = nK;
    f[3] = std::log1p(nK);
    f[4] = g.obh;
    f[5] = g.obw;
    f[6] = g.sbh;
    f[7] = g.sbw;
    f[8] = g.nsb;
    f[9] = g.nob;
    f[10] = g.cores;
    f[11] = g.rx0;
    f[12] = g.rx1;
    f[13] = g.rd0;
    f[14] = g.rd1;
    f[15] = g.src_a;
    f[16] = g.src_b;
    f[17] = g.dst_o;
    f[18] = g.tb_a;
    f[19] = g.tb_b;
    f[20] = g.tb_p;
    f[21] = g.ph;
    f[22] = g.reloads;
    f[23] = g.dbuf ? 1.0 : 0.0;
    f[24] = g.kpad;
    f[25] = g.bk_a;
    f[26] = g.bk_b;
    f[27] = (g.obh * g.tb_a + g.obw * g.tb_b) * g.kb / 1024;
    f[28] = read * nK / tot;
    f[29] = write / tot;
    f[30] = std::log((read + 1) / (comp + 1));
    f[31] = std::log(nan_max(pr.t, 1.0) / 1e3);
    f[32] = g.B;
    f[33] = g.Mt;
    f[34] = g.Kt;
    f[35] = g.Nt;

    // The per-resource terms (cycles per K step)
    const double lk = pr.lk / (device.noc_bytes_per_cycle * device.c.link_eff);
    const double f0 = pr.fetch0, m0 = pr.mcast0, f1 = pr.fetch1, m1 = pr.mcast1, chip = pr.chip;
    out.terms = {f0, m0, f1, m1, chip, lk, comp, write, read};
    const double step = nan_max(read, comp) + 1.0;
    const std::array<double, 8> raw = {f0, m0, f1, m1, chip, lk, comp, write};
    for (std::size_t i = 0; i < raw.size(); ++i) {
        f[36 + i] = std::log1p(raw[i]);
    }
    for (std::size_t i = 0; i < 7; ++i) {
        f[44 + i] = raw[i] / step;
    }
    {
        const std::array<double, 5> cand = {comp, f0 + m0, f1 + m1, chip, lk};
        std::size_t arg = 0;
        for (std::size_t i = 1; i < cand.size(); ++i) {
            if (cand[i] > cand[arg]) {
                arg = i;
            }
        }
        f[51] = static_cast<double>(arg);
    }
    f[52] = lk / (nan_max(nan_max(f0 + m0, f1 + m1), chip) + 1.0);
    // Equation-term features: from the log1p'd terms, round-tripped through expm1 as in training
    {
        auto t = [&](std::size_t i) { return std::expm1(f[36 + i]); };
        const double tf0 = t(0), tm0 = t(1), tf1 = t(2), tm1 = t(3), tchip = t(4), tlink = t(5), tcomp = t(6);
        const double others = nan_max(nan_max(nan_max(tcomp, tchip), tf0 + tm0), tf1 + tm1);
        std::array<double, 5> st = {tcomp, tchip, tf0 + tm0, tf1 + tm1, tlink};
        std::sort(st.begin(), st.end());
        f[53] = std::log((tlink + 1) / (others + 1));
        f[54] = std::log((st[4] + 1) / (st[3] + 1));
        // L1 compute features
        const double ccomp = tcomp * nK;
        f[56] = std::log(out.l1_cycles / nan_max(ccomp, 1e-9));
    }
    return out;
}

}  // namespace ttnn::operations::matmul::model
