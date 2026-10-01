// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_ttnn_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

namespace {

namespace kh = ttnn::kernel_lib::host;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_pre_ttnn/kernels/";

constexpr int64_t TILE = 32;
constexpr int64_t F32_TILE_BYTES = 4096;
constexpr int64_t BF16_TILE_BYTES = 2048;

// ---- CB indices (mhc_pre_program_descriptor.py) ----
constexpr uint8_t CB_X_RESIDENT = 0;
constexpr uint8_t CB_WEIGHT = 1;
constexpr uint8_t CB_BIAS_COEF = 2;
constexpr uint8_t CB_REDUCE_SCALER = 3;
constexpr uint8_t CB_SQ_ACC = 4;
constexpr uint8_t CB_PARTIAL = 5;
constexpr uint8_t CB_GATHERED = 6;
constexpr uint8_t CB_COMBINED = 7;
constexpr uint8_t CB_COEF_IN = 8;
constexpr uint8_t CB_COEF_KEEP = 9;
constexpr uint8_t CB_COMB_COEF = 11;
constexpr uint8_t CB_PRE_COLS = 12;
constexpr uint8_t CB_Y_OUT = 13;
constexpr uint8_t CB_WEIGHT_SPLIT = 15;
constexpr uint8_t CB_X_FP32 = 16;
constexpr uint8_t CB_X_PIECES = 17;
constexpr uint8_t CB_MIX_RUN = 18;
constexpr uint8_t CB_MAX_LANES = 19;
constexpr uint8_t CB_MAX_SCALAR = 20;
constexpr uint8_t CB_GRID = 21;
constexpr uint8_t CB_MAX_SCALER = 22;
constexpr uint8_t CB_W_OWN_READY = 23;
constexpr uint8_t CB_W_OWN_SPLIT = 24;
constexpr uint8_t CB_W_SHARE_LANDED = 25;
constexpr int64_t TOKEN_PAGE_BYTES = 32;
constexpr size_t NUM_CB_SLOTS = 64;
constexpr uint8_t UNPACK_TO_DEST_FP32_CBS[] = {CB_BIAS_COEF, CB_SQ_ACC, CB_GATHERED, CB_COEF_IN, CB_COEF_KEEP};

// ---- semaphores ----
constexpr uint32_t SEM_GATHER = 0;
constexpr uint32_t SEM_MCAST_READY = 1;
constexpr uint32_t SEM_MCAST_CONSUMED = 2;
constexpr uint32_t SEM_W_READY = 3;

// ---- host knobs (the Python module's values; see mhc_pre_program_descriptor.py for the measurements) ----
constexpr int64_t GROUP_CORES_CAP = 32;
constexpr int64_t X_BLOCK_DEPTH_DEFAULT = 2;
constexpr int64_t COEF_IN_BLOCKS = 2;
constexpr uint32_t X_STREAM_CHUNKS = 4;
constexpr uint32_t X_STREAM_INFLIGHT = 2;
constexpr int64_t Y_DEPTH = 2;
constexpr int64_t Y_CHUNK_TILES_CAP = 8;
constexpr uint32_t W_CHUNK_TILES = 8;
constexpr MathFidelity W_LO_FIDELITY = MathFidelity::LoFi;
constexpr uint32_t X_CHUNK_K_TILES = 8;
constexpr int64_t X_PIECE_DEPTH = 2;
constexpr uint32_t X_GRID_BITS = 4;
constexpr uint32_t W_GRID_BITS = 6;
constexpr uint32_t PRODUCT_ORDER_MAX = 2;
constexpr uint32_t PRODUCT_LO_ORDER = 2;
constexpr MathFidelity X_LO_FIDELITY = MathFidelity::HiFi3;
constexpr int64_t DEST_TILES_FP32 = 4;
constexpr int64_t L1_SAFETY_MARGIN = 96 * 1024;
constexpr int64_t BLOCK_TOKEN_TILES_CAP = 1;
constexpr bool NARROW_GROUPS = true;
constexpr double RT_TILES_BASE = 32;
constexpr double RT_TILES_PER_RANK = 0.75;
constexpr double RT_TILES_UNSTREAMED_PROJ = 4;
constexpr double TAIL_FRAC = 0.4;
constexpr double SINKHORN_TILES = 12;
constexpr bool W_BCAST = true;
constexpr NOC READER_NOC = NOC::NOC_0;
constexpr double READER_NOC_FLIP_FRACTION = 0.4;
constexpr std::optional<int64_t> READER_NOC_FLIP_ROWS = std::nullopt;
constexpr bool W_SHARE_ON_READER = true;
constexpr bool W_SHARE_BEFORE_X = false;
constexpr bool PLACEMENT_LEVERS_BF16_X_BF16_W = false;
constexpr bool PLACEMENT_LEVERS_BF16_W_R1 = false;
constexpr int64_t OWNER_C_DISCOUNT = 7;
constexpr uint32_t W_ROLE_DRAM = 0, W_ROLE_SPREAD = 1;
const std::vector<uint32_t> W_MCAST_PLACEHOLDER_CT = {0, SEM_W_READY, 0xFFFFFFFFu, 0, 0x2, 0};

// Address slots of the per-core runtime args, read by override_runtime_arguments.
constexpr size_t READER_RT_X = 0, READER_RT_W = 6, READER_RT_LEN = 10;
constexpr size_t WRITER_RT_Y = 0, WRITER_RT_POST = 1, WRITER_RT_COMB = 2, WRITER_RT_B = 11, WRITER_RT_W = 12;

int64_t w_pieces(DataType w_dtype) { return w_dtype == DataType::FLOAT32 ? 2 : 1; }
int64_t x_pieces(DataType x_dtype) { return x_dtype == DataType::FLOAT32 ? 3 : 1; }

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
// Python floor division (the operands here can be negative only in `(budget - fixed) // per_bt`, which is guarded).
int64_t floordiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -ceil_div(-a, b); }

NOC other_noc(NOC noc) { return noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0; }

bool placement_levers(DataType x_dtype, DataType w_dtype, bool w_bcast) {
    if (w_dtype != DataType::BFLOAT16) {
        return true;
    }
    if (!w_bcast) {
        return PLACEMENT_LEVERS_BF16_W_R1;
    }
    return PLACEMENT_LEVERS_BF16_X_BF16_W || x_dtype != DataType::BFLOAT16;
}

int64_t flip_rows_of(int64_t grid_y, bool levers) {
    if (READER_NOC_FLIP_ROWS.has_value()) {
        return *READER_NOC_FLIP_ROWS;
    }
    // Python round(): half to even
    return levers ? static_cast<int64_t>(std::nearbyint(READER_NOC_FLIP_FRACTION * static_cast<double>(grid_y))) : 0;
}

NOC reader_noc_of(int64_t group_y0, int64_t flip_rows) {
    return group_y0 < flip_rows ? other_noc(READER_NOC) : READER_NOC;
}

// Extra preprocessor defines for all three kernels (MHC_PRE_KERNEL_DEFINES="A=1;B"), as the Python reads them.
KernelDescriptor::Defines kernel_defines() {
    KernelDescriptor::Defines defines;
    const char* v = std::getenv("MHC_PRE_KERNEL_DEFINES");  // diagnostic: perf/measurement knob, same result
    if (v == nullptr) {
        return defines;
    }
    auto strip = [](const std::string& s) {
        const size_t a = s.find_first_not_of(" \t\n\r");
        const size_t b = s.find_last_not_of(" \t\n\r");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    std::string s(v);
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find(';', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        const std::string item = s.substr(start, end - start);
        if (!strip(item).empty()) {
            const size_t eq = item.find('=');
            const std::string name = strip(item.substr(0, eq));
            std::string value = eq == std::string::npos ? std::string() : strip(item.substr(eq + 1));
            defines.emplace_back(name, value.empty() ? "1" : value);
        }
        start = end + 1;
    }
    return defines;
}

uint32_t f32_bits(double x) { return std::bit_cast<uint32_t>(static_cast<float>(x)); }

struct Split {
    std::vector<int64_t> sizes;
    std::vector<int64_t> starts;
};

Split split(int64_t total, int64_t parts) {
    const int64_t base = total / parts, rem = total % parts;
    Split s;
    for (int64_t p = 0; p < parts; ++p) {
        s.sizes.push_back(base + (p < rem ? 1 : 0));
    }
    s.starts.assign(parts, 0);
    for (int64_t p = 1; p < parts; ++p) {
        s.starts[p] = s.starts[p - 1] + s.sizes[p - 1];
    }
    return s;
}

Split c_split(int64_t Ct, int64_t group_cores, bool owner_fixed) {
    const int64_t even = Ct / group_cores;
    const int64_t d = (owner_fixed && group_cores > 1) ? std::min(OWNER_C_DISCOUNT, even - 1) : 0;
    if (d <= 0) {
        return split(Ct, group_cores);
    }
    const Split rest = split(Ct - (even - d), group_cores - 1);
    Split s;
    s.sizes.push_back(even - d);
    s.sizes.insert(s.sizes.end(), rest.sizes.begin(), rest.sizes.end());
    s.starts.push_back(0);
    for (int64_t st : rest.starts) {
        s.starts.push_back(even - d + st);
    }
    return s;
}

struct CbEntry {
    uint8_t index;
    int64_t pages;
    int64_t page_bytes;
    DataType dtype;
    std::vector<std::tuple<uint8_t, int64_t, DataType>> aliases;
};

struct CbArgs {
    int64_t bt, depth, kmax, G, y_chunk, n, x_tile, w_tile, y_tile;
    DataType x_dtype, w_dtype, y_dtype;
};

// _cb_table(): THE per-core CB inventory (same entries, same order).
std::vector<CbEntry> cb_table(const CbArgs& a) {
    const DataType f32 = DataType::FLOAT32;
    const int64_t fT = F32_TILE_BYTES;
    const int64_t pieces = w_pieces(a.w_dtype);
    std::vector<std::tuple<uint8_t, int64_t, DataType>> w_alias;
    if (pieces > 1) {
        w_alias.emplace_back(CB_WEIGHT_SPLIT, a.w_tile / pieces, DataType::BFLOAT16);
    }
    const int64_t xp = x_pieces(a.x_dtype);
    std::vector<std::tuple<uint8_t, int64_t, DataType>> x_alias;
    if (xp > 1) {
        x_alias.emplace_back(CB_X_FP32, a.x_tile, a.x_dtype);
    }
    const int64_t sb_rows = std::min(a.bt, DEST_TILES_FP32);
    std::vector<CbEntry> t = {
        {CB_X_RESIDENT, a.depth * a.bt * a.kmax, a.x_tile, a.x_dtype, x_alias},
        {CB_WEIGHT, a.kmax, a.w_tile, a.w_dtype, w_alias},
        {CB_BIAS_COEF, 1, fT, f32, {}},
        {CB_REDUCE_SCALER, 1, BF16_TILE_BYTES, DataType::BFLOAT16, {}},
        {CB_SQ_ACC, xp > 1 ? a.bt : a.bt * static_cast<int64_t>(X_STREAM_CHUNKS), fT, f32, {}},
        {CB_PARTIAL, 2 * a.bt, fT, f32, {}},
        {CB_GATHERED, a.G * 2 * a.bt, fT, f32, {}},
        {CB_COMBINED, 2 * a.bt, fT, f32, {}},
        {CB_COEF_IN, COEF_IN_BLOCKS * 2 * a.bt, fT, f32, {}},
        {CB_COMB_COEF, 2 * a.bt, fT, f32, {}},
        {CB_COEF_KEEP, 1, fT, f32, {}},
        {CB_PRE_COLS, a.n * a.bt, fT, f32, {}},
        {CB_Y_OUT, Y_DEPTH * a.y_chunk, a.y_tile, a.y_dtype, {}},
        {CB_W_OWN_READY, 1, TOKEN_PAGE_BYTES, DataType::BFLOAT16, {}},
        {CB_W_OWN_SPLIT, 1, TOKEN_PAGE_BYTES, DataType::BFLOAT16, {}},
        {CB_W_SHARE_LANDED, 1, TOKEN_PAGE_BYTES, DataType::BFLOAT16, {}},
    };
    if (xp == 1) {
        t.push_back({CB_MIX_RUN, 1, fT, f32, {}});
    }
    if (xp > 1) {
        t.push_back(
            {CB_X_PIECES,
             X_PIECE_DEPTH * xp * static_cast<int64_t>(X_CHUNK_K_TILES) * sb_rows,
             BF16_TILE_BYTES,
             DataType::BFLOAT16,
             {}});
        t.push_back({CB_MIX_RUN, sb_rows, fT, f32, {}});
        t.push_back({CB_MAX_LANES, 1, fT, f32, {}});
        t.push_back({CB_MAX_SCALAR, 1, fT, f32, {}});
        t.push_back({CB_GRID, std::max<int64_t>(a.bt, 1), fT, f32, {}});
        t.push_back({CB_MAX_SCALER, 1, BF16_TILE_BYTES, DataType::BFLOAT16, {}});
    }
    return t;
}

int64_t l1_bytes(const CbArgs& a) {
    int64_t total = 0;
    for (const auto& e : cb_table(a)) {
        total += e.pages * e.page_bytes;
    }
    return total;
}

struct Fit {
    int64_t group_w = 0, group_h = 0, group_cores = 0, kmax = 0, y_chunk = 0, groups_x = 0, groups_y = 0;
    std::vector<int64_t> core_token_tiles, t_start, c_tiles, c_starts;
    int64_t bt = 0, depth = 0, blocks = 0;
};

double block_schedule_cost(const Fit& f, int64_t n, bool streamed_proj) {
    const int64_t B = f.blocks, d = f.depth, k = f.kmax, G = f.group_cores;
    const double H =
        RT_TILES_BASE + RT_TILES_PER_RANK * static_cast<double>(G) + (streamed_proj ? 0.0 : RT_TILES_UNSTREAMED_PROJ);
    const int64_t ctt_max = *std::max_element(f.core_token_tiles.begin(), f.core_token_tiles.end());
    double cost = static_cast<double>(B * k) + H + static_cast<double>(k) / static_cast<double>(n) -
                  (ctt_max <= 1 ? SINKHORN_TILES : 0.0);
    for (int64_t b = 0; b < B - 1; ++b) {
        const bool pipelined = b + 1 < B && (d >= 3 || b + d >= B);
        cost += pipelined ? std::max(0.0, H - TAIL_FRAC * static_cast<double>(k))
                          : std::max(0.0, H - (1.0 - TAIL_FRAC) * static_cast<double>(k));
    }
    return cost;
}

struct Plan {
    int64_t n = 0, Mt = 0, Ct = 0, Kt = 0, grid_x = 0, grid_y = 0;
    int64_t group_w = 0, group_h = 0, group_cores = 0, groups_x = 0, groups_y = 0, num_groups = 0;
    std::vector<int64_t> core_c_tiles, c_start, core_token_tiles, t_start;
    int64_t core_k_tiles_max = 0, block_token_tiles = 0, x_block_depth = 0, y_chunk_tiles = 0, y_depth = 0;
};

Plan make_plan(tt::tt_metal::IDevice* device, const Tensor& x, const Tensor& w, int64_t n) {
    const auto& padded = x.padded_shape();
    int64_t lead_tiles = 1;
    for (size_t i = 0; i + 2 < padded.rank(); ++i) {
        lead_tiles *= padded[i];
    }
    const int64_t Mt = lead_tiles * (padded[-2] / TILE);
    const int64_t C = x.logical_shape()[-1] / n;
    const int64_t Ct = C / TILE;
    const int64_t Kt = n * Ct;
    const auto grid = device->compute_with_storage_grid_size();
    const int64_t grid_x = grid.x, grid_y = grid.y;
    const int64_t x_tile = x.buffer()->page_size();
    const int64_t w_tile = w.buffer()->page_size();
    const int64_t y_tile = x_tile;
    const int64_t budget =
        static_cast<int64_t>(tt::tt_metal::hal::get_max_worker_l1_unreserved_size()) - L1_SAFETY_MARGIN;

    auto fit = [&](int64_t group_w, int64_t group_h) -> std::optional<Fit> {
        const int64_t group_cores = group_w * group_h;
        const int64_t groups_x = grid_x / group_w, groups_y = grid_y / group_h;
        const Split tok = split(Mt, groups_x * groups_y);
        const int64_t ctt_max = *std::max_element(tok.sizes.begin(), tok.sizes.end());
        const Split cs = c_split(Ct, group_cores, ctt_max <= 1);
        const int64_t cmax = *std::max_element(cs.sizes.begin(), cs.sizes.end());
        const int64_t kmax = n * cmax;
        const int64_t y_chunk = std::min(cmax, Y_CHUNK_TILES_CAP);
        auto l1_at = [&](int64_t bt_, int64_t depth_) {
            return l1_bytes(CbArgs{
                bt_, depth_, kmax, group_cores, y_chunk, n, x_tile, w_tile, y_tile, x.dtype(), w.dtype(), x.dtype()});
        };
        const int64_t bt_cap = std::min(ctt_max, BLOCK_TOKEN_TILES_CAP);
        for (int64_t depth : {X_BLOCK_DEPTH_DEFAULT, int64_t{1}}) {
            const int64_t fixed = l1_at(0, depth);
            const int64_t per_bt = l1_at(1, depth) - fixed;
            const int64_t bt = budget > fixed ? std::min(bt_cap, floordiv(budget - fixed, per_bt)) : 0;
            if (bt >= 1) {
                Fit f;
                f.group_w = group_w;
                f.group_h = group_h;
                f.group_cores = group_cores;
                f.kmax = kmax;
                f.y_chunk = y_chunk;
                f.groups_x = groups_x;
                f.groups_y = groups_y;
                f.core_token_tiles = tok.sizes;
                f.t_start = tok.starts;
                f.c_tiles = cs.sizes;
                f.c_starts = cs.starts;
                f.bt = bt;
                f.depth = depth;
                f.blocks = ceil_div(ctt_max, bt);
                return f;
            }
        }
        return std::nullopt;
    };

    std::optional<Fit> chosen;
    if (Mt >= grid_y && NARROW_GROUPS && x_pieces(x.dtype()) == 1) {
        const bool streamed = w_pieces(w.dtype()) > 1;
        std::optional<std::pair<bool, double>> best_key;
        for (int64_t group_w = std::min(grid_x, Ct); group_w > 0; --group_w) {
            auto f = fit(group_w, 1);
            if (!f.has_value()) {
                continue;
            }
            const std::pair<bool, double> key{f->depth < 2 && f->blocks > 1, block_schedule_cost(*f, n, streamed)};
            if (!best_key.has_value() || key < *best_key) {  // min(): the first of equal keys wins
                best_key = key;
                chosen = f;
            }
        }
    }
    if (!chosen.has_value()) {
        const int64_t group_w = std::min(grid_x, Ct);
        int64_t group_h = 1;
        if (Mt < grid_y) {
            group_h = std::max<int64_t>(1, std::min({grid_y / Mt, Ct / group_w, GROUP_CORES_CAP / group_w}));
        }
        while (true) {
            chosen = fit(group_w, group_h);
            if (chosen.has_value()) {
                break;
            }
            if (group_h * 2 <= grid_y && group_w * group_h * 2 <= GROUP_CORES_CAP && group_w * group_h * 2 <= Ct) {
                group_h *= 2;
                continue;
            }
            TT_THROW(
                "mhc_pre: no blocking fits L1 (C={}, Mt={}, grid={}x{}, budget={} B)", C, Mt, grid_x, grid_y, budget);
        }
    }
    Plan p;
    p.n = n;
    p.Mt = Mt;
    p.Ct = Ct;
    p.Kt = Kt;
    p.grid_x = grid_x;
    p.grid_y = grid_y;
    p.group_w = chosen->group_w;
    p.group_h = chosen->group_h;
    p.group_cores = chosen->group_cores;
    p.groups_x = chosen->groups_x;
    p.groups_y = chosen->groups_y;
    p.num_groups = p.groups_x * p.groups_y;
    p.core_c_tiles = chosen->c_tiles;
    p.c_start = chosen->c_starts;
    p.core_token_tiles = chosen->core_token_tiles;
    p.t_start = chosen->t_start;
    p.core_k_tiles_max = chosen->kmax;
    p.block_token_tiles = chosen->bt;
    p.x_block_depth = chosen->depth;
    p.y_chunk_tiles = chosen->y_chunk;
    p.y_depth = Y_DEPTH;
    return p;
}

struct Group {
    int64_t g, gx0, gy0;
};

// The launched groups (a group with 0 token rows is not launched), the placement levers and the reader NoC sets.
struct Layout {
    std::vector<Group> groups;
    std::vector<CoreRange> ranges;
    int64_t active_rows = 0;
    bool w_bcast = false;
    bool levers = false;
    int64_t flip_rows = 0;
    std::vector<NOC> reader_nocs;  // sorted by value
};

Layout layout_of(const Plan& plan, DataType x_dtype, DataType w_dtype) {
    Layout L;
    for (int64_t g = 0; g < plan.num_groups; ++g) {
        if (plan.core_token_tiles[g] == 0) {
            continue;
        }
        const int64_t gx0 = (g % plan.groups_x) * plan.group_w;
        const int64_t gy0 = (g / plan.groups_x) * plan.group_h;
        L.groups.push_back({g, gx0, gy0});
        L.ranges.emplace_back(CoreCoord(gx0, gy0), CoreCoord(gx0 + plan.group_w - 1, gy0 + plan.group_h - 1));
    }
    L.active_rows = static_cast<int64_t>(L.groups.size()) / plan.groups_x;
    L.w_bcast = W_BCAST && plan.group_h == 1 && static_cast<int64_t>(L.groups.size()) % plan.groups_x == 0 &&
                L.active_rows >= 2;
    L.levers = placement_levers(x_dtype, w_dtype, L.w_bcast);
    L.flip_rows = flip_rows_of(plan.grid_y, L.levers);
    std::set<uint32_t> nocs;
    for (const auto& gr : L.groups) {
        nocs.insert(static_cast<uint32_t>(reader_noc_of(gr.gy0, L.flip_rows)));
    }
    for (uint32_t v : nocs) {
        L.reader_nocs.push_back(static_cast<NOC>(v));
    }
    return L;
}

CBDescriptor make_cb(const CbEntry& e, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = static_cast<uint32_t>(e.pages * e.page_bytes);
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = e.index,
        .data_format = tt::tt_metal::datatype_to_dataformat_converter(e.dtype),
        .page_size = static_cast<uint32_t>(e.page_bytes)});
    for (const auto& [index, page, dtype] : e.aliases) {
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = index,
            .data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype),
            .page_size = static_cast<uint32_t>(page)});
    }
    return cb;
}

std::vector<uint32_t> accessor_ct(const Tensor& t) {
    return tt::tt_metal::TensorAccessorArgs(*t.buffer()).get_compile_time_args();
}

void append(std::vector<uint32_t>& v, const std::vector<uint32_t>& more) {
    v.insert(v.end(), more.begin(), more.end());
}

uint32_t address(const Tensor& t) { return t.buffer()->address(); }

}  // namespace

ProgramDescriptor create_program_descriptor(
    const Tensor& x,
    const Tensor& w,
    const Tensor& b,
    const Tensor& y,
    const Tensor& post,
    const Tensor& comb,
    const MhcPreParams& params) {
    auto* device = x.device();
    const int64_t n = params.n;
    const Plan plan = make_plan(device, x, w, n);
    const int64_t G = plan.group_cores;
    const int64_t bt = plan.block_token_tiles;
    const int64_t kmax = plan.core_k_tiles_max;
    const Layout L = layout_of(plan, x.dtype(), w.dtype());
    const CoreRangeSet all_cores(L.ranges);

    ProgramDescriptor desc;
    // ---- CBs: identical descriptors on every launched core ----
    for (const auto& e : cb_table(CbArgs{
             bt,
             plan.x_block_depth,
             kmax,
             G,
             plan.y_chunk_tiles,
             n,
             static_cast<int64_t>(x.buffer()->page_size()),
             static_cast<int64_t>(w.buffer()->page_size()),
             static_cast<int64_t>(y.buffer()->page_size()),
             x.dtype(),
             w.dtype(),
             y.dtype()})) {
        desc.cbs.push_back(make_cb(e, all_cores));
    }
    for (uint32_t sem : {SEM_GATHER, SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_W_READY}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // ---- W column broadcast (R2): one Mcast1D(PerColumn) over the active rectangle per reader-NoC set ----
    std::map<uint32_t, kh::Mcast1D> w_mcast;
    if (L.w_bcast) {
        const CoreRangeSet w_rect(
            CoreRange(CoreCoord(0, 0), CoreCoord(plan.groups_x * plan.group_w - 1, L.active_rows - 1)));
        for (NOC rnoc : L.reader_nocs) {
            kh::McastConfig w_cfg;
            w_cfg.noc = other_noc(rnoc);
            w_cfg.handshake = false;
            w_cfg.data_ready = kh::DataReadyMode::Counter;
            w_cfg.rotating_sender = true;
            w_cfg.sem_ids = std::vector<uint32_t>{SEM_W_READY};
            w_mcast.emplace(
                static_cast<uint32_t>(rnoc), kh::Mcast1D(device, w_rect, kh::Mcast1DShape::PerColumn, 0, w_cfg));
        }
    }
    const bool w_presplit = !w_mcast.empty() && w_pieces(w.dtype()) > 1 && x_pieces(x.dtype()) == 1;

    // ---- group combine mcast (one Mcast2D per group; identical CT wire within a reader-NoC set) ----
    std::map<int64_t, kh::Mcast2D> helpers;
    std::map<uint32_t, std::vector<uint32_t>> mcast_ct;
    if (G > 1) {
        for (const auto& gr : L.groups) {
            const NOC rnoc = reader_noc_of(gr.gy0, L.flip_rows);
            kh::McastConfig mcast_cfg;
            mcast_cfg.noc = other_noc(rnoc);
            mcast_cfg.handshake = false;
            mcast_cfg.sem_ids = std::vector<uint32_t>{SEM_MCAST_READY};
            const CoreRangeSet rect(
                CoreRange(CoreCoord(gr.gx0, gr.gy0), CoreCoord(gr.gx0 + plan.group_w - 1, gr.gy0 + plan.group_h - 1)));
            auto it = helpers.emplace(gr.g, kh::Mcast2D(device, rect, CoreCoord(gr.gx0, gr.gy0), mcast_cfg)).first;
            const auto ct = it->second.compile_time_args();
            const auto key = static_cast<uint32_t>(rnoc);
            TT_FATAL(
                mcast_ct.find(key) == mcast_ct.end() || mcast_ct.at(key) == ct,
                "mhc_pre: mcast CT wire must be identical across the groups of a set");
            mcast_ct[key] = ct;
        }
    } else {
        for (NOC rnoc : L.reader_nocs) {
            mcast_ct[static_cast<uint32_t>(rnoc)] = {0, SEM_MCAST_READY, SEM_MCAST_CONSUMED, 0, 1, 0};
        }
    }

    // ---- kernel CT args ----
    std::vector<uint32_t> reader_ct = {
        CB_X_RESIDENT,
        CB_REDUCE_SCALER,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(bt),
        static_cast<uint32_t>(kmax),
        static_cast<uint32_t>(plan.Ct),
        CB_MAX_SCALER,
        static_cast<uint32_t>(x_pieces(x.dtype()) > 1),
        X_STREAM_CHUNKS,
        X_STREAM_INFLIGHT,
        CB_WEIGHT,
        CB_W_SHARE_LANDED,
        static_cast<uint32_t>(W_SHARE_BEFORE_X),
    };
    TT_FATAL(reader_ct.size() == 13, "mhc_pre: reader CT base drifted");
    append(reader_ct, accessor_ct(x));
    append(reader_ct, accessor_ct(w));

    const std::vector<uint32_t> compute_ct = {
        CB_X_RESIDENT,
        CB_WEIGHT,
        CB_BIAS_COEF,
        CB_REDUCE_SCALER,
        CB_SQ_ACC,
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COEF_KEEP,
        0,  // CT 10 unused
        CB_COMB_COEF,
        CB_PRE_COLS,
        CB_Y_OUT,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(bt),
        static_cast<uint32_t>(kmax),
        static_cast<uint32_t>(G),
        CB_WEIGHT_SPLIT,
        static_cast<uint32_t>(w_pieces(w.dtype())),
        W_CHUNK_TILES,
        static_cast<uint32_t>(W_LO_FIDELITY),
        static_cast<uint32_t>(params.compute_config.math_fidelity),
        CB_X_FP32,
        CB_X_PIECES,
        CB_MIX_RUN,
        static_cast<uint32_t>(x_pieces(x.dtype())),
        X_CHUNK_K_TILES,
        static_cast<uint32_t>(std::min(bt, DEST_TILES_FP32)),
        CB_MAX_LANES,
        CB_MAX_SCALAR,
        CB_GRID,
        CB_MAX_SCALER,
        X_GRID_BITS,
        W_GRID_BITS,
        PRODUCT_ORDER_MAX,
        PRODUCT_LO_ORDER,
        static_cast<uint32_t>(X_LO_FIDELITY),
        CB_W_OWN_READY,
        CB_W_OWN_SPLIT,
        static_cast<uint32_t>(w_presplit),
        X_STREAM_CHUNKS,
        static_cast<uint32_t>(plan.x_block_depth),
    };

    const std::vector<uint32_t> writer_ct = {
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COMB_COEF,
        CB_Y_OUT,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(bt),
        static_cast<uint32_t>(plan.Ct),
        static_cast<uint32_t>(G),
        static_cast<uint32_t>(plan.y_chunk_tiles),
        static_cast<uint32_t>(plan.y_depth * plan.y_chunk_tiles),
        SEM_GATHER,
        static_cast<uint32_t>(n * (n + 2)),
        CB_BIAS_COEF,
        CB_WEIGHT,
        W_CHUNK_TILES,
        CB_W_OWN_READY,
        CB_W_OWN_SPLIT,
        static_cast<uint32_t>(w_presplit),
        CB_W_SHARE_LANDED,
        static_cast<uint32_t>(plan.x_block_depth),
    };
    TT_FATAL(writer_ct.size() == 22, "mhc_pre: writer MCAST_CT_BASE drifted");
    std::vector<uint32_t> writer_tail_ct;
    for (const Tensor* t : {&y, &post, &comb, &b, &w}) {
        append(writer_tail_ct, accessor_ct(*t));
    }
    auto writer_ct_of = [&](NOC rnoc) {
        std::vector<uint32_t> ct = writer_ct;
        append(ct, mcast_ct.at(static_cast<uint32_t>(rnoc)));
        append(
            ct, w_mcast.empty() ? W_MCAST_PLACEHOLDER_CT : w_mcast.at(static_cast<uint32_t>(rnoc)).compile_time_args());
        append(ct, writer_tail_ct);
        return ct;
    };

    // ---- per-core RT args ----
    const int64_t C = plan.Ct * TILE;
    const std::vector<uint32_t> scalar_bits = {
        f32_bits(params.scale[0]),
        f32_bits(params.scale[1]),
        f32_bits(params.scale[2]),
        f32_bits(params.eps),
        f32_bits(params.norm_eps),
        f32_bits(1.0 / static_cast<double>(n * C)),
        params.sinkhorn_iters,
    };
    std::map<uint32_t, KernelDescriptor::RuntimeArgs> reader_rt, writer_rt;
    KernelDescriptor::RuntimeArgs compute_rt;
    for (const auto& gr : L.groups) {
        const NOC rnoc = reader_noc_of(gr.gy0, L.flip_rows);
        const auto key = static_cast<uint32_t>(rnoc);
        const int64_t ctt = plan.core_token_tiles[gr.g];
        const int64_t ts = plan.t_start[gr.g];
        const int64_t num_blocks = ceil_div(ctt, bt);
        const CoreCoord root_virtual = device->worker_core_from_logical_core(CoreCoord(gr.gx0, gr.gy0));
        for (int64_t dy = 0; dy < plan.group_h; ++dy) {
            for (int64_t dx = 0; dx < plan.group_w; ++dx) {
                const int64_t cx = gr.gx0 + dx, cy = gr.gy0 + dy;
                const CoreCoord core(cx, cy);
                const int64_t rank = dy * plan.group_w + dx;
                const int64_t cc = plan.core_c_tiles[rank];
                const int64_t cs = plan.c_start[rank];
                std::vector<uint32_t> own = {0, 0};
                uint32_t share_on_reader = 0;
                std::vector<uint32_t> w_rt, w_mc_rt;
                if (w_mcast.empty()) {
                    w_rt = {W_ROLE_DRAM, 0, 0, 0, 0};
                    w_mc_rt = {0, 0, 0, 0};
                } else {
                    const Split sh = split(n * cc, L.active_rows);
                    own = {static_cast<uint32_t>(sh.starts[cy]), static_cast<uint32_t>(sh.starts[cy] + sh.sizes[cy])};
                    int64_t events = 0;
                    for (int64_t sz : sh.sizes) {
                        events += sz > 0 ? 1 : 0;
                    }
                    events -= sh.sizes[cy] > 0 ? 1 : 0;
                    share_on_reader = static_cast<uint32_t>(W_SHARE_ON_READER && L.levers);
                    w_rt = {W_ROLE_SPREAD, own[0], own[1], static_cast<uint32_t>(events), share_on_reader};
                    w_mc_rt = w_mcast.at(key).runtime_args(core);
                }
                std::vector<uint32_t> r = {
                    address(x),
                    static_cast<uint32_t>(ts),
                    static_cast<uint32_t>(ctt),
                    static_cast<uint32_t>(cs),
                    static_cast<uint32_t>(cc),
                    static_cast<uint32_t>(num_blocks),
                    address(w),
                    share_on_reader,
                    own[0],
                    own[1]};  // smuggled-rta-ok: patched in override_runtime_arguments
                reader_rt[key].emplace_back(core, std::move(r));
                const std::vector<uint32_t> mcast_rt =
                    G > 1 ? helpers.at(gr.g).runtime_args(core) : std::vector<uint32_t>{0, 0, 0, 0};
                std::vector<uint32_t> wr = {
                    address(y),
                    address(post),
                    address(comb),
                    static_cast<uint32_t>(ts),
                    static_cast<uint32_t>(ctt),
                    static_cast<uint32_t>(cs),
                    static_cast<uint32_t>(cc),
                    static_cast<uint32_t>(num_blocks),
                    static_cast<uint32_t>(rank),
                    static_cast<uint32_t>(root_virtual.x),
                    static_cast<uint32_t>(root_virtual.y),
                    address(b),
                    address(w)};  // smuggled-rta-ok: patched in override_runtime_arguments
                append(wr, w_rt);
                append(wr, mcast_rt);
                append(wr, w_mc_rt);
                writer_rt[key].emplace_back(core, std::move(wr));
                std::vector<uint32_t> cr = {
                    static_cast<uint32_t>(num_blocks),
                    static_cast<uint32_t>(ctt),
                    static_cast<uint32_t>(cc),
                    static_cast<uint32_t>(rank)};
                append(cr, scalar_bits);
                append(cr, own);
                compute_rt.emplace_back(core, std::move(cr));
            }
        }
    }

    const auto defines = kernel_defines();
    for (NOC rnoc : L.reader_nocs) {
        std::vector<CoreRange> set_ranges;
        for (size_t i = 0; i < L.groups.size(); ++i) {
            if (reader_noc_of(L.groups[i].gy0, L.flip_rows) == rnoc) {
                set_ranges.push_back(L.ranges[i]);
            }
        }
        const CoreRangeSet set_cores(set_ranges);
        const auto key = static_cast<uint32_t>(rnoc);
        KernelDescriptor reader;
        reader.kernel_source = std::string(KERNEL_DIR) + "mhc_pre_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = set_cores;
        reader.compile_time_args = reader_ct;
        reader.runtime_args = reader_rt[key];
        reader.defines = defines;
        reader.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = rnoc};
        desc.kernels.push_back(std::move(reader));
        KernelDescriptor writer;
        writer.kernel_source = std::string(KERNEL_DIR) + "mhc_pre_writer.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = set_cores;
        writer.compile_time_args = writer_ct_of(rnoc);
        writer.runtime_args = writer_rt[key];
        writer.defines = defines;
        writer.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = other_noc(rnoc)};
        desc.kernels.push_back(std::move(writer));
    }
    ComputeConfigDescriptor compute_cfg;
    compute_cfg.math_fidelity = params.compute_config.math_fidelity;
    compute_cfg.fp32_dest_acc_en = true;
    compute_cfg.math_approx_mode = params.compute_config.math_approx_mode;
    compute_cfg.unpack_to_dest_mode.assign(NUM_CB_SLOTS, UnpackToDestMode::Default);
    std::vector<uint8_t> fp32_cbs(std::begin(UNPACK_TO_DEST_FP32_CBS), std::end(UNPACK_TO_DEST_FP32_CBS));
    if (w_pieces(w.dtype()) > 1) {
        fp32_cbs.push_back(CB_WEIGHT);
    }
    if (x_pieces(x.dtype()) > 1) {
        fp32_cbs.insert(fp32_cbs.end(), {CB_X_FP32, CB_GRID, CB_MAX_SCALAR});
    }
    fp32_cbs.push_back(CB_MIX_RUN);
    for (uint8_t idx : fp32_cbs) {
        compute_cfg.unpack_to_dest_mode[idx] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.kernel_source = std::string(KERNEL_DIR) + "mhc_pre_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = compute_ct;
    compute.runtime_args = std::move(compute_rt);
    compute.defines = defines;
    compute.config = compute_cfg;
    desc.kernels.push_back(std::move(compute));
    return desc;
}

ProgramDescriptor MhcPreProgramFactory::create_descriptor(
    const MhcPreParams& operation_attributes, const MhcPreInputs& tensor_args, Outputs& outputs) {
    return create_program_descriptor(
        tensor_args.input,
        tensor_args.proj_weight,
        tensor_args.proj_bias,
        std::get<0>(outputs),
        std::get<1>(outputs),
        std::get<2>(outputs),
        operation_attributes);
}

void MhcPreProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcPreParams& operation_attributes,
    const MhcPreInputs& tensor_args,
    Outputs& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Everything the builder reads besides the buffer ADDRESSES is in the program hash, so a cache hit can only move
    // addresses. The kernels are [reader, writer] per reader-NoC set, then compute: the number of sets follows from
    // the plan (deterministic in the hashed specs; a few integer sums).
    const auto& x = tensor_args.input;
    const auto& w = tensor_args.proj_weight;
    const Plan plan = make_plan(x.device(), x, w, operation_attributes.n);
    const size_t sets = layout_of(plan, x.dtype(), w.dtype()).reader_nocs.size();
    const uint32_t x_addr = address(x), w_addr = address(w), b_addr = address(tensor_args.proj_bias);
    const uint32_t y_addr = address(std::get<0>(outputs));
    const uint32_t p_addr = address(std::get<1>(outputs));
    const uint32_t m_addr = address(std::get<2>(outputs));
    for (size_t s = 0; s < sets; ++s) {
        for (auto& col : tt::tt_metal::GetRuntimeArgs(program, static_cast<uint32_t>(2 * s))) {
            for (auto& args : col) {
                if (args.size() < READER_RT_LEN) {
                    continue;
                }
                args[READER_RT_X] = x_addr;
                args[READER_RT_W] = w_addr;
            }
        }
        for (auto& col : tt::tt_metal::GetRuntimeArgs(program, static_cast<uint32_t>(2 * s + 1))) {
            for (auto& args : col) {
                if (args.size() <= WRITER_RT_W) {
                    continue;
                }
                args[WRITER_RT_Y] = y_addr;
                args[WRITER_RT_POST] = p_addr;
                args[WRITER_RT_COMB] = m_addr;
                args[WRITER_RT_B] = b_addr;
                args[WRITER_RT_W] = w_addr;
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
