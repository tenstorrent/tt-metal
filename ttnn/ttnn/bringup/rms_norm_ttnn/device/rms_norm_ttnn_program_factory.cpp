// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The C++ host side of ttnn.bringup.rms_norm: a port of rms_norm_ttnn_program_descriptor.py's
// create_program_descriptor().  The Python module is the documentation -- every knob, every
// deviation (D1..D44) and every measurement behind them is recorded there, next to the Python
// spelling of the same decision -- and it stays importable, so the two can be A/B'd and are
// compared field by field in tests/unit/test_rms_norm_ttnn_cpp_parity.py.  The names below
// follow the Python ones (a leading `_` dropped) so a reader can walk the two side by side.
//
// Only the knobs' SHIPPED values are ported: the Python module constants that exist to be flipped
// by a sweep are constexpr here.  The two environment switches (RMS_STAGE_ZONES, RMS_ABLATE) are
// honoured exactly as the Python reads them.

#include "rms_norm_ttnn_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

namespace {

namespace kh = ttnn::kernel_lib::host;
using tt::tt_metal::BufferType;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::Layout;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorMemoryLayout;

constexpr int64_t TILE_DIM = 32;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/rms_norm_ttnn/kernels/";

// ---------------------------------------------------------------------------
// Primary knobs (their shipped values -- see the Python module for each one's measurement).
// ---------------------------------------------------------------------------
constexpr double L1_SAFETY_FRACTION = 0.85;
constexpr int64_t L1_CB_ARENA_BASE_RESERVE = 70656;
constexpr int64_t CB_RM_STAGE_DEPTH = 2;
constexpr int64_t TRIM_DERIVED = -1;
constexpr int64_t PER_CHANNEL_TRIM_GAMMA = TRIM_DERIVED;
constexpr int64_t PER_CHANNEL_TRIM_BIAS = TRIM_DERIVED;
constexpr bool PC_COMPACT_HOLD = true;
constexpr bool PC_COMPACT_LAZY = true;
constexpr bool ROW_RESIDENT_COMPACT_PC = true;
constexpr int64_t PC_RING_CHUNKS = 1;
constexpr int64_t ROW_RESIDENT_CAP_OVERRIDE = 0;
constexpr double ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION = 0.75;
constexpr int64_t ROW_RESIDENT_COMPACT_MIN_CHUNK_WT = 1;
constexpr bool COMPACT_FIRST = false;
constexpr int64_t DM_TXN_ROWS_MAX = 1;
constexpr uint32_t PASS_A_SQ_BLOCK = 1;
constexpr uint32_t RES_FUSE = 0;
constexpr bool CB_SQ_EXACT = false;
// CB_DEPTH_CANDIDATES = (2,) on the TILE path, (1,) on ROW_MAJOR.
constexpr int64_t CB_DEPTH_TILE = 2;
constexpr int64_t GRID_W = 0;
constexpr int64_t WIDTH_SPLIT_MIN_WT_PER_CORE = 4;
constexpr int64_t WIDTH_SPLIT_MAX_GROUP_CORES = 16;
constexpr int64_t WIDTH_SPLIT_MIN_GAIN = 4;
constexpr bool RAGGED_WIDTH_CHUNK = true;
constexpr int64_t ROW_RESIDENT_MIN_CHUNK_WT = 1;
constexpr uint32_t REDUCE_BULK = 1;
constexpr int64_t REDUCE_ACC_VIA_ADD_MIN_WT = 4;
constexpr int64_t REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT = 2;
constexpr int64_t REDUCE_ACC_VIA_ADD_MIN_CALL_WT = 2;
constexpr int64_t CB_ROW_STAT_DEPTH = 2;
constexpr int64_t CB_R_DEPTH = 0;
constexpr int64_t DEST_ACC_SQUARE_MAX_WT = 8;
constexpr int64_t SQ_FOLD_GROUP = 16;
constexpr uint32_t GATHER_FACES = 2;
constexpr int64_t CB_COMBINE_FLAT_DEPTH = 2;
constexpr int64_t COMBINE_TREE_F0_MIN = 4;
constexpr int64_t COMBINE_TREE_F0_MAX = 10;
constexpr int64_t COMBINE_TREE_MIN_DELETED_FOLD_TILES = 18;
constexpr bool COMBINE_FIN_SPREAD = false;
constexpr bool COMBINE_MCAST_FIRE_AND_FORGET = true;
constexpr uint32_t COMBINE_MCAST_FACES = 3;
constexpr NOC COMBINE_NOC_RESIDENT = NOC::NOC_0;
constexpr NOC COMBINE_NOC_STREAMED = NOC::NOC_1;
constexpr int64_t ROW_RESIDENT_MIN_ROWS_PER_CORE = 2;
// `return_residual_sum`: cb_residual_sum's depth, in compute's pass-B DEST blocks.
constexpr int64_t T_SUM_RING_BLOCKS = 2;

// D40: the per-channel broadcast.  PC_MCAST_MODE is "row" (the shipped choice); the other modes
// of the Python module are sweep-only and are not ported.
constexpr bool PC_MCAST_ENABLED = true;
constexpr bool PC_MCAST_HANDSHAKE = false;
constexpr uint32_t PC_MCAST_SPLIT_N = 1;  // pc_n: always 1 outside "split" mode
constexpr bool PC_MCAST_LATE = true;
constexpr size_t PC_MCAST_MIN_GROUP = 3;
constexpr uint32_t PC_MCAST_SENDER_INDEX = 0;
constexpr bool PC_MCAST_DIAGONAL = true;
constexpr uint32_t PC_OPT_OUT = 0xFFFFFFFFu;

// Circular-buffer slots.
constexpr uint8_t CB_INPUT_STICKS = 0;
constexpr uint8_t CB_INPUT_TILES = 1;
constexpr uint8_t CB_X_SQUARED = 2;
constexpr uint8_t CB_SCALER = 3;
constexpr uint8_t CB_ROW_STAT = 4;
constexpr uint8_t CB_GAMMA_STICKS = 5;
constexpr uint8_t CB_GAMMA_TILES = 6;
constexpr uint8_t CB_NORMALIZED = 7;
constexpr uint8_t CB_OUTPUT_TILES = 8;
constexpr uint8_t CB_OUTPUT_STICKS = 9;
constexpr uint8_t CB_SUM_HANDOFF = 10;
constexpr uint8_t CB_PARTIALS_GATHERED = 11;
constexpr uint8_t CB_STAT_HANDOFF = 12;
constexpr uint8_t CB_ROW_FINAL = 13;
constexpr uint8_t CB_BANK = 14;
constexpr uint8_t CB_COMPACT_HANDOFF = 15;
constexpr uint8_t CB_MCAST_IN = 16;
constexpr uint8_t CB_GATHER_L1 = 17;
constexpr uint8_t CB_NODE_OUT = 18;
constexpr uint8_t CB_RESIDUAL_STICKS = 19;
constexpr uint8_t CB_RESIDUAL_TILES = 20;
constexpr uint8_t CB_X_SUM = 21;
constexpr uint8_t CB_BIAS_STICKS = 22;
constexpr uint8_t CB_BIAS_TILES = 23;
constexpr uint8_t CB_GAMMA_COMPACT = 24;
constexpr uint8_t CB_BIAS_COMPACT = 25;
// `return_residual_sum`: pass A's t = x + r, at the input's dtype, for the writer to store.
constexpr uint8_t CB_RESIDUAL_SUM = 26;

constexpr size_t READER_CT_SCALARS = 33;
constexpr size_t COMPUTE_CT_SCALARS = 27;

// Kernel indices in the descriptor (reader, writer, compute) and the address slots of their
// runtime args -- read by override_runtime_arguments.
constexpr uint32_t READER_KERNEL = 0;
constexpr uint32_t WRITER_KERNEL = 1;
constexpr uint32_t READER_RT_X = 0;
constexpr uint32_t READER_RT_GAMMA = 1;
constexpr uint32_t READER_RT_BIAS = 10;
constexpr uint32_t READER_RT_RESIDUAL = 11;
constexpr uint32_t WRITER_RT_OUT = 0;

// ---------------------------------------------------------------------------
// Environment switches (read once, as the Python reads them at import).
// ---------------------------------------------------------------------------
bool stage_zones() {
    static const bool on = [] {
        const char* v = std::getenv("RMS_STAGE_ZONES");
        return v != nullptr && std::strcmp(v, "") != 0 && std::strcmp(v, "0") != 0;
    }();
    return on;
}

const std::vector<std::string>& ablate() {
    static const std::vector<std::string> names = [] {
        static const char* known[] = {
            "READ_X", "WRITE", "COMPUTE", "PER_CHANNEL", "ROOT_SUM", "ROOT_FINALIZE", "RECONFIG", "GATHER_ZERO"};
        std::vector<std::string> out;
        const char* v = std::getenv("RMS_ABLATE");
        if (v == nullptr) {
            return out;
        }
        std::string s(v);
        size_t start = 0;
        while (start <= s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) {
                end = s.size();
            }
            std::string tok = s.substr(start, end - start);
            // strip + upper
            size_t a = tok.find_first_not_of(" \t\n\r");
            size_t b = tok.find_last_not_of(" \t\n\r");
            tok = (a == std::string::npos) ? "" : tok.substr(a, b - a + 1);
            std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char c) { return std::toupper(c); });
            for (const char* k : known) {
                if (tok == k) {
                    out.push_back(tok);
                    break;
                }
            }
            start = end + 1;
        }
        return out;
    }();
    return names;
}

KernelDescriptor::Defines kernel_defines() {
    KernelDescriptor::Defines d;
    if (stage_zones()) {
        d.emplace_back("RMS_STAGE_ZONES", "1");
    }
    for (const auto& n : ablate()) {
        d.emplace_back("RMS_ABLATE_" + n, "1");
    }
    return d;
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------
int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Python floor division (the solves divide possibly-negative byte counts).
int64_t floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

int64_t largest_divisor_at_most(int64_t n, int64_t cap) {
    cap = std::max<int64_t>(1, std::min(cap, n));
    for (int64_t d = cap; d > 0; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

std::pair<int64_t, int64_t> width_chunk(int64_t wt_core, int64_t cap, bool ragged_ok) {
    cap = std::max<int64_t>(1, std::min(cap, wt_core));
    const int64_t div = largest_divisor_at_most(wt_core, cap);
    if (!RAGGED_WIDTH_CHUNK || !ragged_ok || div >= cap) {
        return {div, wt_core / div};
    }
    const int64_t n = div_up(wt_core, cap);
    const int64_t wtc = div_up(wt_core, n);
    if (wtc <= div) {
        return {div, wt_core / div};
    }
    return {wtc, n};
}

int64_t dm_txn_rows(int64_t block_rows) {
    int64_t cap = DM_TXN_ROWS_MAX == 0 ? block_rows : std::min(DM_TXN_ROWS_MAX, block_rows);
    cap = std::max<int64_t>(1, cap);
    for (int64_t n = cap; n > 0; --n) {
        if (block_rows % n == 0) {
            return n;
        }
    }
    return 1;
}

uint32_t pack_txn_rows(int64_t block_rows, int64_t txn_rows) {
    TT_FATAL(
        0 < block_rows && block_rows < (1 << 16),
        "rms_norm_ttnn: BLOCK_ROWS={} does not fit the packed CT word",
        block_rows);
    TT_FATAL(
        txn_rows >= 1 && block_rows % txn_rows == 0,
        "rms_norm_ttnn: the NoC transaction unit ({}) must divide BLOCK_ROWS ({})",
        txn_rows,
        block_rows);
    return static_cast<uint32_t>(block_rows | ((txn_rows - 1) << 16));
}

// dest_helpers.hpp get_dest_limit(), host mirror (DEST_AUTO_LIMIT).
int64_t dest_tile_limit(const ComputeConfigDescriptor& cfg) {
    if (cfg.dst_full_sync_en) {
        return cfg.fp32_dest_acc_en ? 8 : 16;
    }
    return cfg.fp32_dest_acc_en ? 4 : 8;
}

// The compute kernel's PASS_B_BLK (pass_b_blk / pass_b_blk_small / PASS_B_AUTO), host mirror: the DEST block the
// residual add packs t in, hence cb_residual_sum's push unit.
int64_t pass_b_block(int64_t wt_chunk, int64_t block_rows, int64_t subblock_ct, int64_t cap) {
    if (subblock_ct != 0) {
        return subblock_ct;
    }
    auto largest = [&](int64_t wt) {
        int64_t b = std::min(cap, wt);
        while (b > 1 && wt % b != 0) {
            --b;
        }
        return b;
    };
    if (block_rows > 1) {
        return largest(wt_chunk);
    }
    for (int64_t b = 2; b <= cap && b <= wt_chunk; ++b) {
        if (wt_chunk % b == 0) {
            return b;
        }
    }
    return largest(wt_chunk);
}

int64_t residual_depth(int64_t depth_x) { return CB_R_DEPTH ? CB_R_DEPTH : depth_x; }

int64_t x_squared_wt_of(int64_t wt_chunk, int64_t partial_w) {
    if (partial_w != 0) {
        return wt_chunk;
    }
    if (wt_chunk <= DEST_ACC_SQUARE_MAX_WT) {
        return 1;
    }
    int64_t g = 1;
    for (int64_t d = 2; d <= std::min(SQ_FOLD_GROUP, wt_chunk); ++d) {
        if (wt_chunk % d == 0) {
            g = d;
        }
    }
    return wt_chunk / g;
}

int64_t norm_cb_depth(bool has_gamma, bool has_bias, int64_t block_rows) {
    if (!(has_gamma || has_bias)) {
        return 0;
    }
    if (!(has_gamma && has_bias)) {
        return 1;
    }
    return block_rows == 1 ? 1 : 2;
}

int64_t cb_block_bytes(
    int64_t bt,
    int64_t it,
    int64_t depth_x,
    int64_t depth_out,
    bool has_gamma,
    bool has_bias,
    bool has_residual,
    int64_t depth_r,
    int64_t block_rows) {
    int64_t total = depth_x * bt + it + norm_cb_depth(has_gamma, has_bias, block_rows) * it + depth_out * bt;
    if (has_residual) {
        total += depth_r * bt + it;
    }
    return total;
}

uint32_t f32_bits(double v) {
    const float f = static_cast<float>(v);
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

bool is_bfp_dtype(DataType dt) { return dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B; }

DataType intermediate_dtype(DataType dt) { return is_bfp_dtype(dt) ? DataType::BFLOAT16 : dt; }

uint32_t tile_bytes(DataType dt) { return tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(dt)); }

tt::DataFormat data_format(DataType dt) { return tt::tt_metal::datatype_to_dataformat_converter(dt); }

int64_t stick_elem_bytes(const std::optional<Tensor>& t) {
    if (!t.has_value()) {
        return 0;
    }
    if (is_bfp_dtype(t->dtype())) {
        TT_FATAL(
            t->layout() == Layout::TILE,
            "rms_norm_ttnn: {} is a block-float format and cannot be ROW_MAJOR",
            t->dtype());
        return 0;
    }
    return tt::datum_size(data_format(t->dtype()));
}

std::vector<uint32_t> shape_of(const Tensor& t) {
    const auto& s = t.logical_shape();
    std::vector<uint32_t> v;
    v.reserve(s.rank());
    for (size_t i = 0; i < s.rank(); ++i) {
        v.push_back(s[i]);
    }
    return v;
}

int64_t prod(const std::vector<uint32_t>& v, size_t begin, size_t end) {
    int64_t n = 1;
    for (size_t i = begin; i < end; ++i) {
        n *= v[i];
    }
    return n;
}

// ---- the combine's tree / pages -------------------------------------------------------------
std::vector<int64_t> combine_tree_candidates(int64_t group_size) {
    std::vector<int64_t> ordered;
    for (int64_t f = COMBINE_TREE_F0_MAX; f >= COMBINE_TREE_F0_MIN; --f) {
        if (group_size % f == 0) {
            ordered.push_back(f);
        }
    }
    ordered.push_back(COMBINE_TREE_F0_MAX);
    std::vector<int64_t> out;
    for (int64_t f : ordered) {
        if (std::find(out.begin(), out.end(), f) == out.end()) {
            out.push_back(f);
        }
    }
    return out;
}

std::optional<std::pair<int64_t, int64_t>> combine_tree_arity(int64_t group_size, int64_t rows_per_round) {
    for (int64_t f0 : combine_tree_candidates(group_size)) {
        const int64_t f1 = div_up(group_size, f0);
        if (f1 < 2) {
            continue;
        }
        if (rows_per_round * (group_size - f0 - f1) < COMBINE_TREE_MIN_DELETED_FOLD_TILES) {
            continue;
        }
        return std::make_pair(f0, f1);
    }
    return std::nullopt;
}

bool combine_fin_spread(bool combine) { return combine && COMBINE_FIN_SPREAD; }

uint32_t combine_gather_faces_ct(bool combine, bool compact) {
    const uint32_t mcast_faces = (compact || !combine) ? 0u : (COMBINE_MCAST_FACES & 0xFFu);
    return (GATHER_FACES & 0xFFu) | (mcast_faces << 8);
}

bool combine_mcast_pre_handshake(bool combine, bool single_round) {
    if (!combine) {
        return true;
    }
    return !(COMBINE_MCAST_FIRE_AND_FORGET && single_round);
}

NOC combine_noc(bool native_in) { return native_in ? COMBINE_NOC_RESIDENT : COMBINE_NOC_STREAMED; }

kh::McastConfig mcast_cfg(bool native_in, uint32_t base_sem_id = 0) {
    kh::McastConfig cfg;
    cfg.noc = combine_noc(native_in);
    cfg.handshake = true;
    cfg.base_sem_id = base_sem_id;
    return cfg;
}

// ---- per-channel form ------------------------------------------------------------------------
std::pair<bool, int64_t> per_channel_form(const Tensor& operand, int64_t width) {
    const auto shape = shape_of(operand);
    const int64_t wt = div_up(std::max<int64_t>(1, width), TILE_DIM);
    if (operand.layout() == Layout::ROW_MAJOR && shape.size() >= 2 && shape.back() == TILE_DIM) {
        const int64_t folded = prod(shape, 0, shape.size() - 1);
        if (folded == wt) {
            return {true, folded * TILE_DIM};
        }
    }
    return {false, shape.empty() ? 1 : shape.back()};
}

// ---- placement ---------------------------------------------------------------------------------
enum class Scheme { Rows, ShardH, ShardW };
enum class GroupAxis { None, X, Y };

std::optional<std::pair<int64_t, int64_t>> shard_shape(const Tensor& t) {
    const auto& mc = t.memory_config();
    if (mc.memory_layout() == TensorMemoryLayout::INTERLEAVED || !mc.shard_spec().has_value()) {
        return std::nullopt;
    }
    return std::pair<int64_t, int64_t>{mc.shard_spec()->shape[0], mc.shard_spec()->shape[1]};
}

int64_t shard_l1_bytes(const Tensor& t) {
    const auto sh = shard_shape(t);
    if (!sh.has_value() || t.memory_config().buffer_type() != BufferType::L1) {
        return 0;
    }
    if (t.layout() == Layout::TILE) {
        return (sh->first / TILE_DIM) * (sh->second / TILE_DIM) * tile_bytes(t.dtype());
    }
    const int64_t align = tt::tt_metal::hal::get_l1_alignment();
    const int64_t row_bytes = sh->second * stick_elem_bytes(t);
    return sh->first * (((row_bytes + align - 1) / align) * align);
}

std::pair<int64_t, int64_t> shard_tile_extent(const Tensor& t) {
    const auto sh = shard_shape(t);
    TT_FATAL(sh.has_value(), "rms_norm_ttnn: shard_tile_extent on an interleaved tensor");
    TT_FATAL(sh->first % TILE_DIM == 0 && sh->second % TILE_DIM == 0, "rms_norm_ttnn: TILE shard is not tile-aligned");
    return {sh->first / TILE_DIM, sh->second / TILE_DIM};
}

bool same_shard_spec(const Tensor& a, const Tensor& b) {
    const auto& ma = a.memory_config();
    const auto& mb = b.memory_config();
    if (ma.memory_layout() != mb.memory_layout()) {
        return false;
    }
    if (!ma.shard_spec().has_value() || !mb.shard_spec().has_value()) {
        return !ma.shard_spec().has_value() && !mb.shard_spec().has_value();
    }
    return shard_shape(a) == shard_shape(b) && ma.shard_spec()->grid == mb.shard_spec()->grid &&
           ma.shard_spec()->orientation == mb.shard_spec()->orientation;
}

CoreRangeSet full_grid(tt::tt_metal::IDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
}

std::vector<CoreCoord> cores_in(const CoreRangeSet& crs) {
    return tt::tt_metal::corerange_to_cores(crs, std::nullopt, true);
}

bool shard_row_wise(const Tensor& t) {
    const auto& ss = t.memory_config().shard_spec();
    return !ss.has_value() || ss->orientation == ShardOrientation::ROW_MAJOR;
}

std::vector<CoreCoord> shard_ordered_cores(const Tensor& t, const CoreRangeSet& crs) {
    return tt::tt_metal::corerange_to_cores(crs, std::nullopt, shard_row_wise(t));
}

CBDescriptor make_cb(uint8_t index, int64_t page_size, int64_t num_pages, DataType dtype, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = static_cast<uint32_t>(num_pages * page_size);
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = index, .data_format = data_format(dtype), .page_size = static_cast<uint32_t>(page_size)});
    return cb;
}

struct Work {
    CoreCoord core;
    int64_t row_start = 0;
    int64_t row_count = 0;
    int64_t w_start = 0;
    int64_t w_real = 0;
    bool is_root = false;
    int64_t slot = 0;
    int64_t stick_base = 0;
    int64_t stick_count = 0;
    int64_t w_off_elems = 0;
    int64_t w_real_elems = 0;
};

Work work_tile_axis(
    CoreCoord core,
    int64_t row_start,
    int64_t row_count,
    int64_t w_start,
    int64_t w_real,
    bool is_root,
    int64_t slot,
    int64_t W,
    int64_t R_rm) {
    Work w;
    w.core = core;
    w.row_start = row_start;
    w.row_count = row_count;
    w.w_start = w_start;
    w.w_real = w_real;
    w.is_root = is_root;
    w.slot = slot;
    w.stick_base = row_start * TILE_DIM;
    int64_t sticks = row_count * TILE_DIM;
    if (R_rm) {
        sticks = std::max<int64_t>(0, std::min(sticks, R_rm - w.stick_base));
    }
    w.stick_count = sticks;
    w.w_off_elems = w_start * TILE_DIM;
    w.w_real_elems = std::max<int64_t>(0, std::min(w_real * TILE_DIM, W - w.w_off_elems));
    return w;
}

int64_t band_tile_span(int64_t w_off, int64_t w_real_elems) {
    return div_up((w_off % TILE_DIM) + w_real_elems, TILE_DIM);
}

Work work_band(
    CoreCoord core,
    int64_t stick_base,
    int64_t stick_count,
    int64_t w_off,
    int64_t band_elems,
    int64_t W,
    bool is_root,
    int64_t slot) {
    Work w;
    w.core = core;
    w.w_real_elems = std::max<int64_t>(0, std::min(band_elems, W - w_off));
    w.row_start = 0;
    w.row_count = div_up(stick_count, TILE_DIM);
    w.w_start = w_off / TILE_DIM;
    w.w_real = band_tile_span(w_off, w.w_real_elems);
    w.is_root = is_root;
    w.slot = slot;
    w.stick_base = stick_base;
    w.stick_count = stick_count;
    w.w_off_elems = w_off;
    return w;
}

Work work_inactive(CoreCoord core) {
    Work w;
    w.core = core;
    return w;
}

// The combine's multicast: a Mcast1D line family or one Mcast2D box.
struct CombineMcast {
    std::variant<kh::Mcast1D, kh::Mcast2D> m;

    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake) const {
        return std::visit([&](const auto& x) { return x.compile_time_args(pre_handshake); }, m);
    }
    std::vector<uint32_t> runtime_args(const CoreCoord& core) const {
        return std::visit([&](const auto& x) { return x.runtime_args(core); }, m);
    }
    std::vector<SemaphoreDescriptor> owned_semaphores() const {
        return std::visit([](const auto& x) { return x.owned_semaphores(); }, m);
    }
    uint32_t next_base_sem_id() const {
        return std::visit([](const auto& x) { return x.next_base_sem_id(); }, m);
    }
};

struct Plan {
    Scheme scheme = Scheme::Rows;
    std::vector<Work> assignment;
    CoreRangeSet all_cores;
    bool native_in = false;
    bool native_out = false;
    int64_t wt_per_core = 0;
    bool combine = false;
    int64_t group_size = 1;
    std::optional<CombineMcast> mcast;
    uint32_t gather_sem_id = 0;
    GroupAxis group_axis = GroupAxis::None;
    int64_t l1_reserved = 0;
    bool band = false;
    bool band_out_local = false;
    int64_t shard_row_bytes = 0;
    int64_t out_shard_row_bytes = 0;
};

bool combine_noc_swapped(const Plan& plan) { return plan.combine && combine_noc(plan.native_in) != NOC::NOC_1; }

KernelDescriptor::ConfigDescriptor writer_dm_config(const Plan& plan) {
    if (!combine_noc_swapped(plan)) {
        return tt::tt_metal::WriterConfigDescriptor{};
    }
    return tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = combine_noc(plan.native_in)};
}

NOC reader_noc(const Plan& plan) { return combine_noc_swapped(plan) ? NOC::NOC_1 : NOC::NOC_0; }

KernelDescriptor::ConfigDescriptor reader_dm_config(const Plan& plan) {
    if (!combine_noc_swapped(plan)) {
        return tt::tt_metal::ReaderConfigDescriptor{};
    }
    return tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
}

int64_t width_group_cores(int64_t Wt, int64_t cap) { return largest_divisor_at_most(Wt, std::max<int64_t>(1, cap)); }

std::pair<int64_t, int64_t> auto_width_split(tt::tt_metal::IDevice* device, int64_t Rt, int64_t Wt) {
    const auto grid = device->compute_with_storage_grid_size();
    const int64_t num_cores = static_cast<int64_t>(grid.x) * grid.y;
    const int64_t row_cores = std::max<int64_t>(1, std::min(Rt, num_cores));
    int64_t best_gw = 1, best_gh = 1, best_total = row_cores;
    for (int64_t gh = 1; gh <= std::min<int64_t>(Rt, grid.y); ++gh) {
        int64_t cap = std::min({num_cores / gh, Wt / WIDTH_SPLIT_MIN_WT_PER_CORE, WIDTH_SPLIT_MAX_GROUP_CORES});
        if (gh > 1) {
            cap = std::min<int64_t>(cap, grid.x);
        }
        const int64_t gw = width_group_cores(Wt, cap);
        const int64_t total = gw * gh;
        if (gw >= 2 && total > best_total) {
            best_gw = gw;
            best_gh = gh;
            best_total = total;
        }
    }
    if (best_total < WIDTH_SPLIT_MIN_GAIN * row_cores) {
        return {1, 1};
    }
    return {best_gw, best_gh};
}

std::pair<int64_t, int64_t> resolve_width_split(tt::tt_metal::IDevice* device, int64_t Rt, int64_t Wt) {
    if (GRID_W == 0) {
        return auto_width_split(device, Rt, Wt);
    }
    const auto grid = device->compute_with_storage_grid_size();
    const int64_t num_cores = static_cast<int64_t>(grid.x) * grid.y;
    const int64_t cap = std::min(GRID_W, num_cores);
    const int64_t gh = cap > static_cast<int64_t>(grid.x) ? 1 : std::max<int64_t>(1, std::min<int64_t>(Rt, grid.y));
    const int64_t gw = width_group_cores(Wt, std::min(cap, num_cores / gh));
    return {gw, gh};
}

Plan plan_interleaved_width_split(
    tt::tt_metal::IDevice* device, int64_t Rt, int64_t Wt, int64_t W, int64_t R_rm, int64_t gw, int64_t gh) {
    const auto grid = device->compute_with_storage_grid_size();
    const int64_t wt_per_core = Wt / gw;
    const auto mc_cfg = mcast_cfg(false);
    Plan p;
    p.scheme = Scheme::ShardW;
    if (gw <= static_cast<int64_t>(grid.x)) {
        gh = std::max<int64_t>(1, std::min<int64_t>(gh, grid.y));
        CoreRangeSet crs(CoreRange(CoreCoord(0, 0), CoreCoord(gw - 1, gh - 1)));
        const int64_t base = Rt / gh;
        const int64_t extra = Rt % gh;
        for (const auto& core : cores_in(crs)) {
            const int64_t y = core.y;
            const int64_t rows = base + (y < extra ? 1 : 0);
            const int64_t row_start = y * base + std::min(y, extra);
            const int64_t w_start = static_cast<int64_t>(core.x) * wt_per_core;
            p.assignment.push_back(
                work_tile_axis(core, row_start, rows, w_start, wt_per_core, core.x == 0, core.x, W, R_rm));
        }
        if (gw > 1) {
            p.mcast = CombineMcast{kh::Mcast1D(device, crs, kh::Mcast1DShape::PerRow, 0, mc_cfg)};
        }
        p.group_axis = GroupAxis::Y;
        p.all_cores = crs;
    } else {
        TT_FATAL(gh == 1, "rms_norm_ttnn: a width group wider than the grid must be the only group");
        const int64_t rows_used = div_up(gw, grid.x);
        CoreRangeSet crs(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, rows_used - 1)));
        int64_t i = 0;
        for (const auto& core : cores_in(crs)) {
            if (i >= gw) {
                p.assignment.push_back(work_inactive(core));
            } else {
                p.assignment.push_back(work_tile_axis(core, 0, Rt, i * wt_per_core, wt_per_core, i == 0, i, W, R_rm));
            }
            ++i;
        }
        const auto root = p.assignment[0].core;
        p.mcast = CombineMcast{kh::Mcast2D(device, crs, CoreCoord(root.x, root.y), mc_cfg, gw - 1)};
        p.group_axis = GroupAxis::None;
        p.all_cores = crs;
    }
    p.native_in = false;
    p.native_out = false;
    p.wt_per_core = wt_per_core;
    p.combine = p.mcast.has_value();
    p.group_size = gw;
    p.gather_sem_id = p.mcast.has_value() ? p.mcast->next_base_sem_id() : 0;
    return p;
}

Plan plan_rows(
    tt::tt_metal::IDevice* device,
    const Tensor& input,
    int64_t Rt,
    int64_t Wt,
    int64_t W,
    int64_t R_rm,
    bool allow_width_split) {
    if (allow_width_split && Wt > 1 && input.layout() == Layout::TILE) {
        const auto [gw, gh] = resolve_width_split(device, Rt, Wt);
        if (gw > 1) {
            return plan_interleaved_width_split(device, Rt, Wt, W, R_rm, gw, gh);
        }
    }
    const auto [num_cores, all_cores, g1, g2, rpc1, rpc2] =
        tt::tt_metal::split_work_to_cores(full_grid(device), static_cast<uint32_t>(Rt), true);
    Plan p;
    p.scheme = Scheme::Rows;
    int64_t row_cursor = 0;
    auto take = [&](const CoreRangeSet& group, int64_t rpc) {
        for (const auto& core : cores_in(group)) {
            p.assignment.push_back(work_tile_axis(core, row_cursor, rpc, 0, Wt, true, 0, W, R_rm));
            row_cursor += rpc;
        }
    };
    take(g1, rpc1);
    take(g2, rpc2);
    TT_FATAL(row_cursor == Rt, "rms_norm_ttnn: work split covers {} of {} tile-rows", row_cursor, Rt);
    TT_FATAL(
        p.assignment.size() == num_cores,
        "rms_norm_ttnn: {} cores assigned, expected {}",
        p.assignment.size(),
        num_cores);
    p.all_cores = all_cores;
    p.wt_per_core = Wt;
    p.group_size = 1;
    return p;
}

Plan plan_band(tt::tt_metal::IDevice* device, const Tensor& input, const Tensor& output, int64_t W, int64_t R_rm) {
    const auto in_ml = input.memory_config().memory_layout();
    const auto [shard_h, shard_w] = *shard_shape(input);
    const int64_t shard_row_bytes = input.buffer()->aligned_page_size();
    const auto& shard_grid = input.memory_config().shard_spec()->grid;
    const auto shard_cores = shard_ordered_cores(input, shard_grid);
    const bool row_wise = shard_row_wise(input);

    const bool band_out_local = output.layout() == Layout::ROW_MAJOR && same_shard_spec(input, output);
    const auto out_ml = output.memory_config().memory_layout();
    if (!band_out_local && out_ml != TensorMemoryLayout::INTERLEAVED && out_ml != TensorMemoryLayout::HEIGHT_SHARDED) {
        throw NotImplementedErrorCpp(fmt::format(
            "rms_norm_ttnn: a ROW_MAJOR {} input needs an output that is either the SAME shard spec (written in "
            "place) or stick-paged (INTERLEAVED / HEIGHT_SHARDED); got {} with a different geometry",
            in_ml,
            out_ml));
    }
    const int64_t out_shard_row_bytes = band_out_local ? output.buffer()->aligned_page_size() : 0;

    const auto bbox = shard_grid.bounding_box();
    Plan p;
    p.scheme = Scheme::ShardW;
    if (in_ml == TensorMemoryLayout::WIDTH_SHARDED) {
        CoreRangeSet bbox_crs(CoreRange(bbox.start_coord, bbox.end_coord));
        const int64_t group_size = static_cast<int64_t>(shard_cores.size());
        const auto root = shard_cores[0];
        std::map<std::pair<uint32_t, uint32_t>, int64_t> owned;
        for (size_t i = 0; i < shard_cores.size(); ++i) {
            owned[{shard_cores[i].x, shard_cores[i].y}] = static_cast<int64_t>(i);
        }
        for (const auto& core : cores_in(bbox_crs)) {
            auto it = owned.find({core.x, core.y});
            if (it == owned.end()) {
                p.assignment.push_back(work_inactive(core));
                continue;
            }
            const int64_t i = it->second;
            p.assignment.push_back(work_band(core, 0, R_rm, i * shard_w, shard_w, W, i == 0, i));
        }
        p.all_cores = bbox_crs;
        p.group_axis = GroupAxis::None;
        p.group_size = group_size;
        if (group_size > 1) {
            p.mcast = CombineMcast{
                kh::Mcast2D(device, bbox_crs, CoreCoord(root.x, root.y), mcast_cfg(false), group_size - 1)};
        }
    } else {
        const int64_t nx = bbox.end_coord.x - bbox.start_coord.x + 1;
        const int64_t ny = bbox.end_coord.y - bbox.start_coord.y + 1;
        TT_FATAL(
            static_cast<int64_t>(shard_grid.num_cores()) == nx * ny,
            "rms_norm_ttnn: BLOCK shard grid is not a full rectangle");
        const int64_t group_size = row_wise ? nx : ny;
        for (const auto& core : cores_in(shard_grid)) {
            const int64_t dx = static_cast<int64_t>(core.x) - bbox.start_coord.x;
            const int64_t dy = static_cast<int64_t>(core.y) - bbox.start_coord.y;
            const int64_t h_idx = row_wise ? dy : dx;
            const int64_t w_idx = row_wise ? dx : dy;
            p.assignment.push_back(work_band(
                core,
                std::min(h_idx * shard_h, R_rm),
                std::max<int64_t>(0, std::min(shard_h, R_rm - h_idx * shard_h)),
                w_idx * shard_w,
                shard_w,
                W,
                w_idx == 0,
                w_idx));
        }
        p.all_cores = shard_grid;
        p.group_axis = row_wise ? GroupAxis::Y : GroupAxis::X;
        p.group_size = group_size;
        if (group_size > 1) {
            p.mcast = CombineMcast{kh::Mcast1D(
                device,
                shard_grid,
                row_wise ? kh::Mcast1DShape::PerRow : kh::Mcast1DShape::PerColumn,
                0,
                mcast_cfg(false))};
        }
    }
    int64_t wt_band = 1;
    bool any = false;
    for (const auto& w : p.assignment) {
        if (w.row_count) {
            wt_band = any ? std::max(wt_band, w.w_real) : w.w_real;
            any = true;
        }
    }
    if (!any || wt_band == 0) {
        wt_band = 1;
    }
    p.native_in = false;
    p.native_out = false;
    p.wt_per_core = wt_band;
    p.combine = p.mcast.has_value();
    p.gather_sem_id = p.mcast.has_value() ? p.mcast->next_base_sem_id() : 0;
    p.band = true;
    p.band_out_local = band_out_local;
    p.shard_row_bytes = shard_row_bytes;
    p.out_shard_row_bytes = out_shard_row_bytes;
    return p;
}

Plan plan_placement(
    tt::tt_metal::IDevice* device,
    const Tensor& input,
    const Tensor& output,
    bool is_tile,
    int64_t Rt,
    int64_t Wt,
    int64_t W,
    int64_t R_rm,
    int64_t partial_w,
    bool force_rows) {
    const auto in_ml = input.memory_config().memory_layout();
    if (!force_rows && !is_tile &&
        (in_ml == TensorMemoryLayout::WIDTH_SHARDED || in_ml == TensorMemoryLayout::BLOCK_SHARDED)) {
        return plan_band(device, input, output, W, R_rm);
    }
    if (force_rows || !is_tile || in_ml == TensorMemoryLayout::INTERLEAVED) {
        return plan_rows(device, input, Rt, Wt, W, R_rm, !force_rows);
    }

    const auto [shard_h_t, shard_w_t] = shard_tile_extent(input);
    const auto& shard_grid = input.memory_config().shard_spec()->grid;
    const auto shard_cores = shard_ordered_cores(input, shard_grid);
    const bool row_wise = shard_row_wise(input);
    const bool native_out = output.layout() == Layout::TILE && same_shard_spec(input, output);

    if (in_ml == TensorMemoryLayout::HEIGHT_SHARDED) {
        if (shard_w_t != Wt) {
            return plan_rows(device, input, Rt, Wt, W, R_rm, false);
        }
        Plan p;
        p.scheme = Scheme::ShardH;
        for (size_t i = 0; i < shard_cores.size(); ++i) {
            const int64_t row_start = static_cast<int64_t>(i) * shard_h_t;
            const int64_t row_count = std::max<int64_t>(0, std::min(shard_h_t, Rt - row_start));
            p.assignment.push_back(
                work_tile_axis(shard_cores[i], std::min(row_start, Rt), row_count, 0, Wt, true, 0, W, R_rm));
        }
        p.all_cores = shard_grid;
        p.native_in = true;
        p.native_out = native_out;
        p.wt_per_core = Wt;
        p.group_size = 1;
        return p;
    }

    const bool ragged_w = (Wt % shard_w_t) != 0;
    if (ragged_w && partial_w) {
        return plan_rows(device, input, Rt, Wt, W, R_rm, false);
    }

    const auto bbox = shard_grid.bounding_box();
    const int64_t nx = bbox.end_coord.x - bbox.start_coord.x + 1;
    const int64_t ny = bbox.end_coord.y - bbox.start_coord.y + 1;
    Plan p;
    p.scheme = Scheme::ShardW;
    if (in_ml == TensorMemoryLayout::WIDTH_SHARDED) {
        const int64_t group_size = static_cast<int64_t>(shard_cores.size());
        const auto root = shard_cores[0];
        std::map<std::pair<uint32_t, uint32_t>, int64_t> owned;
        for (size_t i = 0; i < shard_cores.size(); ++i) {
            owned[{shard_cores[i].x, shard_cores[i].y}] = static_cast<int64_t>(i);
        }
        CoreRangeSet bbox_crs(CoreRange(bbox.start_coord, bbox.end_coord));
        for (const auto& core : cores_in(bbox_crs)) {
            auto it = owned.find({core.x, core.y});
            if (it == owned.end()) {
                p.assignment.push_back(work_inactive(core));
                continue;
            }
            const int64_t i = it->second;
            const int64_t w_start = i * shard_w_t;
            p.assignment.push_back(work_tile_axis(
                core, 0, Rt, w_start, std::max<int64_t>(0, std::min(shard_w_t, Wt - w_start)), i == 0, i, W, R_rm));
        }
        p.all_cores = bbox_crs;
        p.group_axis = GroupAxis::None;
        p.group_size = group_size;
        if (group_size > 1) {
            p.mcast =
                CombineMcast{kh::Mcast2D(device, bbox_crs, CoreCoord(root.x, root.y), mcast_cfg(true), group_size - 1)};
        }
    } else {
        TT_FATAL(
            static_cast<int64_t>(shard_grid.num_cores()) == nx * ny,
            "rms_norm_ttnn: BLOCK shard grid is not a full rectangle");
        const int64_t group_size = row_wise ? nx : ny;
        for (const auto& core : cores_in(shard_grid)) {
            const int64_t dx = static_cast<int64_t>(core.x) - bbox.start_coord.x;
            const int64_t dy = static_cast<int64_t>(core.y) - bbox.start_coord.y;
            const int64_t h_idx = row_wise ? dy : dx;
            const int64_t w_idx = row_wise ? dx : dy;
            const int64_t row_start = h_idx * shard_h_t;
            const int64_t w_start = w_idx * shard_w_t;
            p.assignment.push_back(work_tile_axis(
                core,
                std::min(row_start, Rt),
                std::max<int64_t>(0, std::min(shard_h_t, Rt - row_start)),
                w_start,
                std::max<int64_t>(0, std::min(shard_w_t, Wt - w_start)),
                w_idx == 0,
                w_idx,
                W,
                R_rm));
        }
        p.all_cores = shard_grid;
        p.group_axis = row_wise ? GroupAxis::Y : GroupAxis::X;
        p.group_size = group_size;
        if (group_size > 1) {
            p.mcast = CombineMcast{kh::Mcast1D(
                device,
                shard_grid,
                row_wise ? kh::Mcast1DShape::PerRow : kh::Mcast1DShape::PerColumn,
                0,
                mcast_cfg(true))};
        }
    }
    p.native_in = true;
    p.native_out = native_out;
    p.wt_per_core = shard_w_t;
    p.combine = p.mcast.has_value();
    p.gather_sem_id = p.mcast.has_value() ? p.mcast->next_base_sem_id() : 0;
    return p;
}

int64_t combine_fixed_pages(const Plan& plan, bool compact, const std::optional<std::pair<int64_t, int64_t>>& tree) {
    if (!plan.combine) {
        return 0;
    }
    int64_t pages = 0;
    if (!tree.has_value()) {
        pages = plan.group_size + plan.group_size % 2;
    } else {
        const auto [f0, f1] = *tree;
        pages = (f0 + f0 % 2) + (f1 + f1 % 2) + CB_COMBINE_FLAT_DEPTH;
    }
    pages += CB_COMBINE_FLAT_DEPTH;
    if (compact) {
        pages += 2 * CB_COMBINE_FLAT_DEPTH;
    }
    if (combine_fin_spread(plan.combine)) {
        pages += CB_COMBINE_FLAT_DEPTH;
    }
    return pages;
}

std::vector<uint32_t> null_accessor_args() { return tt::tt_metal::TensorAccessorArgs().get_compile_time_args(); }

std::vector<uint32_t> accessor_args(const std::optional<Tensor>& t) {
    if (!t.has_value()) {
        return null_accessor_args();
    }
    return tt::tt_metal::TensorAccessorArgs(*t->buffer()).get_compile_time_args();
}

KernelDescriptor make_kernel(
    const char* file,
    const CoreRangeSet& cores,
    KernelDescriptor::Defines defines,
    std::vector<uint32_t> ct,
    KernelDescriptor::RuntimeArgs rt,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.defines = std::move(defines);
    k.runtime_args = std::move(rt);
    k.config = std::move(config);
    return k;
}

ProgramDescriptor zero_volume_descriptor(const CoreRangeSet& all_cores, const ComputeConfigDescriptor& compute_config) {
    const int64_t bt = tile_bytes(DataType::BFLOAT16);
    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_INPUT_TILES, bt, 1, DataType::BFLOAT16, all_cores));
    desc.cbs.push_back(make_cb(CB_X_SQUARED, bt, 1, DataType::BFLOAT16, all_cores));
    desc.cbs.push_back(make_cb(CB_SCALER, bt, 1, DataType::BFLOAT16, all_cores));
    desc.cbs.push_back(make_cb(CB_ROW_STAT, tile_bytes(DataType::FLOAT32), 1, DataType::FLOAT32, all_cores));
    desc.cbs.push_back(make_cb(CB_OUTPUT_TILES, bt, 1, DataType::BFLOAT16, all_cores));
    const auto null_acc = null_accessor_args();

    std::vector<uint32_t> reader_ct = {1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    reader_ct.insert(reader_ct.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    TT_FATAL(reader_ct.size() == READER_CT_SCALARS, "rms_norm_ttnn: zero-volume reader CT count drifted");
    for (int i = 0; i < 4; ++i) {
        reader_ct.insert(reader_ct.end(), null_acc.begin(), null_acc.end());
    }

    std::vector<uint32_t> writer_ct = {1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, GATHER_FACES, 0, 0};
    writer_ct.insert(writer_ct.end(), {0, 0, 0, 0, 0, 0});
    writer_ct.insert(writer_ct.end(), null_acc.begin(), null_acc.end());

    std::vector<uint32_t> compute_ct = {
        1, 1, 1, 1, 0, 0, 0, f32_bits(1.0), f32_bits(0.0), REDUCE_BULK, 0, 1, 0, 1, 1, 1, 0, 0, 0};
    compute_ct.insert(compute_ct.end(), {0, 0, 0, 0, 0, 0, 0, 0});
    TT_FATAL(compute_ct.size() == COMPUTE_CT_SCALARS, "rms_norm_ttnn: zero-volume compute CT count drifted");

    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    for (const auto& core : cores_in(all_cores)) {
        reader_rt.emplace_back(core, std::vector<uint32_t>(12, 0));
        writer_rt.emplace_back(core, std::vector<uint32_t>(12, 0));
        compute_rt.emplace_back(core, std::vector<uint32_t>(4, 0));
    }
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_reader.cpp",
        all_cores,
        kernel_defines(),
        std::move(reader_ct),
        std::move(reader_rt),
        tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_writer.cpp",
        all_cores,
        kernel_defines(),
        std::move(writer_ct),
        std::move(writer_rt),
        tt::tt_metal::WriterConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_compute.cpp",
        all_cores,
        kernel_defines(),
        std::move(compute_ct),
        std::move(compute_rt),
        compute_config));
    return desc;
}

struct Solved {
    int64_t block_rows;
    int64_t wt_chunk;
    int64_t num_w_chunks;
    int64_t cb_x_depth;
    int64_t cb_out_depth;
    bool x_resident;
    int64_t rm_stage_depth;
    bool narrow_pc_stage;
    bool pc_compact;
};

}  // namespace

ProgramDescriptor create_program_descriptor(
    const Tensor& input,
    const Tensor& output,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& residual,
    double epsilon,
    const ComputeConfigDescriptor& compute_config,
    uint32_t subblock_w,
    const std::optional<Tensor>& residual_sum) {
    auto* device = input.device();
    const auto shape = shape_of(input);
    const size_t rank = shape.size();

    const bool is_tile = input.layout() == Layout::TILE;
    const bool has_gamma = weight.has_value();
    const bool has_bias = bias.has_value();
    const bool has_residual = residual.has_value();
    // `return_residual_sum`: the op's second output, t = x + r (validated host-side: a residual is present and
    // the layout is TILE).  Everything it adds below is guarded by this flag, so an option-off build is the
    // program it was before the option existed.
    const bool has_t_out = residual_sum.has_value();
    TT_FATAL(
        !has_t_out || (has_residual && is_tile),
        "rms_norm_ttnn: return_residual_sum needs a residual_input_tensor and a TILE input");
    const Tensor* per_channel = has_gamma ? &*weight : (has_bias ? &*bias : nullptr);
    const bool per_channel_is_rm = per_channel != nullptr && per_channel->layout() == Layout::ROW_MAJOR;

    const int64_t W = rank >= 1 ? shape.back() : 1;
    const int64_t Wt = div_up(std::max<int64_t>(1, W), TILE_DIM);
    const int64_t partial_w = W % TILE_DIM;

    if (prod(shape, 0, rank) == 0) {
        return zero_volume_descriptor(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), compute_config);
    }

    int64_t Rt = 0;
    int64_t R_rm = 0;
    if (is_tile) {
        Rt = rank >= 2 ? prod(shape, 0, rank - 2) * div_up(shape[rank - 2], TILE_DIM) : 1;
    } else {
        R_rm = rank >= 1 ? prod(shape, 0, rank - 1) : 1;
        Rt = div_up(R_rm, TILE_DIM);
    }

    const int64_t elem_bytes = stick_elem_bytes(input);
    const int64_t gamma_elem_bytes = has_gamma ? stick_elem_bytes(weight) : 0;
    const int64_t bias_elem_bytes = has_bias ? stick_elem_bytes(bias) : 0;

    const int64_t bt = tile_bytes(input.dtype());
    const DataType interm_dtype = intermediate_dtype(input.dtype());
    const int64_t it = tile_bytes(interm_dtype);
    const int64_t gt = has_gamma ? tile_bytes(weight->dtype()) : 0;
    const int64_t bit = has_bias ? tile_bytes(bias->dtype()) : 0;
    const int64_t st = tile_bytes(DataType::BFLOAT16);
    const int64_t ft = tile_bytes(DataType::FLOAT32);

    const bool gamma_blocked = has_gamma ? per_channel_form(*weight, W).first : false;
    const bool bias_blocked = has_bias ? per_channel_form(*bias, W).first : false;

    auto trim_for = [&](int64_t tile_bytes, bool present, int64_t override_) -> int64_t {
        if (!present || per_channel_is_rm) {
            return 0;
        }
        const bool legal_2 = tile_bytes % 4 == 0 && (tile_bytes / 4) % 64 == 0;
        const int64_t derived = legal_2 ? 2 : 1;
        if (override_ == TRIM_DERIVED) {
            return derived;
        }
        if (override_ == 2 && !legal_2) {
            return 1;
        }
        return override_;
    };
    const int64_t gamma_trim = trim_for(gt, has_gamma, PER_CHANNEL_TRIM_GAMMA);
    const int64_t bias_trim = trim_for(bit, has_bias, PER_CHANNEL_TRIM_BIAS);

    const bool pc_compact_ok = PC_COMPACT_HOLD && (has_gamma || has_bias) && !per_channel_is_rm &&
                               (has_gamma ? gamma_trim == 2 : true) && (has_bias ? bias_trim == 2 : true);
    const int64_t pc_compact_bytes =
        (has_gamma ? 2 * TILE_DIM * gamma_elem_bytes : 0) + (has_bias ? 2 * TILE_DIM * bias_elem_bytes : 0);
    const bool pc_compact_rr = pc_compact_ok && ROW_RESIDENT_COMPACT_PC;

    auto make_plan = [&](bool force_rows) {
        Plan p = plan_placement(device, input, output, is_tile, Rt, Wt, W, R_rm, partial_w, force_rows);
        // The resident shards share L1 with the CB arena; each distinct buffer is charged once (the
        // Python dedupes by object identity: under `inplace` the output IS the input).
        std::vector<const tt::tt_metal::Buffer*> seen;
        int64_t reserved = 0;
        for (const Tensor* t :
             {&input, &output, has_residual ? &*residual : nullptr, has_t_out ? &*residual_sum : nullptr}) {
            if (t == nullptr) {
                continue;
            }
            const auto* buf = t->buffer();
            if (std::find(seen.begin(), seen.end(), buf) != seen.end()) {
                continue;
            }
            seen.push_back(buf);
            reserved += shard_l1_bytes(*t);
        }
        p.l1_reserved = reserved;
        return p;
    };

    Plan plan = make_plan(false);

    const int64_t kernel_partial_w = plan.band ? 0 : partial_w;
    const int64_t scaler_pages = kernel_partial_w ? 2 : 1;
    const DataType scaler_dtype = kernel_partial_w ? interm_dtype : DataType::BFLOAT16;
    const int64_t scaler_tile_bytes = tile_bytes(scaler_dtype);
    const int64_t scaler_bytes = scaler_tile_bytes * scaler_pages;
    // `return_residual_sum`: cb_residual_sum is T_SUM_RING_BLOCKS of compute's pass-B DEST blocks -- at most
    // T_SUM_RING_BLOCKS * DEST_AUTO_LIMIT tiles (16-64 kB).  It is charged to the L1 the blocking solve leaves
    // unbudgeted (the 1 - L1_SAFETY_FRACTION margin, ~200 kB), NOT to the solve itself, so the ring never moves a
    // blocking decision and y is bit-identical with the option on or off.  (An L1-SHARDED t is different: its shard
    // really occupies L1 and is charged like any resident shard, above.)  Checked below once the plan is final.
    const int64_t t_ring_bytes = has_t_out ? T_SUM_RING_BLOCKS * dest_tile_limit(compute_config) * bt : 0;
    const int64_t rm_stage_rings = 2 + (has_residual ? 1 : 0);

    auto solve_blocking = [&](const Plan& plan) -> std::optional<Solved> {
        const int64_t wt_core = plan.wt_per_core;
        const bool ragged_ok = kernel_partial_w == 0;
        int64_t avail = static_cast<int64_t>(tt::tt_metal::hal::get_max_worker_l1_unreserved_size());
        if (plan.l1_reserved) {
            avail -= plan.l1_reserved + L1_CB_ARENA_BASE_RESERVE;
        }
        const int64_t budget =
            static_cast<int64_t>(static_cast<double>(std::max<int64_t>(0, avail)) * L1_SAFETY_FRACTION);
        int64_t max_rows = 1;
        {
            bool any = false;
            for (const auto& a : plan.assignment) {
                max_rows = any ? std::max(max_rows, a.row_count) : a.row_count;
                any = true;
            }
            if (!any || max_rows == 0) {
                max_rows = 1;
            }
        }
        if (plan.combine) {
            max_rows = std::min(max_rows, TILE_DIM);
        }
        const std::optional<int64_t> dx0 = plan.native_in ? std::optional<int64_t>(0) : std::nullopt;
        const std::optional<int64_t> do0 = plan.native_out ? std::optional<int64_t>(0) : std::nullopt;
        const std::optional<int64_t> dr0 = (has_residual && plan.native_in) ? std::optional<int64_t>(0) : std::nullopt;
        const std::vector<int64_t> depth_candidates =
            is_tile ? std::vector<int64_t>{CB_DEPTH_TILE} : std::vector<int64_t>{1};
        const std::vector<int64_t>& resident_depths = depth_candidates;

        const auto combine_tree = plan.combine ? combine_tree_arity(plan.group_size, 1) : std::nullopt;
        const int64_t rm_stage_depth_for_pc = CB_RM_STAGE_DEPTH;

        auto f32_terms = [&](bool compact) -> std::pair<int64_t, int64_t> {
            const int64_t per_row = plan.combine ? (CB_ROW_STAT_DEPTH + CB_ROW_STAT_DEPTH) : CB_ROW_STAT_DEPTH;
            const int64_t fixed_pages = combine_fixed_pages(plan, compact, combine_tree);
            int64_t bank = 0;
            if (plan.combine && compact) {
                bank = st;
            }
            return {per_row * ft + bank, fixed_pages * ft};
        };

        auto per_channel_bytes = [&](int64_t width_tiles, int64_t staged_tiles, bool narrow) -> int64_t {
            int64_t total = 0;
            const int64_t staged = !per_channel_is_rm ? 0 : (narrow ? rm_stage_depth_for_pc : staged_tiles);
            if (has_gamma) {
                total += width_tiles * gt + staged * gt;
            }
            if (has_bias) {
                total += width_tiles * bit + staged * bit;
            }
            return total;
        };

        auto resident_fit = [&](int64_t depth,
                                bool compact,
                                int64_t rm_depth = CB_RM_STAGE_DEPTH,
                                int64_t block_rows = 0,
                                bool narrow_pc = false) -> int64_t {
            const int64_t mult = cb_block_bytes(
                bt,
                it,
                dx0.value_or(depth),
                do0.value_or(depth),
                has_gamma,
                has_bias,
                has_residual,
                residual_depth(dr0.value_or(depth)),
                block_rows);
            const auto [per_row_bytes, combine_fixed] = f32_terms(compact);
            const int64_t fixed = per_channel_bytes(wt_core, wt_core, narrow_pc) +
                                  (!is_tile ? rm_stage_rings * rm_depth * wt_core * bt : 0) + scaler_bytes +
                                  combine_fixed;
            const int64_t sq_wt = CB_SQ_EXACT ? x_squared_wt_of(wt_core, kernel_partial_w) : wt_core;
            const int64_t per_tilerow = wt_core * mult - (wt_core - sq_wt) * it + per_row_bytes;
            return std::max<int64_t>(0, floordiv(budget - fixed, std::max<int64_t>(1, per_tilerow)));
        };

        // D41 / D42: RESIDENT, block picked on row-blocks per core.
        std::optional<std::tuple<int64_t, int64_t, int64_t>> best;  // (blocks, depth, br)
        for (auto dit = resident_depths.rbegin(); dit != resident_depths.rend(); ++dit) {
            const int64_t depth = *dit;
            int64_t brmax = resident_fit(depth, true);
            if (brmax < 2) {
                brmax = std::min<int64_t>(1, resident_fit(depth, false));
            }
            if (brmax >= 1) {
                const int64_t top = std::min(max_rows, brmax);
                int64_t br = 0;
                if (plan.native_in || !is_tile) {
                    const int64_t blocks0 = div_up(max_rows, top);
                    const int64_t br0 = div_up(max_rows, blocks0);
                    br = (br0 * blocks0 == max_rows) ? br0 : top;
                } else {
                    br = 1;
                }
                const int64_t blocks = div_up(max_rows, br);
                if (!best.has_value() || blocks > std::get<0>(*best)) {
                    best = std::make_tuple(blocks, depth, br);
                }
            }
        }
        if (best.has_value()) {
            const auto [blocks, depth, br] = *best;
            (void)blocks;
            return Solved{br, wt_core, 1, depth, depth, true, CB_RM_STAGE_DEPTH, false, false};
        }

        if (plan.band) {
            std::vector<int64_t> band_depths = {CB_RM_STAGE_DEPTH};
            if (CB_RM_STAGE_DEPTH != 1) {
                band_depths.push_back(1);
            }
            for (bool narrow_pc : {false, true}) {
                for (int64_t rm_depth : band_depths) {
                    const int64_t fit = resident_fit(depth_candidates[0], false, rm_depth, 1, narrow_pc);
                    if (fit >= 1) {
                        return Solved{
                            1, wt_core, 1, depth_candidates[0], depth_candidates[0], true, rm_depth, narrow_pc, false};
                    }
                }
            }
            return Solved{
                1, wt_core, 1, depth_candidates[0], depth_candidates[0], true, band_depths.back(), true, false};
        }
        if (plan.scheme != Scheme::Rows) {
            return std::nullopt;
        }

        auto row_resident_chunk =
            [&](int64_t depth_x, int64_t depth_out, bool compact) -> std::optional<std::pair<int64_t, int64_t>> {
            auto fixed_of = [&](int64_t hold_wt) -> int64_t {
                int64_t held = (has_residual ? 1 : depth_x) * hold_wt * (has_residual ? it : bt);
                if (compact) {
                    held += hold_wt * pc_compact_bytes;
                } else {
                    held += per_channel_bytes(hold_wt, 0, false);
                }
                const auto [per_row_bytes, combine_fixed] = f32_terms(false);
                return held + scaler_bytes + per_row_bytes + combine_fixed;
            };
            const int64_t per_chunk_tile =
                it * (1 + norm_cb_depth(has_gamma, has_bias, 1)) + bt * depth_out +
                (has_residual ? bt * (depth_x + residual_depth(depth_x)) : 0) + per_channel_bytes(0, 1, false) +
                (!is_tile ? rm_stage_rings * CB_RM_STAGE_DEPTH * bt : 0) +
                (compact ? PC_RING_CHUNKS * ((has_gamma ? gt : 0) + (has_bias ? bit : 0)) : 0);
            const int64_t room = floordiv(budget - fixed_of(wt_core), per_chunk_tile);
            if (room < 1) {
                return std::nullopt;
            }
            int64_t cap = wt_core > 1 ? std::min(room, wt_core - 1) : 1;
            if (ROW_RESIDENT_CAP_OVERRIDE) {
                cap = std::min(cap, ROW_RESIDENT_CAP_OVERRIDE);
            }
            int64_t wtc = 0, n = 0;
            while (true) {
                std::tie(wtc, n) = width_chunk(wt_core, cap, ragged_ok);
                if (wtc < 1 || wtc >= wt_core || n <= 1) {
                    return std::nullopt;
                }
                const int64_t pad = n * wtc - wt_core;
                if (pad == 0) {
                    break;
                }
                const int64_t room_pad = floordiv(budget - fixed_of(wt_core + pad), per_chunk_tile);
                if (wtc <= room_pad) {
                    break;
                }
                cap = std::min(cap - 1, room_pad);
                if (cap < 1) {
                    return std::nullopt;
                }
            }
            if (wtc < (compact ? ROW_RESIDENT_COMPACT_MIN_CHUNK_WT : ROW_RESIDENT_MIN_CHUNK_WT)) {
                return std::nullopt;
            }
            return std::make_pair(wtc, n);
        };

        const int64_t stream_depth = depth_candidates[0];
        const auto grid = device->compute_with_storage_grid_size();
        int64_t active = 0;
        for (const auto& a : plan.assignment) {
            if (a.row_count) {
                ++active;
            }
        }
        const bool compact_ok =
            pc_compact_rr && static_cast<double>(active) >=
                                 ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION * static_cast<double>(grid.x * grid.y);
        std::vector<bool> hold_forms;
        if (compact_ok) {
            hold_forms = COMPACT_FIRST ? std::vector<bool>{true, false} : std::vector<bool>{false, true};
        } else {
            hold_forms = {false};
        }
        // tuple(dict.fromkeys(depth_candidates + (1,)))
        std::vector<int64_t> rr_depths;
        for (int64_t d : depth_candidates) {
            if (std::find(rr_depths.begin(), rr_depths.end(), d) == rr_depths.end()) {
                rr_depths.push_back(d);
            }
        }
        if (std::find(rr_depths.begin(), rr_depths.end(), 1) == rr_depths.end()) {
            rr_depths.push_back(1);
        }
        for (bool compact : hold_forms) {
            for (int64_t depth : rr_depths) {
                if (depth < stream_depth && max_rows < ROW_RESIDENT_MIN_ROWS_PER_CORE) {
                    continue;
                }
                const auto fit = row_resident_chunk(depth, depth, compact);
                if (fit.has_value()) {
                    return Solved{1, fit->first, fit->second, depth, depth, true, CB_RM_STAGE_DEPTH, false, compact};
                }
            }
        }

        // STREAM.
        const int64_t depth = depth_candidates[0];
        const int64_t mult =
            cb_block_bytes(bt, it, depth, depth, has_gamma, has_bias, has_residual, residual_depth(depth), 1);
        const int64_t per_chunk_tile_bytes =
            mult + per_channel_bytes(1, 1, false) + (!is_tile ? rm_stage_rings * CB_RM_STAGE_DEPTH * bt : 0);
        const auto [stream_per_row, stream_combine_fixed] = f32_terms(false);
        const int64_t fixed_stream = scaler_bytes + stream_per_row + stream_combine_fixed;
        const int64_t wt_chunk_l1_max = std::max<int64_t>(1, floordiv(budget - fixed_stream, per_chunk_tile_bytes));
        const auto [wtc, n] = width_chunk(wt_core, wt_chunk_l1_max, ragged_ok);
        return Solved{1, wtc, n, depth, depth, false, CB_RM_STAGE_DEPTH, false, false};
    };

    std::optional<Solved> solved = solve_blocking(plan);
    if (!solved.has_value()) {
        plan = make_plan(true);
        solved = solve_blocking(plan);
        TT_FATAL(solved.has_value(), "rms_norm_ttnn: no admissible blocking even on SCHEME_ROWS");
    }
    const int64_t block_rows = solved->block_rows;
    const int64_t wt_chunk = solved->wt_chunk;
    const int64_t num_w_chunks = solved->num_w_chunks;
    const int64_t cb_x_depth = solved->cb_x_depth;
    const int64_t cb_out_depth = solved->cb_out_depth;
    const bool x_resident = solved->x_resident;
    const int64_t rm_stage_depth = solved->rm_stage_depth;
    const bool narrow_pc_stage = solved->narrow_pc_stage;
    const bool pc_compact = solved->pc_compact;

    const auto& all_cores = plan.all_cores;
    const auto& assignment = plan.assignment;
    const int64_t wt_per_core = plan.wt_per_core;
    const bool combine = plan.combine;
    const bool native_residual = has_residual && plan.native_in && same_shard_spec(input, *residual);
    TT_FATAL(
        !has_residual || !plan.native_in || native_residual,
        "rms_norm_ttnn: a native-in scheme with a residual whose shard spec differs from the input's would silently "
        "skip the residual's read");

    TT_FATAL(x_resident || num_w_chunks > 1, "rms_norm_ttnn: a one-chunk width is resident by definition");
    TT_FATAL(!(combine && num_w_chunks > 1), "rms_norm_ttnn: a width-split core takes its slice in one chunk");
    const bool row_resident = x_resident && num_w_chunks > 1;
    TT_FATAL(!row_resident || block_rows == 1, "rms_norm_ttnn: ROW_RESIDENT holds ONE tile-row of x");
    const bool compact_combine = combine && block_rows > 1;
    TT_FATAL(block_rows <= TILE_DIM || !combine, "rms_norm_ttnn: a compact combine block is at most 32 tile-rows");
    const bool fin_spread = combine_fin_spread(combine);
    int64_t max_row_count = 0;
    for (const auto& a : assignment) {
        max_row_count = std::max(max_row_count, a.row_count);
    }
    const bool single_round = combine && max_row_count <= block_rows;
    const bool mcast_pre_handshake = combine_mcast_pre_handshake(combine, single_round);
    const auto combine_tree = combine ? combine_tree_arity(plan.group_size, 1) : std::nullopt;
    const int64_t tree_f0 = combine_tree ? combine_tree->first : 0;
    const int64_t tree_f1 = combine_tree ? combine_tree->second : 0;
    const int64_t wt_pad = wt_chunk * num_w_chunks - wt_per_core;
    TT_FATAL(wt_pad >= 0, "rms_norm_ttnn: the width chunking cannot be NARROWER than the core's width");
    TT_FATAL(
        wt_pad == 0 || (kernel_partial_w == 0 && block_rows == 1 && !combine),
        "rms_norm_ttnn: a ragged width chunk requires a tile-aligned width, one tile-row per block and no cross-core "
        "width combine");
    const int64_t x_hold_wt = x_resident ? wt_chunk * num_w_chunks : wt_chunk;
    const int64_t pc_hold_wt = wt_chunk * num_w_chunks;
    TT_FATAL(!pc_compact || (x_resident && num_w_chunks > 1), "D39: the compact hold is ROW_RESIDENT's");
    const bool pc_chunked = (!x_resident) || pc_compact;
    const int64_t pc_tile_pages = pc_chunked ? (PC_RING_CHUNKS * wt_chunk) : x_hold_wt;

    if (std::getenv("RMS_TRACE_BLOCKING") != nullptr && std::strlen(std::getenv("RMS_TRACE_BLOCKING")) > 0) {
        const char* scheme_name =
            plan.scheme == Scheme::Rows ? "rows" : (plan.scheme == Scheme::ShardH ? "shard_h" : "shard_w");
        fmt::print(
            "RMS_BLOCKING scheme={} cores={} wt_per_core={} BLOCK_ROWS={} WT_CHUNK={} NUM_W_CHUNKS={} X_RESIDENT={} "
            "depth=({},{}) partial_w={} rows_max={} PC_COMPACT={} PC_CHUNKED={} PC_RING={}\n",
            scheme_name,
            assignment.size(),
            wt_per_core,
            block_rows,
            wt_chunk,
            num_w_chunks,
            int(x_resident),
            cb_x_depth,
            cb_out_depth,
            kernel_partial_w,
            max_row_count,
            int(pc_compact),
            int(pc_chunked),
            pc_tile_pages);
    }

    const int64_t x_squared_wt = x_squared_wt_of(wt_chunk, kernel_partial_w);

    const bool reduce_acc_via_add = REDUCE_BULK == 1 && wt_per_core >= REDUCE_ACC_VIA_ADD_MIN_WT &&
                                    wt_chunk >= REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT &&
                                    !(num_w_chunks == 1 && x_squared_wt < REDUCE_ACC_VIA_ADD_MIN_CALL_WT);
    const int64_t scaler_tiles = reduce_acc_via_add ? 1 : scaler_pages;
    TT_FATAL(scaler_tiles <= scaler_pages, "rms_norm_ttnn: cb_scaler is sized below the tiles the reader pushes");

    uint32_t eps_bits = f32_bits(epsilon);
    if (compact_combine && epsilon == 0.0) {
        eps_bits = f32_bits(1.1754944e-38);
    }

    const int64_t txn_rows = dm_txn_rows(block_rows);

    int64_t pass_b_blk_ct = subblock_w;
    if (pass_b_blk_ct && wt_chunk % pass_b_blk_ct) {
        pass_b_blk_ct = largest_divisor_at_most(wt_chunk, pass_b_blk_ct);
    }
    TT_FATAL(
        pass_b_blk_ct == 0 || wt_chunk % pass_b_blk_ct == 0,
        "rms_norm_ttnn: program_config.subblock_w={} does not divide the resolved WT_CHUNK={}",
        pass_b_blk_ct,
        wt_chunk);

    // ---- circular buffers ----------------------------------------------------------------------
    ProgramDescriptor desc;
    auto& cbs = desc.cbs;
    if (!is_tile) {
        cbs.push_back(make_cb(CB_INPUT_STICKS, bt, rm_stage_depth * wt_chunk, input.dtype(), all_cores));
        cbs.push_back(make_cb(CB_OUTPUT_STICKS, bt, rm_stage_depth * wt_chunk, output.dtype(), all_cores));
        if (has_residual) {
            cbs.push_back(make_cb(CB_RESIDUAL_STICKS, bt, rm_stage_depth * wt_chunk, input.dtype(), all_cores));
        }
    }
    int64_t in_shard_pages = 0;
    int64_t out_shard_pages = 0;
    if (plan.native_in) {
        const auto [sh_t, sw_t] = shard_tile_extent(input);
        in_shard_pages = sh_t * sw_t;
        TT_FATAL(sw_t == wt_chunk, "rms_norm_ttnn: native x CB row stride {} != WT_CHUNK {}", sw_t, wt_chunk);
        cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input));
    } else {
        const int64_t x_span = has_residual ? wt_chunk : x_hold_wt;
        cbs.push_back(make_cb(CB_INPUT_TILES, bt, cb_x_depth * block_rows * x_span, input.dtype(), all_cores));
    }
    if (has_residual) {
        if (native_residual) {
            cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(CB_RESIDUAL_TILES, *residual));
        } else {
            const int64_t r_depth = residual_depth(cb_x_depth);
            cbs.push_back(make_cb(CB_RESIDUAL_TILES, bt, r_depth * block_rows * wt_chunk, input.dtype(), all_cores));
        }
        cbs.push_back(make_cb(CB_X_SUM, it, block_rows * x_hold_wt, interm_dtype, all_cores));
    }
    cbs.push_back(make_cb(CB_X_SQUARED, it, block_rows * x_squared_wt, interm_dtype, all_cores));
    cbs.push_back(make_cb(CB_SCALER, scaler_tile_bytes, scaler_pages, scaler_dtype, all_cores));
    if (!combine) {
        cbs.push_back(make_cb(CB_ROW_STAT, ft, CB_ROW_STAT_DEPTH * block_rows, DataType::FLOAT32, all_cores));
    } else if (fin_spread) {
        cbs.push_back(make_cb(CB_ROW_STAT, ft, CB_COMBINE_FLAT_DEPTH, DataType::FLOAT32, all_cores));
    }
    const int64_t pc_stage_pages = narrow_pc_stage ? rm_stage_depth : wt_chunk;
    if (has_gamma) {
        if (per_channel_is_rm) {
            cbs.push_back(make_cb(CB_GAMMA_STICKS, gt, pc_stage_pages, weight->dtype(), all_cores));
        }
        cbs.push_back(make_cb(CB_GAMMA_TILES, gt, pc_tile_pages, weight->dtype(), all_cores));
        if (pc_compact) {
            cbs.push_back(
                make_cb(CB_GAMMA_COMPACT, 2 * TILE_DIM * gamma_elem_bytes, pc_hold_wt, weight->dtype(), all_cores));
        }
    }
    if (has_bias) {
        if (per_channel_is_rm) {
            cbs.push_back(make_cb(CB_BIAS_STICKS, bit, pc_stage_pages, bias->dtype(), all_cores));
        }
        cbs.push_back(make_cb(CB_BIAS_TILES, bit, pc_tile_pages, bias->dtype(), all_cores));
        if (pc_compact) {
            cbs.push_back(
                make_cb(CB_BIAS_COMPACT, 2 * TILE_DIM * bias_elem_bytes, pc_hold_wt, bias->dtype(), all_cores));
        }
    }
    const int64_t norm_depth = norm_cb_depth(has_gamma, has_bias, block_rows);
    if (norm_depth) {
        cbs.push_back(make_cb(CB_NORMALIZED, it, norm_depth * block_rows * wt_chunk, interm_dtype, all_cores));
    }
    if (plan.native_out) {
        const auto [osh_t, osw_t] = shard_tile_extent(output);
        out_shard_pages = osh_t * osw_t;
        cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output));
    } else {
        cbs.push_back(make_cb(CB_OUTPUT_TILES, bt, cb_out_depth * block_rows * wt_chunk, output.dtype(), all_cores));
    }
    if (combine) {
        cbs.push_back(make_cb(CB_SUM_HANDOFF, ft, CB_ROW_STAT_DEPTH * block_rows, DataType::FLOAT32, all_cores));
        if (!combine_tree.has_value()) {
            cbs.push_back(
                make_cb(CB_PARTIALS_GATHERED, ft, plan.group_size + plan.group_size % 2, DataType::FLOAT32, all_cores));
        } else {
            cbs.push_back(make_cb(CB_PARTIALS_GATHERED, ft, tree_f0 + tree_f0 % 2, DataType::FLOAT32, all_cores));
            cbs.push_back(make_cb(CB_GATHER_L1, ft, tree_f1 + tree_f1 % 2, DataType::FLOAT32, all_cores));
            cbs.push_back(make_cb(CB_NODE_OUT, ft, CB_COMBINE_FLAT_DEPTH, DataType::FLOAT32, all_cores));
        }
        cbs.push_back(make_cb(CB_STAT_HANDOFF, ft, CB_COMBINE_FLAT_DEPTH, DataType::FLOAT32, all_cores));
        cbs.push_back(make_cb(CB_ROW_FINAL, ft, CB_ROW_STAT_DEPTH * block_rows, DataType::FLOAT32, all_cores));
        if (compact_combine) {
            cbs.push_back(make_cb(CB_COMPACT_HANDOFF, ft, CB_COMBINE_FLAT_DEPTH, DataType::FLOAT32, all_cores));
            cbs.push_back(make_cb(CB_MCAST_IN, ft, CB_COMBINE_FLAT_DEPTH, DataType::FLOAT32, all_cores));
            cbs.push_back(make_cb(CB_BANK, st, block_rows, DataType::BFLOAT16, all_cores));
        }
    }
    // `return_residual_sum`: compute pushes t in PASS_B_BLK-tile DEST blocks; the ring is a whole number of them, so
    // no push straddles its end, and the writer drains it one block at a time.
    const int64_t t_blk =
        has_t_out ? pass_b_block(wt_chunk, block_rows, pass_b_blk_ct, dest_tile_limit(compute_config)) : 0;
    if (has_t_out) {
        TT_FATAL(t_blk >= 1 && wt_chunk % t_blk == 0, "rms_norm_ttnn: the t block must divide WT_CHUNK");
        TT_FATAL(
            T_SUM_RING_BLOCKS * t_blk * bt <= t_ring_bytes, "rms_norm_ttnn: cb_residual_sum exceeds its L1 charge");
        int64_t avail = static_cast<int64_t>(tt::tt_metal::hal::get_max_worker_l1_unreserved_size());
        if (plan.l1_reserved) {
            avail -= plan.l1_reserved + L1_CB_ARENA_BASE_RESERVE;
        }
        avail = std::max<int64_t>(0, avail);
        const int64_t margin = avail - static_cast<int64_t>(static_cast<double>(avail) * L1_SAFETY_FRACTION);
        TT_FATAL(
            t_ring_bytes <= margin,
            "rms_norm_ttnn: return_residual_sum's {} B ring does not fit the {} B of L1 the blocking leaves unbudgeted",
            t_ring_bytes,
            margin);
        cbs.push_back(make_cb(CB_RESIDUAL_SUM, bt, T_SUM_RING_BLOCKS * t_blk, residual_sum->dtype(), all_cores));
    }

    // ---- ROW_MAJOR staging-ring zero --------------------------------------------------------------
    const int64_t stage_pad_bytes = wt_chunk * TILE_DIM * elem_bytes;
    bool stage_zero = false;
    if (!is_tile) {
        if (plan.band) {
            for (const auto& a : assignment) {
                if (a.row_count &&
                    ((a.w_off_elems % TILE_DIM) != 0 || a.w_real_elems * elem_bytes != stage_pad_bytes)) {
                    stage_zero = true;
                    break;
                }
            }
        } else {
            stage_zero = partial_w != 0;
        }
    }

    // ---- D40: the per-channel broadcast ("row" lines) ------------------------------------------
    std::optional<kh::Mcast1D> pc_mcast;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> pc_role;
    if (PC_MCAST_ENABLED && (has_gamma || has_bias) && !per_channel_is_rm && !pc_compact) {
        const auto cores = cores_in(all_cores);
        std::set<uint32_t> xs_set, ys_set;
        for (const auto& c : cores) {
            xs_set.insert(c.x);
            ys_set.insert(c.y);
        }
        const std::vector<uint32_t> xs(xs_set.begin(), xs_set.end());
        const std::vector<uint32_t> ys(ys_set.begin(), ys_set.end());
        auto contiguous = [](const std::vector<uint32_t>& v) {
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i] != v[0] + i) {
                    return false;
                }
            }
            return true;
        };
        const bool rect = cores.size() == xs.size() * ys.size() && contiguous(xs) && contiguous(ys);
        std::map<std::pair<uint32_t, uint32_t>, const Work*> act;
        for (const auto& a : assignment) {
            if (a.row_count) {
                act[{a.core.x, a.core.y}] = &a;
            }
        }
        // x_resident || mode in ("col", "row"): always true for the shipped "row" mode.
        bool ok = rect && act.size() == cores.size();
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> live;
        if (ok) {
            for (uint32_t cy : ys) {
                std::vector<std::pair<uint32_t, uint32_t>> g;
                for (uint32_t cx : xs) {
                    g.emplace_back(cx, cy);
                }
                if (g.size() < PC_MCAST_MIN_GROUP) {
                    continue;
                }
                std::set<std::tuple<int64_t, int64_t, int64_t>> keys;
                for (const auto& k : g) {
                    const Work* a = act.at(k);
                    const int64_t blocks = x_resident ? 0 : div_up(a->row_count, block_rows);
                    keys.insert({a->w_start, a->w_real, blocks});
                }
                if (keys.size() == 1) {
                    live.push_back(std::move(g));
                }
            }
            ok = !live.empty();
        }
        if (ok) {
            uint32_t base_sem = 0;
            if (combine) {
                base_sem = plan.gather_sem_id + (combine_tree.has_value() ? 2 : 1);
            }
            const bool handshake = PC_MCAST_HANDSHAKE || !x_resident;
            kh::McastConfig cfg;
            cfg.noc = reader_noc(plan);
            cfg.handshake = handshake;
            cfg.data_ready = handshake ? kh::DataReadyMode::Flag : kh::DataReadyMode::Counter;
            cfg.base_sem_id = base_sem;
            cfg.rotating_sender = false;
            if (PC_MCAST_DIAGONAL) {
                pc_mcast.emplace(
                    device,
                    all_cores,
                    kh::Mcast1DShape::PerRow,
                    PC_MCAST_SENDER_INDEX,
                    cfg,
                    kh::Mcast1DSenderPlacement::Diagonal);
            } else {
                pc_mcast.emplace(device, all_cores, kh::Mcast1DShape::PerRow, PC_MCAST_SENDER_INDEX, cfg);
            }
            std::set<std::pair<uint32_t, uint32_t>> on;
            for (const auto& g : live) {
                on.insert(g.begin(), g.end());
            }
            for (const auto& c : cores) {
                const std::pair<uint32_t, uint32_t> k{c.x, c.y};
                pc_role[k] = on.contains(k) ? (pc_mcast->is_sender(c) ? 1u : 0u) : PC_OPT_OUT;
            }
        }
    }

    // ---- reader -----------------------------------------------------------------------------------
    std::vector<uint32_t> reader_ct = {
        is_tile ? 1u : 0u,
        static_cast<uint32_t>(Wt),
        static_cast<uint32_t>(wt_chunk),
        static_cast<uint32_t>(num_w_chunks),
        pack_txn_rows(block_rows, txn_rows),
        static_cast<uint32_t>(kernel_partial_w),
        has_gamma ? 1u : 0u,
        per_channel_is_rm ? 1u : 0u,
        static_cast<uint32_t>(elem_bytes),
        static_cast<uint32_t>(gamma_elem_bytes),
        static_cast<uint32_t>(R_rm),
        static_cast<uint32_t>(W),
        reduce_acc_via_add ? 1u : 0u,
        plan.native_in ? 1u : 0u,
        static_cast<uint32_t>(in_shard_pages),
        plan.band ? 1u : 0u,
        static_cast<uint32_t>(plan.shard_row_bytes),
        stage_zero ? 1u : 0u,
        x_resident ? 1u : 0u,
        static_cast<uint32_t>(gamma_trim),
        static_cast<uint32_t>(compact_combine ? block_rows : 0),
        has_bias ? 1u : 0u,
        static_cast<uint32_t>(bias_elem_bytes),
        static_cast<uint32_t>(bias_trim),
        has_residual ? 1u : 0u,
        native_residual ? 1u : 0u,
        gamma_blocked ? 1u : 0u,
        bias_blocked ? 1u : 0u,
        narrow_pc_stage ? 1u : 0u,
        static_cast<uint32_t>(wt_pad),
        static_cast<uint32_t>(!pc_compact ? 0 : (PC_COMPACT_LAZY ? 2 : 1)),
        static_cast<uint32_t>(pc_hold_wt),
        pc_chunked ? 1u : 0u,
    };
    TT_FATAL(reader_ct.size() == READER_CT_SCALARS, "rms_norm_ttnn_reader.cpp expects TensorAccessorArgs<33>()");
    const std::optional<Tensor> input_opt = input;
    for (const auto* t : {&input_opt, &weight, &bias, &residual}) {
        const auto a = accessor_args(*t);
        reader_ct.insert(reader_ct.end(), a.begin(), a.end());
    }
    if (pc_mcast.has_value()) {
        const uint32_t inj_g = static_cast<uint32_t>(gamma_trim);
        const uint32_t inj_b = static_cast<uint32_t>(bias_trim);
        const uint32_t pc_late = (PC_MCAST_LATE && num_w_chunks == 1) ? 1u : 0u;
        reader_ct.push_back(PC_MCAST_SPLIT_N | (inj_g << 8) | (inj_b << 16) | (pc_late << 24));
        const auto m = pc_mcast->compile_time_args();
        reader_ct.insert(reader_ct.end(), m.begin(), m.end());
    }

    // ---- writer -----------------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct = {
        is_tile ? 1u : 0u,
        static_cast<uint32_t>(Wt),
        static_cast<uint32_t>(wt_chunk | (wt_pad << 16)),
        static_cast<uint32_t>(num_w_chunks),
        pack_txn_rows(block_rows, txn_rows),
        static_cast<uint32_t>(elem_bytes),
        static_cast<uint32_t>(R_rm),
        static_cast<uint32_t>(W),
        plan.native_out ? 1u : 0u,
        combine ? 1u : 0u,
        combine ? plan.gather_sem_id : 0u,
        static_cast<uint32_t>(plan.group_size),
        static_cast<uint32_t>(out_shard_pages),
        plan.band ? 1u : 0u,
        static_cast<uint32_t>(plan.out_shard_row_bytes),
        combine_gather_faces_ct(combine, compact_combine),
        static_cast<uint32_t>(tree_f0),
        static_cast<uint32_t>(tree_f1),
    };
    TT_FATAL(writer_ct.size() == 18, "rms_norm_ttnn_writer.cpp expects McastArgs<18, 12>()");
    if (combine) {
        const auto m = plan.mcast->compile_time_args(mcast_pre_handshake);
        writer_ct.insert(writer_ct.end(), m.begin(), m.end());
    } else {
        writer_ct.insert(writer_ct.end(), 6, 0u);
    }
    {
        const auto a = tt::tt_metal::TensorAccessorArgs(*output.buffer()).get_compile_time_args();
        writer_ct.insert(writer_ct.end(), a.begin(), a.end());
    }
    if (has_t_out) {
        writer_ct.push_back(static_cast<uint32_t>(t_blk));
        const auto a = tt::tt_metal::TensorAccessorArgs(*residual_sum->buffer()).get_compile_time_args();
        writer_ct.insert(writer_ct.end(), a.begin(), a.end());
    }

    // ---- compute ----------------------------------------------------------------------------------
    std::vector<uint32_t> compute_ct = {
        is_tile ? 1u : 0u,
        static_cast<uint32_t>(wt_chunk),
        static_cast<uint32_t>(num_w_chunks),
        static_cast<uint32_t>(block_rows),
        static_cast<uint32_t>(kernel_partial_w),
        has_gamma ? 1u : 0u,
        per_channel_is_rm ? 1u : 0u,
        f32_bits(1.0 / static_cast<double>(W)),
        eps_bits,
        REDUCE_BULK,
        reduce_acc_via_add ? 1u : 0u,
        static_cast<uint32_t>(scaler_tiles),
        combine ? 1u : 0u,
        static_cast<uint32_t>(plan.group_size),
        static_cast<uint32_t>(x_squared_wt),
        x_resident ? 1u : 0u,
        plan.native_in ? 1u : 0u,
        static_cast<uint32_t>(tree_f0),
        static_cast<uint32_t>(tree_f1),
        has_bias ? 1u : 0u,
        has_residual ? 1u : 0u,
        static_cast<uint32_t>(pass_b_blk_ct),
        narrow_pc_stage ? 1u : 0u,
        fin_spread ? 1u : 0u,
        PASS_A_SQ_BLOCK,
        RES_FUSE,
        pc_chunked ? 1u : 0u,
    };
    TT_FATAL(compute_ct.size() == COMPUTE_CT_SCALARS, "compute CT-arg count drifted");
    TT_FATAL(
        x_squared_wt >= 1 && wt_chunk % x_squared_wt == 0,
        "rms_norm_ttnn: x_squared_wt must divide WT_CHUNK (1 == the flat DEST fold, D43)");

    // ---- the slot tree's level-0 parent coords ---------------------------------------------------
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>> tree_parent;
    if (combine_tree.has_value()) {
        auto group_key = [&](const CoreCoord& c) -> uint32_t {
            if (plan.group_axis == GroupAxis::Y) {
                return c.y;
            }
            if (plan.group_axis == GroupAxis::X) {
                return c.x;
            }
            return 0;
        };
        std::map<std::pair<uint32_t, int64_t>, CoreCoord> slot_core;
        for (const auto& w : assignment) {
            if (w.row_count) {
                slot_core[{group_key(w.core), w.slot}] = w.core;
            }
        }
        for (const auto& w : assignment) {
            if (!w.row_count) {
                continue;
            }
            const auto parent = slot_core.at({group_key(w.core), (w.slot / tree_f0) * tree_f0});
            const auto v = device->worker_core_from_logical_core(CoreCoord(parent.x, parent.y));
            tree_parent[{w.core.x, w.core.y}] = {static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)};
        }
    }

    // ---- runtime args ----------------------------------------------------------------------------
    const uint32_t x_addr = input.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    const uint32_t g_addr = has_gamma ? weight->buffer()->address() : 0;
    const uint32_t b_addr = has_bias ? bias->buffer()->address() : 0;
    const uint32_t r_addr = has_residual ? residual->buffer()->address() : 0;
    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    reader_rt.reserve(assignment.size());
    writer_rt.reserve(assignment.size());
    compute_rt.reserve(assignment.size());
    for (const auto& w : assignment) {
        const auto& core = w.core;
        const uint32_t owns_last_w = (w.w_start + w.w_real >= Wt) ? 1u : 0u;
        std::vector<uint32_t> r = {
            x_addr,
            g_addr,
            static_cast<uint32_t>(w.row_start),
            static_cast<uint32_t>(w.row_count),
            static_cast<uint32_t>(w.w_start),
            static_cast<uint32_t>(w.w_real),
            static_cast<uint32_t>(w.stick_base),
            static_cast<uint32_t>(w.stick_count),
            static_cast<uint32_t>(w.w_off_elems),
            static_cast<uint32_t>(w.w_real_elems),
            b_addr,
            r_addr};
        if (pc_mcast.has_value()) {
            auto it_role = pc_role.find({core.x, core.y});
            r.push_back(it_role == pc_role.end() ? 0u : it_role->second);
            const auto m = pc_mcast->runtime_args(core);
            r.insert(r.end(), m.begin(), m.end());
        }
        reader_rt.emplace_back(core, std::move(r));

        std::vector<uint32_t> wr = {
            out_addr,
            static_cast<uint32_t>(w.row_start),
            static_cast<uint32_t>(w.row_count),
            static_cast<uint32_t>(w.w_start),
            w.is_root ? 1u : 0u,
            static_cast<uint32_t>(w.slot),
            static_cast<uint32_t>(w.stick_base),
            static_cast<uint32_t>(w.stick_count),
            static_cast<uint32_t>(w.w_off_elems),
            static_cast<uint32_t>(w.w_real_elems)};
        auto tp = tree_parent.find({core.x, core.y});
        if (tp != tree_parent.end()) {
            wr.push_back(tp->second.first);
            wr.push_back(tp->second.second);
        } else {
            wr.push_back(0);
            wr.push_back(0);
        }
        if (combine) {
            const auto m = plan.mcast->runtime_args(core);
            wr.insert(wr.end(), m.begin(), m.end());
        }
        writer_rt.emplace_back(core, std::move(wr));

        compute_rt.emplace_back(
            core,
            std::vector<uint32_t>{
                static_cast<uint32_t>(w.row_count), owns_last_w, w.is_root ? 1u : 0u, static_cast<uint32_t>(w.slot)});
    }

    auto reader_defines = kernel_defines();
    if (pc_mcast.has_value()) {
        reader_defines.emplace_back("RMS_PC_MCAST", "1");
    }
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_reader.cpp",
        all_cores,
        std::move(reader_defines),
        std::move(reader_ct),
        std::move(reader_rt),
        reader_dm_config(plan)));
    auto writer_defines = kernel_defines();
    auto compute_defines = kernel_defines();
    if (has_t_out) {
        writer_defines.emplace_back("RMS_RESIDUAL_OUT", "1");
        // Its value is the host's PASS_B_BLK, which the kernel static_asserts against its own.
        compute_defines.emplace_back("RMS_RESIDUAL_OUT", std::to_string(t_blk));
    }
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_writer.cpp",
        all_cores,
        std::move(writer_defines),
        std::move(writer_ct),
        std::move(writer_rt),
        writer_dm_config(plan)));
    if (has_t_out) {
        // t's address: the writer's one common runtime arg, patched on a cache hit by override_runtime_arguments
        // (which this factory defines, so the adapter uses no buffer bindings), like every other address here.
        const uint32_t t_addr = residual_sum->buffer()->address();
        desc.kernels.back().common_runtime_args = {t_addr};  // smuggled-rta-ok: patched in override_runtime_arguments
    }
    desc.kernels.push_back(make_kernel(
        "rms_norm_ttnn_compute.cpp",
        all_cores,
        std::move(compute_defines),
        std::move(compute_ct),
        std::move(compute_rt),
        compute_config));

    if (combine) {
        for (auto& s : plan.mcast->owned_semaphores()) {
            desc.semaphores.push_back(std::move(s));
        }
        const int levels = combine_tree.has_value() ? 2 : 1;
        for (int lvl = 0; lvl < levels; ++lvl) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = plan.gather_sem_id + lvl,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = all_cores,
                .initial_value = 0});
        }
    }
    if (pc_mcast.has_value()) {
        for (auto& s : pc_mcast->owned_semaphores()) {
            desc.semaphores.push_back(std::move(s));
        }
    }
    return desc;
}

tt::tt_metal::ProgramDescriptor RmsNormProgramFactory::create_descriptor(
    const RmsNormParams& operation_attributes, const RmsNormInputs& tensor_args, Tensor& output) {
    return create_program_descriptor(
        tensor_args.input,
        output,
        tensor_args.weight,
        tensor_args.bias,
        tensor_args.residual,
        operation_attributes.epsilon,
        operation_attributes.compute_config,
        operation_attributes.subblock_w,
        tensor_args.residual_sum);
}

void RmsNormProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const RmsNormParams& /*operation_attributes*/,
    const RmsNormInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Everything the builder reads besides the buffer ADDRESSES is in the program hash (the tensors'
    // specs, the attributes, the device), so a cache hit can only move addresses.  The zero-volume
    // program carries no address at all.
    if (tensor_args.input.logical_shape().volume() == 0) {
        return;
    }
    const auto& input = tensor_args.input;
    auto* in_buf = input.buffer();
    auto* out_buf = output.buffer();
    auto* r_buf = tensor_args.residual.has_value() ? tensor_args.residual->buffer() : nullptr;
    const uint32_t x_addr = in_buf->address();
    const uint32_t g_addr = tensor_args.weight.has_value() ? tensor_args.weight->buffer()->address() : 0;
    const uint32_t b_addr = tensor_args.bias.has_value() ? tensor_args.bias->buffer()->address() : 0;
    const uint32_t r_addr = r_buf != nullptr ? r_buf->address() : 0;
    const uint32_t out_addr = out_buf->address();

    auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, READER_KERNEL);
    for (auto& col : reader_args) {
        for (auto& args : col) {
            if (args.size() <= READER_RT_RESIDUAL) {
                continue;
            }
            args[READER_RT_X] = x_addr;
            args[READER_RT_GAMMA] = g_addr;
            args[READER_RT_BIAS] = b_addr;
            args[READER_RT_RESIDUAL] = r_addr;
        }
    }
    auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, WRITER_KERNEL);
    for (auto& col : writer_args) {
        for (auto& args : col) {
            if (args.size() <= WRITER_RT_OUT) {
                continue;
            }
            args[WRITER_RT_OUT] = out_addr;
        }
    }
    if (tensor_args.residual_sum.has_value()) {
        tt::tt_metal::GetCommonRuntimeArgs(program, WRITER_KERNEL)[0] = tensor_args.residual_sum->buffer()->address();
    }
    // The zero-copy CBs alias a resident shard: x (slot 1), the residual (slot 20), the output (slot 8).
    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& idx = cb->buffer_indices();
        if (idx.contains(CB_INPUT_TILES)) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *in_buf);
        } else if (idx.contains(CB_RESIDUAL_TILES) && r_buf != nullptr) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *r_buf);
        } else if (idx.contains(CB_OUTPUT_TILES)) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *out_buf);
        }
    }
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn
