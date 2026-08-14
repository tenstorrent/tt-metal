// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu_geometry.hpp"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry {
namespace {

// Tuning overrides for the blocking knobs. Each defaults to the shipped constant, so an unset
// environment reproduces the shipped geometry byte for byte.
//
// They do NOT participate in the program-cache key, so a shape whose program is already cached
// keeps the geometry it was compiled with. Reading them ONCE per process (the statics below) makes
// "set them before the first dispatch" an enforceable rule rather than a convention.
//
// Out-of-domain values are REFUSED, not clamped: a silently clamped knob makes a tuning sweep
// report a number for a configuration it never ran. The domains are hardware limits, not taste —
// see each knob's declaration in the header.
uint32_t env_u32(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 0);
    TT_FATAL(
        end != raw && *end == '\0' && parsed >= lo && parsed <= hi,
        "moe_fused_swiglu: {}='{}' must be an integer in [{}, {}]",
        name,
        raw,
        lo,
        hi);
    return static_cast<uint32_t>(parsed);
}

bool env_bool(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    const std::string_view value{raw};
    if (value == "0" || value == "false" || value == "False") {
        return false;
    }
    TT_FATAL(value == "1" || value == "true" || value == "True", "moe_fused_swiglu: {}='{}' must be 0/1", name, raw);
    return true;
}

// Read once per process, so the geometry cannot drift between one dispatch and the next.
struct Knobs {
    uint32_t depth_h = env_u32("MOE_DEPTH_H", DEPTH_H, 2, MAX_DEPTH_H);
    uint32_t depth_x = env_u32("MOE_DEPTH_X", DEPTH_X, 1, 4);
    uint32_t hack_ahead = env_u32("MOE_HACK_AHEAD", HACK_AHEAD, 1, MAX_DEPTH_H);
    uint32_t wd_ahead = env_u32("MOE_WD_AHEAD", WD_AHEAD, 1, 8);
    uint32_t wd_split = env_u32("MOE_WD_SPLIT", WD_SPLIT, 0, 8);
    uint32_t gu_chunks = env_u32("MOE_GU_CHUNKS", GU_CHUNKS, 1, 64);
    uint32_t wd_mgroup_min_blocks = env_u32("MOE_WD_MGROUP_MIN_BLOCKS", WD_MGROUP_MIN_BLOCKS, 1, 1u << 20);
    bool wd_resident = env_bool("MOE_WD_RESIDENT", WD_RESIDENT);
    bool wd_mrow = env_bool("MOE_WD_MROW", WD_MROW_ROUNDS);
    bool wd_mgroups = env_bool("MOE_WD_MGROUPS", WD_MGROUPS);
};

const Knobs& knobs() {
    static const Knobs instance;
    return instance;
}

uint32_t pow2_ceil(uint32_t value) {
    uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> split(uint32_t total, uint32_t groups) {
    TT_FATAL(groups > 0, "moe_fused_swiglu: split group count must be positive");
    const uint32_t base = total / groups;
    const uint32_t remainder = total % groups;
    std::vector<uint32_t> sizes(groups);
    std::vector<uint32_t> starts(groups);
    uint32_t start = 0;
    for (uint32_t i = 0; i < groups; ++i) {
        sizes[i] = base + (i < remainder ? 1u : 0u);
        starts[i] = start;
        start += sizes[i];
    }
    return {std::move(sizes), std::move(starts)};
}

uint32_t largest_divisor_le(uint32_t n, uint32_t cap) {
    for (uint32_t candidate = std::min(n, cap); candidate > 0; --candidate) {
        if (n % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

std::optional<ScatterPlan> scatter_plan(uint32_t m_block, uint32_t m_eff_min, uint32_t hn_pad, uint32_t kgroups) {
    if (kgroups < 2) {
        return std::nullopt;
    }
    ScatterPlan plan;
    for (uint32_t m = m_eff_min; m <= m_block; m <<= 1) {
        const uint32_t tiles = m * hn_pad;
        plan.sizes.push_back(tiles / largest_divisor_le(tiles, kgroups));
    }
    plan.slice_pages = 1;
    for (const uint32_t size : plan.sizes) {
        plan.slice_pages = std::lcm(plan.slice_pages, size);
    }
    plan.gather_pages = kgroups * *std::max_element(plan.sizes.begin(), plan.sizes.end());
    for (const uint32_t size : plan.sizes) {
        if (plan.slice_pages % size != 0 || plan.gather_pages % size != 0) {
            return std::nullopt;
        }
    }
    return plan;
}

}  // namespace

Blocking::Blocking(
    uint32_t hgroups_,
    uint32_t kgroups_,
    uint32_t emb_,
    uint32_t hidden_,
    uint32_t m_t_max_,
    uint32_t w_tile_,
    uint32_t bfp8_tile_,
    uint32_t bf16_tile_,
    uint32_t x_stick_,
    uint32_t l1_budget_,
    uint32_t out_tile_,
    bool enable_phase_alias_,
    bool x_is_rm_) :
    hgroups(hgroups_),
    kgroups(kgroups_),
    num_cores(hgroups_ * kgroups_),
    emb(emb_),
    hidden(hidden_),
    emb_t(emb_ / TILE),
    hid_t(hidden_ / TILE),
    m_t_max(m_t_max_),
    m_eff_min(pow2_ceil(OUT_SUBBLOCK_H_GU)),
    w_tile(w_tile_),
    bfp8_tile(bfp8_tile_),
    bf16_tile(bf16_tile_),
    x_stick(x_stick_ == 0 ? bfp8_tile_ : x_stick_),
    out_tile(out_tile_ == 0 ? bfp8_tile_ : out_tile_),
    enable_phase_alias(enable_phase_alias_),
    x_is_rm(x_is_rm_),
    l1_budget(l1_budget_) {
    TT_FATAL(kgroups >= 2, "moe_fused_swiglu: grid must be at least two rows tall");
    TT_FATAL(emb % TILE == 0 && hidden % TILE == 0, "moe_fused_swiglu: embedding and hidden must be tile-aligned");
    TT_FATAL(pow2_ceil(M_BLOCK) == M_BLOCK, "moe_fused_swiglu: M_BLOCK must be a power of two");
    TT_FATAL(m_eff_min <= M_BLOCK, "moe_fused_swiglu: gate/up subblock height exceeds M_BLOCK");

    gu_chunks_target = knobs().gu_chunks;
    // Runtime gate the kernels apply: grouped mode engages only when m_blocks >= this.
    wd_mgroup_min_blocks = knobs().wd_mgroup_min_blocks;

    std::tie(kr_sizes, kr_starts) = split(emb_t, kgroups);
    kr_pad = *std::max_element(kr_sizes.begin(), kr_sizes.end());

    const auto choice = choose_hn_pad();
    hn_pad = choice.hn_pad;
    gu_chunks = choice.chunks;
    balanced_hn = choice.balanced;
    gather_pages = choice.plan.gather_pages;
    slice_pages = choice.plan.slice_pages;
    gu_chunk_w = hn_pad / gu_chunks;
    hn_block = gu_chunk_w;
    gu_in1_subblocks = gu_chunk_w / hn_block;
    if (balanced_hn) {
        std::tie(hn_sizes, hn_starts) = split(hid_t, hgroups);
    } else {
        hn_sizes.resize(hgroups);
        hn_starts.resize(hgroups);
        for (uint32_t x = 0; x < hgroups; ++x) {
            hn_starts[x] = x * hn_pad;
            hn_sizes[x] = hn_starts[x] >= hid_t ? 0 : std::min(hn_pad, hid_t - hn_starts[x]);
        }
    }
    wd_mrow_rounds = knobs().wd_mrow && kgroups == M_BLOCK;

    std::tie(ec_sizes, ec_starts) = split(emb_t, num_cores);
    ec_max = *std::max_element(ec_sizes.begin(), ec_sizes.end());
    mgroup_rows = M_BLOCK / 2;
    mgroup_cores = hgroups * mgroup_rows;
    std::tie(ec_group_sizes, ec_group_starts) = split(emb_t, mgroup_cores);
    ec_group_max = *std::max_element(ec_group_sizes.begin(), ec_group_sizes.end());
    wd_mgroups =
        knobs().wd_mgroups && wd_mrow_rounds && M_BLOCK % 2 == 0 && kgroups == M_BLOCK && ec_group_max <= DEST_LIMIT;
    wd_ec_max = wd_mgroups ? ec_group_max : ec_max;

    TT_FATAL(
        ec_max <= DEST_LIMIT,
        "moe_fused_swiglu: down output width {} tiles exceeds DEST limit {} on grid {}x{}",
        ec_max,
        DEST_LIMIT,
        hgroups,
        kgroups);
    out_subblock_h_dn = 1;
    while (out_subblock_h_dn * 2 <= std::min(OUT_SUBBLOCK_H_DN_MAX, M_BLOCK) &&
           out_subblock_h_dn * 2 * ec_max <= DEST_LIMIT) {
        out_subblock_h_dn *= 2;
    }
    TT_FATAL(
        M_BLOCK % OUT_SUBBLOCK_H_GU == 0 && M_BLOCK % out_subblock_h_dn == 0,
        "moe_fused_swiglu: M_BLOCK must divide both subblock heights");

    const uint32_t depth_h_req = knobs().depth_h;
    const uint32_t hack_ahead_req = knobs().hack_ahead;

    max_m_blocks = (m_t_max + M_BLOCK - 1) / M_BLOCK;
    depth_x = max_m_blocks > 1 ? knobs().depth_x : 1;
    depth_w = W_RESIDENT ? 1 : DEPTH_W;
    // Bounded by hgroups - 2, not hgroups: depth_wd must be >= wd_ahead + 2, so anything larger
    // leaves min_depth_wd() with no legal depth to return.
    wd_ahead = std::max(1u, std::min(knobs().wd_ahead, hgroups > 2 ? hgroups - 2 : 1u));
    depth_h = depth_h_req;
    hack_ahead = std::max(1u, std::min(hack_ahead_req, depth_h - 1));
    wd_resident = knobs().wd_resident;
    depth_wd = wd_resident ? hgroups : min_depth_wd();

    // ---- THE L1 FALLBACK LADDER ----------------------------------------------------------
    // Ordered CHEAPEST-CONCESSION-FIRST: each step gives up the least performance for the bytes it
    // reclaims, and only runs if the previous ones left us over budget. Nothing here can fail — if
    // the ladder bottoms out still over, the program factory refuses with both numbers.
    //
    // Steps 1-4 deliberately evaluate `l1_bytes(true, ...)`, i.e. AS IF the input were row-major.
    // That overstates a tiled input by (32 * x_stick - bfp8_tile) bytes of CB_X_IN, so the early
    // concessions fire slightly earlier than a tiled shape strictly needs. Conservative on purpose:
    // it keeps the resident-weight decisions identical for both activation formats, so the two
    // formats share one geometry and one set of measured numbers. Only steps 5-6 use the real
    // `x_is_rm`, because by then the decision is about x's own CBs.
    // ---------------------------------------------------------------------------------------

    // 1. Grouped mode wants h depth 3 but pays a wider wd_ec_max; drop the h slot before the mode.
    if (wd_mgroups && depth_h > 2 && l1_bytes(true, out_tile, enable_phase_alias) > l1_budget) {
        depth_h = 2;
        hack_ahead = std::max(1u, std::min(hack_ahead_req, depth_h - 1));
    }
    // 2. Still over: give up grouped mode entirely and restore the h depth it was paying for.
    if (wd_mgroups && l1_bytes(true, out_tile, enable_phase_alias) > l1_budget) {
        wd_mgroups = false;
        wd_ec_max = ec_max;
        depth_h = depth_h_req;
        hack_ahead = std::max(1u, std::min(hack_ahead_req, depth_h - 1));
    }
    // 3. Drop the whole-h-row schedule: h_fast falls from hid_t to M_BLOCK*hn_pad, shrinking
    //    CB_H and CB_H_LOCAL. Costs the phase-2 fast path but is cheaper than losing residency.
    if (wd_mrow_rounds && l1_bytes(true, out_tile, enable_phase_alias) > l1_budget) {
        wd_mrow_rounds = false;
    }
    // 4. The expensive concession: stop holding W_down resident and stream it instead. Streaming
    //    is what SHRINKS CB_W_DOWN — resident depth is hgroups, streamed starts at wd_ahead + 2 —
    //    then walk depth_wd down through the remaining legal depths.
    if (l1_bytes(true, out_tile, enable_phase_alias) > l1_budget) {
        wd_resident = false;
        depth_wd = min_depth_wd();
        while (l1_bytes(true, out_tile, enable_phase_alias) > l1_budget) {
            const uint32_t next = next_smaller_depth_wd(depth_wd);
            if (next == depth_wd) {
                break;
            }
            depth_wd = next;
        }
    }
    // Consistency fixups, not budget steps: these keep mode flags that imply each other in sync
    // after the ladder above may have cleared one of them.
    wd_packed = balanced_hn && wd_resident;
    if (balanced_hn && !wd_packed) {
        wd_mrow_rounds = false;
    }
    if (!(wd_mrow_rounds && wd_resident)) {
        wd_mgroups = false;
        wd_ec_max = ec_max;
    }
    // 5. Give up the cross-M-block x prefetch. Row-major ONLY: a tiled prefetch reserves a second
    //    cb_x_tiles slot (PREFETCH_RESERVES_X_SLOT in the reader), and at depth 1 that reserve
    //    deadlocks against compute on the second M-block.
    if (x_is_rm && depth_x > 1 && l1_bytes(x_is_rm, out_tile, enable_phase_alias) > l1_budget) {
        depth_x = 1;
    }
    // 6. Last resort: two h slots, the smallest useful producer/consumer window.
    if (depth_h > 2 && l1_bytes(x_is_rm, out_tile, enable_phase_alias) > l1_budget) {
        depth_h = 2;
        hack_ahead = std::max(1u, std::min(hack_ahead_req, depth_h - 1));
    }
    wd_split = wd_resident && depth_wd == hgroups ? std::min(8u, knobs().wd_split) : 0;
    if (wd_split != 0 && hgroups > NOC_MAX_TRANSACTION_ID) {
        wd_split = 0;
    }
}

Blocking::HnChoice Blocking::choose_hn_pad() const {
    const uint32_t floor = (hid_t + hgroups - 1) / hgroups;
    const uint32_t ceiling = std::max(floor, (hid_t - 1) / std::max(1u, hgroups - 1)) + 1;
    const auto try_width = [&](uint32_t candidate, bool require_uniform) -> std::optional<HnChoice> {
        if (candidate * hgroups < hid_t) {
            return std::nullopt;
        }
        if (require_uniform && candidate * (hgroups - 1) >= hid_t) {
            return std::nullopt;
        }
        std::vector<uint32_t> chunks(candidate);
        std::iota(chunks.begin(), chunks.end(), 1);
        const uint32_t target = gu_chunks_target;
        std::sort(chunks.begin(), chunks.end(), [target](uint32_t a, uint32_t b) {
            const uint32_t da = a > target ? a - target : target - a;
            const uint32_t db = b > target ? b - target : target - b;
            return da == db ? a < b : da < db;
        });
        for (const uint32_t chunk_count : chunks) {
            if (candidate % chunk_count != 0 || OUT_SUBBLOCK_H_GU * (candidate / chunk_count) > DEST_LIMIT) {
                continue;
            }
            auto plan = scatter_plan(M_BLOCK, m_eff_min, candidate, kgroups);
            if (plan.has_value()) {
                return HnChoice{candidate, chunk_count, std::move(*plan), !require_uniform};
            }
        }
        return std::nullopt;
    };

    for (uint32_t candidate = floor; candidate <= ceiling; ++candidate) {
        if (auto result = try_width(candidate, true); result.has_value()) {
            return *result;
        }
    }
    if (hgroups <= hid_t) {
        if (auto result = try_width(floor, false); result.has_value()) {
            return *result;
        }
    }
    TT_FATAL(
        false,
        "moe_fused_swiglu: hidden {} ({} tiles) cannot be split across {} columns with the scatter lattice",
        hidden,
        hid_t,
        hgroups);
    return {};
}

bool Blocking::depth_wd_legal(uint32_t depth) const {
    if (depth < wd_ahead + 2) {
        return false;
    }
    if ((wd_resident || wd_ahead > 1) && hgroups % depth != 0) {
        return false;
    }
    return true;
}

// Shallowest legal streamed W_down depth. Falls back to hgroups when the grid is too short to hold
// wd_ahead + 2 slots — only reachable on a 2-row grid, where the streamed path has no legal depth.
uint32_t Blocking::min_depth_wd() const {
    for (uint32_t depth = wd_ahead + 2; depth <= hgroups; ++depth) {
        if (depth_wd_legal(depth)) {
            return depth;
        }
    }
    return hgroups;
}

uint32_t Blocking::next_smaller_depth_wd(uint32_t depth) const {
    for (uint32_t candidate = depth - 1; candidate > wd_ahead + 1; --candidate) {
        if (depth_wd_legal(candidate)) {
            return candidate;
        }
    }
    return depth;
}

// The full CB size table, annotated with what each term scales with, is in the header. This is that
// table as code — every page count here is a product of M_BLOCK, a depth, and one of the four
// derived widths (kr_pad, hn_pad, ec_max, wd_ec_max).
std::vector<CbView> Blocking::cb_layout(
    bool input_is_rm, uint32_t requested_out_tile, uint32_t idx_page, uint32_t counts_page) const {
    const uint32_t output_tile = requested_out_tile == 0 ? bfp8_tile : requested_out_tile;
    //: One whole gate/up output block, and the h block phase 2 broadcasts. `h_fast` is the whole
    //: hidden row under the wd_mrow schedule and just this column's block otherwise.
    const uint32_t gu = M_BLOCK * hn_pad;
    const uint32_t h_fast = wd_mrow_rounds ? hid_t : gu;
    const uint32_t out_block = std::max(M_BLOCK * ec_max, wd_mgroups ? mgroup_rows * ec_group_max : 0u);
    const uint32_t out_interm = (wd_mrow_rounds ? M_BLOCK / 2 : M_BLOCK) * ec_max;
    return {
        {CB_X_IN, input_is_rm ? XSTICK_ROWS * TILE : 1, x_stick, FormatKey::XIn},
        {CB_X_TILES, depth_x * M_BLOCK * kr_pad, bfp8_tile, FormatKey::Bfp8},
        {CB_X_STAGE, 1, 64, FormatKey::U32},
        {CB_MAILBOX_WRITER, 1, 64, FormatKey::U32},
        {CB_MAILBOX_COMPUTE, 1, 64, FormatKey::U32},
        {CB_W_GATE, depth_w * kr_pad * hn_pad, w_tile, FormatKey::Weight},
        {CB_W_UP, depth_w * kr_pad * hn_pad, w_tile, FormatKey::Weight},
        {CB_W_DOWN, depth_wd * hn_pad * wd_ec_max, w_tile, FormatKey::Weight},
        {CB_H, depth_h * h_fast, bfp8_tile, FormatKey::Bfp8},
        {CB_IDX_SCRATCH, 1, idx_page, FormatKey::U32},
        {CB_COUNTS_SCRATCH, 1, counts_page, FormatKey::U32},
        {CB_GATHER_GATE, gather_pages, bfp8_tile, FormatKey::Bfp8},
        {CB_GATHER_UP, gather_pages, bfp8_tile, FormatKey::Bfp8},
        {CB_SLICE_GATE, slice_pages, bf16_tile, FormatKey::Bf16},
        {CB_SLICE_UP, slice_pages, bf16_tile, FormatKey::Bf16},
        {CB_H_SLICE, slice_pages, bfp8_tile, FormatKey::Bfp8},
        {CB_OUT_TILES, DEPTH_OUT * out_block, output_tile, FormatKey::Out},
        {CB_GATE_ACC, gu, bfp8_tile, FormatKey::Bfp8},
        {CB_UP_ACC, gu, bfp8_tile, FormatKey::Bfp8},
        {CB_GATE_SILU, slice_pages, bf16_tile, FormatKey::Bf16},
        {CB_H_LOCAL, std::max(gu, h_fast), bfp8_tile, FormatKey::Bfp8},
        {CB_OUT_INTERM, out_interm, bf16_tile, FormatKey::Bf16},
    };
}

uint32_t Blocking::phase_cb_alias_pages(uint32_t requested_out_tile) const {
    const auto layout = cb_layout(true, requested_out_tile, 64, 64);
    uint32_t pages = 1;
    for (const uint32_t wanted : {CB_GATHER_GATE, CB_H_SLICE, CB_OUT_TILES}) {
        const auto it =
            std::find_if(layout.begin(), layout.end(), [&](const CbView& view) { return view.index == wanted; });
        pages = std::lcm(pages, it->pages);
    }
    return pages;
}

bool Blocking::phase_cb_alias(uint32_t requested_out_tile) const {
    const uint32_t output_tile = requested_out_tile == 0 ? bfp8_tile : requested_out_tile;
    if (output_tile != bfp8_tile) {
        return false;
    }
    const auto layout = cb_layout(true, requested_out_tile, 64, 64);
    uint32_t separate_pages = 0;
    for (const uint32_t wanted : {CB_GATHER_GATE, CB_H_SLICE, CB_OUT_TILES}) {
        separate_pages +=
            std::find_if(layout.begin(), layout.end(), [&](const CbView& view) { return view.index == wanted; })->pages;
    }
    return phase_cb_alias_pages(requested_out_tile) < separate_pages;
}

std::vector<CbAllocation> Blocking::cb_allocations(
    bool input_is_rm,
    uint32_t requested_out_tile,
    uint32_t idx_page,
    uint32_t counts_page,
    bool aliases_enabled) const {
    const auto layout = cb_layout(input_is_rm, requested_out_tile, idx_page, counts_page);
    std::unordered_map<uint32_t, CbView> by_index;
    for (const auto& view : layout) {
        by_index.emplace(view.index, view);
    }

    std::vector<std::vector<uint32_t>> aliases{{CB_X_STAGE, CB_MAILBOX_WRITER, CB_MAILBOX_COMPUTE}};
    if (aliases_enabled) {
        if (phase_cb_alias(requested_out_tile)) {
            aliases.push_back({CB_GATHER_GATE, CB_H_SLICE, CB_OUT_TILES});
        }
        const uint32_t bf16_lcm = std::lcm(by_index.at(CB_GATE_SILU).pages, by_index.at(CB_OUT_INTERM).pages);
        if (bf16_lcm < by_index.at(CB_GATE_SILU).pages + by_index.at(CB_OUT_INTERM).pages) {
            aliases.push_back({CB_GATE_SILU, CB_OUT_INTERM});
        }
    }

    std::unordered_map<uint32_t, uint32_t> alias_root;
    std::unordered_map<uint32_t, CbAllocation> alias_allocations;
    for (const auto& alias : aliases) {
        uint32_t pages = 1;
        const uint32_t page_size = by_index.at(alias.front()).page_size;
        std::vector<CbView> views;
        for (const uint32_t index : alias) {
            const auto& view = by_index.at(index);
            TT_FATAL(view.page_size == page_size, "moe_fused_swiglu: aliased CB page sizes disagree");
            TT_FATAL(!alias_root.contains(index), "moe_fused_swiglu: CB {} is in two alias groups", index);
            pages = std::lcm(pages, view.pages);
            alias_root[index] = alias.front();
            views.push_back(view);
        }
        alias_allocations.emplace(alias.front(), CbAllocation{pages * page_size, std::move(views)});
    }

    std::vector<CbAllocation> allocations;
    allocations.reserve(layout.size());
    for (const auto& view : layout) {
        if (alias_allocations.contains(view.index)) {
            allocations.push_back(alias_allocations.at(view.index));
        } else if (!alias_root.contains(view.index)) {
            allocations.push_back(CbAllocation{view.pages * view.page_size, {view}});
        }
    }
    return allocations;
}

uint64_t Blocking::l1_bytes(bool input_is_rm, uint32_t requested_out_tile, bool aliases_enabled) const {
    uint64_t total = 0;
    for (const auto& allocation : cb_allocations(input_is_rm, requested_out_tile, 64, 64, aliases_enabled)) {
        total += allocation.total_size;
    }
    return total;
}

std::string Blocking::describe() const {
    std::ostringstream stream;
    stream << hgroups << 'x' << kgroups << " grid, emb " << emb << ", hidden " << hidden << ": kr_pad " << kr_pad
           << ", hn_pad " << hn_pad << ", gu_chunks " << gu_chunks << ", ec_max " << ec_max << ", depth_wd " << depth_wd
           << ", depth_x " << depth_x << ", depth_h " << depth_h << ", wd_split " << wd_split << ", wd_mrow "
           << (wd_mrow_rounds && wd_resident);
    return stream.str();
}

uint32_t nd_shard_n_tiles(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    if (memory_config.buffer_type() != tt::tt_metal::BufferType::DRAM || !memory_config.created_with_nd_shard_spec()) {
        return 0;
    }
    const auto& spec = memory_config.nd_shard_spec();
    if (!spec.has_value() || spec->shard_shape.rank() < 2 || spec->shard_shape[-1] % TILE != 0) {
        return 0;
    }
    return spec->shard_shape[-1] / TILE;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry
