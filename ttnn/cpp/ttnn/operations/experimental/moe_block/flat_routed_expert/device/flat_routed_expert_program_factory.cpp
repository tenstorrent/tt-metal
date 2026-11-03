// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_program_factory.hpp"

#include "flat_drisc.hpp"
#include "flat_routed_expert_plan.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace factory_detail {
using tt::tt_metal::CoreCoord;
using Defines = std::map<std::string, std::string>;

constexpr uint32_t XRD_BATCH = 2, DW_BATCH = 2;
// semaphore ids (16 created in order on every role core)
constexpr uint32_t DATA = 4, HARR = 5, GO = 6, DONE = 7, GATH = 8, XARR = 9, SFREE = 10, HFREE = 11, HSFREE = 12,
                   WORD = 13, GATH1 = 14, GATH2 = 15;
// the gate/up cores' grant counter (polled by their DRISC sender; id 1 is otherwise unused on them)
constexpr uint32_t GRANT = 1;
constexpr const char* KDIR = "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/device/kernels/";

// A runtime-arg list whose buffer-address words are recorded for the cache-hit patch.
struct Args {
    std::vector<uint32_t> v;
    std::vector<std::tuple<uint32_t, FlatAddr, uint32_t>> addr;
    Args& lit(uint32_t x) {
        v.push_back(x);
        return *this;
    }
    Args& lits(const std::vector<uint32_t>& xs) {
        v.insert(v.end(), xs.begin(), xs.end());
        return *this;
    }
    Args& a(FlatAddr s, uint32_t off = 0) {
        addr.emplace_back(v.size(), s, off);
        v.push_back(off);
        return *this;
    }
    Args& cat(const Args& o) {
        for (const auto& [i, s, off] : o.addr) {
            addr.emplace_back(v.size() + i, s, off);
        }
        v.insert(v.end(), o.v.begin(), o.v.end());
        return *this;
    }
};

tt::DataFormat w_format(bool bf8) { return bf8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Bfp4_b; }
}  // namespace factory_detail
using namespace factory_detail;

std::vector<std::pair<std::string, std::string>> flat_schedule_defines(
    const FlatRoutedExpertConfig& cfg, const FlatRoutedExpertPlan& p) {
    const uint32_t se_max_e = p.E <= 16 ? 16 : (p.E <= 32 ? 32 : 64);
    std::vector<std::pair<std::string, std::string>> d{
        {"SE_RPS", std::to_string(p.mt * 32)},
        {"SE_MAX_E", std::to_string(se_max_e)},
        {"SE_GU_NREG", std::to_string(p.ring_g / p.nk_gu)}};
    if (cfg.pin) {
        d.emplace_back("SE_PIN_MIN", std::to_string(cfg.pin));
        d.emplace_back("SE_PIN_SMALL", "2");
        // pin only a dominant expert (se_dyn_pin); layout.pin_ratio (0: always pin)
        d.emplace_back("SE_PIN_RATIO", std::to_string(cfg.layout.pin_ratio));
    }
    return d;
}

FlatProgramHandles add_flat_routed_expert_program(
    tt::tt_metal::Program& program,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertConfig& cfg,
    const FlatRoutedExpertPlan& p,
    const FlatRoutedExpertInputs& t) {
    using namespace tt::tt_metal;
    auto* device = &mesh_device;
    const FlatLayoutOptions& lo = cfg.layout;
    FlatProgramHandles sv;
    const auto phys = [device](const CoreCoord& c) { return device->worker_core_from_logical_core(c); };
    const auto pk = [&](const CoreCoord& c) {
        const auto q = phys(c);
        return static_cast<uint32_t>((q.x << 16) | q.y);
    };
    const uint32_t E = p.E, MT = p.mt, MTG = p.mtg, NP = p.np, G = p.g, S = p.s, V = p.v, It = p.It, Ht = p.Ht;
    const uint32_t w_tile = p.w_tile, banks = p.banks, NR = p.rects.size();
    const uint32_t n_rd = p.readers.size(), ngu = p.gu.size(), ND = p.down.size();
    const uint32_t x_blk = MT * kKBlk, x_bytes = x_blk * kBf8Tile;
    const uint32_t PIN = cfg.pin;
    TT_FATAL(NR == 2, "flat_routed_expert: two gate/up rectangles");

    // ---- per-core pieces ----
    std::vector<uint32_t> head_xy, gu_xy;
    const uint32_t coord_xy = pk(p.coords[0]);
    for (uint32_t d : p.d_heads) {
        head_xy.push_back(pk(p.down[d]));
    }
    for (uint32_t i = 0; i < ngu; ++i) {
        gu_xy.push_back(pk(p.gu[i]));
    }
    std::vector<std::vector<uint32_t>> rect_cores(NR);
    for (uint32_t ci = 0; ci < ngu; ++ci) {
        rect_cores[p.rect_of(p.gu[ci])].push_back(ci);
    }
    // x delivery: x read once by the helpers (the plan's two full rectangles, one helper each) and unicast snakes, no
    // multicast: the NOC0 wedge of a relay's multicast into the gate/up rectangle beside the DRISC weight writes and
    // DRAM traffic. Each primary unicasts its blocks into its rectangle's snake head; the gate/up cores forward them
    // along the snake (down a column, wrapping inside the rectangle, one column over, on down), each piece once its
    // successor has freed the block's slot. One snake a rectangle (2: the relay writes every block twice - GLM-5.3 M
    // 1024 238.7 vs 205.5 us / expert, multicast 211.5), 4 cut-through pieces, the forwarder on the gate/up NCRISC
    // (se5_xfwd.cpp) on NOC0 with 3 pieces in flight, the receiver (se5_recv.cpp, BRISC) on NOC1.
    const auto aligned = [&](uint32_t n) { return x_bytes % n == 0 && (x_bytes / n) % 16 == 0; };
    const uint32_t x_pieces = aligned(4) ? 4 : (aligned(2) ? 2 : 1);
    constexpr uint32_t X_FWD_DEPTH = 3;
    log_info(
        tt::LogOp,
        "flat_routed_expert: x read once (helpers), unicast snakes (snakes 1 pieces {} NCRISC forwarder NOC0 depth {})",
        x_pieces,
        X_FWD_DEPTH);
    struct XLink {
        uint32_t pred = 0, succ0 = 0;
    };
    std::vector<XLink> xlink(ngu);
    std::vector<uint32_t> x_head(NR, 0);  // each rectangle's snake head (gate/up index)
    for (uint32_t k = 0; k < NR; ++k) {
        const auto [x0, x1, y0, y1] = p.rects[k];
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> at;
        for (uint32_t ci : rect_cores[k]) {
            at[{p.gu[ci].x, p.gu[ci].y}] = ci;
        }
        const uint32_t W = x1 - x0 + 1, H = y1 - y0 + 1;
        TT_FATAL(at.size() == W * H, "flat x snakes: rectangle {} is not full of gate/up cores", k);
        // the snake: from the relay's row, down a column (wrapping inside the rectangle, one NoC hop a step, NOC0:
        // east / south), one column over, on down from that row - every core forwards once
        const uint32_t yr = std::clamp<uint32_t>(p.relays[k].y, y0, y1);
        std::vector<uint32_t> seq;
        uint32_t row = yr;
        for (uint32_t i = 0; i < W; ++i) {
            for (uint32_t r = 0; r < H; ++r) {
                const uint32_t y = y0 + (row - y0 + r) % H;
                seq.push_back(at.at({x0 + i, y}));
                if (r + 1 == H) {
                    row = y;  // the next column starts on this row
                }
            }
        }
        x_head[k] = seq.front();
        for (size_t j = 0; j < seq.size(); ++j) {
            XLink& l = xlink[seq[j]];
            l.pred = j ? pk(p.gu[seq[j - 1]]) : 0;
            l.succ0 = j + 1 < seq.size() ? pk(p.gu[seq[j + 1]]) : 0;
        }
    }
    const uint32_t ng_bytes = 4 * p.NG;
    Args dyn;
    dyn.a(FlatAddr::Counts).a(FlatAddr::Regions).lit(ng_bytes).lit(1).lit(1000000000).a(FlatAddr::Ids).lit(4 * E);

    // ---- defines ----
    const uint32_t se_max_e = E <= 16 ? 16 : (E <= 32 ? 32 : 64);
    const uint32_t meta_bytes = se_max_e == 16 ? 512 : (se_max_e == 32 ? 1024 : 2048);
    const uint32_t dyn_need = (ng_bytes + 63) / 64 * 64 + (4 * E + 63) / 64 * 64;
    uint32_t dyn_half = 512;
    while (dyn_half < dyn_need) {
        dyn_half *= 2;
    }
    const uint32_t nreg_gu = p.ring_g / p.nk_gu;
    bool dn_reg = PIN != 0;
    for (uint32_t d = 0; d < ND && dn_reg; ++d) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(p.pcds[d], It);
        dn_reg = static_cast<uint32_t>(std::nearbyint(p.dring * (It / kd))) == nreg_gu * (It / kd);
    }
    Defines dyn_def = {{"SE_DYN_HALF", std::to_string(dyn_half)}, {"SE_META_BYTES", std::to_string(meta_bytes)}};
    for (const auto& [k, v] : flat_schedule_defines(cfg, p)) {  // (SE_RPS, SE_MAX_E, SE_GU_NREG, pinning)
        dyn_def[k] = v;
    }
    // The schedule's inputs (counts, regions, expert ids: one DRAM page each, all in bank 0) read once per call (the
    // plain op): the launcher's BRISC (se_drisc_go.cpp) reads them, compacts this chip's per-expert count / region
    // into its CB 9 and sets its semaphore SE_DYN_LSEM (reset every launch); every other se_dyn_load polls that
    // semaphore over the NoC and pulls the compact copy (instead of ~150 tiny DRAM reads of bank 0 behind the startup
    // weight streams: 5-16 us each). Inside a combine overlap (cmb_rt) every kernel reads DRAM.
    const bool dyn_pull = cfg.cmb_rt == 0;
    if (dyn_pull) {
        const CoreCoord lv = phys(p.launcher), l0 = flat_worker_noc0(mesh_device, p.launcher);
        dyn_def["SE_DYN_PULL"] = "1";
        dyn_def["SE_DYN_LX"] = std::to_string(lv.x);
        dyn_def["SE_DYN_LY"] = std::to_string(lv.y);
        dyn_def["SE_DYN_LPX"] = std::to_string(l0.x);
        dyn_def["SE_DYN_LPY"] = std::to_string(l0.y);
        dyn_def["SE_DYN_LSEM"] = "13";
        dyn_def["SE_DYN_CB"] = "9";
    }
    if (dn_reg) {
        dyn_def["SE_DN_REG"] = "1";
    }
    // h pieces per sub-block down the chains
    constexpr uint32_t H_PIECES = 32;
    auto with = [](Defines base, std::initializer_list<std::pair<const std::string, std::string>> extra) {
        for (const auto& kv : extra) {
            base[kv.first] = kv.second;
        }
        return base;
    };
    const std::string sbt = std::to_string(p.sbt);

    // ---- kernels + runtime args ----
    // combine overlap (cfg.cmb_rt): a y writer's args padded to cmb_rt, then 4 report args the overlap fills
    auto add_writer_rt = [&](KernelHandle k, const CoreCoord& c, Args args) {
        if (cfg.cmb_rt) {
            TT_FATAL(
                args.v.size() <= cfg.cmb_rt,
                "flat_routed_expert: {} writer args > cmb_rt {}",
                args.v.size(),
                cfg.cmb_rt);
            args.v.resize(cfg.cmb_rt + 4 + (cfg.y_l1 ? kFlatYL1Args : 0), 0);
            sv.y_writers.emplace_back(k, c);
        }
        return args;
    };
    // the steps the writers report: slot order (flat_with_program: cmb_slot_order with row-major y)
    TT_FATAL(
        !cfg.cmb_rt || !cfg.y_row_major || cfg.cmb_slot_order,
        "flat_routed_expert: a combine overlap with row-major y reports in slot order (cmb_slot_order)");
    Defines cmb_def = cfg.cmb_rt ? Defines{{"SE_CMB_DONE_RT", std::to_string(cfg.cmb_rt)}} : Defines{};
    if (cfg.cmb_rt && cfg.cmb_no_go) {  // (flat_with_program: cumulative counts, no go)
        cmb_def["SE_CMB_NO_GO"] = "1";
    }
    if (cfg.cmb_rt && cfg.y_l1) {  // (flat_with_program: y rows straight into the combine's L1)
        cmb_def["SE_Y_L1"] = std::to_string(cfg.cmb_rt + 4);
    }
    auto add_rt = [&](KernelHandle k, const CoreCoord& c, const Args& args) {
        SetRuntimeArgs(program, k, c, args.v);
        for (const auto& [i, s, off] : args.addr) {
            sv.patches.push_back({k, c, i, s, off});
        }
    };
    auto dm = [&](const std::string& name,
                  const std::vector<CoreCoord>& cores,
                  DataMovementProcessor proc,
                  NOC noc,
                  const std::vector<uint32_t>& ct,
                  const Defines& defs) {
        return CreateKernel(
            program,
            std::string(KDIR) + name,
            rect_ranges(cores),
            DataMovementConfig{
                .processor = proc,
                .noc = noc,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                .compile_args = ct,
                .defines = defs});
    };
    const auto fidelity = static_cast<MathFidelity>(cfg.math_fidelity);
    auto cp = [&](const std::string& name,
                  const std::vector<CoreCoord>& cores,
                  const std::vector<uint32_t>& ct,
                  const Defines& defs,
                  bool fp32 = false,
                  bool full_sync = false) {
        Defines d = defs;
        if (cfg.pack_stochastic_rounding) {
            d["SE_PACK_SRND"] = "1";
        }
        return CreateKernel(
            program,
            std::string(KDIR) + name,
            rect_ranges(cores),
            ComputeConfig{
                .math_fidelity = fidelity,
                .fp32_dest_acc_en = fp32,
                .dst_full_sync_en = full_sync,
                .compile_args = ct,
                .defines = d});
    };
    // the down computes' accumulation (cfg.down_fp32): fp32 DEST in a full-sync DST
    const auto dn_def = [&](Defines d) {
        if (cfg.down_fp32) {
            d = with(d, {{"SE_DN_FP32", "1"}, {"SE_DN_FULL_SYNC", "1"}});
        }
        return d;
    };

    // row-major y (cfg.y_row_major): the down compute pack-untilizes, the writers send row segments (se_yrm.hpp)
    const bool yrm = cfg.y_row_major;
    // the row-major out CB (bf16 row tiles of PCD pages) takes what the down cores' arena holds after the out region's
    // start (the plan sizes the arena for the largest role; the plan's 2 KB margin is kept), at most 2 MT (a virtual
    // expert ahead) and 8 (the writers' transaction ids 8..15)
    auto yrm_rows = [&](uint32_t pw) {
        const uint32_t room = p.arena_tiles * 2048 - p.o_off - 2048;
        return std::min({room / (pw * 2048), 2 * MT, 8u});
    };
    auto yrm_def = [&](Defines d, uint32_t pw = 0) {
        if (cfg.y_bf16) {
            d["SE_Y_BF16"] = "1";  // (the y writers: bf16 tile pages)
        }
        if (yrm) {
            d["SE_Y_RM"] = "1";
            if (pw) {
                d["SE_Y_RM_ROWS"] = std::to_string(yrm_rows(pw));
            }
        }
        return d;
    };
    // the DRAM cores' persistent senders read and forward the gate/up weights (flat_drisc.hpp); the launcher only posts
    // each call's jobs (se_drisc_go.cpp). gu_source tensix*: the gate/up cores read their own blocks (se_gu_read.hpp)
    // from the same bank layout; the senders stay up (idle) and the launcher posts nothing
    const bool gu_tensix = std::any_of(p.gu_how.begin(), p.gu_how.end(), [](uint32_t h) { return h != 0; });
    // (hybrid plans: some slots DRISC, the others read by their cores; a slot with no DRISC gets a skip each call)
    const bool any_drisc = std::any_of(p.slot_drisc.begin(), p.slot_drisc.end(), [](bool d) { return d; });
    const FlatDriscTopology drisc = FlatDriscTopology::of(mesh_device, p.drisc_tiles);
    const uint32_t drisc_base = drisc.base;
    auto drisc_xy = [&](uint32_t r) { return drisc.virtual_xy(drisc.logical(r)); };
    {
        TT_FATAL(p.r_ <= 8, "flat DRISC: <= 8 receivers a slot (se_drisc.hpp SE_DRISC_MAX_R)");
        TT_FATAL(
            FlatDriscSenders::running_on(mesh_device, p.drisc_n, p.drisc_tiles, p.drisc_probe),
            "flat_routed_expert: the DRISC senders of {} slots ({}) are not running on this mesh device "
            "(FlatDriscSenders::get)",
            p.drisc_n,
            to_string(p.drisc_tiles));
        // the job posting args (se_drisc_post.hpp); the GDDR arbitration weights (SD_MPFE: layout.drisc_mpfe, 0
        // untouched)
        Args post;
        post.lit(drisc_base)
            .a(FlatAddr::GateUp)
            .a(FlatAddr::Arena)
            .lit(p.drisc_n)
            .lit(p.r_)
            .lit(p.drisc_mpfe)
            .lits(
                {p.drisc_dma_cap,
                 p.slot * w_tile,
                 p.drisc_ub | (p.drisc_early_free ? 1u << 8 : 0u),  // (se_drisc.hpp SD_UB_EARLY_FREE)
                 G,
                 p.nk_gu,
                 GRANT,
                 DATA});
        // per slot (se_drisc_post.hpp): its DRISC / alternate, region, and its chunk / blocks / receivers
        for (uint32_t r = 0; r < p.drisc_n; ++r) {
            const auto [ac, aj] = drisc.alt(r);
            const uint32_t axy = aj != ~0u ? drisc.virtual_xy(ac) : 0;
            post.lit(drisc_xy(r))
                .lit(FlatDriscTopology::job(r, banks))
                .lit((r / banks) * p.region_bytes)
                .lit(axy)
                .lit(aj == ~0u ? 0 : aj)
                .lit(p.rg * p.slot * w_tile)
                .lit(p.rg)
                .lit(p.slot_drisc[r] ? p.r_ : 0u);  // (0 receivers: the slot's cores read it; a skip)
        }
        for (uint32_t r = 0; r < p.drisc_n; ++r) {
            for (uint32_t j = 0; j < p.r_; ++j) {
                post.lit(pk(p.gu[p.gu_at(r, j)]));
            }
        }
        // (gate/up cores reading their own weights: the launcher only leads the schedule pull, it posts nothing)
        const auto k =
            dm("se_drisc_go.cpp",
               {p.launcher},
               DataMovementProcessor::RISCV_0,
               NOC::NOC_0,
               {E},
               !any_drisc ? with(dyn_def, {{"SE_GU_TENSIX", "1"}}) : dyn_def);
        Args a;
        a.cat(post).cat(dyn);
        add_rt(k, p.launcher, a);
    }

    // x relays: primaries (feed their snake) + helpers (read / tilize half of x each, into both primaries' rings)
    auto rl_off = [&](uint32_t idx) { return idx < NR ? 0 : (idx - NR) / NR + 1; };
    auto rl_sb = [&](uint32_t idx) {
        uint32_t n = 0;
        for (uint32_t g = 0; g < V * p.nsb; ++g) {
            n += g % p.vstride == rl_off(idx);
        }
        return n;
    };
    {
        // indexed mode: the token index address is the relay reader's last runtime arg (after dyn)
        const bool indexed = t.token_index.has_value();
        const uint32_t idx_arg = 3 + static_cast<uint32_t>(dyn.v.size());
        // the relay reads x in segments of sbt x 32 bf16 (one super-block row): a segment must not straddle two x
        // pages (they sit in different banks), so the page is a multiple of the segment or divides it
        const uint32_t x_page_bytes = t.x.logical_shape()[-1] * 2, seg_bytes = p.sbt * 64;
        TT_FATAL(
            x_page_bytes % seg_bytes == 0 || seg_bytes % x_page_bytes == 0,
            "flat_routed_expert: x page of {} B (hidden {} / x_pages_per_row {}) must be a multiple or a divisor of "
            "the {} B read segment",
            x_page_bytes,
            p.H,
            p.H / t.x.logical_shape()[-1],
            seg_bytes);
        Defines xrd_def = with(dyn_def, {{"SE_SBT", sbt}});
        if (indexed) {
            xrd_def["XRD_INDEXED"] = "1";
            if (lo.x_ready > 0) {  // (x lands during the program: wait for the caller's count, layout.x_ready)
                xrd_def["XRD_READY"] = std::to_string(lo.x_ready);
            }
        }
        TT_FATAL(kRmChunks % XRD_BATCH == 0, "flat_routed_expert: XRD_BATCH must divide the row-major chunks");
        // x reads on NOC1, the relays' sends on NOC0 (one RISC per NoC on a relay: dedicated-NoC mode)
        const std::vector<CoreCoord> xrel_p(p.relays.begin(), p.relays.begin() + NR);
        const std::vector<CoreCoord> xrel_h(p.relays.begin() + NR, p.relays.end());
        const std::vector<uint32_t> xrd_ct = {
            0, t.x.logical_shape()[-1] * 2, E, MT, p.nsb, S, XRD_BATCH, idx_arg, p.H / t.x.logical_shape()[-1]};
        // (the primaries read no x: their kernel idles; the helpers read / tilize all of it)
        const auto kxr =
            dm("se11_xrd.cpp",
               xrel_p,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_1,
               xrd_ct,
               with(xrd_def, {{"XRD_HELP_ALL_PRIM", "1"}}));
        const auto kxr_h = dm("se11_xrd.cpp", xrel_h, DataMovementProcessor::RISCV_0, NOC::NOC_1, xrd_ct, xrd_def);
        const auto ktz = cp("se11_tz.cpp", p.relays, {0, 1, MT}, with(dyn_def, {{"SE_SBT", sbt}}));
        const Defines xmc_def =
            with(dyn_def, {{"SE_SBT", sbt}, {"SE_XNH", "2"}, {"XMC_CHAIN_PIECES", std::to_string(x_pieces)}});
        // one kernel per primary, its credits to the helpers on its own semaphore (5 / 6 there)
        std::vector<KernelHandle> kxms;
        for (uint32_t q = 0; q < NR; ++q) {
            kxms.push_back(
                dm("se11_xmc.cpp",
                   {p.relays[q]},
                   DataMovementProcessor::RISCV_1,
                   NOC::NOC_0,
                   {1, MT, kBf8Tile, p.x_slots, XARR, WORD, kKBlk, E, p.nsb},
                   with(xmc_def, {{"XMC_CREDIT_SEM", std::to_string(5 + q)}})));
        }
        // one kernel per helper k: its arrivals are semaphore {4, 7}[k] on both primaries, its credits come back on 5
        // (primary 0) and 6 (primary 1)
        std::vector<KernelHandle> khl;
        for (uint32_t k = 0; k < NR; ++k) {
            khl.push_back(
                dm("se13_xhelp.cpp",
                   {p.relays[NR + k]},
                   DataMovementProcessor::RISCV_1,
                   NOC::NOC_0,
                   {1, MT, kBf8Tile, p.land_slots, k ? 7u : 4u, 5, E, p.nsb},
                   with(dyn_def, {{"SE_SBT", sbt}, {"SE_XDUAL", "6"}})));
        }
        const uint32_t sbb = MT * p.sbt * kBf8Tile;
        for (uint32_t idx = 0; idx < p.relays.size(); ++idx) {
            const CoreCoord rl = p.relays[idx];
            const uint32_t k = idx % NR;
            // helper k: the super-blocks g % 2 == k of the whole stream; the primaries none
            Args xr;
            xr.a(FlatAddr::X).lit(idx < NR ? 1u : NR).lit(idx < NR ? 0u : k).cat(dyn);
            if (indexed) {
                xr.a(FlatAddr::TokenIndex);
                if (lo.x_ready > 0) {  // (the slot after the index: this relay's x-ready word, XRD_READY)
                    xr.a(FlatAddr::Words, X_READY_OFF);
                } else {
                    xr.a(FlatAddr::Arena, p.ti_off);
                }
            }
            add_rt(idx >= NR ? kxr_h : kxr, rl, xr);
            SetRuntimeArgs(program, ktz, rl, std::vector<uint32_t>{rl_sb(idx)});
            if (idx < NR) {
                const uint32_t hxy = pk(p.gu[x_head[k]]);
                Args xm;
                xm.a(FlatAddr::Arena, p.x_off)
                    .lit(hxy)
                    .lit(hxy)
                    .lit(1)
                    .a(FlatAddr::Words)
                    .lit(1)
                    .lits({V * p.nsb, XARR, 1u, 0u, 0u})
                    .lits({0, 0, 0});
                // helper 0's xy, the landing rings, their slots, helper 1's xy (rectangle 0's first - helper k's ring
                // k), then the snake head, listed from the arg RT 2 names
                xm.lit(pk(p.relays[NR])).a(FlatAddr::Arena, p.land_off).lit(p.land_slots).lit(pk(p.relays[NR + 1]));
                xm.cat(dyn);
                xm.v[2] = static_cast<uint32_t>(xm.v.size());
                xm.lit(hxy);
                add_rt(kxms[idx], rl, xm);
            } else {
                Args hl;
                hl.lit(pk(p.relays[0]))  // (primary 0 first, ring k on both primaries)
                    .a(FlatAddr::Arena, p.land_off + k * p.land_slots * sbb)
                    .lit(rl_sb(idx))
                    .lit(1)
                    .lit(0)
                    .cat(dyn);
                hl.lit(NR).lit(k);        // (se13_xhelp.cpp SE_X2: every NR-th super-block from k)
                hl.lit(pk(p.relays[1]));  // (SE_XDUAL: the second primary)
                add_rt(khl[k], rl, hl);
            }
        }
    }

    // gate/up cores
    {
        // the receiver (BRISC) on NOC1: on NOC0 the gate/up cores' h writes, the DRAM weight reads and x deliveries
        // into these same cores wedged the NoC (a relay stuck in noc_cmd_buf_ready or in its write barrier; ~1 in
        // 1e4..2e5 launches, any chip), a hardware hang the NoC probe test reproduces from the traffic alone
        // (tests/ttnn/unit_tests/operations/noc_hang_probe, op_skeleton). The receivers count their DRISC grants
        // locally; the x forwarder (NCRISC) on NOC0 reads their consumed count / end marker in semaphores 2 / 3 (free
        // on gate/up cores), one successor, X_FWD_DEPTH pieces in flight
        Defines recv_def = with(
            dyn_def,
            {{"SE_X_FWD_NC_CONS", "2"}, {"SE_X_FWD_NC_END", "3"}, {"SE_X_FWD_DEPTH", std::to_string(X_FWD_DEPTH)}});
        // gate/up weights read by the gate/up cores (se_gu_read.hpp): the receiver's grant counter and DATA count as
        // the DRISC protocol, semaphore 0 the loads-list mailbox to the NCRISC reader
        Defines gu_def;
        if (gu_tensix) {
            gu_def = {
                {"SE_GU_TENSIX", "1"},
                {"SE_GU_BLK", std::to_string(p.slot * w_tile)},
                {"SE_GU_PER_E", std::to_string(p.nk_gu)},
                {"SE_GU_DEPTH", std::to_string(p.gu_depth)},
                {"SE_GU_PIECES", std::to_string(p.gu_pieces)},
                {"SE_GU_MBOX_SEM", "0"},
                {"SE_GU_GRANT_SEM", std::to_string(GRANT)},
                {"SE_GU_DATA_SEM", std::to_string(DATA)},
                {"SE_DYN_HALF", dyn_def.at("SE_DYN_HALF")}};
            recv_def.insert(gu_def.begin(), gu_def.end());
        }
        if (p.nu) {  // 16-column units: the h run from K-tile cg NU / 2, half tiles at the ends
            recv_def["SE_HG_NU"] = std::to_string(p.nu);
            recv_def["SE_HPC"] = std::to_string(p.hpc);
        }
        const std::vector<uint32_t> recv_ct = {
            0,         x_blk,   1,    p.slot, E * p.nk_gu, V,        p.ring_g, 16,    1,     3,        2,    MTG,
            kBf8Tile,  ngu,     DATA, HARR,   GO,          DONE,     kKBlk,    HARR,  SFREE, p.hbuf,   XARR, HFREE,
            p.x_slots, x_bytes, S,    NP,     HSFREE,      H_PIECES, p.nk_gu,  GATH1, GATH2, x_pieces, G,    E};
        TT_FATAL(
            x_bytes % x_pieces == 0 && (x_bytes / x_pieces) % 16 == 0,
            "flat_routed_expert: x pieces {} of a {} B x block must be 16 B aligned",
            x_pieces,
            x_bytes);
        const KernelHandle kr = dm("se5_recv.cpp", p.gu, DataMovementProcessor::RISCV_0, NOC::NOC_1, recv_ct, recv_def);
        const KernelHandle kxf =
            dm("se5_xfwd.cpp",
               p.gu,
               DataMovementProcessor::RISCV_1,
               NOC::NOC_0,
               {p.x_slots, x_bytes, x_pieces, XARR, SFREE, 2, 3, X_FWD_DEPTH},
               gu_def);
        const bool gu_rp = p.gu_rp && !p.gu_full_sync;
        Defines cdef = with(
            dyn_def,
            {{"SE_DST_TILES", std::to_string(gu_rp ? 4 : p.dst_tiles)},
             {"SE_ACT", std::to_string(cfg.activation)},
             {"SE_XMT", std::to_string(MT)}});
        if (gu_rp) {
            cdef["SE_GU_RP"] = std::to_string(std::max(1u, 4 / p.gwt));
            cdef["SE_XSLOTS"] = std::to_string(p.x_slots);
            if (p.gu_kg > 1) {  // gate/up K blocks per CB handshake
                cdef["SE_GU_KG"] = std::to_string(p.gu_kg);
            }
        }
        TT_FATAL(
            !p.hg || cfg.activation <= 3,
            "flat_routed_expert: half-width gate / up tiles take SiLU / SwiGLU-OAI / SiTU / clamped SiLU GLU only");
        if (p.hg) {  // half-width gate / up tiles: the face-wise activation
            cdef["SE_HG"] = "1";
        }
        if (p.nu) {  // 16-column units per core (RT 1: the core's first unit odd)
            cdef["SE_HG_NU"] = std::to_string(p.nu);
            cdef["SE_HPC"] = std::to_string(p.hpc);
        }
        const auto kc =
            cp("se3_compute.cpp",
               p.gu,
               {kKBlk, MTG, p.nk_gu, 0, 1, E, S, p.slot, 1, 1, NP, 0, p.ring_g},
               cdef,
               true,
               p.gu_full_sync);
        for (uint32_t r = 0; r < n_rd; ++r) {
            for (uint32_t j = 0; j < p.r_; ++j) {
                const uint32_t ci = p.gu_at(r, j);
                const CoreCoord gc = p.gu[ci];
                const uint32_t cg_ = r * p.rg + j / G;  // (the core's column group)
                Args a;
                // the forwarder and its credit word: the DRISC's L1 word, the grant counter
                a.lit(drisc_xy(r))
                    .lit(GRANT)
                    .lit(cg_)
                    .lit(head_xy.size())
                    .lit(0)
                    .lit(coord_xy)
                    .lit(GATH)
                    .a(FlatAddr::Arena, p.x_off)
                    .lits({0, 0, 0})  // (x snake links: the forwarder's)
                    .lit(pk(p.relays[p.rect_of(gc)]))
                    .a(FlatAddr::Words, 0)
                    .a(FlatAddr::Arena, p.h_off)
                    .lit(j % G)
                    .lit(0)
                    .lits({2, 3, 0, SFREE, WORD})
                    .lits(head_xy)
                    .cat(dyn);
                // the gate/up weight reads (se_gu_read.hpp): mode (1 NCRISC, 2 BRISC), weights, bank, this core's
                // first block (slot region + block j / G), chunk bytes, landing ring (the arena's start)
                const uint32_t how = p.gu_how.at(ci);
                const uint32_t gmode = how == FlatGuPlace::tensix_noc0 ? 1u : (how == FlatGuPlace::tensix_noc1 ? 2u : 0u);
                Args gw;
                gw.lit(gmode)
                    .a(FlatAddr::GateUp)
                    .lit(r % banks)
                    .lit((r / banks) * p.region_bytes + (j / G) * p.slot * w_tile)
                    .lit(p.rg * p.slot * w_tile)
                    .a(FlatAddr::Arena, 0);
                if (gu_tensix) {
                    a.cat(gw);
                }
                add_rt(kr, gc, a);
                // x ring, predecessor (+ its free word's sem), successor, relay + this core's word there
                Args f;
                f.a(FlatAddr::Arena, p.x_off)
                    .lit(xlink[ci].pred)
                    .lit(SFREE)
                    .lit(xlink[ci].succ0)
                    .lit(pk(p.relays[p.rect_of(gc)]))
                    .a(FlatAddr::Words, 0);
                if (gu_tensix) {
                    f.cat(gw);
                }
                add_rt(kxf, gc, f);
                SetRuntimeArgs(program, kc, gc, std::vector<uint32_t>{j % G, (cg_ * p.nu) & 1u});
            }
        }
    }

    // down cores, one kernel set per column width
    std::map<uint32_t, std::vector<uint32_t>> dgroups;
    for (uint32_t d = 0; d < ND; ++d) {
        dgroups[p.pcds[d]].push_back(d);
    }
    for (const auto& [pw, ds] : dgroups) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(pw, It), nblk = It / kd, slot = kd * pw;
        const uint32_t ring = static_cast<uint32_t>(std::nearbyint(p.dring * nblk)), out = MT * pw;
        std::vector<CoreCoord> cores;
        for (uint32_t d : ds) {
            cores.push_back(p.down[d]);
        }
        // row-major y written from the NCRISC on its NOC (se6_dw.cpp) instead of the BRISC, whose NOC carries the h
        // chain: bfp8 x + bf16 h at M 512 / 2048 / 5120 -9 / -9 / -11%, bfp8 / bfp8 -4 / -1 / -0%; +1% (~0.4 us) at
        // M <= 64. The h chain / y writes (BRISC) on NOC1, the down-weight reads (NCRISC) on NOC0.
        const bool y_nc = yrm;
        const Defines ync = y_nc ? Defines{{"SE_Y_NC", "1"}} : Defines{};
        auto plus = [](Defines a, const Defines& b) {
            a.insert(b.begin(), b.end());
            return a;
        };
        const auto kr =
            dm("se6_drecv.cpp",
               cores,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_1,
               {2,    p.h_tiles, kBf8Tile, H_PIECES, 16, out, V,  S,    ngu,   ND,    HARR, HSFREE,
                GATH, DONE,      GO,       p.hbuf,   MT, pw,  Ht, GATH, GATH1, GATH2, E},
               plus(plus(yrm_def(dyn_def, pw), ync), y_nc ? Defines{} : cmb_def));
        const auto kw =
            dm("se6_dw.cpp",
               cores,
               DataMovementProcessor::RISCV_1,
               NOC::NOC_0,
               {1, slot, w_tile, E * nblk, DW_BATCH, cfg.weights_bf8 ? 1u : 0u, E, 0, 0, 0, MT, pw, Ht},
               plus(
                   plus(
                       y_nc ? plus(plus(yrm_def(dyn_def, pw), ync), cmb_def) : dyn_def,
                       Defines{{"SE_DW_ROT", std::to_string(banks)}}),
                   Defines{{"SE_DW_RING", std::to_string(ring)}}));
        const auto kc =
            cp("se6_dcompute.cpp",
               cores,
               {MTG, G, It, kd, pw, E, S, slot, ring},
               dn_def(yrm_def(dyn_def)),
               cfg.down_fp32,
               cfg.down_fp32);
        for (uint32_t d : ds) {
            const CoreCoord dc = p.down[d];
            const uint32_t pred = p.d_pred.contains(d) ? pk(p.down[p.d_pred.at(d)]) : 0;
            const uint32_t succ = p.d_succ.contains(d) ? pk(p.down[p.d_succ.at(d)]) : 0;
            Args a;
            a.a(FlatAddr::Arena, p.h_off)
                .lit(pred)
                .lit(succ)
                .lit(coord_xy)
                .lit(d == 0 ? 1 : 0)
                .lit(ngu)
                .a(FlatAddr::Y)
                .lit(p.col0s[d])
                .a(FlatAddr::Done)
                .lit(d)
                .lits(gu_xy)
                .cat(dyn);
            add_rt(kr, dc, y_nc ? a : add_writer_rt(kr, dc, a));
            Args w;
            w.a(FlatAddr::Down).lit(d % banks).lit(d * p.wd_rot_region).cat(dyn);
            if (y_nc) {
                w.a(FlatAddr::Y).lit(p.col0s[d]);
            }
            add_rt(kw, dc, y_nc ? add_writer_rt(kw, dc, w) : w);
            SetRuntimeArgs(program, kc, dc, std::vector<uint32_t>{0});
        }
    }

    // ---- semaphores, CBs ----
    const auto arena_cores = p.arena_cores();
    const auto all_crs = rect_ranges(arena_cores);
    for (uint32_t i = 0; i < 16; ++i) {
        CreateSemaphore(program, all_crs, 0);
    }
    const Buffer& arena = *t.arena.buffer();
    auto arena_cb = [&](uint8_t idx,
                        uint32_t off,
                        uint32_t size,
                        const std::vector<CoreCoord>& cores,
                        tt::DataFormat fmt,
                        uint32_t page) {
        // (every arena-backed CB must lie inside the arena: one past its end overlaps whatever L1 sits above it)
        TT_FATAL(
            off + size <= arena.aligned_size_per_bank(),
            "flat_routed_expert: CB {} at {} + {} B runs past the {} B arena",
            idx,
            off,
            size,
            arena.aligned_size_per_bank());
        CircularBufferConfig c(size, {{idx, fmt}});
        c.set_page_size(idx, page);
        c.set_globally_allocated_address(arena);
        c.set_address_offset(off);
        sv.arena_cbs.emplace_back(CreateCircularBuffer(program, rect_ranges(cores), c), off);
    };
    uint32_t static_bytes = 0;  // (every static CB, summed: at most the plan's static_l1_bytes, checked below)
    auto static_cb =
        [&](uint8_t idx, uint32_t size, const std::vector<CoreCoord>& cores, tt::DataFormat fmt, uint32_t page) {
            CircularBufferConfig c(size, {{idx, fmt}});
            c.set_page_size(idx, page);
            CreateCircularBuffer(program, rect_ranges(cores), c);
            static_bytes += (size + 63) / 64 * 64;
        };
    const auto wf = w_format(cfg.weights_bf8);
    const auto bfp8 = tt::DataFormat::Bfp8_b;
    // the down output CB in the arena's out region (bytes = the bfp8 double block): bfp8 tiles, or with row-major y
    // bf16 row tiles of PCD pages each (as many whole row tiles as the region holds: MT for MT <= 16)
    auto out_cb = [&](const std::vector<CoreCoord>& cores, uint32_t pw, uint32_t bytes) {
        if (!yrm) {
            if (cfg.y_bf16) {  // bf16 tiles: one block (the bfp8 double buffer's bytes hold one bf16 block)
                const uint32_t tiles = bytes / (2 * kBf8Tile);
                TT_FATAL(tiles * 2048 <= bytes, "flat_routed_expert: y_bf16 out CB");
                arena_cb(16, p.o_off, tiles * 2048, cores, tt::DataFormat::Float16_b, 2048);
                return;
            }
            arena_cb(16, p.o_off, bytes, cores, bfp8, kBf8Tile);
            return;
        }
        TT_FATAL(yrm_rows(pw) >= 1, "flat_routed_expert: row-major y: no room for one row tile");
        arena_cb(16, p.o_off, yrm_rows(pw) * pw * 2048, cores, tt::DataFormat::Float16_b, 2048);
    };
    arena_cb(0, 0, kRmChunks * 32 * p.seg, p.relays, tt::DataFormat::Float16_b, p.seg);
    arena_cb(1, p.sb_off, kSbSlots * MT * p.sbt * kBf8Tile, p.relays, bfp8, kBf8Tile);
    arena_cb(1, 0, p.ring_g * p.slot * w_tile, p.gu, wf, w_tile);
    arena_cb(0, p.x_off, p.x_slots * x_bytes, p.gu, bfp8, kBf8Tile);
    static_cb(3, p.hbuf * MTG * p.hpc * kBf8Tile, p.gu, bfp8, kBf8Tile);
    static_cb(16, 2048, p.gu, tt::DataFormat::Float16_b, 2048);
    arena_cb(2, p.h_off, p.hbuf * p.h_tiles * kBf8Tile, p.down, bfp8, kBf8Tile);
    for (const auto& [pw, ds] : dgroups) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(pw, It);
        const uint32_t ring = static_cast<uint32_t>(std::nearbyint(p.dring * (It / kd)));
        std::vector<CoreCoord> cores;
        for (uint32_t d : ds) {
            cores.push_back(p.down[d]);
        }
        arena_cb(1, 0, ring * kd * pw * w_tile, cores, wf, w_tile);
        out_cb(cores, pw, 2 * MT * pw * kBf8Tile);
    }
    static_cb(6, meta_bytes, arena_cores, tt::DataFormat::UInt32, meta_bytes);
    static_cb(7, 4 * dyn_half, arena_cores, tt::DataFormat::UInt32, 4 * dyn_half);
    if (dyn_pull) {  // the leader's compact schedule inputs (2 x SE_MAX_E words; the CB exists on every core)
        static_cb(9, 1024, arena_cores, tt::DataFormat::UInt32, 1024);
    }
    TT_FATAL(
        static_bytes <= p.static_l1_bytes,
        "flat_routed_expert: static CBs {} B > the plan's static_l1_bytes {} B (callers size the arena with it)",
        static_bytes,
        p.static_l1_bytes);

    override_flat_routed_expert_addresses(program, sv, t);
    return sv;
}

void override_flat_routed_expert_addresses(
    tt::tt_metal::Program& program, const FlatProgramHandles& handles, const FlatRoutedExpertInputs& t) {
    using namespace tt::tt_metal;
    std::array<uint32_t, static_cast<size_t>(FlatAddr::Count)> addr{};
    addr[size_t(FlatAddr::X)] = t.x.buffer()->address();
    addr[size_t(FlatAddr::Y)] = t.output.buffer()->address();
    addr[size_t(FlatAddr::Counts)] = t.counts.buffer()->address();
    addr[size_t(FlatAddr::Regions)] = t.regions.buffer()->address();
    addr[size_t(FlatAddr::Ids)] = t.global_expert_ids.buffer()->address();
    addr[size_t(FlatAddr::GateUp)] = t.gate_up_weights.buffer()->address();
    addr[size_t(FlatAddr::Down)] = t.down_weights.buffer()->address();
    addr[size_t(FlatAddr::Done)] = t.done_words.buffer()->address();
    addr[size_t(FlatAddr::Arena)] = t.arena.buffer()->address();
    addr[size_t(FlatAddr::Words)] = t.words.buffer()->address();
    addr[size_t(FlatAddr::TokenIndex)] = t.token_index ? t.token_index->buffer()->address() : 0;
    // (a source's args - the arena's CBs with it - only when its address differs from the one they were last written
    // with: FlatProgramHandles::last)
    std::array<bool, static_cast<size_t>(FlatAddr::Count)> moved{};
    for (size_t s = 0; s < moved.size(); ++s) {
        moved[s] = handles.last[s] != addr[s];
        handles.last[s] = addr[s];
    }
    for (const auto& pa : handles.patches) {
        if (moved[size_t(pa.src)]) {
            GetRuntimeArgs(program, pa.kernel, pa.core)[pa.index] = addr[size_t(pa.src)] + pa.offset;
        }
    }
    if (moved[size_t(FlatAddr::Arena)]) {
        for (const auto& [cb, off] : handles.arena_cbs) {
            UpdateDynamicCircularBufferAddress(program, cb, *t.arena.buffer(), off);
        }
    }
}

FlatRoutedExpertConfig with_combine_overlap(FlatRoutedExpertConfig cfg, const FlatRoutedExpertPlan& plan, bool y_l1) {
    cfg.y_row_major = true;
    cfg.y_bf16 = false;
    cfg.cmb_rt = 48 + static_cast<uint32_t>(plan.gu.size()) + 2 * cfg.experts_per_chip;
    // the writers report local experts in slot order, with cumulative counts and no go handshake
    cfg.cmb_slot_order = true;
    cfg.cmb_no_go = true;
    cfg.y_l1 = y_l1;
    return cfg;
}

void set_flat_combine_report(
    tt::tt_metal::Program& program,
    const FlatProgramHandles& handles,
    const FlatRoutedExpertConfig& cfg,
    const FlatCombineReport& report) {
    using namespace tt::tt_metal;
    TT_FATAL(cfg.cmb_rt > 0, "flat_routed_expert: set_flat_combine_report on a program built without cmb_rt");
    TT_FATAL(
        report.l1_args.size() <= kFlatYL1Args,
        "flat_routed_expert: {} L1 hand-off args > {}",
        report.l1_args.size(),
        kFlatYL1Args);
    TT_FATAL(cfg.y_l1 || report.l1_args.empty(), "flat_routed_expert: L1 hand-off args without y_l1");
    uint32_t widx = 0;
    for (const auto& [k, core] : handles.y_writers) {
        auto& a = GetRuntimeArgs(program, k, core);
        a[cfg.cmb_rt + 0] = static_cast<uint32_t>(report.collector.x);
        a[cfg.cmb_rt + 1] = static_cast<uint32_t>(report.collector.y);
        a[cfg.cmb_rt + 2] = report.counts_addr;
        a[cfg.cmb_rt + 3] = 0;  // (no go handshake: SE_CMB_NO_GO)
        if (cfg.y_l1 && !report.l1_args.empty()) {
            TT_FATAL(a.size() >= cfg.cmb_rt + 4 + kFlatYL1Args, "flat_routed_expert: writer args too short (y_l1)");
            for (uint32_t j = 0; j < report.l1_args.size(); ++j) {
                a[cfg.cmb_rt + 4 + j] = report.l1_args[j];
            }
            a[cfg.cmb_rt + 4 + 4] = widx++;
        }
    }
}

std::vector<tt::tt_metal::CoreCoord> flat_y_writers(const FlatRoutedExpertPlan& plan) {
    std::map<uint32_t, std::vector<uint32_t>> dgroups;
    for (uint32_t d = 0; d < plan.down.size(); ++d) {
        dgroups[plan.pcds[d]].push_back(d);
    }
    std::vector<tt::tt_metal::CoreCoord> out;
    for (const auto& [pw, ds] : dgroups) {
        for (const uint32_t d : ds) {
            out.push_back(plan.down[d]);
        }
    }
    return out;
}

FlatRoutedExpertProgramFactory::cached_program_t FlatRoutedExpertProgramFactory::create(
    const FlatRoutedExpertParams& cfg, const FlatRoutedExpertInputs& t, Tensor& /*output*/) {
    auto& mesh_device = *t.x.device();
    tt::tt_metal::Program program{};
    auto handles =
        add_flat_routed_expert_program(program, mesh_device, cfg, *flat_routed_expert_plan(mesh_device, cfg), t);
    return cached_program_t{std::move(program), std::move(handles)};
}

void FlatRoutedExpertProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const FlatRoutedExpertParams&, const FlatRoutedExpertInputs& t, Tensor&) {
    override_flat_routed_expert_addresses(cached.program, cached.shared_variables, t);
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
