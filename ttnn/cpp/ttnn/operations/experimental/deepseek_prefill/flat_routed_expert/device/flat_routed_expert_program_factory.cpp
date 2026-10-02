// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_program_factory.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

namespace factory_detail {
using tt::tt_metal::CoreCoord;
using Defines = std::map<std::string, std::string>;

constexpr uint32_t H_PIECES = 32, XRD_BATCH = 2, FWD = 2, DW_BATCH = 2;
// semaphore ids (16 created in order on every role core)
constexpr uint32_t DATA = 4, HARR = 5, GO = 6, DONE = 7, GATH = 8, XARR = 9, SFREE = 10, HFREE = 11, HSFREE = 12,
                   WORD = 13, GATH1 = 14, GATH2 = 15;
constexpr const char* KDIR =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/flat_routed_expert/device/kernels/";

// A runtime-arg list whose buffer-address words are recorded for the cache-hit patch.
struct Args {
    std::vector<uint32_t> v;
    std::vector<std::tuple<uint32_t, AddrSrc, uint32_t>> addr;
    Args& lit(uint32_t x) {
        v.push_back(x);
        return *this;
    }
    Args& lits(const std::vector<uint32_t>& xs) {
        v.insert(v.end(), xs.begin(), xs.end());
        return *this;
    }
    Args& a(AddrSrc s, uint32_t off = 0) {
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
constexpr uint32_t KBLK = kKBlk, BF8_TILE = kBf8Tile, H_TILE = kBf8Tile, RM_CHUNKS = kRmChunks, SB_SLOTS = kSbSlots;

FlatRoutedExpertProgramFactory::cached_program_t FlatRoutedExpertProgramFactory::create(
    const FlatRoutedExpertParams& cfg, const FlatRoutedExpertInputs& t, Tensor& output) {
    using namespace tt::tt_metal;
    auto* device = t.x.device();
    const FlatRoutedExpertPlan p = make_flat_routed_expert_plan(device, cfg);
    Program program{};
    shared_variables_t sv;
    const auto phys = [device](const CoreCoord& c) { return device->worker_core_from_logical_core(c); };
    const auto pk = [&](const CoreCoord& c) {
        const auto q = phys(c);
        return static_cast<uint32_t>((q.x << 16) | q.y);
    };
    const uint32_t E = p.E, MT = p.mt, MTG = p.mtg, NP = p.np, G = p.g, S = p.s, V = p.v, It = p.It, Ht = p.Ht;
    const uint32_t w_tile = p.w_tile, banks = p.banks, NR = p.rects.size();
    const uint32_t n_rd = p.readers.size(), ngu = p.gu.size(), ND = p.down.size(), ngu_sg = ngu / p.nsg;
    const uint32_t x_blk = MT * KBLK, x_bytes = x_blk * BF8_TILE;
    const uint32_t PIN = cfg.pin;
    const auto sgx = [&](uint32_t k) { return p.nsg > 1 ? std::vector<uint32_t>{k} : std::vector<uint32_t>{}; };

    // ---- per-core pieces ----
    std::vector<std::vector<uint32_t>> head_xy_sg(p.nsg), gu_xy_sg(p.nsg);
    std::vector<uint32_t> coord_sg;
    for (uint32_t d : p.d_heads) {
        head_xy_sg[p.sg_dn(d)].push_back(pk(p.down[d]));
    }
    for (uint32_t k = 0; k < p.nsg; ++k) {
        coord_sg.push_back(pk(p.down[k * p.nd_sg]));
        for (uint32_t i = k * ngu_sg; i < (k + 1) * ngu_sg; ++i) {
            gu_xy_sg[k].push_back(pk(p.gu[i]));
        }
    }
    std::vector<std::vector<uint32_t>> rect_cores(NR);
    for (uint32_t ci = 0; ci < ngu; ++ci) {
        rect_cores[p.rect_of(p.gu[ci])].push_back(ci);
    }
    auto word_rel = [&](uint32_t ci) {
        const auto& rc = rect_cores[p.rect_of(p.gu[ci])];
        return 4 * static_cast<uint32_t>(std::find(rc.begin(), rc.end(), ci) - rc.begin());
    };
    const uint32_t ng_bytes = 4 * p.NG;
    Args dyn;
    dyn.a(AddrSrc::Counts).a(AddrSrc::Regions).lit(ng_bytes).lit(1).lit(1000000000).a(AddrSrc::Ids).lit(4 * E);

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
    dn_reg = dn_reg && (!p.rdown || p.ring_dr == nreg_gu * p.nblk_r);
    Defines dyn_def = {
        {"SE_DYN", "1"},
        {"SE_DYN_HALF", std::to_string(dyn_half)},
        {"SE_RPS", std::to_string(MT * 32)},
        {"SE_MAX_E", std::to_string(se_max_e)},
        {"SE_META_BYTES", std::to_string(meta_bytes)},
        {"SE_GU_NREG", std::to_string(nreg_gu)}};
    if (p.nsg > 1) {
        dyn_def["SE_SG"] = std::to_string(p.nsg);
    }
    if (PIN) {
        dyn_def["SE_PIN_MIN"] = std::to_string(PIN);
        dyn_def["SE_PIN_SMALL"] = "2";
    }
    if (dn_reg) {
        dyn_def["SE_DN_REG"] = "1";
    }
    auto with = [](Defines base, std::initializer_list<std::pair<const std::string, std::string>> extra) {
        for (const auto& kv : extra) {
            base[kv.first] = kv.second;
        }
        return base;
    };
    const std::string sbt = std::to_string(p.sbt);

    // ---- kernels + runtime args ----
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
            DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct, .defines = defs});
    };
    auto cp = [&](const std::string& name,
                  const std::vector<CoreCoord>& cores,
                  const std::vector<uint32_t>& ct,
                  const Defines& defs,
                  bool fp32 = false) {
        return CreateKernel(
            program,
            std::string(KDIR) + name,
            rect_ranges(cores),
            ComputeConfig{
                .math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = fp32, .compile_args = ct, .defines = defs});
    };

    // readers (read NOC0, forward NOC1): the chain-tail readers also compute down columns (se9_rdown)
    std::vector<CoreCoord> rdn_cores, plain;
    for (const auto& [r, tail] : p.rdn) {
        rdn_cores.push_back(p.readers[r]);
    }
    for (const auto& c : p.readers) {
        if (std::find(rdn_cores.begin(), rdn_cores.end(), c) == rdn_cores.end()) {
            plain.push_back(c);
        }
    }
    std::map<uint32_t, uint32_t> tail_succ;  // down core -> its reader tail's xy
    for (const auto& [r, tail] : p.rdn) {
        tail_succ[tail] = pk(p.readers[r]);
    }
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
        if (yrm) {
            d["SE_Y_RM"] = "1";
            if (pw) {
                d["SE_Y_RM_ROWS"] = std::to_string(yrm_rows(pw));
            }
        }
        return d;
    };
    if (p.rdown) {
        const Defines rdn_def = yrm_def(with(dyn_def, {{"SE9_TRID", "1"}, {"SE_E2E", "1"}}), p.pcd_r);
        const auto k =
            dm("se9_rdown.cpp",
               rdn_cores,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_0,
               {0,        w_tile, p.rg * p.slot, E * p.nk_gu, 1,    p.slot_dr, E * p.nblk_r, 2,  p.h_tiles, H_TILE,
                H_PIECES, 16,     p.out_tiles_r, V,           HARR, HSFREE,    p.hbuf,       MT, p.pcd_r,   Ht,
                S,        E,      p.rd_slots,    p.ring_dr},
               rdn_def);
        const auto kc =
            cp("se6_dcompute.cpp",
               rdn_cores,
               {MTG, G, It, p.kd_r, p.pcd_r, E, S, p.slot_dr, p.ring_dr},
               yrm_def(with(dyn_def, {{"SE_EARLY_POP", "1"}})));
        for (uint32_t i = 0; i < p.rdn.size(); ++i) {
            const auto [r, t_] = p.rdn[i];
            Args a;
            a.a(AddrSrc::GateUp)
                .lit(r % banks)
                .lit((r / banks) * p.region_bytes)
                .a(AddrSrc::ReaderDown)
                .lit(i % banks)
                .lit((i / banks) * p.wr_region)
                .lit(pk(p.down[t_]))
                .lit(coord_sg[p.sg_rd(r)])
                .a(AddrSrc::Done)
                .lit(p.nd_sg + i % p.n_rdn)
                .a(AddrSrc::Y)
                .lit(p.rem_cols + (i % p.n_rdn) * p.pcd_r)
                .cat(dyn)
                .lits(sgx(p.sg_rd(r)));
            add_rt(k, p.readers[r], a);
            SetRuntimeArgs(program, kc, p.readers[r], std::vector<uint32_t>{0});
        }
    }
    if (!plain.empty()) {
        const auto k =
            dm("se_reader.cpp",
               plain,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_0,
               {0, w_tile, p.rg * p.slot, 1, p.nk_gu, 0, 1, E, p.rg * p.slot, 0},
               dyn_def);
        for (uint32_t r = 0; r < n_rd; ++r) {
            if (std::find(plain.begin(), plain.end(), p.readers[r]) == plain.end()) {
                continue;
            }
            Args a;
            a.a(AddrSrc::GateUp).lit(r % banks).lit((r / banks) * p.region_bytes).lit(0).cat(dyn).lits(sgx(p.sg_rd(r)));
            add_rt(k, p.readers[r], a);
        }
    }
    {
        const auto k =
            dm("se10_fwd.cpp",
               p.readers,
               DataMovementProcessor::RISCV_1,
               NOC::NOC_1,
               {0, p.r_, w_tile, p.rg * p.slot, p.slot, p.ring_g, 0, DATA, p.nk_gu, p.rd_slots, FWD, E, G},
               dyn_def);
        for (uint32_t r = 0; r < n_rd; ++r) {
            Args a;
            a.a(AddrSrc::Arena);  // land_addr: the gate/up cores' in1 ring (arena offset 0)
            for (uint32_t j = 0; j < p.r_; ++j) {
                a.lit(pk(p.gu[r * p.r_ + j]));
            }
            a.cat(dyn).lits(sgx(p.sg_rd(r)));
            add_rt(k, p.readers[r], a);
        }
    }

    // x relays: primaries (multicast) + helpers (read / tilize every vstride-th super-block)
    auto rl_off = [&](uint32_t idx) { return idx < NR ? 0 : (idx - NR) / NR + 1; };
    auto rl_sb = [&](uint32_t idx) {
        uint32_t n = 0;
        for (uint32_t g = 0; g < V * p.nsb; ++g) {
            n += g % p.vstride == rl_off(idx);
        }
        return n;
    };
    {
        // indexed mode: the token index address is the relay reader's last runtime arg (after dyn + subgrid)
        const bool indexed = t.token_index.has_value();
        const uint32_t idx_arg = 3 + static_cast<uint32_t>(dyn.v.size()) + static_cast<uint32_t>(sgx(0).size());
        // the relay reads x in segments of sbt x 32 bf16 (one super-block row): a segment must not straddle two x
        // pages (they sit in different banks), so the page is a multiple of the segment or divides it
        const uint32_t x_page_bytes = t.x.logical_shape()[-1] * 2, seg_bytes = p.sbt * 64;
        TT_FATAL(
            x_page_bytes % seg_bytes == 0 || seg_bytes % x_page_bytes == 0,
            "flat_routed_expert: x page of {} B (hidden {} / x_pages_per_row {}) must be a multiple or a divisor of "
            "the "
            "{} B read segment",
            x_page_bytes,
            p.H,
            p.H / t.x.logical_shape()[-1],
            seg_bytes);
        Defines xrd_def = with(dyn_def, {{"SE_SBT", sbt}, {"XHELP_SMALL", "1"}});
        if (indexed) {
            xrd_def["XRD_INDEXED"] = "1";
        }
        const auto kxr =
            dm("se11_xrd.cpp",
               p.relays,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_1,
               {0, t.x.logical_shape()[-1] * 2, E, MT, p.nsb, S, XRD_BATCH, idx_arg, p.H / t.x.logical_shape()[-1]},
               xrd_def);
        const auto ktz = cp("se11_tz.cpp", p.relays, {0, 1, MT}, with(dyn_def, {{"SE_SBT", sbt}}));
        const std::vector<CoreCoord> prim(p.relays.begin(), p.relays.begin() + NR);
        const auto kxm =
            dm("se11_xmc.cpp",
               prim,
               DataMovementProcessor::RISCV_1,
               NOC::NOC_0,
               {1, MT, BF8_TILE, p.x_slots, XARR, WORD, KBLK, E, p.nsb},
               with(
                   dyn_def,
                   {{"SE_SBT", sbt},
                    {"XMC_WHOLE_SB", "1"},
                    {"XMC_HELPER", "1"},
                    {"SE_XNH", std::to_string(p.nh)},
                    {"XMC_ZERO_WORDS", "1"}}));
        std::vector<KernelHandle> khl;
        for (uint32_t j = 0; j < p.nh; ++j) {
            const std::vector<CoreCoord> hl(p.relays.begin() + NR + NR * j, p.relays.begin() + NR + NR * (j + 1));
            const uint32_t data_sem = std::vector<uint32_t>{4, 7, 8, 10}[j];
            khl.push_back(
                dm("se13_xhelp.cpp",
                   hl,
                   DataMovementProcessor::RISCV_1,
                   NOC::NOC_0,
                   {1, MT, BF8_TILE, p.land_slots, data_sem, 5, E, p.nsb},
                   with(dyn_def, {{"SE_SBT", sbt}})));
        }
        const uint32_t sbb = MT * p.sbt * BF8_TILE;
        for (uint32_t idx = 0; idx < p.relays.size(); ++idx) {
            const CoreCoord rl = p.relays[idx];
            const uint32_t k = idx % NR;
            const auto [x0, x1, y0, y1] = p.rects[k];
            Args xr;
            xr.a(AddrSrc::X).lit(p.vstride).lit(rl_off(idx)).cat(dyn).lits(sgx(k));
            if (indexed) {
                xr.a(AddrSrc::TokenIndex);
            }
            add_rt(kxr, rl, xr);
            SetRuntimeArgs(program, ktz, rl, std::vector<uint32_t>{rl_sb(idx)});
            if (idx < NR) {
                Args xm;
                xm.a(AddrSrc::Arena, p.x_off)
                    .lit(pk(CoreCoord(x0, y0)))
                    .lit(pk(CoreCoord(x1, y1)))
                    .lit((x1 - x0 + 1) * (y1 - y0 + 1))
                    .a(AddrSrc::Words)
                    .lit(rect_cores[k].size())
                    .lits({V * p.nsb, XARR, 1, 0, p.group_rect ? ((idx * MTG) | ((idx + 1) * MTG << 8)) : 0, 0, 0, 0})
                    .lit(pk(p.relays[idx + NR]))
                    .a(AddrSrc::Arena, p.land_off)
                    .lit(p.land_slots);
                for (uint32_t j = 1; j < p.nh; ++j) {
                    xm.lit(pk(p.relays[NR + NR * j + idx]));
                }
                xm.cat(dyn).lits(sgx(k));
                add_rt(kxm, rl, xm);
            } else {
                const uint32_t j = (idx - NR) / NR;
                Args hl;
                hl.lit(pk(p.relays[idx % NR]))
                    .a(AddrSrc::Arena, p.land_off + j * p.land_slots * sbb)
                    .lit(rl_sb(idx))
                    .lit(1 + p.nh)
                    .lit(j + 1)
                    .cat(dyn)
                    .lits(sgx(k));
                add_rt(khl[j], rl, hl);
            }
        }
    }

    // gate/up cores
    {
        // NOC1: on NOC0 the gate/up cores' h writes, the DRAM weight reads and the x relays' multicast into these same
        // cores wedge the NoC (a relay stuck in noc_cmd_buf_ready or in its write barrier; ~1 in 1e4..2e5 launches,
        // any chip), a hardware hang the NoC probe test reproduces from the traffic alone
        // (tests/ttnn/unit_tests/operations/noc_hang_probe, op_skeleton); with these writes on NOC1 it never did.
        const auto kr =
            dm("se5_recv.cpp",
               p.gu,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_1,
               {0,         x_blk,   1,    p.slot, E * p.nk_gu, V,        p.ring_g, 16,    1,     3,      2,    MTG,
                H_TILE,    ngu_sg,  DATA, HARR,   GO,          DONE,     KBLK,     HARR,  SFREE, p.hbuf, XARR, HFREE,
                p.x_slots, x_bytes, S,    NP,     HSFREE,      H_PIECES, p.nk_gu,  GATH1, GATH2, 1,      G,    E},
               with(dyn_def, {{"SE_GU_ONLY", "1"}, {"SE_X_RELAY", "1"}, {"SE_NO_PARTNER", "1"}}));
        Defines cdef = with(
            dyn_def,
            {{"SE_DST_TILES", std::to_string(p.gu_rp ? 4 : p.dst_tiles)},
             {"SE_GU_ONLY", "1"},
             {"SE_ACT", std::to_string(cfg.activation)},
             {"SE_XMT", std::to_string(MT)}});
        if (p.gu_rp) {
            cdef["SE_GU_RP"] = std::to_string(std::max(1u, 4 / (2 * NP)));
            cdef["SE_XSLOTS"] = std::to_string(p.x_slots);
        }
        if (p.gu_l1acc) {
            cdef["SE_GU_L1ACC"] = "1";
        }
        const auto kc = cp(
            "se3_compute.cpp", p.gu, {KBLK, MTG, p.nk_gu, 0, 1, E, S, p.slot, 1, 1, NP, 0, p.ring_g}, cdef, p.gu_fp32);
        for (uint32_t r = 0; r < n_rd; ++r) {
            const uint32_t sg = p.sg_rd(r);
            for (uint32_t j = 0; j < p.r_; ++j) {
                const uint32_t ci = r * p.r_ + j;
                const CoreCoord gc = p.gu[ci];
                Args a;
                a.lit(pk(p.readers[r]))
                    .lit(j)
                    .lit((r % p.n_rd_sg) * p.rg + j / G)
                    .lit(head_xy_sg[sg].size())
                    .lit(0)
                    .lit(coord_sg[sg])
                    .lit(GATH)
                    .a(AddrSrc::Arena, p.x_off)
                    .lits({0, 0, 0})
                    .lit(pk(p.relays[p.rect_of(gc)]))
                    .a(AddrSrc::Words, word_rel(ci))
                    .a(AddrSrc::Arena, p.h_off)
                    .lit(j % G)
                    .lits({0, 2, 3, 0, SFREE, WORD})
                    .lits(head_xy_sg[sg])
                    .cat(dyn)
                    .lits(sgx(sg));
                add_rt(kr, gc, a);
                SetRuntimeArgs(program, kc, gc, std::vector<uint32_t>{j % G});
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
        const auto kr =
            dm("se6_drecv.cpp",
               cores,
               DataMovementProcessor::RISCV_0,
               NOC::NOC_1,
               {2,     p.h_tiles, H_TILE, H_PIECES, 16, out,    V,  S,  ngu_sg, p.nd_sg + p.n_rdn,
                HARR,  HSFREE,    GATH,   DONE,     GO, p.hbuf, MT, pw, Ht,     GATH,
                GATH1, GATH2,     E},
               yrm_def(with(dyn_def, {{"SE_E2E", "1"}}), pw));
        const auto kw =
            dm("se6_dw.cpp",
               cores,
               DataMovementProcessor::RISCV_1,
               NOC::NOC_0,
               {1, slot, w_tile, E * nblk, DW_BATCH, cfg.weights_bf8 ? 1u : 0u, E},
               dyn_def);
        const auto kc =
            cp("se6_dcompute.cpp",
               cores,
               {MTG, G, It, kd, pw, E, S, slot, ring},
               yrm_def(with(dyn_def, {{"SE_EARLY_POP", "1"}})));
        for (uint32_t d : ds) {
            const CoreCoord dc = p.down[d];
            const uint32_t sg = p.sg_dn(d);
            const uint32_t pred = p.d_pred.contains(d) ? pk(p.down[p.d_pred.at(d)]) : 0;
            const uint32_t succ =
                tail_succ.contains(d) ? tail_succ.at(d) : (p.d_succ.contains(d) ? pk(p.down[p.d_succ.at(d)]) : 0);
            Args a;
            a.a(AddrSrc::Arena, p.h_off)
                .lit(pred)
                .lit(succ)
                .lit(coord_sg[sg])
                .lit(p.dl(d) == 0 ? 1 : 0)
                .lit(ngu_sg)
                .a(AddrSrc::Y)
                .lit(p.col0s[d])
                .a(AddrSrc::Done)
                .lit(p.dl(d))
                .lits(gu_xy_sg[sg])
                .cat(dyn)
                .lits(sgx(sg));
            add_rt(kr, dc, a);
            Args w;
            w.a(AddrSrc::Down).lit(d % banks).lit((d / banks) * p.wd_region).cat(dyn).lits(sgx(sg));
            add_rt(kw, dc, w);
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
        CircularBufferConfig c(size, {{idx, fmt}});
        c.set_page_size(idx, page);
        c.set_globally_allocated_address(arena);
        c.set_address_offset(off);
        sv.arena_cbs.emplace_back(CreateCircularBuffer(program, rect_ranges(cores), c), off);
    };
    auto static_cb =
        [&](uint8_t idx, uint32_t size, const std::vector<CoreCoord>& cores, tt::DataFormat fmt, uint32_t page) {
            CircularBufferConfig c(size, {{idx, fmt}});
            c.set_page_size(idx, page);
            CreateCircularBuffer(program, rect_ranges(cores), c);
        };
    const auto wf = w_format(cfg.weights_bf8);
    // the down output CB in the arena's out region (bytes = the bfp8 double block): bfp8 tiles, or with row-major y
    // bf16 row tiles of PCD pages each (as many whole row tiles as the region holds: MT for MT <= 16)
    auto out_cb = [&](const std::vector<CoreCoord>& cores, uint32_t pw, uint32_t bytes) {
        if (!yrm) {
            arena_cb(16, p.o_off, bytes, cores, tt::DataFormat::Bfp8_b, BF8_TILE);
            return;
        }
        TT_FATAL(yrm_rows(pw) >= 1, "flat_routed_expert: row-major y: no room for one row tile");
        arena_cb(16, p.o_off, yrm_rows(pw) * pw * 2048, cores, tt::DataFormat::Float16_b, 2048);
    };
    arena_cb(0, 0, p.rd_slots * p.rg * p.slot * w_tile, p.readers, wf, w_tile);
    arena_cb(0, 0, RM_CHUNKS * 32 * p.seg, p.relays, tt::DataFormat::Float16_b, p.seg);
    arena_cb(1, p.sb_off, SB_SLOTS * MT * p.sbt * BF8_TILE, p.relays, tt::DataFormat::Bfp8_b, BF8_TILE);
    arena_cb(1, 0, p.ring_g * p.slot * w_tile, p.gu, wf, w_tile);
    arena_cb(0, p.x_off, p.x_slots * x_bytes, p.gu, tt::DataFormat::Bfp8_b, BF8_TILE);
    static_cb(3, p.hbuf * MTG * NP * H_TILE, p.gu, tt::DataFormat::Bfp8_b, H_TILE);
    static_cb(16, 2048, p.gu, tt::DataFormat::Float16_b, 2048);
    if (p.gu_l1acc) {
        arena_cb(5, p.p_off, MTG * 2 * NP * 2048, p.gu, tt::DataFormat::Float16_b, 2048);
    }
    arena_cb(2, p.h_off, p.hbuf * p.h_tiles * H_TILE, p.down, tt::DataFormat::Bfp8_b, H_TILE);
    for (const auto& [pw, ds] : dgroups) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(pw, It);
        const uint32_t ring = static_cast<uint32_t>(std::nearbyint(p.dring * (It / kd)));
        std::vector<CoreCoord> cores;
        for (uint32_t d : ds) {
            cores.push_back(p.down[d]);
        }
        arena_cb(1, 0, ring * kd * pw * w_tile, cores, wf, w_tile);
        out_cb(cores, pw, 2 * MT * pw * BF8_TILE);
    }
    if (p.rdown) {
        arena_cb(1, p.rd_off, p.ring_dr * p.slot_dr * w_tile, rdn_cores, wf, w_tile);
        arena_cb(2, p.h_off, p.hbuf * p.h_tiles * H_TILE, rdn_cores, tt::DataFormat::Bfp8_b, H_TILE);
        out_cb(rdn_cores, p.pcd_r, 2 * p.out_tiles_r * BF8_TILE);
    }
    static_cb(6, meta_bytes, arena_cores, tt::DataFormat::UInt32, meta_bytes);
    static_cb(7, 4 * dyn_half, arena_cores, tt::DataFormat::UInt32, 4 * dyn_half);

    cached_program_t cached{std::move(program), std::move(sv)};
    override_runtime_arguments(cached, cfg, t, output);
    return cached;
}

void FlatRoutedExpertProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const FlatRoutedExpertParams&, const FlatRoutedExpertInputs& t, Tensor& output) {
    using namespace tt::tt_metal;
    auto& program = cached.program;
    std::array<uint32_t, static_cast<size_t>(AddrSrc::Count)> addr{};
    addr[size_t(AddrSrc::X)] = t.x.buffer()->address();
    addr[size_t(AddrSrc::Y)] = output.buffer()->address();
    addr[size_t(AddrSrc::Counts)] = t.counts.buffer()->address();
    addr[size_t(AddrSrc::Regions)] = t.regions.buffer()->address();
    addr[size_t(AddrSrc::Ids)] = t.global_expert_ids.buffer()->address();
    addr[size_t(AddrSrc::GateUp)] = t.gate_up_weights.buffer()->address();
    addr[size_t(AddrSrc::Down)] = t.down_weights.buffer()->address();
    addr[size_t(AddrSrc::ReaderDown)] = t.reader_down_weights ? t.reader_down_weights->buffer()->address() : 0;
    addr[size_t(AddrSrc::Done)] = t.done_words.buffer()->address();
    addr[size_t(AddrSrc::Arena)] = t.arena.buffer()->address();
    addr[size_t(AddrSrc::Words)] = t.words.buffer()->address();
    addr[size_t(AddrSrc::TokenIndex)] = t.token_index ? t.token_index->buffer()->address() : 0;
    for (const auto& pa : cached.shared_variables.patches) {
        GetRuntimeArgs(program, pa.kernel, pa.core)[pa.index] = addr[size_t(pa.src)] + pa.offset;
    }
    for (const auto& [cb, off] : cached.shared_variables.arena_cbs) {
        UpdateDynamicCircularBufferAddress(program, cb, *t.arena.buffer(), off);
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
