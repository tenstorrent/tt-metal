// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_program_factory.hpp"
#include "flat_drisc.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::bringup::flat_routed_expert {

namespace factory_detail {
using tt::tt_metal::CoreCoord;
using Defines = std::map<std::string, std::string>;

constexpr uint32_t XRD_BATCH = 2, FWD = 2, DW_BATCH = 2;
// semaphore ids (16 created in order on every role core)
constexpr uint32_t DATA = 4, HARR = 5, GO = 6, DONE = 7, GATH = 8, XARR = 9, SFREE = 10, HFREE = 11, HSFREE = 12,
                   WORD = 13, GATH1 = 14, GATH2 = 15;
// h buffer 3's gather semaphore on the down chain heads (4 h buffers; ids 0..3 are the weight readers' credits, unused
// on down cores)
constexpr uint32_t GATH3 = 3;
// DRISC mode: the gate/up cores' grant counter (polled by their DRISC sender; id 1 is otherwise unused on them)
constexpr uint32_t GRANT = 1;
constexpr const char* KDIR = "ttnn/ttnn/bringup/flat_routed_expert_ttnn/device/kernels/";

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
constexpr uint32_t KBLK = kKBlk, BF8_TILE = kBf8Tile, RM_CHUNKS = kRmChunks;

uint32_t flat_combine_overlap_walk_steps(uint32_t experts_per_chip) {
    const uint32_t se_max_e = experts_per_chip <= 16 ? 16 : (experts_per_chip <= 32 ? 32 : 64);
    TT_FATAL(2 * experts_per_chip <= 2 * se_max_e, "flat_combine_overlap: more schedule entries than SE_MAX_V");
    return 2 * experts_per_chip;  // a schedule has at most E + (pinned chunks - 1) <= 2 E entries (se_dyn_pin)
}

std::vector<std::pair<std::string, std::string>> flat_schedule_defines(
    const FlatRoutedExpertConfig& cfg, const FlatRoutedExpertPlan& p) {
    TT_FATAL(p.nsg == 1, "flat_schedule_defines: one subgrid only (the overlapped plans)");
    const uint32_t se_max_e = p.E <= 16 ? 16 : (p.E <= 32 ? 32 : 64);
    std::vector<std::pair<std::string, std::string>> d{
        {"SE_RPS", std::to_string(p.mt * 32)},
        {"SE_MAX_E", std::to_string(se_max_e)},
        {"SE_GU_NREG", std::to_string(p.ring_g / p.nk_gu)}};
    if (cfg.pin) {
        d.emplace_back("SE_PIN_MIN", std::to_string(cfg.pin));
        d.emplace_back("SE_PIN_SMALL", "2");
        // pin only a dominant expert (se_dyn_pin); MIMO_FL_PIN_RATIO (read at program build; 0: always pin)
        const char* r = std::getenv("MIMO_FL_PIN_RATIO");
        d.emplace_back("SE_PIN_RATIO", r != nullptr ? std::string(r) : std::string("8"));
    }
    return d;
}

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
    const uint32_t x_blk = MT * KBLK, x_bytes = x_blk * p.x_tile;
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
    // MIMO_FL_X1=1 (prototype, read at program build): x is read from DRAM once. Rectangle 0's relays (primary +
    // helpers) read and tilize every super-block and the primary multicasts each one into BOTH rectangles
    // (se11_xmc.cpp's second-rectangle args), gated on all gate/up cores' freed words; rectangle 1's relays idle.
    // (Each rectangle's relays read all of x before: 2x the x DRAM bytes; NoC trace, worklog 8e.)
    const bool x_once = std::getenv("MIMO_FL_X1") && std::atoi(std::getenv("MIMO_FL_X1")) != 0;
    TT_FATAL(
        !x_once || (NR == 2 && p.nsg == 1 && !p.group_rect),
        "MIMO_FL_X1: two rectangles, one subgrid, no M-groups by rectangle only");
    // MIMO_FL_X2=1: x read once with both multicasters: rectangle k's relays read and tilize the super-blocks g with
    // g % 2 == k (helpers sharing them as before) and its primary multicasts each into BOTH rectangles, setting its own
    // arrival counter (rectangle 0's XARR, rectangle 1's semaphore 3); the gate/up cores merge the two streams
    // (se5_recv.cpp SE_X_RELAY2) and report their freed slots to both primaries.
    const bool x_two = std::getenv("MIMO_FL_X2") && std::atoi(std::getenv("MIMO_FL_X2")) != 0;
    TT_FATAL(
        !x_two || (NR == 2 && p.nsg == 1 && !p.group_rect && !x_once),
        "MIMO_FL_X2: two rectangles, one subgrid, no M-groups by rectangle, not with MIMO_FL_X1");
    auto word_rel = [&](uint32_t ci) {
        if (x_once || x_two) {
            return 4 * ci;  // every gate/up core's word on rectangle 0's primary
        }
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
        {"SE_DYN", "1"}, {"SE_DYN_HALF", std::to_string(dyn_half)}, {"SE_META_BYTES", std::to_string(meta_bytes)}};
    for (const auto& [k, v] : flat_schedule_defines(cfg, p)) {  // (SE_RPS, SE_MAX_E, SE_GU_NREG, pinning)
        dyn_def[k] = v;
    }
    TT_FATAL(
        dyn_def.at("SE_MAX_E") == std::to_string(se_max_e) && dyn_def.at("SE_GU_NREG") == std::to_string(nreg_gu) &&
            dyn_def.at("SE_RPS") == std::to_string(MT * 32),
        "flat_routed_expert: schedule defines out of step");
    if (p.nsg > 1) {
        dyn_def["SE_SG"] = std::to_string(p.nsg);
    }
    // perf probe (read at program build): MIMO_FL_W_NOREAD=1 - the weight readers skip their DRAM reads (garbage
    // weights), every other transfer stays: the flat expert's NoC pattern without most of its DRAM traffic
    if (std::getenv("MIMO_FL_W_NOREAD")) {
        dyn_def["SE_W_NOREAD"] = "1";
    }
    if (dn_reg) {
        dyn_def["SE_DN_REG"] = "1";
    }
    if (p.hu) {  // h units of HU row tiles (gather semaphores of up to 8 unit buffers: se_dyn.hpp)
        dyn_def["SE_HU"] = std::to_string(p.hu);
    } else if (p.hbuf == 4) {  // the 4th h buffer's gather semaphore on the down chain heads
        dyn_def["SE_GATH3"] = std::to_string(GATH3);
    }
    // perf probe (read at program build, not hashed): MIMO_FL_X_RESIDENT=1 - x never moves. The relay kernels
    // (read / tilize / multicast / helpers) return at once and the gate/up receivers treat a landing slot's x as
    // present as soon as the slot is free: the compute runs on whatever the landing ring holds (garbage outputs).
    // h pieces per sub-block down the chains (perf probe MIMO_FL_HPIECES; must divide the h tiles)
    // (32 per sub-block; h units keep the piece size: 32 HU / MT per unit, else each chain link, LINK_DEPTH pieces in
    // flight, is round-trip bound: 1-row units with 32 pieces ran bfp8 M 2048 at 655 vs 275 us per expert)
    const uint32_t H_PIECES = std::getenv("MIMO_FL_HPIECES") ? std::atoi(std::getenv("MIMO_FL_HPIECES"))
                                                             : (p.hu ? std::max(1u, 32 * p.hu / MT) : 32);
    if (const char* ld = std::getenv("MIMO_FL_LINK_DEPTH")) {  // (perf probe: h pieces in flight per chain link)
        dyn_def["SE_LINK_DEPTH"] = ld;
    }
    if (std::getenv("MIMO_FL_ZONES")) {  // device-profiler zones in the kernels that have them
        dyn_def["SE_ZONES"] = "1";
    }
    if (const char* wz = std::getenv("MIMO_FL_WAITZ")) {  // gate/up x / weight wait zones on sub-block WAITZ
        dyn_def["SE_WAITZ"] = wz;
    }
    if (std::getenv("MIMO_FL_DZ")) {  // detail zones in the down compute (a sample of sub-blocks)
        dyn_def["SE_DZ"] = "1";
    }
    if (const char* hv = std::getenv("MIMO_FL_H_VC")) {  // (perf probe: the h exchange's unicast VC)
        dyn_def["SE_H_VC"] = hv;
    }
    if (const char* fv = std::getenv("MIMO_FL_FWD_VC")) {  // (perf probe: the weight forward's unicast VC)
        dyn_def["SE_FWD_VC"] = fv;
    }
    if (std::getenv("MIMO_FL_X_RESIDENT")) {
        dyn_def["SE_X_RESIDENT"] = "1";
    }
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
            args.v.resize(cfg.cmb_rt + 4, 0);
            sv.cmb_writers.emplace_back(k, c);
        }
        return args;
    };
    Defines cmb_def = cfg.cmb_rt ? Defines{{"SE_CMB_DONE_RT", std::to_string(cfg.cmb_rt)}} : Defines{};
    if (cfg.cmb_rt && std::getenv("MIMO_FL_CMB_EARLY")) {  // (perf probe: combine released at launch, garbage)
        cmb_def["SE_CMB_EARLY"] = "1";
    }
    // the steps the writers report must be the order combine walks: flat_combine_overlap's completion order with
    // row-major y, slot order with bfp8 tiles (combine's untilizers) or the MIMO_FL_CMB_SLOT_ORDER probe
    if (cfg.cmb_rt && (!cfg.y_row_major || std::getenv("MIMO_FL_CMB_SLOT_ORDER"))) {
        cmb_def["SE_CMB_SLOT_ORDER"] = "1";
    } else if (cfg.cmb_rt) {  // steps = schedule entries, at most 2 E (flat_combine_overlap_walk_steps)
        cmb_def["SE_CMB_STEPS"] = std::to_string(flat_combine_overlap_walk_steps(E));
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
            DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct, .defines = defs});
    };
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
                .math_fidelity = MathFidelity::LoFi,
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

    // dual DRISC slots (plan.dual): the per-call switch, the largest per-expert count served by the DRAM cores
    // (MIMO_FL_DRISC_MAXCNT; default 256 tokens, 128 with reader down columns). us / expert Tensix -> dual, GLM-Flash
    // M 32 / 64 / 128 / 256 33.0 / 35.4 / 40.2 / 49.7 -> 31.5 / 33.9 / 39.3 / 49.6 (384: DRISC 58.8 vs 57.4: Tensix
    // there); K2.7 E12 M 32 / 64 / 128 58.0 / 62.1 / 68.7 -> 56.5 / 60.6 / 67.6 (256: DRISC 85.0 vs 83.7)
    Defines drisc_dual_def = dyn_def;
    if (p.dual) {
        const char* mc = std::getenv("MIMO_FL_DRISC_MAXCNT");
        drisc_dual_def["SE_DRISC_DUAL"] = "1";
        drisc_dual_def["SE_DRISC_MAXCNT"] =
            mc ? std::string(mc) : std::string(flat_drisc_pure() ? "0xFFFFFFFF" : (p.rdown ? "128" : "256"));
        if (flat_drisc_tiles() == 3) {  // switch mode: the alternate tiles serve calls with counts up to this
            const char* am = std::getenv("MIMO_FL_DRISC_ALT_MAXCNT");
            // (two tiles up to: GLM-Flash 224, K2.7 E12 (reader down columns) 384: M 256 85.2 two tiles vs 89.1 / 98.1
            // one tile, job 1 on NOC0 / NOC1)
            drisc_dual_def["SE_DRISC_ALT_MAXCNT"] = am ? std::string(am) : std::string(p.rdown ? "384" : "224");
        }
        drisc_dual_def["SE_GRANT_SEM"] = std::to_string(GRANT);
    }
    if (p.drisc && flat_drisc_tiles() == 2) {  // both tiles of a bank every call (se_drisc.hpp SD_PAIRED)
        drisc_dual_def["SE_DRISC_PAIRED_ALWAYS"] = "1";
    }
    // readers (read NOC0, forward NOC1): the chain-tail readers also compute down columns (se9_rdown)
    std::vector<CoreCoord> rdn_cores, plain;
    for (const auto& [r, tail] : p.rdn) {
        rdn_cores.push_back(p.readers[r]);
    }
    std::vector<CoreCoord> trd;  // the Tensix readers (DRISC mode: the slots it leaves to them)
    for (uint32_t r = 0; r < n_rd; ++r) {
        if (p.tensix_reader(r)) {
            trd.push_back(p.readers[r]);
        }
    }
    for (const auto& c : trd) {
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
            // perf probes (read at program build): MIMO_FL_YRM_NOWRITE - the writers drop the y writes;
            // MIMO_FL_YRM_PACKTILE - the down compute packs tiles instead of untilizing (y is garbage either way)
            if (std::getenv("MIMO_FL_YRM_NOWRITE")) {
                d["SE_YRM_NOWRITE"] = "1";
            }
            if (const char* vc = std::getenv("MIMO_FL_YRM_VC")) {
                d["SE_YRM_VC"] = vc;
            }
            if (std::getenv("MIMO_FL_YRM_PACKTILE")) {
                d["SE_YRM_PACKTILE"] = "1";
            }
            if (pw) {
                d["SE_Y_RM_ROWS"] = std::to_string(yrm_rows(pw));
            }
        }
        return d;
    };
    if (p.rdown) {
        // (dual: the DRISC slots' reader tails skip their gate/up reads on DRISC calls: their own kernel)
        for (const bool dual_set : {false, true}) {
            const auto in_set = [&](uint32_t r) { return (p.dual && p.drisc_slot(r)) == dual_set; };
            std::vector<CoreCoord> cores;
            for (const auto& [r, tail] : p.rdn) {
                if (in_set(r)) {
                    cores.push_back(p.readers[r]);
                }
            }
            if (cores.empty()) {
                continue;
            }
            Defines rdn_def =
                yrm_def(with(dual_set ? drisc_dual_def : dyn_def, {{"SE9_TRID", "1"}, {"SE_E2E", "1"}}), p.pcd_r);
            rdn_def.insert(cmb_def.begin(), cmb_def.end());
            const auto k =
                dm("se9_rdown.cpp",
                   cores,
                   DataMovementProcessor::RISCV_0,
                   NOC::NOC_0,
                   {0,        w_tile, p.rg * p.slot, E * p.nk_gu, 1,    p.slot_dr, E * p.nblk_r, 2,  p.h_tiles, p.h_tile,
                    H_PIECES, 16,     p.out_tiles_r, V,           HARR, HSFREE,    p.hbuf,       MT, p.pcd_r,   Ht,
                    S,        E,      p.rd_slots,    p.ring_dr},
                   rdn_def);
            const auto kc =
                cp("se6_dcompute.cpp",
                   cores,
                   {MTG, G, It, p.kd_r, p.pcd_r, E, S, p.slot_dr, p.ring_dr},
                   dn_def(yrm_def(with(dyn_def, {{"SE_EARLY_POP", "1"}}))),
                   cfg.down_fp32,
                   cfg.down_fp32);
            for (uint32_t i = 0; i < p.rdn.size(); ++i) {
                const auto [r, t_] = p.rdn[i];
                if (!in_set(r)) {
                    continue;
                }
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
                add_rt(k, p.readers[r], add_writer_rt(k, p.readers[r], a));
                SetRuntimeArgs(program, kc, p.readers[r], std::vector<uint32_t>{0});
            }
        }
    }
    // DRISC mode (flat_drisc.hpp): the DRAM cores' persistent senders read and forward the gate/up weights; the
    // readers only post each call's job (se_drisc_go.cpp)
    const uint32_t drisc_base = p.drisc ? flat_drisc_base() : 0;
    // perf probe MIMO_FL_DRISC_XNOC=1 (DRISC mode): the x relays' multicast on NOC1 (x reads on NOC0) and the gate/up
    // receivers on NOC0, off the DRISCs' NOC0 weight writes. Much slower (M 512 78 -> 122 us / expert: the relays sit
    // for NOC0 multicast)
    const char* xnoc_env = std::getenv("MIMO_FL_DRISC_XNOC");
    const bool xswap = p.drisc && xnoc_env != nullptr && std::atoi(xnoc_env) != 0;
    const NOC x_rd_noc = xswap ? NOC::NOC_0 : NOC::NOC_1, x_mc_noc = xswap ? NOC::NOC_1 : NOC::NOC_0;
    auto drisc_xy = [&](uint32_t r) {
        const auto v = device->virtual_core_from_logical_core(flat_drisc_logical(device, r, banks), tt::CoreType::DRAM);
        return static_cast<uint32_t>((v.x << 16) | v.y);
    };
    Args drisc_post;
    if (p.drisc) {
        TT_FATAL(p.nsg == 1 && p.r_ <= 8, "MIMO_FL_DRISC: one subgrid");
        for (const auto& [r, tail] : p.rdn) {
            TT_FATAL(!p.drisc_slot(r) || p.dual, "MIMO_FL_DRISC: a reader down column on a DRISC-only slot");
        }
        auto* mesh = dynamic_cast<distributed::MeshDevice*>(device);
        TT_FATAL(mesh != nullptr, "MIMO_FL_DRISC: a mesh device");
        flat_drisc_ensure(mesh, p.drisc_n, banks, p.rdown);
        // the job posting args (se_drisc_post.hpp): by a launcher kernel, or (dual) by slot 0's own reader
        drisc_post.lit(drisc_base)
            .a(AddrSrc::GateUp)
            .a(AddrSrc::Arena)
            .lit(p.drisc_n)
            .lit(p.r_)
            .lit(flat_drisc_mpfe(p.rdown))
            .lits({p.rg * p.slot * w_tile, p.slot * w_tile, p.rg, G, p.nk_gu, GRANT, DATA});
        for (uint32_t r = 0; r < p.drisc_n; ++r) {
            const auto [ac, aj] = flat_drisc_alt(device, r, banks);
            uint32_t axy = 0;
            if (aj != ~0u) {
                const auto v = device->virtual_core_from_logical_core(ac, tt::CoreType::DRAM);
                axy = static_cast<uint32_t>((v.x << 16) | v.y);
            }
            drisc_post.lit(drisc_xy(r))
                .lit(flat_drisc_job(r, banks))
                .lit((r / banks) * p.region_bytes)
                .lit(axy)
                .lit(aj == ~0u ? 0 : aj);
        }
        for (uint32_t r = 0; r < p.drisc_n; ++r) {
            for (uint32_t j = 0; j < p.r_; ++j) {
                drisc_post.lit(pk(p.gu[r * p.r_ + j]));
            }
        }
        TT_FATAL(
            !p.dual || std::find(plain.begin(), plain.end(), p.launcher) != plain.end(),
            "MIMO_FL_DRISC dual: the launcher must be a plain reader");
        if (!p.dual) {
            const auto k =
                dm("se_drisc_go.cpp", {p.launcher}, DataMovementProcessor::RISCV_0, NOC::NOC_0, {E}, drisc_dual_def);
            Args a;
            a.cat(drisc_post).cat(dyn);
            add_rt(k, p.launcher, a);
        }
    }
    // Tensix readers + forwarders; dual DRISC slots' ones (SE_DRISC_DUAL) stand down on DRISC calls
    for (const bool dual_set : {false, true}) {
        const auto in_set = [&](uint32_t r) { return p.tensix_reader(r) && (p.dual && p.drisc_slot(r)) == dual_set; };
        std::vector<CoreCoord> rd_cores, fwd_cores;
        for (uint32_t r = 0; r < n_rd; ++r) {
            if (in_set(r)) {
                fwd_cores.push_back(p.readers[r]);
                if (std::find(plain.begin(), plain.end(), p.readers[r]) != plain.end()) {
                    rd_cores.push_back(p.readers[r]);
                }
            }
        }
        const Defines def = dual_set ? drisc_dual_def : dyn_def;
        // dual: slot 0's reader also posts the DRISC jobs (its own kernel: SE_DRISC_LAUNCH = the post args' index)
        const uint32_t post0 = 4 + static_cast<uint32_t>(dyn.v.size()) + static_cast<uint32_t>(sgx(0).size());
        for (const bool launch : {false, true}) {
            std::vector<CoreCoord> cores;
            for (const auto& c : rd_cores) {
                if ((dual_set && c == p.launcher) == launch) {
                    cores.push_back(c);
                }
            }
            if (cores.empty()) {
                continue;
            }
            const auto k =
                dm("se_reader.cpp",
                   cores,
                   DataMovementProcessor::RISCV_0,
                   NOC::NOC_0,
                   {0, w_tile, p.rg * p.slot, 1, p.nk_gu, 0, 1, E, p.rg * p.slot, 0},
                   launch ? with(def, {{"SE_DRISC_LAUNCH", std::to_string(post0)}}) : def);
            for (uint32_t r = 0; r < n_rd; ++r) {
                if (!in_set(r) || std::find(cores.begin(), cores.end(), p.readers[r]) == cores.end()) {
                    continue;
                }
                Args a;
                a.a(AddrSrc::GateUp)
                    .lit(r % banks)
                    .lit((r / banks) * p.region_bytes)
                    .lit(0)
                    .cat(dyn)
                    .lits(sgx(p.sg_rd(r)));
                if (launch) {
                    a.cat(drisc_post);
                }
                add_rt(k, p.readers[r], a);
            }
        }
        if (!fwd_cores.empty()) {
            const auto k =
                dm("se10_fwd.cpp",
                   fwd_cores,
                   DataMovementProcessor::RISCV_1,
                   NOC::NOC_1,
                   {0, p.r_, w_tile, p.rg * p.slot, p.slot, p.ring_g, 0, DATA, p.nk_gu, p.rd_slots, FWD, E, G},
                   def);
            for (uint32_t r = 0; r < n_rd; ++r) {
                if (!in_set(r)) {
                    continue;
                }
                Args a;
                a.a(AddrSrc::Arena);  // land_addr: the gate/up cores' in1 ring (arena offset 0)
                for (uint32_t j = 0; j < p.r_; ++j) {
                    a.lit(pk(p.gu[r * p.r_ + j]));
                }
                a.cat(dyn).lits(sgx(p.sg_rd(r)));
                add_rt(k, p.readers[r], a);
            }
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
        // perf probes (read at program build; not in the program hash, so a process uses one setting):
        // MIMO_FL_XRD_SKIP=n: the relays read no x rows from DRAM (garbage tiles), MIMO_FL_XRD_BATCH: chunks per read
        // barrier (must divide RM_CHUNKS)
        if (const char* skip = std::getenv("MIMO_FL_XRD_SKIP")) {
            xrd_def["XRD_SKIP_READS"] = skip;
        }
        uint32_t xrd_batch = XRD_BATCH;
        if (const char* b = std::getenv("MIMO_FL_XRD_BATCH")) {
            xrd_batch = static_cast<uint32_t>(std::stoul(b));
            TT_FATAL(xrd_batch > 0 && RM_CHUNKS % xrd_batch == 0, "MIMO_FL_XRD_BATCH must divide {}", RM_CHUNKS);
        }
        std::vector<CoreCoord> xrel;  // the relays that read x (x1: rectangle 0's only)
        for (uint32_t idx = 0; idx < p.relays.size(); ++idx) {
            if (!x_once || idx % NR == 0) {
                xrel.push_back(p.relays[idx]);
            }
        }
        const auto kxr =
            dm("se11_xrd.cpp",
               xrel,
               DataMovementProcessor::RISCV_0,
               x_rd_noc,
               {0, t.x.logical_shape()[-1] * 2, E, MT, p.nsb, S, xrd_batch, idx_arg, p.H / t.x.logical_shape()[-1]},
               xrd_def);
        const auto ktz = cp("se11_tz.cpp", xrel, {0, 1, MT}, with(dyn_def, {{"SE_SBT", sbt}}));
        const std::vector<CoreCoord> prim(p.relays.begin(), p.relays.begin() + (x_once ? 1 : NR));
        const Defines xmc_def = with(
            dyn_def,
            {{"SE_SBT", sbt},
             {"XMC_WHOLE_SB", "1"},
             {"XMC_HELPER", "1"},
             {"SE_XNH", std::to_string(p.nh)},
             {"XMC_ZERO_WORDS", "1"}});
        // (x read once with forwarding: one kernel per primary, its half XMC_FWD_K)
        std::vector<KernelHandle> kxms;
        for (uint32_t q = 0; q < (p.xfwd ? prim.size() : 1u); ++q) {
            kxms.push_back(
                dm("se11_xmc.cpp",
                   p.xfwd ? std::vector<CoreCoord>{prim[q]} : prim,
                   DataMovementProcessor::RISCV_1,
                   x_mc_noc,
                   {1, MT, p.x_tile, p.x_slots, XARR, WORD, KBLK, E, p.nsb},
                   p.xfwd ? with(xmc_def, {{"XMC_FWD", "1"}, {"XMC_FWD_K", std::to_string(q)}}) : xmc_def));
        }
        std::vector<KernelHandle> khl;
        for (uint32_t j = 0; j < p.nh; ++j) {
            const std::vector<CoreCoord> hl(
                p.relays.begin() + NR + NR * j, p.relays.begin() + NR + NR * j + (x_once ? 1 : NR));
            const uint32_t data_sem = std::vector<uint32_t>{4, 7, 8, 10}[j];
            khl.push_back(
                dm("se13_xhelp.cpp",
                   hl,
                   DataMovementProcessor::RISCV_1,
                   x_mc_noc,
                   {1, MT, p.x_tile, p.land_slots, data_sem, 5, E, p.nsb},
                   x_two || p.xfwd ? with(dyn_def, {{"SE_SBT", sbt}, {"SE_X2", "1"}})
                                   : with(dyn_def, {{"SE_SBT", sbt}})));
        }
        const uint32_t sbb = MT * p.sbt * p.x_tile;
        for (uint32_t idx = 0; idx < p.relays.size(); ++idx) {
            const CoreCoord rl = p.relays[idx];
            const uint32_t k = idx % NR;
            if (x_once && k != 0) {
                continue;  // rectangle 1's relays: no x work
            }
            const auto [x0, x1, y0, y1] = p.rects[k];
            Args xr;
            if (x_two || p.xfwd) {  // this rectangle's half of the stream (g % 2 == k), shared with its helpers
                xr.a(AddrSrc::X).lit(NR * p.vstride).lit(NR * rl_off(idx) + k).cat(dyn).lits(sgx(k));
            } else {
                xr.a(AddrSrc::X).lit(p.vstride).lit(rl_off(idx)).cat(dyn).lits(sgx(k));
            }
            if (indexed) {
                xr.a(AddrSrc::TokenIndex);
            }
            add_rt(kxr, rl, xr);
            SetRuntimeArgs(program, ktz, rl, std::vector<uint32_t>{rl_sb(idx)});
            if (idx < NR) {
                Args xm;
                xm.a(AddrSrc::Arena, p.x_off)
                    .lit(pk(xswap ? CoreCoord(x1, y1) : CoreCoord(x0, y0)))  // (NOC1 multicast: start is the far corner)
                    .lit(pk(xswap ? CoreCoord(x0, y0) : CoreCoord(x1, y1)))
                    .lit((x1 - x0 + 1) * (y1 - y0 + 1))
                    .a(AddrSrc::Words)
                    .lit(x_once || x_two ? ngu : static_cast<uint32_t>(rect_cores[k].size()))
                    .lits(
                        {V * p.nsb,
                         x_two && k == 1 ? 3u : XARR,
                         x_two ? NR : 1u,
                         x_two ? k : 0u,
                         p.group_rect ? ((idx * MTG) | ((idx + 1) * MTG << 8)) : 0});
                if (x_once || x_two) {  // the second rectangle this primary multicasts into (se11_xmc.cpp RT 11-13)
                    const auto [u0, u1, v0, v1] = p.rects[x_two ? 1 - k : 1];
                    xm.lit((u1 - u0 + 1) * (v1 - v0 + 1))
                        .lit(pk(xswap ? CoreCoord(u1, v1) : CoreCoord(u0, v0)))
                        .lit(pk(xswap ? CoreCoord(u0, v0) : CoreCoord(u1, v1)));
                } else {
                    xm.lits({0, 0, 0});
                }
                xm
                    .lit(pk(p.relays[idx + NR]))
                    .a(AddrSrc::Arena, p.land_off)
                    .lit(p.land_slots);
                for (uint32_t j = 1; j < p.nh; ++j) {
                    xm.lit(pk(p.relays[NR + NR * j + idx]));
                }
                xm.cat(dyn).lits(sgx(k));
                if (p.xfwd) {  // the forward ring, the other primary, the ring's slots (se11_xmc.cpp XMC_FWD)
                    xm.a(AddrSrc::Arena, p.fw_off).lit(pk(p.relays[1 - idx])).lit(p.fw_slots);
                }
                add_rt(kxms[p.xfwd ? idx : 0], rl, xm);
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
                if (x_two || p.xfwd) {  // (se13_xhelp.cpp SE_X2: the primary's own stream: every NR-th from k)
                    hl.lit(NR).lit(k);
                }
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
        // DRISC mode: the DRISC slots' receivers count their grants locally (SE_DRISC), the others credit their reader
        Defines recv_def = with(dyn_def, {{"SE_GU_ONLY", "1"}, {"SE_X_RELAY", "1"}, {"SE_NO_PARTNER", "1"}});
        if (x_two) {  // two x streams alternating by super-block (runs of SBT / KBLK blocks)
            recv_def["SE_X_RELAY2"] = std::to_string(p.sbt / KBLK);
            recv_def["SE_X_NRELAY"] = "2";
        }
        std::vector<CoreCoord> gu_t, gu_d;
        for (uint32_t ci = 0; ci < ngu; ++ci) {
            (p.drisc_slot(ci / p.r_) ? gu_d : gu_t).push_back(p.gu[ci]);
        }
        const std::vector<uint32_t> recv_ct = {
            0,         x_blk,   1,    p.slot, E * p.nk_gu, V,        p.ring_g, 16,    1,     3,      2,    MTG,
            p.h_tile,  ngu_sg,  DATA, HARR,   GO,          DONE,     KBLK,     HARR,  SFREE, p.hbuf, XARR, HFREE,
            p.x_slots, x_bytes, S,    NP,     HSFREE,      H_PIECES, p.nk_gu,  GATH1, GATH2, 1,      G,    E};
        const NOC recv_noc = xswap ? NOC::NOC_0 : NOC::NOC_1;
        const KernelHandle kr_t =
            gu_t.empty() ? 0 : dm("se5_recv.cpp", gu_t, DataMovementProcessor::RISCV_0, recv_noc, recv_ct, recv_def);
        Defines recv_d_def = recv_def;
        if (p.dual) {  // both paths (the reader's credits / the local grant counter), chosen per call
            for (const auto& [k, v] : drisc_dual_def) {
                recv_d_def[k] = v;
            }
        } else {
            recv_d_def["SE_DRISC"] = "1";
        }
        const KernelHandle kr_d =
            gu_d.empty() ? 0
                         : dm("se5_recv.cpp", gu_d, DataMovementProcessor::RISCV_0, recv_noc, recv_ct, recv_d_def);
        // perf probe (read at program build): MIMO_FL_GU_FULL_SYNC=1 - fp32 gate/up in a full-sync DST (8 tiles: the
        // whole MT x 2 NP sub-block in one K loop, no row passes; pack no longer overlaps math, once per sub-block)
        const bool gu_full_sync = p.gu_fp32 && p.gu_rp && (p.gu_full_sync || std::getenv("MIMO_FL_GU_FULL_SYNC"));
        const bool gu_rp = p.gu_rp && !gu_full_sync;
        Defines cdef = with(
            dyn_def,
            {{"SE_DST_TILES", std::to_string(gu_rp ? 4 : p.dst_tiles)},
             {"SE_GU_ONLY", "1"},
             {"SE_ACT", std::to_string(cfg.activation)},
             {"SE_XMT", std::to_string(MT)}});
        if (gu_rp) {
            cdef["SE_GU_RP"] = std::to_string(std::max(1u, 4 / (2 * NP)));
            cdef["SE_XSLOTS"] = std::to_string(p.x_slots);
        }
        if (p.gu_l1acc) {
            cdef["SE_GU_L1ACC"] = "1";
        }
        if (std::getenv("MIMO_FL_NO_ACT")) {  // probe: no gate/up activation (wrong h), the bare matmul + pack
            cdef["SE_NO_ACT"] = "1";
        }
        const auto kc =
            cp("se3_compute.cpp",
               p.gu,
               {KBLK, MTG, p.nk_gu, 0, 1, E, S, p.slot, 1, 1, NP, 0, p.ring_g},
               cdef,
               p.gu_fp32,
               gu_full_sync);
        for (uint32_t r = 0; r < n_rd; ++r) {
            const uint32_t sg = p.sg_rd(r);
            for (uint32_t j = 0; j < p.r_; ++j) {
                const uint32_t ci = r * p.r_ + j;
                const CoreCoord gc = p.gu[ci];
                Args a;
                // the forwarder and its credit word: the reader's semaphore j, or the DRISC's L1 word j
                const bool dr = p.drisc_slot(r);  // (dual: RT 0 / 1 are the reader's, the grant counter is SE_GRANT_SEM)
                a.lit(dr && !p.dual ? drisc_xy(r) : pk(p.readers[r]))
                    .lit(dr && !p.dual ? GRANT : j)
                    .lit((r % p.n_rd_sg) * p.rg + j / G)
                    .lit(head_xy_sg[sg].size())
                    .lit(0)
                    .lit(coord_sg[sg])
                    .lit(GATH)
                    .a(AddrSrc::Arena, p.x_off)
                    .lits({0, 0, 0})
                    .lit(pk(p.relays[x_once || x_two ? 0 : p.rect_of(gc)]))
                    .a(AddrSrc::Words, word_rel(ci))
                    .a(AddrSrc::Arena, p.h_off)
                    .lit(j % G)
                    .lit(x_two ? pk(p.relays[1]) : 0)  // (RT 15: X2, the second primary for the freed reports)
                    .lits({2, 3, 0, SFREE, WORD})
                    .lits(head_xy_sg[sg])
                    .cat(dyn)
                    .lits(sgx(sg));
                add_rt(dr ? kr_d : kr_t, gc, a);
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
        // the down side's row tiles per h buffer: the sub-block, or the h unit
        const uint32_t MTD = p.hu ? p.hu : MT, MTGD = p.hu ? p.hu : MTG;
        const uint32_t ring = static_cast<uint32_t>(std::nearbyint(p.dring * nblk)), out = MTD * pw;
        std::vector<CoreCoord> cores;
        for (uint32_t d : ds) {
            cores.push_back(p.down[d]);
        }
        // perf probe (read at program build): MIMO_FL_DN_NOC0=1 - the down cores' h chain / y writes on NOC0, their
        // weight reads on NOC1 (default: the other way round)
        const bool dn_noc0 = std::getenv("MIMO_FL_DN_NOC0") != nullptr;
        // y writes from the NCRISC on its NOC (se6_dw.cpp) instead of the BRISC, whose NOC carries the h chain (one
        // subgrid): bfp8 x + bf16 h at M 512 / 2048 / 5120 -9 / -9 / -11%, bf16 x + h -4 / -3 / -4% (the doubled h
        // traffic the y writes contended with), bfp8 / bfp8 -4 / -1 / -0%; +1% (~0.4 us) at M <= 64 (perf probe
        // MIMO_FL_Y_NC=0: the BRISC writes y). Not with reader down columns (H > 4992): there the NCRISC's NOC also
        // carries the larger down weight stream, H 7168 / 6144 bf16 x + h at M 2048 +7 / +6%
        const char* ync_env = std::getenv("MIMO_FL_Y_NC");
        const bool y_nc = yrm && p.nsg == 1 && (ync_env ? std::atoi(ync_env) != 0 : !p.rdown);
        const Defines ync = y_nc ? Defines{{"SE_Y_NC", "1"}} : Defines{};
        auto plus = [](Defines a, const Defines& b) {
            a.insert(b.begin(), b.end());
            return a;
        };
        const auto kr =
            dm("se6_drecv.cpp",
               cores,
               DataMovementProcessor::RISCV_0,
               dn_noc0 ? NOC::NOC_0 : NOC::NOC_1,
               {2,     p.h_tiles, p.h_tile, H_PIECES, 16, out,    V,   S,  ngu_sg, p.nd_sg + p.n_rdn,
                HARR,  HSFREE,    GATH,     DONE,     GO, p.hbuf, MTD, pw, Ht,     GATH,
                GATH1, GATH2,     E},
               plus(plus(yrm_def(with(dyn_def, {{"SE_E2E", "1"}}), pw), ync), y_nc ? Defines{} : cmb_def));
        const auto kw =
            dm("se6_dw.cpp",
               cores,
               DataMovementProcessor::RISCV_1,
               dn_noc0 ? NOC::NOC_1 : NOC::NOC_0,
               {1, slot, w_tile, E * nblk, DW_BATCH, cfg.weights_bf8 ? 1u : 0u, E, 0, 0, 0, MTD, pw, Ht},
               plus(y_nc ? plus(plus(yrm_def(dyn_def, pw), ync), cmb_def) : dyn_def,
                    p.dw_rot ? Defines{{"SE_DW_ROT", std::to_string(banks)}} : Defines{}));
        const auto kc =
            cp("se6_dcompute.cpp",
               cores,
               {MTGD, G, It, kd, pw, E, S, slot, ring},
               dn_def(yrm_def(with(dyn_def, {{"SE_EARLY_POP", "1"}}))),
               cfg.down_fp32,
               cfg.down_fp32);
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
            add_rt(kr, dc, y_nc ? a : add_writer_rt(kr, dc, a));
            Args w;
            w.a(AddrSrc::Down)
                .lit(d % banks)
                .lit(p.dw_rot ? d * p.wd_rot_region : (d / banks) * p.wd_region)
                .cat(dyn)
                .lits(sgx(sg));
            if (y_nc) {
                w.a(AddrSrc::Y).lit(p.col0s[d]);
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
    // x / h tile formats (the plan's precision probes: bf16 tiles instead of bfp8)
    const auto x_fmt = p.x_tile == BF8_TILE ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const auto h_fmt = p.h_tile == BF8_TILE ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
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
    if (!trd.empty()) {
        arena_cb(0, 0, p.rd_slots * p.rg * p.slot * w_tile, trd, wf, w_tile);
    }
    arena_cb(0, 0, RM_CHUNKS * 32 * p.seg, p.relays, tt::DataFormat::Float16_b, p.seg);
    arena_cb(1, p.sb_off, p.sb_slots * MT * p.sbt * p.x_tile, p.relays, x_fmt, p.x_tile);
    arena_cb(1, 0, p.ring_g * p.slot * w_tile, p.gu, wf, w_tile);
    arena_cb(0, p.x_off, p.x_slots * x_bytes, p.gu, x_fmt, p.x_tile);
    if (p.hl_in_arena) {  // bf16 x / h: gate/up h_local in the arena (out of the program-wide static CB region)
        arena_cb(3, p.hl_off, p.hl_slots * MTG * NP * p.h_tile, p.gu, h_fmt, p.h_tile);
    } else {
        static_cb(3, p.hl_slots * MTG * NP * p.h_tile, p.gu, h_fmt, p.h_tile);
    }
    static_cb(16, 2048, p.gu, tt::DataFormat::Float16_b, 2048);
    if (p.gu_l1acc) {
        arena_cb(5, p.p_off, MTG * 2 * NP * 2048, p.gu, tt::DataFormat::Float16_b, 2048);
    }
    arena_cb(2, p.h_off, p.hbuf * p.h_tiles * p.h_tile, p.down, h_fmt, p.h_tile);
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
        arena_cb(2, p.h_off, p.hbuf * p.h_tiles * p.h_tile, rdn_cores, h_fmt, p.h_tile);
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

}  // namespace ttnn::operations::bringup::flat_routed_expert
