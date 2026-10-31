// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_program.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace combine_program_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using RtArgs = std::vector<std::variant<uint32_t, Buffer*>>;

constexpr const char* kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/";

CoreRangeSet cores_of(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> r;
    r.reserve(cores.size());
    for (const auto& c : cores) {
        r.emplace_back(c, c);
    }
    return CoreRangeSet(r);
}

bool same_cell(const CoreCoord& a, const CoreCoord& b) { return a.x == b.x && a.y == b.y; }

struct ChipPort {
    Dir dir;
    uint32_t li;
    CoreCoord core;
};

std::vector<CoreCoord> port_cells_of(const PortCores& pc, ChipCoord ch) {
    std::vector<CoreCoord> v;
    for (const auto& [k, c] : pc) {
        if (k.chip == ch && std::none_of(v.begin(), v.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
            v.push_back(c);
        }
    }
    return v;
}

CBDescriptor cb(uint8_t index, const CoreRangeSet& cores, uint32_t total, tt::DataFormat fmt, uint32_t page) {
    CBDescriptor d;
    d.total_size = total;
    d.core_ranges = cores;
    d.format_descriptors.push_back(CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page});
    return d;
}

KernelDescriptor dm_kernel(
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    bool riscv0,
    KernelDescriptor::Defines defines = {}) {
    KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.defines = std::move(defines);
    // dedicated NoCs, fixed per RISC: RISCV_0 on NoC1, RISCV_1 on NoC0
    k.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor =
            riscv0 ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = riscv0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0};
    return k;
}

}  // namespace combine_program_detail

std::shared_ptr<const CombinePlan> make_combine_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, MoEBlockResources& resources, const CombinePlanInputs& in) {
    using namespace combine_program_detail;
    const auto& cfg = resources.config();
    const auto& mesh = resources.mesh_plan();
    auto plan = std::make_shared<CombinePlan>();
    plan->topo = mesh.topo;
    plan->mesh = mesh;
    plan->L = mesh.num_links;
    plan->tie = cfg.tie_rule;
    plan->tuning = cfg.combine_tuning;
    plan->snap_bits = cfg.snap_bits();
    plan->S = in.S;
    plan->W = in.W;
    const uint32_t N = mesh.topo.num_chips();
    TT_FATAL(N <= kMaxChips, "routed_experts_combine: {} chips > {}", N, kMaxChips);
    const uint32_t piece = in.piece_max.value_or(std::max<uint32_t>(kPieceMax, 2 * in.W));
    plan->pc = row_pieces(in.W, piece);
    // (the L1 hand-off's y writers write a row's columns at slot + column offset and its trailer at slot + row bytes:
    // one piece a row, and a fabric payload >= row + 32 B, else the combine waits for trailers that never come)
    TT_FATAL(
        plan->pc.NP == 1,
        "routed_experts_combine: {} B rows need one piece (piece_max {} < row)",
        plan->pc.row_bytes,
        piece);
    TT_FATAL(
        resources.fabric_max_payload_bytes() >= plan->pc.row_bytes + 32,
        "routed_experts_combine: the fabric payload {} B < a row + its trailer ({} B)",
        resources.fabric_max_payload_bytes(),
        plan->pc.row_bytes + 32);

    // the cells: the flat plan's free ones; the port count the ports-apart accumulators lose (combine_port_count)
    const auto grid = mesh_device.compute_with_storage_grid_size();
    std::vector<CoreCoord> pool = in.cells;
    if (pool.empty()) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                pool.emplace_back(x, y);
            }
        }
    }
    const auto place = cfg.combine_tuning.place;
    uint32_t n_ports = 0;
    {
        const auto& pp = resources.ports(place, pool);
        std::vector<CoreCoord> uni;
        for (const auto& [k, c] : pp) {
            if (std::none_of(uni.begin(), uni.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
                uni.push_back(c);
            }
        }
        if (uni.size() >= pool.size()) {
            n_ports = uni.size();
        } else {
            for (const auto& [ch, _] : mesh.chips) {
                n_ports = std::max<uint32_t>(n_ports, port_cells_of(pp, ch).size());
            }
        }
    }
    const auto au = combine_auto(
        N, in.S, in.W, pool.size(), n_ports, in.arena_bytes, in.depth, piece, in.n_acc.value_or(kAccMax), in.mode);
    TT_FATAL(
        au.fits,
        "routed_experts_combine: does not fit ({} cells, {} KB): {}",
        pool.size(),
        in.arena_bytes >> 10,
        au.why);
    TT_FATAL(
        !in.n_acc || au.best->n_acc == *in.n_acc,
        "routed_experts_combine: {} accumulators, {} planned",
        au.best->n_acc,
        *in.n_acc);
    plan->choice = *au.best;
    plan->fused = au.best->acc_on_ports;
    plan->perchip = !plan->fused;
    const uint32_t n_acc = au.best->n_acc;

    // ports and accumulators (sparse_combine_rf_prepare)
    if (plan->fused) {  // the accumulator cells: every chip's nearest-to-Ethernet port cells, then the pool's first
                        // others
        const auto& free_ports = resources.ports(place, pool);
        std::vector<CoreCoord> aset;
        for (const auto& c : pool) {
            if (std::any_of(
                    free_ports.begin(), free_ports.end(), [&](const auto& kv) { return same_cell(kv.second, c); })) {
                aset.push_back(c);
            }
        }
        TT_FATAL(
            aset.size() <= n_acc, "acc_on_ports: {} port cells over the chips > {} accumulators", aset.size(), n_acc);
        for (const auto& c : pool) {
            if (aset.size() >= n_acc) {
                break;
            }
            if (std::none_of(aset.begin(), aset.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
                aset.push_back(c);
            }
        }
        plan->port_core = resources.ports(place, aset);
        for (const auto& [ch, _] : mesh.chips) {
            plan->accs[ch] = aset;
        }
    } else {
        plan->port_core = resources.ports(place, pool);
        std::vector<CoreCoord> never;
        for (const auto& c : pool) {
            if (std::none_of(plan->port_core.begin(), plan->port_core.end(), [&](const auto& kv) {
                    return same_cell(kv.second, c);
                })) {
                never.push_back(c);
            }
        }
        TT_FATAL(!never.empty(), "routed_experts_combine: no cell free of every chip's ports (the collector's)");
        size_t A_ = SIZE_MAX;
        for (const auto& [ch, _] : mesh.chips) {
            const auto own = port_cells_of(plan->port_core, ch);
            std::vector<CoreCoord> v{never[0]};
            for (const auto& c : pool) {
                if (v.size() >= n_acc) {
                    break;
                }
                if (!same_cell(c, never[0]) &&
                    std::none_of(own.begin(), own.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
                    v.push_back(c);
                }
            }
            A_ = std::min(A_, v.size());
            plan->accs[ch] = std::move(v);
        }
        for (auto& [ch, v] : plan->accs) {
            v.resize(A_);
        }
    }
    const uint32_t A = plan->accs.begin()->second.size();
    TT_FATAL(A <= kAccMax, "routed_experts_combine: {} accumulators > {}", A, kAccMax);
    plan->A = A;
    plan->MAXT = (in.S + A - 1) / A;
    plan->collector = plan->accs.begin()->second.at(0);
    for (const auto& [ch, v] : plan->accs) {
        for (const auto& c : v) {
            if (std::none_of(plan->acc_cells_union.begin(), plan->acc_cells_union.end(), [&](const CoreCoord& x) {
                    return same_cell(x, c);
                })) {
                plan->acc_cells_union.push_back(c);
            }
        }
    }

    // the layout (the arena budget's sizing, as sparse_combine_rf_prepare)
    plan->st = state_layout(N, A);
    const uint32_t SLOT = plan->pc.slot, ISLOT = SLOT;
    const uint32_t acc_cbs = acc_cb_bytes(in.S, A, plan->pc.NT);
    const uint32_t QP0 = au.best->inj_slots, keep_slots = au.best->local_keep;
    const int64_t keep = plan->fused ? static_cast<int64_t>(keep_slots + std::min(32u, QP0)) * ISLOT
                                     : static_cast<int64_t>(keep_slots) * ISLOT;
    const int64_t fit = (static_cast<int64_t>(in.arena_bytes) - plan->st.RINGS - acc_cbs - keep) / SLOT;
    const int64_t PSL = std::min<int64_t>(static_cast<int64_t>(in.depth) * (N - 1), fit);
    TT_FATAL(
        PSL >= static_cast<int64_t>(N) - 1, "routed_experts_combine: no room for one landing slot per source chip");
    TT_FATAL(PSL < 4096, "routed_experts_combine: a ring's base / depth are 12-bit fields");
    plan->PSL = static_cast<uint32_t>(PSL);
    plan->D = N > 1 ? plan->PSL / (N - 1) : 0;
    plan->cred = credit_policy(N, in.S, A, plan->PSL, plan->tuning);
    plan->la = std::min<uint32_t>(plan->tuning.la, 64);
    plan->far = plan->tuning.la > 1 ? plan->tuning.far : 0;
    const uint32_t acc_area = plan->st.RINGS + plan->PSL * SLOT;
    plan->loc_off = align64(acc_area);
    uint32_t QP = QP0, QL = 0;
    if (plan->fused) {  // (local ring up to 16 slots, then the injection ring up to QP, from what is left)
        const int64_t a_end = align64(acc_area) + acc_cbs;
        QL = static_cast<uint32_t>(std::min<int64_t>(16, (in.arena_bytes - a_end - std::min(32u, QP) * ISLOT) / ISLOT));
        QP = static_cast<uint32_t>(
            std::min<int64_t>(QP, (in.arena_bytes - a_end - static_cast<int64_t>(QL) * ISLOT) / ISLOT));
        TT_FATAL(QL >= 2 && QP >= 8, "acc_on_ports: local ring {} / injection ring {} slots", QL, QP);
    } else {
        QL = std::min<uint32_t>(16, (in.arena_bytes - align64(acc_area) - acc_cbs) / ISLOT);
        TT_FATAL(QL >= 2, "routed_experts_combine: no room for the accumulators' local ring");
    }
    plan->QP = QP;
    plan->QL = QL;
    plan->acc_arena = plan->loc_off + QL * ISLOT;
    plan->lut_off = align64(plan->st.RINGS);
    plan->lut_bytes = plan->perchip ? align64(2 * N * A) : 0;
    plan->port_arena = plan->fused ? plan->acc_arena : plan->lut_off + plan->lut_bytes;
    plan->inj_off = plan->port_arena;

    // device-derived
    const auto& topo = mesh.topo;
    plan->chip_ids.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        const auto ch = topo.coord(i);
        const auto node = mesh_device.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(ch.r, ch.c));
        plan->chip_ids[i] = node.chip_id;
        if (i == 0) {
            plan->mesh_id = *node.mesh_id;
        }
    }
    if (plan->perchip) {
        plan->lut.assign(plan->lut_bytes / 2, 0);
        for (const auto& [ch, v] : plan->accs) {
            for (uint32_t o = 0; o < v.size(); ++o) {
                const auto vc = mesh_device.worker_core_from_logical_core(v[o]);
                TT_FATAL(
                    vc.x < 256 && vc.y < 256, "routed_experts_combine: virtual core {} beyond the LUT's bytes", vc);
                plan->lut[topo.index(ch) * A + o] = static_cast<uint16_t>(vc.x | vc.y << 8);
            }
        }
    }
    return plan;
}

void add_combine_arena_cbs(tt::tt_metal::ProgramDescriptor& desc, const CombinePlan& plan, ChipCoord ch) {
    using namespace combine_program_detail;
    std::vector<CoreCoord> pcells;
    for (const auto& [k, c] : plan.port_core) {
        if (k.chip == ch &&
            std::none_of(pcells.begin(), pcells.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
            pcells.push_back(c);
        }
    }
    const auto& accs = plan.accs.at(ch);
    std::vector<CoreCoord> only_ports, only_accs;
    for (const auto& c : pcells) {
        if (std::none_of(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
            only_ports.push_back(c);
        }
    }
    // (arena CBs: per role on disjoint cells, each the first CB of its cores -> the same base A0 everywhere)
    if (!only_ports.empty()) {
        desc.cbs.push_back(
            cb(kCbArena, cores_of(only_ports), plan.port_arena, tt::DataFormat::UInt32, plan.port_arena));
    }
    desc.cbs.push_back(cb(kCbArena, cores_of(accs), plan.acc_arena, tt::DataFormat::UInt32, plan.acc_arena));
    // (the injection ring right after the arena: A0 + inj_off on every port core - the y writers address it)
    desc.cbs.push_back(cb(kCbInj, cores_of(pcells), plan.QP * plan.pc.slot, tt::DataFormat::UInt32, plan.pc.slot));
}

HandoffArgs add_combine_program(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombinePlan& plan,
    ChipCoord ch,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& buf,
    const FlatHandoff& flat) {
    using namespace combine_program_detail;
    const auto& topo = plan.topo;
    const uint32_t N = topo.num_chips(), L = plan.L, A = plan.A;
    const auto& st = plan.st;
    const auto& pc = plan.pc;
    const uint32_t SLOT = pc.slot, ISLOT = pc.slot;
    TT_FATAL(flat.writers.size() <= 32, "routed_experts_combine: {} y writers > 32", flat.writers.size());
    auto virt = [&](const CoreCoord& c) { return mesh_device.worker_core_from_logical_core(c); };
    auto idx = [&](ChipCoord c) { return topo.index(c); };
    auto fh = [&](ChipCoord s, ChipCoord d) { return *first_hop(s, d, topo, plan.tie); };
    auto node = [&](ChipCoord c) {
        return mesh_device.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(c.r, c.c));
    };

    // this chip's ports: (direction, link) order = port ids (port_luts numbering)
    std::vector<ChipPort> my;
    for (const Dir d : kDirs) {
        for (uint32_t li = 0; li < L; ++li) {
            auto it = plan.port_core.find(PortKey{ch, d, li});
            if (it != plan.port_core.end()) {
                my.push_back({d, li, it->second});
            }
        }
    }
    const uint32_t n_pid = my.size();
    std::vector<CoreCoord> pcells;
    for (const auto& p : my) {
        if (std::none_of(pcells.begin(), pcells.end(), [&](const CoreCoord& x) { return same_cell(x, p.core); })) {
            pcells.push_back(p.core);
        }
    }
    const auto& accs = plan.accs.at(ch);
    const auto pset = cores_of(pcells), aset = cores_of(accs);
    auto is_port = [&](const CoreCoord& c) {
        return std::any_of(pcells.begin(), pcells.end(), [&](const CoreCoord& x) { return same_cell(x, c); });
    };

    // circular buffers (the arena and the injection rings: add_combine_arena_cbs); CB_TAB on both kinds
    const uint32_t NT = pc.NT;
    {
        std::vector<CoreCoord> tab = accs;
        for (const auto& c : pcells) {
            if (!std::any_of(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
                tab.push_back(c);
            }
        }
        desc.cbs.push_back(cb(kCbTab, cores_of(tab), kTabCbBytes, tt::DataFormat::UInt32, kTabCbBytes));
    }
    desc.cbs.push_back(cb(kCbRow, aset, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile));
    desc.cbs.push_back(cb(kCbMeta, aset, 2 * 32, tt::DataFormat::UInt32, 32));
    desc.cbs.push_back(cb(kCbAcc, aset, plan.MAXT * NT * 4096, tt::DataFormat::Float32, 4096));
    desc.cbs.push_back(cb(kCbOut, aset, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile));
    desc.cbs.push_back(cb(kCbW, aset, 2 * kTile, tt::DataFormat::Float16_b, kTile));

    // the zero gate: accumulator 0 releases every combine core once the zeroing kernels counted in (ports' senders
    // unless they share an accumulator's core, the accumulators' readers, the flat expert's y writers)
    const auto gate = virt(accs.at(0));
    std::vector<CoreCoord> comb = accs;
    for (const auto& c : pcells) {
        if (!std::any_of(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, c); })) {
            comb.push_back(c);
        }
    }
    const uint32_t nz = (plan.fused ? 0u : static_cast<uint32_t>(pcells.size())) + A + flat.writers.size();

    std::vector<uint32_t> wxy;
    for (const auto& w : flat.writers) {
        const auto v = virt(w);
        wxy.push_back(v.x);
        wxy.push_back(v.y);
    }
    const uint32_t W = flat.writers.size();
    // the RISC split (SPC_SNOC): the sender (and the accumulator writer) on RISCV_0 / NoC1 by default
    const bool s_riscv0 = plan.tuning.snoc != 0, r_riscv0 = !s_riscv0;
    const uint32_t s_slot = s_riscv0 ? 0 : 1, r_slot = 1 - s_slot;  // (bar call-count slots per RISC)

    std::vector<uint32_t> o_ct;
    tt::tt_metal::TensorAccessorArgs(*buf.out).append_to(o_ct);

    // port readers (ports apart only: fused ports run the hand-off on the sender)
    if (!plan.fused) {
        auto k = dm_kernel(
            "dataflow/rf_port_reader.cpp",
            pset,
            {kCbInj, kCbArena, kCbTab, st.MISC, flat.words_offset, r_slot},
            r_riscv0);
        for (const auto& p : my) {
            const uint32_t pid = &p - my.data();
            RtArgs a{buf.plan, buf.bar, W, pid, plan.QP};
            for (const uint32_t v : wxy) {
                a.emplace_back(v);
            }
            k.emplace_runtime_args(p.core, a);
        }
        desc.kernels.push_back(std::move(k));
    }

    // port senders
    {
        std::vector<uint32_t> ct = {
            kCbInj,
            pc.row_bytes,
            SLOT,
            plan.PSL,
            A,
            N,
            L,
            ISLOT,
            kCbArena,
            st.CRED,
            st.USED,
            st.RET,
            st.MISC,
            st.HDRS,
            st.RINGS,
            st.LAYR,
            st.LAYO,
            st.U0,
            plan.cred.cred_thr,
            plan.cred.cred_idle,
            plan.cred.cred_age,
            plan.far,
            pc.NP,
            pc.PD,
            pc.PS,
            kNumHeaders,
            kHeaderStride,
            plan.la,
            1u /*BAR*/,
            plan.tuning.gbgate ? 1u : 0u,
            kMiscRtot,
            kMiscRdone,
            s_slot,
            plan.fused ? 0u : 1u /*ZERO*/,
            flat.words_offset,
            plan.lut_off,
            plan.lut_bytes};
        KernelDescriptor::Defines defs;
        if (plan.fused) {
            defs.emplace_back("REC_FUSED", "1");
            ct.push_back(kCbOut);
            ct.push_back(NT);
            ct.insert(ct.end(), o_ct.begin(), o_ct.end());
        }
        if (plan.perchip) {
            defs.emplace_back("REC_ACC_LUT", "1");
        }
        auto k = dm_kernel("dataflow/rf_port_sender.cpp", pset, std::move(ct), s_riscv0, std::move(defs));
        std::vector<uint32_t> acc_xy;
        for (const auto& c : accs) {
            const auto v = virt(c);
            acc_xy.push_back(v.x);
            acc_xy.push_back(v.y);
        }
        for (const auto& p : my) {
            const uint32_t pid = &p - my.data();
            RtArgs a{
                static_cast<uint32_t>(plan.mesh_id),
                idx(ch),
                p.li,
                buf.bar,
                static_cast<uint32_t>(gate.x),
                static_cast<uint32_t>(gate.y),
                buf.plan,
                pid};
            for (uint32_t i = 0; i < N; ++i) {
                a.emplace_back(plan.chip_ids[i]);
            }
            if (plan.perchip) {
                a.emplace_back(buf.lut);
            } else {
                for (const uint32_t v : acc_xy) {
                    a.emplace_back(v);
                }
            }
            for (uint32_t i = 0; i < N; ++i) {  // per source chip: the port on it that feeds me on my link
                const auto s = topo.coord(i);
                if (s != ch && fh(ch, s) == p.dir) {
                    const auto v = virt(plan.port_core.at(PortKey{s, fh(s, ch), p.li}));
                    a.emplace_back(static_cast<uint32_t>(v.x));
                    a.emplace_back(static_cast<uint32_t>(v.y));
                } else {
                    a.emplace_back(0u);
                    a.emplace_back(0u);
                }
            }
            if (plan.far) {  // (hops to each home chip)
                for (uint32_t i = 0; i < N; ++i) {
                    a.emplace_back(static_cast<uint32_t>(route(ch, topo.coord(i), topo, plan.tie).size()));
                }
            }
            a.emplace_back(buf.hcache);
            a.emplace_back(static_cast<uint32_t>(p.dir) * 2 + p.li);
            {
                std::vector<uint32_t> fab;
                const auto& mp = *plan.mesh.chips.at(ch)[static_cast<size_t>(p.dir)];
                tt::tt_fabric::append_fabric_connection_rt_args(
                    node(ch), node(mp.peer), mp.links.at(p.li), desc, p.core, fab);
                for (const uint32_t v : fab) {
                    a.emplace_back(v);
                }
            }
            if (plan.fused) {  // (the hand-off receiver's args, then the accumulator writer's)
                const auto ai = static_cast<uint32_t>(
                    std::find_if(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, p.core); }) -
                    accs.begin());
                a.emplace_back(W);
                a.emplace_back(pid);
                a.emplace_back(plan.QP);
                a.emplace_back(buf.out);
                a.emplace_back(ai < plan.S ? (plan.S - ai + A - 1) / A : 0u);
                a.emplace_back(ai);
                for (const uint32_t v : wxy) {
                    a.emplace_back(v);
                }
            }
            {  // barrier: the port facing each other chip increments every port core there
                std::vector<uint32_t> bt;
                if (p.li == 0) {
                    for (uint32_t i = 0; i < N; ++i) {
                        const auto s = topo.coord(i);
                        if (s == ch || fh(ch, s) != p.dir) {
                            continue;
                        }
                        auto cells = port_cells_of(plan.port_core, s);
                        std::sort(cells.begin(), cells.end(), [](const CoreCoord& x, const CoreCoord& y) {
                            return std::tie(x.x, x.y) < std::tie(y.x, y.y);
                        });
                        for (const auto& c : cells) {
                            const auto v = virt(c);
                            bt.insert(bt.end(), {i, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                        }
                    }
                }
                a.emplace_back(static_cast<uint32_t>(bt.size() / 3));
                for (const uint32_t v : bt) {
                    a.emplace_back(v);
                }
            }
            k.emplace_runtime_args(p.core, a);
        }
        desc.kernels.push_back(std::move(k));
    }

    // accumulators: reader, compute, writer
    {
        auto rd = dm_kernel(
            "dataflow/rf_acc_reader.cpp",
            aset,
            {kCbRow,
             kCbMeta,
             kCbTab,
             pc.row_bytes,
             SLOT,
             plan.PSL,
             A,
             NT,
             N,
             L,
             kCbArena,
             st.RET,
             st.MISC,
             st.RINGS,
             st.LAYO,
             pc.NP,
             pc.PD,
             pc.PS,
             kCbW,
             kMiscTaken,
             kMiscRtot,
             plan.fused ? 160u : 128u,
             plan.fused ? 103u : 101u,
             plan.loc_off,
             flat.words_offset,
             r_slot},
            r_riscv0);
        KernelDescriptor cp;
        cp.kernel_source = std::string(kKernelDir) + "compute/rf_acc_compute.cpp";
        cp.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cp.core_ranges = aset;
        cp.compile_time_args = {kCbRow, kCbMeta, kCbAcc, kCbOut, NT, plan.MAXT, kCbW, plan.snap_bits};
        {
            tt::tt_metal::ComputeConfigDescriptor cc;
            cc.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
            cc.fp32_dest_acc_en = true;
            // (the fp32 sums unpacked straight to DEST: the default narrows a Float32 CB to TF32 on its way)
            cc.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
            cc.unpack_to_dest_mode[kCbAcc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            cp.config = cc;
        }
        std::vector<CoreCoord> aw_cells;
        for (const auto& c : accs) {
            if (!plan.fused || !is_port(c)) {
                aw_cells.push_back(c);
            }
        }
        std::vector<uint32_t> aw_ct = {kCbOut, pc.row_bytes, NT, A};
        aw_ct.insert(aw_ct.end(), o_ct.begin(), o_ct.end());
        auto aw = dm_kernel("dataflow/rf_acc_writer.cpp", cores_of(aw_cells), std::move(aw_ct), s_riscv0);
        for (uint32_t i = 0; i < A; ++i) {
            const auto& core = accs[i];
            const uint32_t n_tok = i < plan.S ? (plan.S - i + A - 1) / A : 0;
            RtArgs a{buf.plan, i, idx(ch), buf.bar, static_cast<uint32_t>(gate.x), static_cast<uint32_t>(gate.y)};
            for (uint32_t kk = 0; kk < N; ++kk) {  // per source chip: the port returning its credits
                const auto s = topo.coord(kk);
                if (s != ch) {  // (every source: its ring's layout goes out through that port even with no rows)
                    const auto v = virt(plan.port_core.at(PortKey{ch, fh(ch, s), link_of(i, kk, idx(ch), L)}));
                    a.emplace_back(static_cast<uint32_t>(v.x));
                    a.emplace_back(static_cast<uint32_t>(v.y));
                } else {
                    a.emplace_back(0u);
                    a.emplace_back(0u);
                }
            }
            a.emplace_back(i == 0 ? 1u : 0u);  // (the gate)
            a.emplace_back(nz);
            a.emplace_back(i == 0 ? static_cast<uint32_t>(comb.size()) : 0u);
            if (i == 0) {
                for (const auto& c : comb) {
                    const auto v = virt(c);
                    a.emplace_back(static_cast<uint32_t>(v.x));
                    a.emplace_back(static_cast<uint32_t>(v.y));
                }
            }
            a.emplace_back(plan.QL);
            a.emplace_back(W);
            a.emplace_back(n_pid + i);
            for (const uint32_t v : wxy) {
                a.emplace_back(v);
            }
            rd.emplace_runtime_args(core, a);
            cp.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{n_tok});
            if (!plan.fused || !is_port(core)) {
                aw.emplace_runtime_args(core, RtArgs{buf.out, n_tok, i});
            }
        }
        desc.kernels.push_back(std::move(rd));
        desc.kernels.push_back(std::move(cp));
        if (!aw_cells.empty()) {
            desc.kernels.push_back(std::move(aw));
        }
    }

    // the y writers' hand-off args
    HandoffArgs h;
    h.collector = accs.at(0);
    h.counts_offset = st.CNT;
    h.gate = gate;
    auto& la = h.l1_args;
    la = {n_pid + A, ISLOT, buf.table, flat.words_offset, 0u, 0u, 0u, 0u};
    for (const auto& p : my) {
        const auto v = virt(p.core);
        la.insert(
            la.end(), {static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), plan.inj_off, plan.QP, st.MISC + 512});
    }
    for (const auto& c : accs) {
        const auto v = virt(c);
        la.insert(
            la.end(),
            {static_cast<uint32_t>(v.x),
             static_cast<uint32_t>(v.y),
             plan.loc_off,
             plan.QL,
             st.MISC + (plan.fused ? 640u : 512u)});
    }
    return h;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
