// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_program.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_routed_expert_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace combine_program_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

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

uint32_t addr_of(const CombineBuffers& b, CombineBuf which) {
    const Buffer* buf = b[which];
    TT_FATAL(buf != nullptr, "routed_experts_combine: buffer {} not given", static_cast<int>(which));
    return static_cast<uint32_t>(buf->address());
}

// A kernel's runtime args under construction: literals and buffer addresses (recorded for the cache-hit patches).
struct Args {
    std::vector<CombineArg> v;
    void lit(uint32_t x) { v.push_back({x, std::nullopt}); }
    void buf(CombineBuf b, uint32_t off = 0) { v.push_back({off, b}); }
    void lits(const std::vector<uint32_t>& xs) {
        for (const uint32_t x : xs) {
            lit(x);
        }
    }
};

void set_args(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle k,
    const CoreCoord& core,
    const Args& a,
    const CombineBuffers& bufs,
    CombineHandles& h) {
    std::vector<uint32_t> vals(a.v.size());
    for (uint32_t i = 0; i < a.v.size(); ++i) {
        const auto& x = a.v[i];
        vals[i] = x.buf ? addr_of(bufs, *x.buf) + x.value : x.value;
        if (x.buf) {
            h.patches.push_back({k, core, i, *x.buf, x.value});
        }
    }
    tt::tt_metal::SetRuntimeArgs(program, k, core, vals);
}

tt::tt_metal::KernelHandle dm_kernel(
    tt::tt_metal::Program& program,
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    bool riscv0,
    std::map<std::string, std::string> defines = {}) {
    // dedicated NoCs, fixed per RISC: RISCV_0 on NoC1, RISCV_1 on NoC0
    return tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + file,
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor =
                riscv0 ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = riscv0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0,
            .compile_args = std::move(ct),
            .defines = std::move(defines)});
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
    plan->snap_bits = cfg.snap_bits();
    plan->S = in.S;
    plan->W = in.W;
    const uint32_t N = mesh.topo.num_chips();
    TT_FATAL(N <= kMaxChips, "routed_experts_combine: {} chips > {}", N, kMaxChips);
    plan->pc = row_slot(in.W);
    // (the L1 hand-off's y writers write a row's columns at slot + column offset and its trailer at slot + row bytes:
    // one fabric packet a row, so the payload must hold a row + 32 B, else the combine waits for trailers that never
    // come)
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
    uint32_t n_ports = 0;
    {
        const auto& pp = resources.ports(pool);
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
        N, in.S, in.W, pool.size(), n_ports, in.arena_bytes, in.depth, in.n_acc.value_or(kAccMax), in.mode);
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
        const auto& free_ports = resources.ports(pool);
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
        plan->port_core = resources.ports(aset);
        for (const auto& [ch, _] : mesh.chips) {
            plan->accs[ch] = aset;
        }
    } else {
        plan->port_core = resources.ports(pool);
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
    // (the ports' end fence checks one layout word of each home on their link: an accumulator o with
    // (o + src + home) % L == link exists for every residue when A >= L)
    TT_FATAL(N == 1 || A >= plan->L, "routed_experts_combine: {} accumulators < {} links", A, plan->L);
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
    plan->cred = credit_policy();
    const uint32_t acc_area = plan->st.RINGS + plan->PSL * SLOT;
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
    // the placement (sparse_combine_rf_prepare.build): ports apart - the injection ring at RINGS, the LUT after it,
    // then CB_TAB; accumulators - CB_TAB and their CBs after the landing pool, then the local ring (acc_on_ports: then
    // the injection ring)
    plan->arena_bytes = in.arena_bytes;
    plan->lut_bytes = plan->perchip ? align64(2 * N * A) : 0;
    const uint32_t port_area = plan->st.RINGS + QP * ISLOT;
    plan->lut_off = align64(port_area);
    plan->tab_off_port = align64(plan->lut_off + plan->lut_bytes);
    uint32_t e = align64(acc_area);
    auto take = [&](uint32_t bytes) {
        const uint32_t o = e;
        e = align64(e + bytes);
        return o;
    };
    const uint32_t NT = plan->pc.NT;
    plan->tab_off_acc = take(kTabCbBytes);
    plan->row_off = take(2 * NT * kTile);
    plan->meta_off = take(2 * 32);
    plan->acc_off = take(plan->MAXT * NT * 4096);
    plan->out_off = take(2 * NT * kTile);
    plan->w_off = take(2 * kTile);
    plan->loc_off = e;
    TT_FATAL(
        std::max(e, plan->fused ? 0u : plan->tab_off_port + kTabCbBytes) <= in.arena_bytes,
        "routed_experts_combine: needs {} KB a core, the arena has {} KB",
        std::max(e, plan->tab_off_port + kTabCbBytes) >> 10,
        in.arena_bytes >> 10);
    if (plan->fused) {
        plan->inj_off = plan->loc_off + QL * ISLOT;
        TT_FATAL(plan->inj_off + QP * ISLOT <= in.arena_bytes, "acc_on_ports: no room for the injection ring");
    } else {
        QL = std::min<uint32_t>(16, (in.arena_bytes - plan->loc_off) / ISLOT);
        TT_FATAL(QL >= 2, "routed_experts_combine: no room for the accumulators' local ring");
        plan->QL = QL;
        plan->inj_off = plan->st.RINGS;
    }

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

CombineHandles add_combine_program(
    tt::tt_metal::Program& program,
    const CombinePlan& plan,
    ChipCoord ch,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& bufs,
    const FlatHandoff& flat) {
    using namespace combine_program_detail;
    const auto& topo = plan.topo;
    const uint32_t N = topo.num_chips(), L = plan.L, A = plan.A;
    const auto& st = plan.st;
    const auto& pc = plan.pc;
    const uint32_t SLOT = pc.slot, ISLOT = pc.slot, NT = pc.NT;
    const uint32_t words_off = plan.arena_bytes;  // (the y writers' words, past the combine's bytes)
    TT_FATAL(flat.writers.size() <= 32, "routed_experts_combine: {} y writers > 32", flat.writers.size());
    auto virt = [&](const CoreCoord& c) { return mesh_device.worker_core_from_logical_core(c); };
    auto idx = [&](ChipCoord c) { return topo.index(c); };
    auto fh = [&](ChipCoord s, ChipCoord d) { return *first_hop(s, d, topo); };
    auto node = [&](ChipCoord c) {
        return mesh_device.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(c.r, c.c));
    };
    CombineHandles h;

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
    auto is_acc = [&](const CoreCoord& c) {
        return std::any_of(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, c); });
    };
    auto is_port = [&](const CoreCoord& c) {
        return std::any_of(pcells.begin(), pcells.end(), [&](const CoreCoord& x) { return same_cell(x, c); });
    };

    // circular buffers, all in the arena at the reference's offsets
    const Buffer& arena = *bufs[CombineBuf::Arena];
    auto arena_cb =
        [&](uint8_t i, const CoreRangeSet& cores, uint32_t off, uint32_t total, tt::DataFormat fmt, uint32_t page) {
            TT_FATAL(
                off + total <= plan.arena_bytes,
                "routed_experts_combine: CB {} at {} + {} B past the arena",
                i,
                off,
                total);
            tt::tt_metal::CircularBufferConfig c(total, {{i, fmt}});
            c.set_page_size(i, page);
            c.set_globally_allocated_address(arena);
            c.set_address_offset(off);
            h.arena_cbs.emplace_back(tt::tt_metal::CreateCircularBuffer(program, cores, c), off);
        };
    arena_cb(kCbInj, pset, plan.inj_off, plan.QP * ISLOT, tt::DataFormat::UInt32, ISLOT);
    if (!plan.fused) {
        std::vector<CoreCoord> only_ports;
        for (const auto& c : pcells) {
            if (!is_acc(c)) {
                only_ports.push_back(c);
            }
        }
        arena_cb(kCbTab, cores_of(only_ports), plan.tab_off_port, kTabCbBytes, tt::DataFormat::UInt32, kTabCbBytes);
    }
    arena_cb(kCbTab, aset, plan.tab_off_acc, kTabCbBytes, tt::DataFormat::UInt32, kTabCbBytes);
    arena_cb(kCbRow, aset, plan.row_off, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbMeta, aset, plan.meta_off, 2 * 32, tt::DataFormat::UInt32, 32);
    arena_cb(kCbAcc, aset, plan.acc_off, plan.MAXT * NT * 4096, tt::DataFormat::Float32, 4096);
    arena_cb(kCbOut, aset, plan.out_off, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbW, aset, plan.w_off, 2 * kTile, tt::DataFormat::Float16_b, kTile);

    // the zero gate: accumulator 0 releases every combine core once the zeroing kernels counted in (ports' senders
    // unless they share an accumulator's core, the accumulators' readers, the flat expert's y writers)
    const auto gate = virt(accs.at(0));
    std::vector<CoreCoord> comb = accs;
    for (const auto& c : pcells) {
        if (!is_acc(c)) {
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
    // the RISC split: the sender (and the accumulator writer) on RISCV_0 / NoC1, the port reader on RISCV_1 / NoC0
    constexpr bool s_riscv0 = true, r_riscv0 = false;
    constexpr uint32_t s_slot = 0, r_slot = 1;  // (bar call-count slots per RISC)

    std::vector<uint32_t> o_ct;
    tt::tt_metal::TensorAccessorArgs(*bufs[CombineBuf::Out]).append_to(o_ct);

    // port readers (ports apart only: fused ports run the hand-off on the sender)
    if (!plan.fused) {
        std::vector<uint32_t> ct = {kCbInj, kCbTab, st.MISC, words_off, r_slot};
        const auto k = dm_kernel(program, "dataflow/combine_port_reader.cpp", pset, std::move(ct), r_riscv0);
        for (uint32_t pid = 0; pid < n_pid; ++pid) {
            Args a;
            a.buf(CombineBuf::Arena);
            a.buf(CombineBuf::Plan);
            a.buf(CombineBuf::Bar);
            a.lits({W, pid, plan.QP});
            a.lits(wxy);
            set_args(program, k, my[pid].core, a, bufs, h);
        }
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
            kNumHeaders,
            kHeaderStride,
            kMiscRtot,
            kMiscRdone,
            s_slot,
            plan.fused ? 0u : 1u /*ZERO*/,
            words_off,
            plan.lut_off,
            plan.lut_bytes};
        std::map<std::string, std::string> defs;
        if (plan.fused) {
            defs["REC_FUSED"] = "1";
            ct.push_back(kCbOut);
            ct.push_back(NT);
            ct.insert(ct.end(), o_ct.begin(), o_ct.end());
        }
        if (plan.perchip) {
            defs["REC_ACC_LUT"] = "1";
        }
        const auto k =
            dm_kernel(program, "dataflow/combine_port_writer.cpp", pset, std::move(ct), s_riscv0, std::move(defs));
        std::vector<uint32_t> acc_xy;
        for (const auto& c : accs) {
            const auto v = virt(c);
            acc_xy.push_back(v.x);
            acc_xy.push_back(v.y);
        }
        for (uint32_t pid = 0; pid < n_pid; ++pid) {
            const auto& p = my[pid];
            Args a;
            a.buf(CombineBuf::Arena);
            a.lits({plan.mesh_id, idx(ch), p.li});
            a.buf(CombineBuf::Bar);
            a.lits({static_cast<uint32_t>(gate.x), static_cast<uint32_t>(gate.y)});
            a.buf(CombineBuf::Plan);
            a.lit(pid);
            a.lits(plan.chip_ids);
            if (plan.perchip) {
                a.buf(CombineBuf::Lut);
            } else {
                a.lits(acc_xy);
            }
            for (uint32_t i = 0; i < N; ++i) {  // per source chip: the port on it that feeds me on my link
                const auto s = topo.coord(i);
                if (s != ch && fh(ch, s) == p.dir) {
                    const auto v = virt(plan.port_core.at(PortKey{s, fh(s, ch), p.li}));
                    a.lits({static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                } else {
                    a.lits({0u, 0u});
                }
            }
            a.buf(CombineBuf::Hcache);
            a.lit(static_cast<uint32_t>(p.dir) * 2 + p.li);
            {
                std::vector<uint32_t> fab;
                const auto& mp = *plan.mesh.chips.at(ch)[static_cast<size_t>(p.dir)];
                tt::tt_fabric::append_fabric_connection_rt_args(
                    node(ch), node(mp.peer), mp.links.at(p.li), program, p.core, fab);
                a.lits(fab);
            }
            if (plan.fused) {  // (the hand-off receiver's args, then the accumulator writer's)
                const auto ai = static_cast<uint32_t>(
                    std::find_if(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_cell(x, p.core); }) -
                    accs.begin());
                a.lits({W, pid, plan.QP});
                a.buf(CombineBuf::Out);
                a.lits({ai < plan.S ? (plan.S - ai + A - 1) / A : 0u, ai});
                a.lits(wxy);
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
                a.lit(static_cast<uint32_t>(bt.size() / 3));
                a.lits(bt);
            }
            set_args(program, k, p.core, a, bufs, h);
        }
    }

    // accumulators: reader, compute, writer
    {
        const auto rd = dm_kernel(
            program,
            "dataflow/accumulator_reader.cpp",
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
             st.RET,
             st.MISC,
             st.RINGS,
             st.LAYO,
             kCbW,
             kMiscTaken,
             kMiscRtot,
             plan.fused ? 160u : 128u,
             plan.fused ? 103u : 101u,
             plan.loc_off,
             words_off,
             r_slot},
            r_riscv0);
        // (the fp32 sums unpacked straight to DEST: the default narrows a Float32 CB to TF32 on its way)
        std::vector<tt::tt_metal::UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        u2d[kCbAcc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        const auto cp = tt::tt_metal::CreateKernel(
            program,
            std::string(kKernelDir) + "compute/accumulator_compute.cpp",
            aset,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = u2d,
                .compile_args = {kCbRow, kCbMeta, kCbAcc, kCbOut, NT, plan.MAXT, kCbW, plan.snap_bits}});
        std::vector<CoreCoord> aw_cells;
        for (const auto& c : accs) {
            if (!plan.fused || !is_port(c)) {
                aw_cells.push_back(c);
            }
        }
        std::vector<uint32_t> aw_ct = {kCbOut, pc.row_bytes, NT, A};
        aw_ct.insert(aw_ct.end(), o_ct.begin(), o_ct.end());
        std::optional<tt::tt_metal::KernelHandle> aw;
        if (!aw_cells.empty()) {
            aw = dm_kernel(program, "dataflow/accumulator_writer.cpp", cores_of(aw_cells), std::move(aw_ct), s_riscv0);
        }
        for (uint32_t i = 0; i < A; ++i) {
            const auto& core = accs[i];
            const uint32_t n_tok = i < plan.S ? (plan.S - i + A - 1) / A : 0;
            Args a;
            a.buf(CombineBuf::Arena);
            a.buf(CombineBuf::Plan);
            a.lits({i, idx(ch)});
            a.buf(CombineBuf::Bar);
            a.lits({static_cast<uint32_t>(gate.x), static_cast<uint32_t>(gate.y)});
            for (uint32_t kk = 0; kk < N; ++kk) {  // per source chip: the port returning its credits
                const auto s = topo.coord(kk);
                if (s != ch) {  // (every source: its ring's layout goes out through that port even with no rows)
                    const auto v = virt(plan.port_core.at(PortKey{ch, fh(ch, s), link_of(i, kk, idx(ch), L)}));
                    a.lits({static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                } else {
                    a.lits({0u, 0u});
                }
            }
            a.lits({i == 0 ? 1u : 0u, nz, i == 0 ? static_cast<uint32_t>(comb.size()) : 0u});  // (the gate)
            if (i == 0) {
                for (const auto& c : comb) {
                    const auto v = virt(c);
                    a.lits({static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                }
            }
            a.lits({plan.QL, W, n_pid + i});
            a.lits(wxy);
            set_args(program, rd, core, a, bufs, h);
            tt::tt_metal::SetRuntimeArgs(program, cp, core, {n_tok});
            if (!plan.fused || !is_port(core)) {
                Args w;
                w.buf(CombineBuf::Out);
                w.lits({n_tok, i});
                set_args(program, *aw, core, w, bufs, h);
            }
        }
    }

    // the y writers' report: the collector's counts and the L1 hand-off (destinations: ports, then accumulators)
    h.collector = gate;
    h.counts_offset = st.CNT;
    auto& la = h.l1_args;
    auto lit = [&](uint32_t x) { la.push_back({x, std::nullopt}); };
    auto at = [&](CombineBuf b, uint32_t off) { la.push_back({off, b}); };
    lit(n_pid + A);
    lit(ISLOT);
    at(CombineBuf::Table, 0);
    at(CombineBuf::Arena, words_off);
    lit(0);  // (the writer index: per writer)
    lit(static_cast<uint32_t>(gate.x) | static_cast<uint32_t>(gate.y) << 16);
    at(CombineBuf::Bar, 0);
    lit(kWriterWordsRows * 64);  // (words to zero, then count in on the gate)
    for (const auto& p : my) {
        const auto v = virt(p.core);
        lit(v.x);
        lit(v.y);
        at(CombineBuf::Arena, plan.inj_off);
        lit(plan.QP);
        at(CombineBuf::Arena, st.MISC + 512);
    }
    for (const auto& c : accs) {
        const auto v = virt(c);
        lit(v.x);
        lit(v.y);
        at(CombineBuf::Arena, plan.loc_off);
        lit(plan.QL);
        at(CombineBuf::Arena, st.MISC + (plan.fused ? 640u : 512u));
    }
    return h;
}

void override_combine_addresses(tt::tt_metal::Program& program, const CombineHandles& h, const CombineBuffers& bufs) {
    using namespace combine_program_detail;
    // (a source's args - the arena's CBs with it - only when its address differs from the one they were last written
    // with: CombineHandles::last)
    std::array<bool, static_cast<size_t>(CombineBuf::Count)> moved{};
    for (size_t b = 0; b < moved.size(); ++b) {
        if (bufs.buf[b] != nullptr) {
            const uint32_t a = addr_of(bufs, static_cast<CombineBuf>(b));
            moved[b] = h.last[b] != a;
            h.last[b] = a;
        }
    }
    for (const auto& p : h.patches) {
        if (moved[static_cast<size_t>(p.buf)]) {
            tt::tt_metal::GetRuntimeArgs(program, p.kernel, p.core)[p.index] = addr_of(bufs, p.buf) + p.offset;
        }
    }
    if (moved[static_cast<size_t>(CombineBuf::Arena)]) {
        for (const auto& [cb, off] : h.arena_cbs) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb, *bufs[CombineBuf::Arena], off);
        }
    }
}

std::vector<uint32_t> resolve_l1_args(const CombineHandles& h, const CombineBuffers& bufs) {
    using namespace combine_program_detail;
    std::vector<uint32_t> v(h.l1_args.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const auto& x = h.l1_args[i];
        v[i] = x.buf ? addr_of(bufs, *x.buf) + x.value : x.value;
    }
    return v;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
