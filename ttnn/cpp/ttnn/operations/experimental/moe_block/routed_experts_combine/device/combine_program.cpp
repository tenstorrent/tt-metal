// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_program.hpp"
#include <tuple>
#include <optional>
#include <map>

#include <algorithm>
#include <limits>
#include <set>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
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

bool same_core(const CoreCoord& a, const CoreCoord& b) { return a.x == b.x && a.y == b.y; }

struct ChipPort {
    Dir dir;
    uint32_t li;
    CoreCoord core;
};

std::vector<CoreCoord> port_cores_of(const PortCores& pc, ChipCoord ch) {
    std::vector<CoreCoord> v;
    for (const auto& [k, c] : pc) {
        if (k.chip == ch && std::none_of(v.begin(), v.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
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

// A kernel into the program descriptor: its handle is its index in the descriptor's kernels (the Program built from
// the descriptor creates the kernels in that order).
tt::tt_metal::KernelHandle add_kernel(
    tt::tt_metal::ProgramDescriptor& program, tt::tt_metal::KernelDescriptor kernel_descriptor) {
    program.kernels.push_back(std::move(kernel_descriptor));
    return static_cast<tt::tt_metal::KernelHandle>(program.kernels.size() - 1);
}

// A kernel's runtime args under construction: literals and buffer addresses (recorded for the cache-hit patches).
struct Args {
    std::vector<CombineArg> v;
    std::optional<std::pair<uint32_t, uint32_t>> fabric;  // (a sender's fabric connection args: first, count)
    void lit(uint32_t x) { v.push_back({x, std::nullopt}); }
    void buf(CombineBuf b, uint32_t off = 0) { v.push_back({off, b}); }
    void lits(const std::vector<uint32_t>& xs) {
        for (const uint32_t x : xs) {
            lit(x);
        }
    }
};

void set_args(
    tt::tt_metal::ProgramDescriptor& program,
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
    // (a second call for the same core replaces its args: the Program API's SetRuntimeArgs)
    auto& runtime_args = program.kernels[k].runtime_args;
    const auto existing = std::find_if(
        runtime_args.begin(), runtime_args.end(), [&](const auto& core_args) { return core_args.first == core; });
    if (existing != runtime_args.end()) {
        existing->second = std::move(vals);
    } else {
        runtime_args.emplace_back(core, std::move(vals));
    }
    if (a.fabric) {
        h.fabric_args[{static_cast<uint32_t>(k), core.x, core.y}] = *a.fabric;
    }
}

tt::tt_metal::KernelHandle dm_kernel(
    tt::tt_metal::ProgramDescriptor& program,
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    bool riscv0,
    std::map<std::string, std::string> defines,
    const std::vector<std::pair<std::string, uint32_t>>& options) {
    // dedicated NoCs, fixed per RISC: RISCV_0 on NoC1, RISCV_1 on NoC0
    tt::tt_metal::KernelDescriptor kernel_descriptor;
    kernel_descriptor.kernel_source = std::string(kKernelDir) + file;
    kernel_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel_descriptor.core_ranges = cores;
    kernel_descriptor.compile_time_args = std::move(ct);
    kernel_descriptor.named_compile_time_args =
        options;  // (the combine's options: kernels/dataflow/combine_options.hpp)
    kernel_descriptor.defines =
        tt::tt_metal::KernelDescriptor::Defines(defines.begin(), defines.end());  // (sorted by name)
    kernel_descriptor.opt_level = tt::tt_metal::KernelBuildOptLevel::O2;  // (DataMovementConfig's default)
    kernel_descriptor.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor =
            riscv0 ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = riscv0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
    return add_kernel(program, std::move(kernel_descriptor));
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
    plan->hops = resources.first_hops();
    plan->L = mesh.num_links;
    plan->port_luts = resources.port_luts();
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

    // the cores: the flat plan's free ones; the port count the ports-apart accumulators lose (combine_port_count)
    const auto grid = mesh_device.compute_with_storage_grid_size();
    std::vector<CoreCoord> pool = in.free_cores;
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
            if (std::none_of(uni.begin(), uni.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
                uni.push_back(c);
            }
        }
        if (uni.size() >= pool.size()) {
            n_ports = uni.size();
        } else {
            for (const auto& [ch, _] : mesh.chips) {
                n_ports = std::max<uint32_t>(n_ports, port_cores_of(pp, ch).size());
            }
        }
    }
    // (FlatCores: the flat expert cores' free arena tails, whole slots a core)
    // (the pool: every free span's whole slots, classes of cores with the same spans -
    // kernels/dataflow/combine_pool.hpp)
    uint32_t pool_slots = 0;
    std::vector<uint32_t> pool_args = {0, 0};
    for (const auto& fc : in.flat_free) {
        std::vector<uint32_t> spans;
        uint32_t spc = 0;
        for (const auto& [b, e] : fc.spans) {
            const uint32_t n = std::min<uint32_t>((e - b) / plan->pc.slot, 0xFFFF);
            if (n > 0) {
                TT_FATAL(b % 64 == 0 && b / 64 <= 0xFFFF, "routed_experts_combine: pool span at {}", b);
                spans.push_back(b / 64 | n << 16);
                spc += n;
            }
        }
        if (spans.empty() || fc.cores.empty()) {
            continue;
        }
        TT_FATAL(fc.cores.size() < 256 && spans.size() < 256 && spc <= 0xFFFF, "routed_experts_combine: pool class");
        pool_args.push_back(static_cast<uint32_t>(fc.cores.size()) | spans.size() << 8 | spc << 16);
        pool_args.insert(pool_args.end(), spans.begin(), spans.end());
        for (size_t c = 0; c < fc.cores.size(); c += 2) {
            uint32_t w = 0;
            for (size_t k = c; k < std::min(c + 2, fc.cores.size()); ++k) {
                const auto v = mesh_device.worker_core_from_logical_core(fc.cores[k]);
                TT_FATAL(v.x < 256 && v.y < 256, "routed_experts_combine: port-or-accumulator core {}", v);
                w |= (static_cast<uint32_t>(v.x) | static_cast<uint32_t>(v.y) << 8) << ((k - c) * 16);
            }
            pool_args.push_back(w);
        }
        pool_slots += static_cast<uint32_t>(fc.cores.size()) * spc;
        ++pool_args[1];
    }
    pool_args[0] = static_cast<uint32_t>(pool_args.size());
    auto plan_with = [&](CombineLanding l) {
        return combine_auto(
            N,
            in.S,
            in.W,
            pool.size(),
            n_ports,
            in.arena_bytes,
            in.depth,
            in.n_acc.value_or(kAccMax),
            in.mode,
            l,
            pool_slots);
    };
    // no landing given: the rings in the accumulators' L1 or on the flat expert cores, whichever ranks higher
    // (combine_choice_key). 2x4 GLM-5.3-Flash experts + combine: chunk 5120 D 1 QP 8 -> D 8 QP 16 on the flat expert
    // cores 2762 -> 1543 us, 1280 D 8 QL 3 -> D 8 QL 16 513 -> 489, 2560 702 -> 708
    CombineLanding landing = in.landing.value_or(CombineLanding::Rings);
    auto au = plan_with(landing);
    if (!in.landing && pool_slots > 0) {
        auto fc = plan_with(CombineLanding::FlatCores);
        const uint32_t a_ref = std::max(au.fits ? au.best->n_acc : 0u, fc.fits ? fc.best->n_acc : 0u);
        auto key = [&](const CombineAuto& x) { return combine_choice_key(*x.best, a_ref); };
        if (fc.fits && (!au.fits || key(fc) > key(au))) {
            au = std::move(fc);
            landing = CombineLanding::FlatCores;
        }
    }
    const bool flat_pool = landing_on_flat(landing);
    TT_FATAL(
        !flat_pool || pool_slots > 0,
        "routed_experts_combine: FlatCores landing - no free slot on the flat expert cores (arena {} KB)",
        in.arena_bytes >> 10);
    TT_FATAL(
        au.fits,
        "routed_experts_combine: does not fit ({} cores, {} KB): {}",
        pool.size(),
        in.arena_bytes >> 10,
        au.why);
    TT_FATAL(
        !in.n_acc || au.best->n_acc == *in.n_acc,
        "routed_experts_combine: {} accumulators, {} planned",
        au.best->n_acc,
        *in.n_acc);
    plan->choice = *au.best;
    plan->acc_on_ports = au.best->acc_on_ports;
    plan->perchip = !plan->acc_on_ports;
    plan->shared = au.best->shared;
    const uint32_t n_acc = au.best->n_acc;

    // ports and accumulators (sparse_combine_rf_prepare)
    if (plan->acc_on_ports) {  // the accumulator cores: every chip's nearest-to-Ethernet port cores, then
                               // the pool's first others
        const auto& free_ports = resources.ports(pool);
        std::vector<CoreCoord> aset;
        for (const auto& c : pool) {
            if (std::any_of(
                    free_ports.begin(), free_ports.end(), [&](const auto& kv) { return same_core(kv.second, c); })) {
                aset.push_back(c);
            }
        }
        TT_FATAL(
            aset.size() <= n_acc, "acc_on_ports: {} port cores over the chips > {} accumulators", aset.size(), n_acc);
        for (const auto& c : pool) {
            if (aset.size() >= n_acc) {
                break;
            }
            if (std::none_of(aset.begin(), aset.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
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
                    return same_core(kv.second, c);
                })) {
                never.push_back(c);
            }
        }
        TT_FATAL(!never.empty(), "routed_experts_combine: no core free of every chip's ports (the collector's)");
        size_t A_ = SIZE_MAX;
        for (const auto& [ch, _] : mesh.chips) {
            const auto own = port_cores_of(plan->port_core, ch);
            std::vector<CoreCoord> v{never[0]};
            for (const auto& c : pool) {
                if (v.size() >= n_acc) {
                    break;
                }
                if (!same_core(c, never[0]) &&
                    std::none_of(own.begin(), own.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
                    v.push_back(c);
                }
            }
            A_ = std::min(A_, v.size());
            plan->accs[ch] = std::move(v);
        }
        for (auto& [ch, v] : plan->accs) {
            v.resize(A_);
        }
        // (the port-or-accumulator cores: every chip's ports and accumulators)
        auto add_port_or_accumulator_core = [&](const CoreCoord& c) {
            if (std::none_of(
                    plan->port_or_accumulator_cores.begin(),
                    plan->port_or_accumulator_cores.end(),
                    [&](const CoreCoord& x) { return same_core(x, c); })) {
                plan->port_or_accumulator_cores.push_back(c);
            }
        };
        for (const auto& [key, c] : plan->port_core) {
            add_port_or_accumulator_core(c);
        }
        for (const auto& [ch, v] : plan->accs) {
            for (const auto& c : v) {
                add_port_or_accumulator_core(c);
            }
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
            if (std::none_of(
                    plan->accumulator_cores_union.begin(),
                    plan->accumulator_cores_union.end(),
                    [&](const CoreCoord& x) { return same_core(x, c); })) {
                plan->accumulator_cores_union.push_back(c);
            }
        }
    }

    // the layout (the arena budget's sizing, as sparse_combine_rf_prepare)
    plan->st = state_layout(N, A, plan->shared ? kGrants : 0);
    const uint32_t SLOT = plan->pc.slot, ISLOT = SLOT;
    plan->row_depth = flat_pool ? kPoolRows : 2u;
    uint32_t acc_cbs = acc_cb_bytes(in.S, A, plan->pc.NT, plan->row_depth);
    const uint32_t QP0 = au.best->inj_slots, keep_slots = au.best->local_keep;
    const int64_t keep = plan->acc_on_ports ? static_cast<int64_t>(keep_slots + std::min(32u, QP0)) * ISLOT
                                            : static_cast<int64_t>(keep_slots) * ISLOT;
    const int64_t fit = flat_pool ? static_cast<int64_t>(pool_slots / A)
                                  : (static_cast<int64_t>(in.arena_bytes) - plan->st.RINGS - acc_cbs - keep) / SLOT;
    const int64_t PSL = std::min<int64_t>(
        {static_cast<int64_t>(in.depth) * (N - 1),
         fit,
         !plan->shared ? std::numeric_limits<int64_t>::max() : int64_t(flat_pool ? kSharedFlatPslMax : kSharedPslMax)});
    TT_FATAL(
        PSL >= static_cast<int64_t>(N) - 1, "routed_experts_combine: no room for one landing slot per source chip");
    TT_FATAL(PSL < 4096, "routed_experts_combine: a ring's base / depth are 12-bit fields");
    plan->PSL = static_cast<uint32_t>(PSL);
    plan->tab_acc = acc_tab_bytes(plan->shared, plan->PSL);
    acc_cbs = acc_cb_bytes(in.S, A, plan->pc.NT, plan->row_depth, plan->tab_acc);
    plan->D = N > 1 ? plan->PSL / (N - 1) : 0;
    plan->cred = credit_policy(N, plan->D, plan->shared, in.credit_batch);
    const uint32_t acc_area = plan->st.RINGS + (flat_pool ? 0u : plan->PSL * SLOT);
    if (flat_pool) {  // (every flat expert core, in the flat plan's order)
        plan->flat_pool = true;
        plan->pool_slots = pool_slots;
        plan->pool_args = std::move(pool_args);
    }
    uint32_t QP = QP0, QL = 0;
    if (plan->acc_on_ports) {  // (local ring up to 16 slots, then the injection ring up to QP, from what is left)
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
    plan->tab_off_acc = take(plan->tab_acc);
    plan->row_off = take(plan->row_depth * NT * kTile);
    plan->meta_off = take(plan->row_depth * 32);
    plan->acc_off = take(plan->MAXT * NT * 4096);
    plan->out_off = take(2 * NT * kTile);
    plan->w_off = take(plan->row_depth * kTile);
    plan->loc_off = e;
    TT_FATAL(
        std::max(e, plan->acc_on_ports ? 0u : plan->tab_off_port + kTabCbBytes) <= in.arena_bytes,
        "routed_experts_combine: needs {} KB a core, the arena has {} KB",
        std::max(e, plan->tab_off_port + kTabCbBytes) >> 10,
        in.arena_bytes >> 10);
    if (plan->acc_on_ports) {
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
        const auto node = mesh_device.get_fabric_node_id(mesh_coord(ch, topo));
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
    tt::tt_metal::ProgramDescriptor& program,
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
    auto fh = [&](ChipCoord s, ChipCoord d) { return plan.hops.at(s, d, topo); };  // (the fabric's own first hop)
    auto node = [&](ChipCoord c) { return mesh_device.get_fabric_node_id(mesh_coord(c, topo)); };
    CombineHandles h;

    // this chip's ports by port id: the resources' port LUTs (the numbering the MoE head's planner writes the plan's
    // n_inj / n_ret by)
    const auto& lut = plan.port_luts.lut.at(ch);
    const uint32_t n_pid = plan.port_luts.num_ports.at(ch);
    std::vector<ChipPort> my(n_pid);
    std::vector<bool> got(n_pid, false);
    for (const Dir d : kDirs) {
        for (uint32_t li = 0; li < L; ++li) {
            const uint8_t pid = lut.at(static_cast<size_t>(d) * kMaxLinks + li);
            if (pid != 0xFF) {
                TT_FATAL(
                    pid < n_pid && !got[pid], "routed_experts_combine: port id {} of chip ({}, {})", pid, ch.r, ch.c);
                my[pid] = {d, li, plan.port_core.at(PortKey{ch, d, li})};
                got[pid] = true;
            }
        }
    }
    TT_FATAL(
        std::all_of(got.begin(), got.end(), [](bool g) { return g; }),
        "routed_experts_combine: chip ({}, {}) has a port id with no port",
        ch.r,
        ch.c);
    std::vector<CoreCoord> port_cores;
    for (const auto& p : my) {
        if (std::none_of(
                port_cores.begin(), port_cores.end(), [&](const CoreCoord& x) { return same_core(x, p.core); })) {
            port_cores.push_back(p.core);
        }
    }
    const auto& accs = plan.accs.at(ch);
    // ports apart: every port-or-accumulator core runs both roles' binaries (plan.port_or_accumulator_cores;
    // combine_port_or_accumulator_reader / _writer.cpp)
    const bool dual = !plan.acc_on_ports;
    const auto pset = dual ? cores_of(plan.port_or_accumulator_cores) : cores_of(port_cores),
               aset = dual ? cores_of(plan.port_or_accumulator_cores) : cores_of(accs);
    auto is_acc = [&](const CoreCoord& c) {
        return std::any_of(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_core(x, c); });
    };
    auto is_port = [&](const CoreCoord& c) {
        return std::any_of(port_cores.begin(), port_cores.end(), [&](const CoreCoord& x) { return same_core(x, c); });
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
            // (the CB's handle until finalize_combine_handles: its index in the descriptor's cbs)
            h.arena_cbs.emplace_back(program.cbs.size(), off);
            tt::tt_metal::CBDescriptor circular_buffer{.total_size = total, .core_ranges = cores};
            circular_buffer.format_descriptors.push_back(
                tt::tt_metal::CBFormatDescriptor{.buffer_index = i, .data_format = fmt, .page_size = page});
            circular_buffer.buffer = const_cast<Buffer*>(&arena);
            circular_buffer.address_offset = off;
            program.cbs.push_back(std::move(circular_buffer));
        };
    arena_cb(kCbInj, pset, plan.inj_off, plan.QP * ISLOT, tt::DataFormat::UInt32, ISLOT);
    if (dual) {  // (the port reader's table at its own index: the core's accumulator table is kCbTab)
        arena_cb(kCbTabP, pset, plan.tab_off_port, kTabCbBytes, tt::DataFormat::UInt32, kTabCbBytes);
    }
    arena_cb(kCbTab, aset, plan.tab_off_acc, plan.tab_acc, tt::DataFormat::UInt32, plan.tab_acc);
    arena_cb(kCbRow, aset, plan.row_off, plan.row_depth * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbMeta, aset, plan.meta_off, plan.row_depth * 32, tt::DataFormat::UInt32, 32);
    arena_cb(kCbAcc, aset, plan.acc_off, plan.MAXT * NT * 4096, tt::DataFormat::Float32, 4096);
    arena_cb(kCbOut, aset, plan.out_off, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbW, aset, plan.w_off, plan.row_depth * kTile, tt::DataFormat::Float16_b, kTile);

    // the zero gate: accumulator 0 releases every combine core once the zeroing kernels counted in (ports' senders
    // unless they share an accumulator's core, the accumulators' readers, the flat expert's y writers)
    const auto gate = virt(accs.at(0));
    std::vector<CoreCoord> comb = accs;
    for (const auto& c : port_cores) {
        if (!is_acc(c)) {
            comb.push_back(c);
        }
    }
    const uint32_t nz = (plan.acc_on_ports ? 0u : static_cast<uint32_t>(port_cores.size())) + A + flat.writers.size();

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
    // (dual: each role's kernel is recorded - its compile-time args, defines and each core's args - and the cores' two
    // binaries composed after the accumulators)
    struct Pend {
        std::vector<uint32_t> ct;
        std::map<std::string, std::string> defs;
        std::map<std::pair<size_t, size_t>, Args> args;
    };
    enum : uint32_t { R_PR = 0, R_PW = 1, R_AR = 2, R_AW = 3 };
    constexpr tt::tt_metal::KernelHandle kPend = 0x10000;
    std::array<Pend, 4> pend;
    // the combine's options, every combine data-movement kernel's named compile-time args
    // (kernels/dataflow/combine_options.hpp)
    const std::vector<std::pair<std::string, uint32_t>> options = {
        {"accumulators_on_ports", plan.acc_on_ports ? 1u : 0u},
        {"per_chip_accumulator_table", plan.perchip ? 1u : 0u},
        {"shared_landing", plan.shared ? 1u : 0u},
        {"shared_landing_grants_per_source", plan.shared ? st.G : 1u},
        {"shared_landing_grant_ids_offset", plan.shared ? st.GIDH : 0u},
        {"shared_landing_grant_snapshots_offset", plan.shared ? st.GREC : 0u},
        {"landing_on_flat_expert_cores", plan.flat_pool ? 1u : 0u},
        {"landing_reads_in_flight", plan.flat_pool ? plan.row_depth : 2u}};
    auto kern = [&](uint32_t role,
                    const std::string& file,
                    const CoreRangeSet& cores,
                    std::vector<uint32_t> ct,
                    bool riscv0,
                    std::map<std::string, std::string> defs) -> tt::tt_metal::KernelHandle {
        if (dual) {
            pend[role].ct = std::move(ct);
            pend[role].defs = std::move(defs);
            return kPend + role;
        }
        return dm_kernel(program, file, cores, std::move(ct), riscv0, std::move(defs), options);
    };
    auto put = [&](tt::tt_metal::KernelHandle k, const CoreCoord& core, const Args& a) {
        if (k >= kPend) {
            pend[k - kPend].args[{core.x, core.y}] = a;
        } else {
            set_args(program, k, core, a, bufs, h);
        }
    };
    std::map<std::pair<size_t, size_t>, uint32_t> acc_tok;  // (dual: the compute's tokens a core)

    std::vector<uint32_t> o_ct;
    tt::tt_metal::TensorAccessorArgs(*bufs[CombineBuf::Out]).append_to(o_ct);
    // a buffer's accessor args at the end of a kernel's compile-time args, their first index as the define `name`
    auto accessor =
        [&](std::vector<uint32_t>& ct, std::map<std::string, std::string>& defs, const char* name, CombineBuf b) {
            defs[name] = std::to_string(ct.size());
            tt::tt_metal::TensorAccessorArgs(*bufs[b]).append_to(ct);
        };

    // port readers (ports apart only: ports that also accumulate run the hand-off on the sender)
    if (!plan.acc_on_ports) {
        std::vector<uint32_t> ct = {kCbInj, dual ? kCbTabP : kCbTab, st.MISC, words_off, r_slot};
        std::map<std::string, std::string> defs;
        accessor(ct, defs, "COMBINE_PLAN_ACCESSOR_FIRST_ARG", CombineBuf::Plan);
        const auto k = kern(R_PR, "dataflow/combine_port_reader.cpp", pset, std::move(ct), r_riscv0, std::move(defs));
        for (uint32_t pid = 0; pid < n_pid; ++pid) {
            Args a;
            a.buf(CombineBuf::Arena);
            a.buf(CombineBuf::Plan);
            a.buf(CombineBuf::Bar);
            a.lits({W, pid, plan.QP});
            a.lits(wxy);
            put(k, my[pid].core, a);
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
            plan.acc_on_ports ? 0u : 1u /*ZERO*/,
            words_off,
            plan.lut_off,
            plan.lut_bytes};
        std::map<std::string, std::string> defs;
        // (the output CB, its tiles a row and the output's accessor: written from here with accumulators_on_ports,
        // passed always)
        ct.push_back(kCbOut);
        ct.push_back(NT);
        ct.insert(ct.end(), o_ct.begin(), o_ct.end());
        accessor(ct, defs, "COMBINE_PLAN_ACCESSOR_FIRST_ARG", CombineBuf::Plan);
        accessor(ct, defs, "COMBINE_HEADER_CACHE_ACCESSOR_FIRST_ARG", CombineBuf::Hcache);
        // (the per-chip accumulator table's accessor; without per_chip_accumulator_table the plan's stands in, unread)
        accessor(
            ct,
            defs,
            "COMBINE_ACCUMULATOR_TABLE_ACCESSOR_FIRST_ARG",
            plan.perchip ? CombineBuf::Lut : CombineBuf::Plan);
        const auto k = kern(R_PW, "dataflow/combine_port_writer.cpp", pset, std::move(ct), s_riscv0, std::move(defs));
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
            if (plan.flat_pool) {
                a.lits(plan.pool_args);
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
                    mesh_device, node(ch), node(mp.peer), mp.links.at(p.li), program, p.core, fab);
                TT_FATAL(
                    fab.size() == 3,
                    "routed_experts_combine: a fabric connection of {} args (build_from_eth_channel takes 3: eth "
                    "channel, "
                    "teardown and buffer-index semaphores)",
                    fab.size());
                a.fabric = std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(a.v.size()), static_cast<uint32_t>(fab.size())};
                a.lits(fab);
            }
            if (plan.acc_on_ports) {  // (the hand-off receiver's args, then the accumulator writer's)
                const auto ai = static_cast<uint32_t>(
                    std::find_if(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_core(x, p.core); }) -
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
                        auto source_ports = port_cores_of(plan.port_core, s);
                        std::sort(source_ports.begin(), source_ports.end(), [](const CoreCoord& x, const CoreCoord& y) {
                            return std::tie(x.x, x.y) < std::tie(y.x, y.y);
                        });
                        for (const auto& c : source_ports) {
                            const auto v = virt(c);
                            bt.insert(bt.end(), {i, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                        }
                    }
                }
                a.lit(static_cast<uint32_t>(bt.size() / 3));
                a.lits(bt);
            }
            put(k, p.core, a);
        }
        if (dual) {  // (the template of a core's port block where the core is no port of this chip: its buffers at a
                     // sender's args)
            Args a;
            a.buf(CombineBuf::Arena);
            a.lits({plan.mesh_id, idx(ch), kIdlePort});
            a.buf(CombineBuf::Bar);
            a.lits({0u, 0u});
            a.buf(CombineBuf::Plan);
            a.lit(0u);
            a.lits(plan.chip_ids);
            if (plan.perchip) {
                a.buf(CombineBuf::Lut);
            } else {
                a.lits(std::vector<uint32_t>(acc_xy.size(), 0u));
            }
            if (plan.flat_pool) {
                a.lits(std::vector<uint32_t>(plan.pool_args.size(), 0u));
            }
            a.lits(std::vector<uint32_t>(2 * N, 0u));
            a.buf(CombineBuf::Hcache);
            pend[R_PW].args[{SIZE_MAX, SIZE_MAX}] = a;
        }
    }

    // accumulators: reader, compute, writer
    {
        std::vector<uint32_t> rd_ct = {
            kCbRow,
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
            combine_formats::acc_landed_word(plan.acc_on_ports),
            combine_formats::acc_handoff_base_word(plan.acc_on_ports),
            plan.loc_off,
            words_off,
            r_slot};
        std::map<std::string, std::string> rd_defs;
        accessor(rd_ct, rd_defs, "COMBINE_PLAN_ACCESSOR_FIRST_ARG", CombineBuf::Plan);
        const auto rd =
            kern(R_AR, "dataflow/accumulator_reader.cpp", aset, std::move(rd_ct), r_riscv0, std::move(rd_defs));
        // (the fp32 sums unpacked straight to DEST: the default narrows a Float32 CB to TF32 on its way)
        std::vector<tt::tt_metal::UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        u2d[kCbAcc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        // (no runtime args - one program for the mesh needs them alike, and a core's accumulator and its tokens differ
        // by chip: its reader sends them as the first meta page, accumulator_reader.cpp /
        // combine_port_or_accumulator_reader.cpp)
        tt::tt_metal::KernelDescriptor compute_kernel;
        compute_kernel.kernel_source = std::string(kKernelDir) + "compute/accumulator_compute.cpp";
        compute_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_kernel.core_ranges = aset;
        compute_kernel.compile_time_args = {kCbRow, kCbMeta, kCbAcc, kCbOut, NT, plan.MAXT, kCbW, plan.snap_bits};
        compute_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;  // (ComputeConfig's default)
        compute_kernel.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = true, .unpack_to_dest_mode = u2d};
        add_kernel(program, std::move(compute_kernel));
        std::vector<CoreCoord> accumulator_writer_cores;
        for (const auto& c : accs) {
            if (!plan.acc_on_ports || !is_port(c)) {
                accumulator_writer_cores.push_back(c);
            }
        }
        std::vector<uint32_t> aw_ct = {kCbOut, pc.row_bytes, NT, A};
        aw_ct.insert(aw_ct.end(), o_ct.begin(), o_ct.end());
        std::optional<tt::tt_metal::KernelHandle> aw;
        if (!accumulator_writer_cores.empty()) {
            aw = kern(
                R_AW,
                "dataflow/accumulator_writer.cpp",
                cores_of(accumulator_writer_cores),
                std::move(aw_ct),
                s_riscv0,
                {});
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
            if (plan.flat_pool) {
                a.lits(plan.pool_args);
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
            a.lits({n_tok});  // (the compute's tokens: the reader's first meta page)
            put(rd, core, a);
            if (!plan.acc_on_ports || !is_port(core)) {
                Args w;
                w.buf(CombineBuf::Out);
                w.lits({n_tok, i});
                put(*aw, core, w);
            }
        }
        // (a port-or-accumulator core that is no accumulator of this chip: its reader's first meta page is 0 - the
        // compute returns)
    }

    // (dual) the port-or-accumulator cores' binaries: the NCRISC's port reader + accumulator reader, the BRISC's sender
    // + accumulator writer (combine_port_or_accumulator_reader / _writer.cpp). A core's args: its role (0 none, 1 port,
    // 2 accumulator), the accumulator block (padded to one length on every chip), the port block - a block not this
    // chip's role holds a template with the buffers at a real block's args, so every chip's address args sit at the
    // same args and the host's cache-hit rewrite keeps every chip's right (one program for the mesh)
    if (dual) {
        auto append = [](Args& to, const Args& b) { to.v.insert(to.v.end(), b.v.begin(), b.v.end()); };
        auto pad = [](Args& a, size_t n) {
            TT_FATAL(a.v.size() <= n, "routed_experts_combine: a role block of {} args over its {}", a.v.size(), n);
            while (a.v.size() < n) {
                a.lit(0u);
            }
        };
        Args pr_t, ar_t, aw_t;
        pr_t.buf(CombineBuf::Arena);
        pr_t.buf(CombineBuf::Plan);
        pr_t.buf(CombineBuf::Bar);
        pr_t.lits({W, kIdlePort});
        ar_t.buf(CombineBuf::Arena);
        ar_t.buf(CombineBuf::Plan);
        ar_t.lits({0u, 0u});
        ar_t.buf(CombineBuf::Bar);
        aw_t.buf(CombineBuf::Out);
        const Args& pw_t = pend[R_PW].args.at({SIZE_MAX, SIZE_MAX});
        // (the accumulator reader's block: its args but the gate's core list - the compute's tokens last - plus room
        // for that list on any core)
        const size_t amax = 7 + 2 * N + (plan.flat_pool ? plan.pool_args.size() : 0) + 3 + 3 + 2 * W + 1 +
                            2 * plan.port_or_accumulator_cores.size();
        constexpr size_t awmax = 3;
        auto merged_defs = [](std::map<std::string, std::string> d,
                              const std::vector<std::pair<std::string, std::string>>& rename) {
            for (const auto& [from, to] : rename) {
                if (const auto it = d.find(from); it != d.end()) {
                    d[to] = it->second;
                    d.erase(it);
                }
            }
            return d;
        };
        // reader: the port reader's compile-time args, then the accumulator reader's
        std::vector<uint32_t> r_ct = pend[R_PR].ct;
        r_ct.insert(r_ct.end(), pend[R_AR].ct.begin(), pend[R_AR].ct.end());
        auto r_defs = merged_defs(
            pend[R_AR].defs, {{"COMBINE_PLAN_ACCESSOR_FIRST_ARG", "ACCUMULATOR_READER_PLAN_ACCESSOR_FIRST_ARG"}});
        r_defs["PORT_READER_PLAN_ACCESSOR_FIRST_ARG"] = pend[R_PR].defs.at("COMBINE_PLAN_ACCESSOR_FIRST_ARG");
        r_defs["ACCUMULATOR_READER_FIRST_COMPILE_TIME_ARG"] = std::to_string(pend[R_PR].ct.size());
        r_defs["PORT_ROLE_FIRST_RUNTIME_ARG"] = std::to_string(1 + amax);
        const auto rk = dm_kernel(
            program,
            "dataflow/combine_port_or_accumulator_reader.cpp",
            pset,
            std::move(r_ct),
            r_riscv0,
            r_defs,
            options);
        // writer: the sender's, then the accumulator writer's
        std::vector<uint32_t> w_ct = pend[R_PW].ct;
        w_ct.insert(w_ct.end(), pend[R_AW].ct.begin(), pend[R_AW].ct.end());
        auto w_defs = merged_defs(
            pend[R_PW].defs,
            {{"COMBINE_PLAN_ACCESSOR_FIRST_ARG", "PORT_WRITER_PLAN_ACCESSOR_FIRST_ARG"},
             {"COMBINE_HEADER_CACHE_ACCESSOR_FIRST_ARG", "PORT_WRITER_HEADER_CACHE_ACCESSOR_FIRST_ARG"},
             {"COMBINE_ACCUMULATOR_TABLE_ACCESSOR_FIRST_ARG", "PORT_WRITER_ACCUMULATOR_TABLE_ACCESSOR_FIRST_ARG"}});
        w_defs["ACCUMULATOR_WRITER_FIRST_COMPILE_TIME_ARG"] = std::to_string(pend[R_PW].ct.size());
        w_defs["PORT_ROLE_FIRST_RUNTIME_ARG"] = std::to_string(1 + awmax);
        const auto wk = dm_kernel(
            program,
            "dataflow/combine_port_or_accumulator_writer.cpp",
            pset,
            std::move(w_ct),
            s_riscv0,
            w_defs,
            options);
        for (const auto& c : plan.port_or_accumulator_cores) {
            const std::pair<size_t, size_t> key{c.x, c.y};
            const uint32_t role = is_port(c) ? 1u : is_acc(c) ? 2u : 0u;
            Args r, w;
            r.lit(role);
            Args ab = role == 2 ? pend[R_AR].args.at(key) : ar_t;
            pad(ab, amax);
            append(r, ab);
            append(r, role == 1 ? pend[R_PR].args.at(key) : pr_t);
            w.lit(role);
            Args wb = role == 2 ? pend[R_AW].args.at(key) : aw_t;
            pad(wb, awmax);
            append(w, wb);
            append(w, role == 1 ? pend[R_PW].args.at(key) : pw_t);
            if (role == 1) {  // (the sender's fabric args, past the role and the accumulator block:
                              // PORT_ROLE_FIRST_RUNTIME_ARG = 1 + awmax)
                if (const auto& pb = pend[R_PW].args.at(key); pb.fabric) {
                    w.fabric = std::pair<uint32_t, uint32_t>{1 + awmax + pb.fabric->first, pb.fabric->second};
                }
            }
            set_args(program, rk, c, r, bufs, h);
            set_args(program, wk, c, w, bufs, h);
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
        at(CombineBuf::Arena, st.MISC + 4 * combine_formats::kMiscPortLanded);
    }
    for (const auto& c : accs) {
        const auto v = virt(c);
        lit(v.x);
        lit(v.y);
        at(CombineBuf::Arena, plan.loc_off);
        lit(plan.QL);
        at(CombineBuf::Arena, st.MISC + 4 * combine_formats::acc_landed_word(plan.acc_on_ports));
    }
    return h;
}

void finalize_combine_handles(tt::tt_metal::Program& program, CombineHandles& h, const CombineBuffers& bufs) {
    // (a CB's handle in the descriptor is its index in its cbs: the program's CB built from it, in that order)
    const auto circular_buffers = program.circular_buffers();
    for (auto& [cb, off] : h.arena_cbs) {
        cb = circular_buffers.at(cb)->id();
    }
    override_combine_addresses(program, h, bufs);
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
