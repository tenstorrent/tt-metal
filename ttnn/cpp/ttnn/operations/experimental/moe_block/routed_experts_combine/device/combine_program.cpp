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
#include "ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/combine_names.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace names = combine_names;  // (the kernels' named compile-time args, shared with the kernels)

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
    std::erase_if(h.fabric_args, [&](const auto& f) { return f.kernel == k && f.core == core; });
    if (a.fabric) {
        h.fabric_args.push_back({k, core, a.fabric->first, a.fabric->second});
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
}

// A kernel's named compile-time args (combine_names.hpp), in the order they were added
using NamedArgs = std::vector<std::pair<std::string, uint32_t>>;

// `into` with `more` added: a name already there must have the same value (the two roles of a two-role kernel share
// names with equal values; the combine's options and a role's names are distinct)
void merge_named(NamedArgs& into, const NamedArgs& more) {
    for (const auto& [name, value] : more) {
        const auto existing =
            std::find_if(into.begin(), into.end(), [&](const auto& arg) { return arg.first == name; });
        if (existing == into.end()) {
            into.emplace_back(name, value);
        } else {
            TT_FATAL(
                existing->second == value,
                "routed_experts_combine: named compile-time arg {} is both {} and {}",
                name,
                existing->second,
                value);
        }
    }
}

// A data-movement kernel's accessor args end the positional compile-time args `ct`: their first index, `first_arg` past
// the other role's args in a two-role kernel, is the named arg `name`
void append_accessor(
    std::vector<uint32_t>& ct,
    NamedArgs& named,
    const char* name,
    const tt::tt_metal::Buffer& buffer,
    uint32_t first_arg) {
    named.emplace_back(name, first_arg + static_cast<uint32_t>(ct.size()));
    tt::tt_metal::TensorAccessorArgs(buffer).append_to(ct);
}

tt::tt_metal::KernelHandle dm_kernel(
    tt::tt_metal::ProgramDescriptor& program,
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    bool riscv0,
    NamedArgs named,
    const NamedArgs& options) {
    // dedicated NoCs, fixed per RISC: RISCV_0 on NoC1, RISCV_1 on NoC0
    tt::tt_metal::KernelDescriptor kernel_descriptor;
    kernel_descriptor.kernel_source = std::string(kKernelDir) + file;
    kernel_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel_descriptor.core_ranges = cores;
    // (the positional compile-time args are the accessor args only; the kernel's named args, then the combine's
    // options: kernels/dataflow/combine_options.hpp)
    merge_named(named, options);
    kernel_descriptor.compile_time_args = std::move(ct);
    kernel_descriptor.named_compile_time_args = std::move(named);
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
    plan->fabric_links = mesh.num_links;
    plan->port_luts = resources.port_luts();
    plan->snap_bits = cfg.snap_bits();
    plan->home_tokens = in.home_tokens;
    plan->row_elements = in.row_elements;
    const uint32_t N = mesh.topo.num_chips();
    TT_FATAL(N <= kMaxChips, "routed_experts_combine: {} chips > {}", N, kMaxChips);
    plan->row_slot = row_slot(in.row_elements);
    // (the L1 hand-off's y writers write a row's columns at slot + column offset and its trailer at slot + row bytes:
    // one fabric packet a row, so the payload must hold a row + 32 B, else the combine waits for trailers that never
    // come)
    TT_FATAL(
        resources.fabric_max_payload_bytes() >= plan->row_slot.row_bytes + 32,
        "routed_experts_combine: the fabric payload {} B < a row + its trailer ({} B)",
        resources.fabric_max_payload_bytes(),
        plan->row_slot.row_bytes + 32);

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
            const uint32_t n = std::min<uint32_t>((e - b) / plan->row_slot.slot_bytes, 0xFFFF);
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
            in.home_tokens,
            in.row_elements,
            pool.size(),
            n_ports,
            in.arena_bytes,
            in.depth,
            in.accumulator_count.value_or(kAccMax),
            in.mode,
            l,
            pool_slots);
    };
    // no landing given: the rings in the accumulators' L1 or on the flat expert cores, whichever ranks higher
    // (combine_choice_key). 2x4 GLM-5.3-Flash experts + combine: chunk 5120 with 1 landing slot a source and an 8-slot
    // injection ring -> 8 and 16 on the flat expert cores 2762 -> 1543 us, 1280 with 8 a source and a 3-slot local
    // ring -> 8 and 16 513 -> 489, 2560 702 -> 708
    CombineLanding landing = in.landing.value_or(CombineLanding::Rings);
    auto au = plan_with(landing);
    if (!in.landing && pool_slots > 0) {
        auto fc = plan_with(CombineLanding::FlatCores);
        const uint32_t a_ref =
            std::max(au.fits ? au.best->accumulator_count : 0u, fc.fits ? fc.best->accumulator_count : 0u);
        auto key = [&](const CombineAuto& x) { return combine_choice_key(*x.best, a_ref); };
        if (fc.fits && (!au.fits || key(fc) > key(au))) {
            au = std::move(fc);
            landing = CombineLanding::FlatCores;
        }
    }
    const bool landing_on_flat_expert_cores = landing_on_flat(landing);
    TT_FATAL(
        !landing_on_flat_expert_cores || pool_slots > 0,
        "routed_experts_combine: FlatCores landing - no free slot on the flat expert cores (arena {} KB)",
        in.arena_bytes >> 10);
    TT_FATAL(
        au.fits,
        "routed_experts_combine: does not fit ({} cores, {} KB): {}",
        pool.size(),
        in.arena_bytes >> 10,
        au.why);
    TT_FATAL(
        !in.accumulator_count || au.best->accumulator_count == *in.accumulator_count,
        "routed_experts_combine: {} accumulators, {} planned",
        au.best->accumulator_count,
        *in.accumulator_count);
    plan->choice = *au.best;
    plan->accumulators_on_ports = au.best->accumulators_on_ports;
    plan->per_chip_accumulator_table = !plan->accumulators_on_ports;
    plan->shared_landing = au.best->shared_landing;
    const uint32_t accumulator_count = au.best->accumulator_count;

    // ports and accumulators (sparse_combine_rf_prepare)
    if (plan->accumulators_on_ports) {  // the accumulator cores: every chip's nearest-to-Ethernet port cores, then
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
            aset.size() <= accumulator_count,
            "accumulators_on_ports: {} port cores over the chips > {} accumulators",
            aset.size(),
            accumulator_count);
        for (const auto& c : pool) {
            if (aset.size() >= accumulator_count) {
                break;
            }
            if (std::none_of(aset.begin(), aset.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
                aset.push_back(c);
            }
        }
        plan->port_core = resources.ports(aset);
        for (const auto& [ch, _] : mesh.chips) {
            plan->accumulator_cores[ch] = aset;
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
                if (v.size() >= accumulator_count) {
                    break;
                }
                if (!same_core(c, never[0]) &&
                    std::none_of(own.begin(), own.end(), [&](const CoreCoord& x) { return same_core(x, c); })) {
                    v.push_back(c);
                }
            }
            A_ = std::min(A_, v.size());
            plan->accumulator_cores[ch] = std::move(v);
        }
        for (auto& [ch, v] : plan->accumulator_cores) {
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
        for (const auto& [ch, v] : plan->accumulator_cores) {
            for (const auto& c : v) {
                add_port_or_accumulator_core(c);
            }
        }
    }
    const uint32_t A = plan->accumulator_cores.begin()->second.size();
    TT_FATAL(A <= kAccMax, "routed_experts_combine: {} accumulators > {}", A, kAccMax);
    plan->accumulator_count = A;
    // (the ports' end fence checks one layout word of each home on their link: an accumulator o with
    // (o + src + home) % L == link exists for every residue when A >= L)
    TT_FATAL(
        N == 1 || A >= plan->fabric_links, "routed_experts_combine: {} accumulators < {} links", A, plan->fabric_links);
    plan->max_tokens_per_accumulator = (in.home_tokens + A - 1) / A;
    plan->collector = plan->accumulator_cores.begin()->second.at(0);
    for (const auto& [ch, v] : plan->accumulator_cores) {
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
    plan->state = state_layout(N, A, plan->shared_landing ? kGrants : 0);
    const uint32_t slot_bytes = plan->row_slot.slot_bytes;
    plan->row_depth = landing_on_flat_expert_cores ? kPoolRows : 2u;
    uint32_t accumulator_cb_bytes = acc_cb_bytes(in.home_tokens, A, plan->row_slot.row_tiles, plan->row_depth);
    const uint32_t injection_ring_slots_wanted = au.best->injection_ring_slots,
                   local_ring_slots_reserved = au.best->local_ring_slots_reserved;
    const int64_t kept =
        plan->accumulators_on_ports
            ? static_cast<int64_t>(local_ring_slots_reserved + std::min(32u, injection_ring_slots_wanted)) * slot_bytes
            : static_cast<int64_t>(local_ring_slots_reserved) * slot_bytes;
    const int64_t fit =
        landing_on_flat_expert_cores
            ? static_cast<int64_t>(pool_slots / A)
            : (static_cast<int64_t>(in.arena_bytes) - plan->state.rings_offset - accumulator_cb_bytes - kept) /
                  slot_bytes;
    const int64_t landing_slots = std::min<int64_t>(
        {static_cast<int64_t>(in.depth) * (N - 1),
         fit,
         !plan->shared_landing ? std::numeric_limits<int64_t>::max()
                               : int64_t(landing_on_flat_expert_cores ? kSharedFlatPslMax : kSharedPslMax)});
    TT_FATAL(
        landing_slots >= static_cast<int64_t>(N) - 1,
        "routed_experts_combine: no room for one landing slot per source chip");
    TT_FATAL(landing_slots < 4096, "routed_experts_combine: a ring's base / depth are 12-bit fields");
    plan->landing_slots_per_accumulator = static_cast<uint32_t>(landing_slots);
    plan->accumulator_table_cb_bytes = acc_tab_bytes(plan->shared_landing, plan->landing_slots_per_accumulator);
    accumulator_cb_bytes =
        acc_cb_bytes(in.home_tokens, A, plan->row_slot.row_tiles, plan->row_depth, plan->accumulator_table_cb_bytes);
    plan->landing_slots_per_source = N > 1 ? plan->landing_slots_per_accumulator / (N - 1) : 0;
    plan->credit_policy = credit_policy(N, plan->landing_slots_per_source, plan->shared_landing, in.credit_batch);
    const uint32_t accumulator_area =
        plan->state.rings_offset +
        (landing_on_flat_expert_cores ? 0u : plan->landing_slots_per_accumulator * slot_bytes);
    if (landing_on_flat_expert_cores) {  // (every flat expert core, in the flat plan's order)
        plan->landing_on_flat_expert_cores = true;
        plan->pool_slots = pool_slots;
        plan->pool_args = std::move(pool_args);
    }
    uint32_t injection_ring_slots = injection_ring_slots_wanted, local_ring_slots = 0;
    if (plan->accumulators_on_ports) {  // (local ring up to 16 slots, then the injection ring, from what is left)
        const int64_t accumulator_area_end = align64(accumulator_area) + accumulator_cb_bytes;
        local_ring_slots = static_cast<uint32_t>(std::min<int64_t>(
            16,
            (in.arena_bytes - accumulator_area_end - std::min(32u, injection_ring_slots) * slot_bytes) / slot_bytes));
        injection_ring_slots = static_cast<uint32_t>(std::min<int64_t>(
            injection_ring_slots,
            (in.arena_bytes - accumulator_area_end - static_cast<int64_t>(local_ring_slots) * slot_bytes) /
                slot_bytes));
        TT_FATAL(
            local_ring_slots >= 2 && injection_ring_slots >= 8,
            "routed_experts_combine: accumulators on the ports: local ring {} / injection ring {} slots",
            local_ring_slots,
            injection_ring_slots);
    } else {
        local_ring_slots =
            std::min<uint32_t>(16, (in.arena_bytes - align64(accumulator_area) - accumulator_cb_bytes) / slot_bytes);
        TT_FATAL(local_ring_slots >= 2, "routed_experts_combine: no room for the accumulators' local ring");
    }
    plan->injection_ring_slots = injection_ring_slots;
    plan->local_ring_slots = local_ring_slots;
    // the placement (sparse_combine_rf_prepare.build): ports apart - the injection ring at the state block's end, the
    // accumulator table after it, then CB_TAB; accumulators - CB_TAB and their CBs after the landing pool, then the
    // local ring (accumulators on the ports: then the injection ring)
    plan->arena_bytes = in.arena_bytes;
    plan->accumulator_table_bytes = plan->per_chip_accumulator_table ? align64(2 * N * A) : 0;
    const uint32_t port_area = plan->state.rings_offset + injection_ring_slots * slot_bytes;
    plan->accumulator_table_offset = align64(port_area);
    plan->port_table_cb_offset = align64(plan->accumulator_table_offset + plan->accumulator_table_bytes);
    uint32_t e = align64(accumulator_area);
    auto take = [&](uint32_t bytes) {
        const uint32_t o = e;
        e = align64(e + bytes);
        return o;
    };
    const uint32_t row_tiles = plan->row_slot.row_tiles;
    plan->accumulator_table_cb_offset = take(plan->accumulator_table_cb_bytes);
    plan->row_cb_offset = take(plan->row_depth * row_tiles * kTile);
    plan->meta_cb_offset = take(plan->row_depth * 32);
    plan->accumulator_fp32_offset = take(plan->max_tokens_per_accumulator * row_tiles * 4096);
    plan->output_cb_offset = take(2 * row_tiles * kTile);
    plan->weight_cb_offset = take(plan->row_depth * kTile);
    plan->local_ring_offset = e;
    TT_FATAL(
        std::max(e, plan->accumulators_on_ports ? 0u : plan->port_table_cb_offset + kTabCbBytes) <= in.arena_bytes,
        "routed_experts_combine: needs {} KB a core, the arena has {} KB",
        std::max(e, plan->port_table_cb_offset + kTabCbBytes) >> 10,
        in.arena_bytes >> 10);
    if (plan->accumulators_on_ports) {
        plan->injection_ring_offset = plan->local_ring_offset + local_ring_slots * slot_bytes;
        TT_FATAL(
            plan->injection_ring_offset + injection_ring_slots * slot_bytes <= in.arena_bytes,
            "routed_experts_combine: accumulators on the ports: no room for the injection ring");
    } else {
        local_ring_slots = std::min<uint32_t>(16, (in.arena_bytes - plan->local_ring_offset) / slot_bytes);
        TT_FATAL(local_ring_slots >= 2, "routed_experts_combine: no room for the accumulators' local ring");
        plan->local_ring_slots = local_ring_slots;
        plan->injection_ring_offset = plan->state.rings_offset;
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
    if (plan->per_chip_accumulator_table) {
        plan->accumulator_table.assign(plan->accumulator_table_bytes / 2, 0);
        for (const auto& [ch, v] : plan->accumulator_cores) {
            for (uint32_t o = 0; o < v.size(); ++o) {
                const auto vc = mesh_device.worker_core_from_logical_core(v[o]);
                TT_FATAL(
                    vc.x < 256 && vc.y < 256, "routed_experts_combine: virtual core {} beyond the LUT's bytes", vc);
                plan->accumulator_table[topo.index(ch) * A + o] = static_cast<uint16_t>(vc.x | vc.y << 8);
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
    const uint32_t N = topo.num_chips(), L = plan.fabric_links, A = plan.accumulator_count;
    const auto& st = plan.state;
    const auto& pc = plan.row_slot;
    const uint32_t SLOT = pc.slot_bytes, ISLOT = pc.slot_bytes, NT = pc.row_tiles;
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
    const auto& accs = plan.accumulator_cores.at(ch);
    // ports apart: every port-or-accumulator core runs both roles' binaries (plan.port_or_accumulator_cores;
    // combine_port_or_accumulator_reader / _writer.cpp)
    const bool dual = !plan.accumulators_on_ports;
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
            tt::tt_metal::CBDescriptor circular_buffer{.total_size = total, .core_ranges = cores};
            circular_buffer.format_descriptors.push_back(
                tt::tt_metal::CBFormatDescriptor{.buffer_index = i, .data_format = fmt, .page_size = page});
            circular_buffer.buffer = const_cast<Buffer*>(&arena);
            circular_buffer.address_offset = off;
            program.cbs.push_back(std::move(circular_buffer));
        };
    arena_cb(
        kCbInj, pset, plan.injection_ring_offset, plan.injection_ring_slots * ISLOT, tt::DataFormat::UInt32, ISLOT);
    if (dual) {  // (the port reader's table at its own index: the core's accumulator table is kCbTab)
        arena_cb(kCbTabP, pset, plan.port_table_cb_offset, kTabCbBytes, tt::DataFormat::UInt32, kTabCbBytes);
    }
    arena_cb(
        kCbTab,
        aset,
        plan.accumulator_table_cb_offset,
        plan.accumulator_table_cb_bytes,
        tt::DataFormat::UInt32,
        plan.accumulator_table_cb_bytes);
    arena_cb(kCbRow, aset, plan.row_cb_offset, plan.row_depth * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbMeta, aset, plan.meta_cb_offset, plan.row_depth * 32, tt::DataFormat::UInt32, 32);
    arena_cb(
        kCbAcc,
        aset,
        plan.accumulator_fp32_offset,
        plan.max_tokens_per_accumulator * NT * 4096,
        tt::DataFormat::Float32,
        4096);
    arena_cb(kCbOut, aset, plan.output_cb_offset, 2 * NT * kTile, tt::DataFormat::Float16_b, kTile);
    arena_cb(kCbW, aset, plan.weight_cb_offset, plan.row_depth * kTile, tt::DataFormat::Float16_b, kTile);

    // the zero gate: accumulator 0 releases every combine core once the zeroing kernels counted in (ports' senders
    // unless they share an accumulator's core, the accumulators' readers, the flat expert's y writers)
    const auto gate = virt(accs.at(0));
    std::vector<CoreCoord> comb = accs;
    for (const auto& c : port_cores) {
        if (!is_acc(c)) {
            comb.push_back(c);
        }
    }
    const uint32_t nz =
        (plan.accumulators_on_ports ? 0u : static_cast<uint32_t>(port_cores.size())) + A + flat.writers.size();

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
    // (dual: each role's kernel is recorded - its compile-time args, named args and each core's args - and the cores'
    // two binaries composed after the accumulators)
    struct Pend {
        std::vector<uint32_t> ct;
        NamedArgs named;
        std::map<std::pair<size_t, size_t>, Args> args;
    };
    enum : uint32_t { R_PR = 0, R_PW = 1, R_AR = 2, R_AW = 3 };
    constexpr tt::tt_metal::KernelHandle kPend = 0x10000;
    std::array<Pend, 4> pend;
    // the combine's options, every combine data-movement kernel's named compile-time args
    // (kernels/dataflow/combine_options.hpp)
    const NamedArgs options = {
        {names::kAccumulatorsOnPorts, plan.accumulators_on_ports ? 1u : 0u},
        {names::kPerChipAccumulatorTable, plan.per_chip_accumulator_table ? 1u : 0u},
        {names::kSharedLanding, plan.shared_landing ? 1u : 0u},
        {names::kSharedLandingGrantsPerSource, plan.shared_landing ? st.grants_per_source : 1u},
        {names::kSharedLandingGrantIdsOffset, plan.shared_landing ? st.grant_ids_offset : 0u},
        {names::kSharedLandingGrantSnapshotsOffset, plan.shared_landing ? st.grant_snapshots_offset : 0u},
        {names::kLandingOnFlatExpertCores, plan.landing_on_flat_expert_cores ? 1u : 0u},
        {names::kLandingReadsInFlight, plan.landing_on_flat_expert_cores ? plan.row_depth : 2u}};
    auto kern = [&](uint32_t role,
                    const std::string& file,
                    const CoreRangeSet& cores,
                    std::vector<uint32_t> ct,
                    bool riscv0,
                    NamedArgs named) -> tt::tt_metal::KernelHandle {
        if (dual) {
            pend[role].ct = std::move(ct);
            pend[role].named = std::move(named);
            return kPend + role;
        }
        return dm_kernel(program, file, cores, std::move(ct), riscv0, std::move(named), options);
    };
    auto put = [&](tt::tt_metal::KernelHandle k, const CoreCoord& core, const Args& a) {
        if (k >= kPend) {
            pend[k - kPend].args[{core.x, core.y}] = a;
        } else {
            set_args(program, k, core, a, bufs, h);
        }
    };
    std::map<std::pair<size_t, size_t>, uint32_t> acc_tok;  // (dual: the compute's tokens a core)

    // port readers (ports apart only: ports that also accumulate run the hand-off on the sender)
    if (!plan.accumulators_on_ports) {
        // (positional compile-time args: the device plan's accessor args)
        std::vector<uint32_t> ct;
        NamedArgs named = {
            {names::kInjectionCb, kCbInj},
            {names::kPortTableCb, dual ? kCbTabP : kCbTab},
            {names::kMiscWordsOffset, st.misc_words_offset},
            {names::kYWriterWordsOffset, words_off},
            {names::kBarrierCallSlot, r_slot}};
        append_accessor(ct, named, names::kPortReaderPlanAccessorFirstArg, *bufs[CombineBuf::Plan], 0);
        const auto k = kern(R_PR, "dataflow/combine_port_reader.cpp", pset, std::move(ct), r_riscv0, std::move(named));
        for (uint32_t pid = 0; pid < n_pid; ++pid) {
            Args a;
            a.buf(CombineBuf::Arena);
            a.buf(CombineBuf::Plan);
            a.buf(CombineBuf::Bar);
            a.lits({W, pid, plan.injection_ring_slots});
            a.lits(wxy);
            put(k, my[pid].core, a);
        }
    }

    // port senders
    {
        // (the output CB, its tiles a row and the output's accessor: written from here with accumulators_on_ports,
        // passed always; the zero flag the kernel derives from accumulators_on_ports)
        NamedArgs named = {
            {names::kInjectionCb, kCbInj},
            {names::kRowBytes, pc.row_bytes},
            {names::kLandingSlotBytes, SLOT},
            {names::kLandingSlotsPerAccumulator, plan.landing_slots_per_accumulator},
            {names::kAccumulatorCount, A},
            {names::kChipCount, N},
            {names::kFabricLinks, L},
            {names::kInjectionSlotBytes, ISLOT},
            {names::kCreditsReceivedOffset, st.credits_received_offset},
            {names::kRowsSentOffset, st.rows_sent_offset},
            {names::kCreditsToReturnOffset, st.credits_to_return_offset},
            {names::kMiscWordsOffset, st.misc_words_offset},
            {names::kRowHeadersOffset, st.row_headers_offset},
            {names::kRingsOffset, st.rings_offset},
            {names::kRingLayoutsReceivedOffset, st.ring_layouts_received_offset},
            {names::kRingLayoutsToForwardOffset, st.ring_layouts_to_forward_offset},
            {names::kRowsSentAtCallStartOffset, st.rows_sent_at_call_start_offset},
            {names::kReturnCreditsAfterRows, plan.credit_policy.return_credits_after_rows},
            {names::kIdleSnapshotRows, plan.credit_policy.idle_snapshot_rows},
            {names::kIdleSnapshotAgeCycles, plan.credit_policy.idle_snapshot_age_cycles},
            {names::kRowHeaderCount, kNumHeaders},
            {names::kRowHeaderStride, kHeaderStride},
            {names::kBarrierCallSlot, s_slot},
            {names::kYWriterWordsOffset, words_off},
            {names::kAccumulatorTableOffset, plan.accumulator_table_offset},
            {names::kAccumulatorTableBytes, plan.accumulator_table_bytes},
            {names::kOutputCb, kCbOut},
            {names::kRowTiles, NT}};
        // (positional compile-time args: the accessor args - the output's, the device plan's, the routed-header
        // cache's and the per-chip accumulator table's; without per_chip_accumulator_table the plan's stands in,
        // unread)
        std::vector<uint32_t> ct;
        append_accessor(ct, named, names::kPortWriterOutputAccessorFirstArg, *bufs[CombineBuf::Out], 0);
        append_accessor(ct, named, names::kPortWriterPlanAccessorFirstArg, *bufs[CombineBuf::Plan], 0);
        append_accessor(ct, named, names::kPortWriterHeaderCacheAccessorFirstArg, *bufs[CombineBuf::Hcache], 0);
        append_accessor(
            ct,
            named,
            names::kPortWriterAccumulatorTableAccessorFirstArg,
            *bufs[plan.per_chip_accumulator_table ? CombineBuf::Lut : CombineBuf::Plan],
            0);
        const auto k = kern(R_PW, "dataflow/combine_port_writer.cpp", pset, std::move(ct), s_riscv0, std::move(named));
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
            if (plan.per_chip_accumulator_table) {
                a.buf(CombineBuf::Lut);
            } else {
                a.lits(acc_xy);
            }
            if (plan.landing_on_flat_expert_cores) {
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
                    "routed_experts_combine: a fabric connection of {} args (expected 3: eth channel, teardown and "
                    "buffer-index semaphores)",
                    fab.size());
                a.fabric =
                    std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(a.v.size()), static_cast<uint32_t>(fab.size())};
                a.lits(fab);
            }
            if (plan.accumulators_on_ports) {  // (the hand-off receiver's args, then the accumulator writer's)
                const auto ai = static_cast<uint32_t>(
                    std::find_if(accs.begin(), accs.end(), [&](const CoreCoord& x) { return same_core(x, p.core); }) -
                    accs.begin());
                a.lits({W, pid, plan.injection_ring_slots});
                a.buf(CombineBuf::Out);
                a.lits({ai < plan.home_tokens ? (plan.home_tokens - ai + A - 1) / A : 0u, ai});
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
            if (plan.per_chip_accumulator_table) {
                a.buf(CombineBuf::Lut);
            } else {
                a.lits(std::vector<uint32_t>(acc_xy.size(), 0u));
            }
            if (plan.landing_on_flat_expert_cores) {
                a.lits(std::vector<uint32_t>(plan.pool_args.size(), 0u));
            }
            a.lits(std::vector<uint32_t>(2 * N, 0u));
            a.buf(CombineBuf::Hcache);
            pend[R_PW].args[{SIZE_MAX, SIZE_MAX}] = a;
        }
    }

    // accumulators: reader, compute, writer
    {
        // (positional compile-time args: the device plan's accessor args, past the port reader's in a two-role kernel)
        std::vector<uint32_t> rd_ct;
        NamedArgs rd_named = {
            {names::kRowCb, kCbRow},
            {names::kMetaCb, kCbMeta},
            {names::kAccumulatorTableCb, kCbTab},
            {names::kRowBytes, pc.row_bytes},
            {names::kLandingSlotBytes, SLOT},
            {names::kLandingSlotsPerAccumulator, plan.landing_slots_per_accumulator},
            {names::kAccumulatorCount, A},
            {names::kRowTiles, NT},
            {names::kChipCount, N},
            {names::kFabricLinks, L},
            {names::kCreditsToReturnOffset, st.credits_to_return_offset},
            {names::kMiscWordsOffset, st.misc_words_offset},
            {names::kRingsOffset, st.rings_offset},
            {names::kRingLayoutsToForwardOffset, st.ring_layouts_to_forward_offset},
            {names::kWeightCb, kCbW},
            {names::kLocalRingLandedWord, combine_formats::acc_landed_word(plan.accumulators_on_ports)},
            {names::kLocalRingHandoffBaseWord, combine_formats::acc_handoff_base_word(plan.accumulators_on_ports)},
            {names::kLocalRingOffset, plan.local_ring_offset},
            {names::kYWriterWordsOffset, words_off},
            {names::kBarrierCallSlot, r_slot}};
        append_accessor(
            rd_ct,
            rd_named,
            names::kAccumulatorReaderPlanAccessorFirstArg,
            *bufs[CombineBuf::Plan],
            dual ? static_cast<uint32_t>(pend[R_PR].ct.size()) : 0u);
        const auto rd =
            kern(R_AR, "dataflow/accumulator_reader.cpp", aset, std::move(rd_ct), r_riscv0, std::move(rd_named));
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
        compute_kernel.named_compile_time_args = {
            {names::kRowCb, kCbRow},
            {names::kMetaCb, kCbMeta},
            {names::kAccumulatorCb, kCbAcc},
            {names::kOutputCb, kCbOut},
            {names::kRowTiles, NT},
            {names::kMaxTokensPerAccumulator, plan.max_tokens_per_accumulator},
            {names::kWeightCb, kCbW},
            {names::kSnapOffsetBits, plan.snap_bits}};
        compute_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;  // (ComputeConfig's default)
        compute_kernel.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = true, .unpack_to_dest_mode = u2d};
        add_kernel(program, std::move(compute_kernel));
        std::vector<CoreCoord> accumulator_writer_cores;
        for (const auto& c : accs) {
            if (!plan.accumulators_on_ports || !is_port(c)) {
                accumulator_writer_cores.push_back(c);
            }
        }
        // (positional compile-time args: the output's accessor args, past the sender's in a two-role kernel)
        std::vector<uint32_t> aw_ct;
        NamedArgs aw_named = {
            {names::kOutputCb, kCbOut},
            {names::kRowBytes, pc.row_bytes},
            {names::kRowTiles, NT},
            {names::kAccumulatorCount, A}};
        append_accessor(
            aw_ct,
            aw_named,
            names::kAccumulatorWriterOutputAccessorFirstArg,
            *bufs[CombineBuf::Out],
            dual ? static_cast<uint32_t>(pend[R_PW].ct.size()) : 0u);
        std::optional<tt::tt_metal::KernelHandle> aw;
        if (!accumulator_writer_cores.empty()) {
            aw = kern(
                R_AW,
                "dataflow/accumulator_writer.cpp",
                cores_of(accumulator_writer_cores),
                std::move(aw_ct),
                s_riscv0,
                std::move(aw_named));
        }
        for (uint32_t i = 0; i < A; ++i) {
            const auto& core = accs[i];
            const uint32_t n_tok = i < plan.home_tokens ? (plan.home_tokens - i + A - 1) / A : 0;
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
            if (plan.landing_on_flat_expert_cores) {
                a.lits(plan.pool_args);
            }
            a.lits({i == 0 ? 1u : 0u, nz, i == 0 ? static_cast<uint32_t>(comb.size()) : 0u});  // (the gate)
            if (i == 0) {
                for (const auto& c : comb) {
                    const auto v = virt(c);
                    a.lits({static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
                }
            }
            a.lits({plan.local_ring_slots, W, n_pid + i});
            a.lits(wxy);
            a.lits({n_tok});  // (the compute's tokens: the reader's first meta page)
            put(rd, core, a);
            if (!plan.accumulators_on_ports || !is_port(core)) {
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
        const size_t amax = 7 + 2 * N + (plan.landing_on_flat_expert_cores ? plan.pool_args.size() : 0) + 3 + 3 +
                            2 * W + 1 + 2 * plan.port_or_accumulator_cores.size();
        constexpr size_t awmax = 3;
        // reader: the port reader's compile-time args, then the accumulator reader's (its accessor first arg already
        // past the port reader's); the roles' named args merged
        std::vector<uint32_t> r_ct = pend[R_PR].ct;
        r_ct.insert(r_ct.end(), pend[R_AR].ct.begin(), pend[R_AR].ct.end());
        NamedArgs r_named = pend[R_PR].named;
        merge_named(r_named, pend[R_AR].named);
        r_named.emplace_back(names::kPortRoleFirstRuntimeArg, static_cast<uint32_t>(1 + amax));
        const auto rk = dm_kernel(
            program,
            "dataflow/combine_port_or_accumulator_reader.cpp",
            pset,
            std::move(r_ct),
            r_riscv0,
            std::move(r_named),
            options);
        // writer: the sender's, then the accumulator writer's
        std::vector<uint32_t> w_ct = pend[R_PW].ct;
        w_ct.insert(w_ct.end(), pend[R_AW].ct.begin(), pend[R_AW].ct.end());
        NamedArgs w_named = pend[R_PW].named;
        merge_named(w_named, pend[R_AW].named);
        w_named.emplace_back(names::kPortRoleFirstRuntimeArg, static_cast<uint32_t>(1 + awmax));
        const auto wk = dm_kernel(
            program,
            "dataflow/combine_port_or_accumulator_writer.cpp",
            pset,
            std::move(w_ct),
            s_riscv0,
            std::move(w_named),
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
                              // port_role_first_runtime_arg = 1 + awmax)
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
    h.counts_offset = st.counts_offset;
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
        at(CombineBuf::Arena, plan.injection_ring_offset);
        lit(plan.injection_ring_slots);
        at(CombineBuf::Arena, st.misc_words_offset + 4 * combine_formats::kMiscPortLanded);
    }
    for (const auto& c : accs) {
        const auto v = virt(c);
        lit(v.x);
        lit(v.y);
        at(CombineBuf::Arena, plan.local_ring_offset);
        lit(plan.local_ring_slots);
        at(CombineBuf::Arena, st.misc_words_offset + 4 * combine_formats::acc_landed_word(plan.accumulators_on_ports));
    }
    return h;
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
