// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_half_merge.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_geometry.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

// Both halves' semaphores live in ONE id space and the two passes reuse it.
//
// They can, because the passes are strictly ordered: pass A is finished everywhere before pass B
// begins anywhere, so no id is ever live for both. The pass barrier is what makes that true, and
// it also has to zero the shared ids -- pass A leaves them at arbitrary values and pass B's
// waits assume they start at zero.
//
// The barrier's own semaphore is therefore the one id that must NOT be reused, so it sits above
// both halves' blocks and is never reset.
uint32_t merge_semaphores(const ProgramDescriptor& fused, const ProgramDescriptor& unified, ProgramDescriptor& out) {
    std::map<uint32_t, SemaphoreDescriptor> by_id;
    for (const auto* half : {&fused, &unified}) {
        for (const auto& sem : half->semaphores) {
            auto [it, inserted] = by_id.emplace(sem.id, sem);
            if (!inserted) {
                // Sharing an id is the design; sharing it with a different shape is not, and the
                // program would silently keep whichever half was emitted first.
                TT_FATAL(
                    it->second.core_ranges == sem.core_ranges && it->second.core_type == sem.core_type &&
                        it->second.initial_value == sem.initial_value,
                    "the two halves disagree on semaphore {}: one id cannot carry two configurations",
                    sem.id);
            }
        }
    }
    TT_FATAL(!by_id.empty(), "neither half declared a semaphore; the pass barrier has nothing to model itself on");

    const uint32_t barrier_id = by_id.rbegin()->first + 1;
    TT_FATAL(
        barrier_id <
            ::ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry::NUM_DEVICE_SEMAPHORES,
        "the merged program needs {} semaphores (both halves' shared block plus the pass barrier) but a core has "
        "{}; the halves' blocks are {} and {} ids wide",
        barrier_id + 1,
        ::ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry::NUM_DEVICE_SEMAPHORES,
        fused.semaphores.size(),
        unified.semaphores.size());

    SemaphoreDescriptor barrier = by_id.begin()->second;
    barrier.id = barrier_id;
    barrier.initial_value = 0;

    for (const auto& [id, sem] : by_id) {
        out.semaphores.push_back(sem);
    }
    out.semaphores.push_back(barrier);
    return barrier_id;
}

// Lays both halves' circular buffers over one L1 arena.
//
// Each half keeps its own descriptors, page sizes and formats untouched and is laid out from the
// arena's base, so the two overlap exactly. The alternative -- folding both halves' indices into
// shared alias groups -- would force every group's total to a multiple of every member's page
// size, and a bf8 tile (1088 B) against a bf16 one (2048 B) gives a 34816 B granule.
//
// Returns the arena bytes a core needs: the larger half, not the sum.
uint32_t overlay_circular_buffers(
    ProgramDescriptor& fused,
    ProgramDescriptor& unified,
    bool run_fused_pass,
    tt::tt_metal::Buffer* l1_arena,
    ProgramDescriptor& out) {
    const uint32_t alignment = tt::tt_metal::hal::get_l1_alignment();

    // Pass A off: its buffers are never touched, so they are not placed and no arena is needed.
    // Every remaining CB is the unified half's and is allocated the ordinary way.
    if (!run_fused_pass) {
        for (auto& cb : unified.cbs) {
            out.cbs.push_back(cb);
        }
        return 0;
    }

    // What one core actually owns in the arena. Every placed byte is checked against it: the
    // offsets below are computed here, while the arena is sized independently by the caller, and
    // nothing else compares the two. Overrunning it would walk into whatever the L1 allocator put
    // next -- no fault, no watcher trip, just corruption.
    const uint32_t arena_bytes_per_core = static_cast<uint32_t>(l1_arena->aligned_size_per_bank());

    auto place_half = [&](ProgramDescriptor& half, const char* which) {
        uint32_t offset = 0;
        for (auto& cb : half.cbs) {
            TT_FATAL(
                cb.buffer == nullptr && cb.tensor == nullptr && cb.global_circular_buffer == nullptr,
                "a half handed over a circular buffer that is already backed by its own allocation; the merge owns "
                "CB placement");
            cb.buffer = l1_arena;
            cb.address_offset = offset;
            offset += (cb.total_size + alignment - 1) / alignment * alignment;
            TT_FATAL(
                offset <= arena_bytes_per_core,
                "the {} half's circular buffers need {} bytes of the shared L1 arena but a core owns only {}; "
                "either half growing its buffers, or a wider model shape, lands here",
                which,
                offset,
                arena_bytes_per_core);
            out.cbs.push_back(cb);
        }
        return offset;
    };

    const uint32_t fused_bytes = place_half(fused, "fused");
    const uint32_t unified_bytes = place_half(unified, "unified");
    // The fused half sets the floor: it sizes its buffers against hal::get_max_worker_l1_unreserved_size(),
    // a static ceiling rather than this device's allocator base, so it does not shrink when the base
    // rises. The unified half does re-fit, and sits well under. Anything that wants L1 back from this
    // op -- a larger kernel-config ring, say -- has to come from the fused half's blocking.
    return std::max(fused_bytes, unified_bytes);
}

// One binary gets one processor configuration, so where the halves disagree the merge has to
// choose, and a wrong choice here is wrong numbers with no error.
//
// bfp8_pack_precise is the one field they genuinely differ on: the fused half sets it, the unified
// half leaves it clear. The merged program keeps it SET -- the more precise packing of the two --
// which means the unified half's bf8 output in a merged program is not bit-for-bit what the
// standalone unified op produces. That is a deliberate, accepted difference; grade that half on
// PCC against the standalone op, not on equality.
//
// Every other field must already agree. Taking a silent majority on math fidelity or the dst
// accumulator would change one half's arithmetic without any signal.
void reconcile_config(const KernelDescriptor& fused, const KernelDescriptor& unified, KernelDescriptor& merged) {
    const auto* f = std::get_if<tt::tt_metal::ComputeConfigDescriptor>(&fused.config);
    if (f == nullptr) {
        return;  // Data-movement kernels carry nothing that can disagree.
    }
    const auto& u = std::get<tt::tt_metal::ComputeConfigDescriptor>(unified.config);
    TT_FATAL(
        f->math_fidelity == u.math_fidelity,
        "the halves disagree on math fidelity ({} vs {}); one compute binary has one setting",
        static_cast<int>(f->math_fidelity),
        static_cast<int>(u.math_fidelity));
    TT_FATAL(
        f->fp32_dest_acc_en == u.fp32_dest_acc_en,
        "the halves disagree on fp32_dest_acc_en ({} vs {}); it also reaches the kernels as a define",
        f->fp32_dest_acc_en,
        u.fp32_dest_acc_en);
    TT_FATAL(
        f->dst_full_sync_en == u.dst_full_sync_en,
        "the halves disagree on dst_full_sync_en ({} vs {})",
        f->dst_full_sync_en,
        u.dst_full_sync_en);
    TT_FATAL(
        f->math_approx_mode == u.math_approx_mode,
        "the halves disagree on math_approx_mode ({} vs {})",
        f->math_approx_mode,
        u.math_approx_mode);

    auto& out = std::get<tt::tt_metal::ComputeConfigDescriptor>(merged.config);
    out.bfp8_pack_precise = f->bfp8_pack_precise || u.bfp8_pack_precise;
    TT_FATAL(f->unpack_to_dest_mode == u.unpack_to_dest_mode, "the halves disagree on unpack_to_dest_mode");
}

// One merged kernel from the two halves' descriptors for the same RISC-V role.
//
// The fused half keeps index 0 of both argument lists so its body needs no rebasing; the unified
// half's indices shift by the length of the fused block, which is exactly what the merged binary
// was compiled with. Runtime args are padded to a uniform fused block length per role so that
// shift is one compile-time constant rather than a per-core value.
KernelDescriptor merge_kernel(
    const KernelDescriptor& fused,
    const KernelDescriptor& unified,
    const std::string& source,
    bool run_fused_pass,
    const PassBarrierPlan& barrier,
    bool is_coordinator_kernel,
    uint32_t barrier_semaphore_id,
    uint32_t shared_semaphore_count,
    MergeReport::Bases& bases_out) {
    TT_FATAL(
        fused.core_ranges == unified.core_ranges,
        "the two halves place this kernel on different cores; a merged binary runs on one grid");
    TT_FATAL(
        fused.config.index() == unified.config.index(),
        "the two halves disagree on this kernel's processor class, so they cannot share a binary");

    KernelDescriptor merged = fused;
    merged.kernel_source = source;
    merged.source_type = KernelDescriptor::SourceType::FILE_PATH;

    // Compile-time args: fused block, then unified's at the base the binary was built with.
    const uint32_t ct_base = static_cast<uint32_t>(fused.compile_time_args.size());
    merged.compile_time_args.insert(
        merged.compile_time_args.end(), unified.compile_time_args.begin(), unified.compile_time_args.end());
    merged.defines.emplace_back("HYB_UNIFIED_CT_BASE", std::to_string(ct_base));

    // Only the unified half uses named compile-time args, so there is nothing to collide with.
    TT_FATAL(
        fused.named_compile_time_args.empty(),
        "the fused half grew named compile-time args; the merge would have to disambiguate them against the unified "
        "half's");
    merged.named_compile_time_args = unified.named_compile_time_args;

    // Defines: both halves derive them from the same op-level activation and bias choice, so a key
    // they share must agree. One that does not would silently compile one half wrong.
    for (const auto& [key, value] : unified.defines) {
        auto it =
            std::find_if(merged.defines.begin(), merged.defines.end(), [&](const auto& d) { return d.first == key; });
        if (it == merged.defines.end()) {
            merged.defines.emplace_back(key, value);
        } else {
            TT_FATAL(
                it->second == value,
                "the two halves define {} differently ({} vs {}); one binary can only carry one value",
                key,
                it->second,
                value);
        }
    }

    // A uniform fused block per role, so the unified half's runtime base is a compile-time
    // constant. Per-core fused lists differ in length (the mcast and per-row blocks vary), and a
    // per-core base would have to be read from L1 before any argument could be fetched.
    uint32_t fused_block = 0;
    for (const auto& [core, args] : fused.runtime_args) {
        fused_block = std::max(fused_block, static_cast<uint32_t>(args.size()));
    }
    merged.defines.emplace_back("HYB_UNIFIED_RT_BASE", std::to_string(fused_block));
    bases_out.ct = ct_base;
    bases_out.rt = fused_block;

    uint32_t unified_block = 0;
    for (const auto& [core, args] : unified.runtime_args) {
        unified_block = std::max(unified_block, static_cast<uint32_t>(args.size()));
    }

    std::map<CoreCoord, const KernelDescriptor::CoreRuntimeArgs*> unified_by_core;
    for (const auto& [core, args] : unified.runtime_args) {
        unified_by_core.emplace(core, &args);
    }
    for (auto& [core, args] : merged.runtime_args) {
        args.resize(fused_block, 0);
        const auto it = unified_by_core.find(core);
        TT_FATAL(
            it != unified_by_core.end(),
            "core {} carries fused runtime args but no unified ones; the halves must cover the same cores",
            core.str());
        args.insert(args.end(), it->second->begin(), it->second->end());
    }
    TT_FATAL(
        merged.runtime_args.size() == unified.runtime_args.size(),
        "the halves disagree on how many cores carry runtime args ({} vs {})",
        merged.runtime_args.size(),
        unified.runtime_args.size());

    // Buffer bindings name an argument POSITION, so the unified half's move with its block. Left
    // unshifted they would patch the fused half's slots on every program-cache hit.
    for (const auto& binding : unified.buffer_bindings) {
        merged.buffer_bindings.push_back(
            tt::tt_metal::BufferBinding{binding.core, binding.arg_idx + fused_block, binding.buffer});
    }
    TT_FATAL(
        unified.common_runtime_args.empty() && fused.common_runtime_args.empty(),
        "a half grew common runtime args; the merge does not rebase them yet");

    reconcile_config(fused, unified, merged);
    if (run_fused_pass) {
        merged.defines.emplace_back("HYB_RUN_FUSED_PASS", "1");
    }
    // The barrier is a NoC rendezvous and a TRISC has no NoC, so hybrid_pass_barrier() is empty on
    // the compute kernel. Handing it the block anyway spends ring on arguments nothing reads, and
    // the ring is what this op is short of. The define goes with the args: without a block to
    // point at, a base is a lie.
    const bool runs_barrier = !std::holds_alternative<tt::tt_metal::ComputeConfigDescriptor>(merged.config);
    if (run_fused_pass && runs_barrier) {
        // The barrier's own runtime-arg block, after both halves'. Appended per core because only
        // one core is the master; everything else in it is grid-wide.
        const uint32_t barrier_base = static_cast<uint32_t>(fused_block + unified_block);
        merged.defines.emplace_back("HYB_BARRIER_RT_BASE", std::to_string(barrier_base));
        for (auto& [core, args] : merged.runtime_args) {
            args.resize(barrier_base, 0);
            // Coordinator, not just master CORE: both data-movement kernels run this body, and a
            // second one entering the master block would re-zero the shared semaphores after
            // pass B had already started on them.
            args.push_back(static_cast<uint32_t>(is_coordinator_kernel && core == barrier.master_logical));
            args.push_back(barrier.master_noc_x);
            args.push_back(barrier.master_noc_y);
            args.push_back(barrier.rect_x_start);
            args.push_back(barrier.rect_y_start);
            args.push_back(barrier.rect_x_end);
            args.push_back(barrier.rect_y_end);
            args.push_back(barrier.num_receivers);
            args.push_back(barrier.total_arrivals);
            args.push_back(barrier_semaphore_id);
            args.push_back(shared_semaphore_count);
        }
    }
    return merged;
}

// The kernels a half emits, keyed by the processor class they run on.
std::map<size_t, const KernelDescriptor*> index_by_role(const ProgramDescriptor& half, const char* which) {
    std::map<size_t, const KernelDescriptor*> by_role;
    for (const auto& kernel : half.kernels) {
        const auto [it, inserted] = by_role.emplace(kernel.config.index(), &kernel);
        TT_FATAL(inserted, "the {} half emits two kernels for the same processor class", which);
    }
    return by_role;
}

}  // namespace

tt::tt_metal::ProgramDescriptor merge_halves(
    ProgramDescriptor fused,
    ProgramDescriptor unified,
    const MergedKernelSources& sources,
    bool run_fused_pass,
    tt::tt_metal::Buffer* l1_arena,
    const PassBarrierPlan& barrier,
    MergeReport& report) {
    TT_FATAL(
        !run_fused_pass || l1_arena != nullptr,
        "pass A runs, so both halves' circular buffers need the shared L1 arena to be laid over");

    ProgramDescriptor merged;
    report.barrier_semaphore_id = merge_semaphores(fused, unified, merged);
    report.semaphore_count = static_cast<uint32_t>(merged.semaphores.size());
    report.arena_bytes_per_core = overlay_circular_buffers(fused, unified, run_fused_pass, l1_arena, merged);

    const auto fused_by_role = index_by_role(fused, "fused");
    const auto unified_by_role = index_by_role(unified, "unified");
    TT_FATAL(
        fused_by_role.size() == unified_by_role.size(),
        "the halves run on a different number of processor classes ({} vs {}), so they cannot be paired",
        fused_by_role.size(),
        unified_by_role.size());

    // Reader, writer and compute, matched by processor class rather than by emission order.
    const std::map<size_t, const std::string*> source_for_role = {
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::ReaderConfigDescriptor{}).index(), &sources.reader},
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::WriterConfigDescriptor{}).index(), &sources.writer},
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::ComputeConfigDescriptor{}).index(), &sources.compute},
    };
    // Keyed by processor class, never by emission or map order: the bases differ per role and
    // attributing one role's to another would be silent.
    const std::map<size_t, MergeReport::Bases*> bases_for_role = {
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::ReaderConfigDescriptor{}).index(), &report.reader},
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::WriterConfigDescriptor{}).index(), &report.writer},
        {KernelDescriptor::ConfigDescriptor(tt::tt_metal::ComputeConfigDescriptor{}).index(), &report.compute},
    };

    for (const auto& [role, fused_kernel] : fused_by_role) {
        const auto unified_it = unified_by_role.find(role);
        TT_FATAL(unified_it != unified_by_role.end(), "the unified half has no kernel for processor class {}", role);
        const auto source_it = source_for_role.find(role);
        TT_FATAL(source_it != source_for_role.end(), "no merged kernel source for processor class {}", role);
        const auto bases_it = bases_for_role.find(role);
        TT_FATAL(bases_it != bases_for_role.end(), "no report slot for processor class {}", role);

        merged.kernels.push_back(merge_kernel(
            *fused_kernel,
            *unified_it->second,
            *source_it->second,
            run_fused_pass,
            barrier,
            /*is_coordinator_kernel=*/role ==
                KernelDescriptor::ConfigDescriptor(tt::tt_metal::ReaderConfigDescriptor{}).index(),
            report.barrier_semaphore_id,
            /*shared_semaphore_count=*/report.barrier_semaphore_id,
            *bases_it->second));
    }
    return merged;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
