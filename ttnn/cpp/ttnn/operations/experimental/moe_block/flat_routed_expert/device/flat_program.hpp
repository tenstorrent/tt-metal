// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The flat routed expert as a program fragment another op composes into its own program (one program per chip):
// routed_experts_combine builds each chip's Program, adds the flat expert's kernels / CBs / semaphores with
// add_flat_routed_expert_program, then its own kernels on the cells the flat plan leaves free, and re-applies the
// flat expert's buffer addresses on every program-cache hit with override_flat_routed_expert_addresses.
//
// Contract (what the caller provides and may rely on):
//  * Cores: the flat expert occupies FlatRoutedExpertPlan::arena_cores() (gate/up, down, relays, idle gate/up cells,
//    an idle reader cell). The caller's kernels go on other cores. The flat expert creates 16 semaphores (ids
//    0..15) on its cores only, and its CBs on its cores only (ids 0, 1, 2, 3, 6, 7, 16).
//  * L1: every arena-backed CB lies inside `tensors.arena` (>= plan.arena_tiles x 32 rows a core, sharded over at
//    least the plan's arena cores). The flat expert's static CBs take at most plan.static_l1_bytes a core below it.
//    The caller may lay its own buffers in the same arena on its own cells (e.g. a persistent arena the caller owns).
//  * Combine overlap (config.cmb_rt > 0, see with_combine_overlap): every y writer (y_writers, in program order) has 4
//    report args at config.cmb_rt (collector x, y, counts address, unused) and, with config.y_l1, kFlatYL1Args
//    L1 hand-off args after them (y_row_major.hpp SE_Y_L1); set_flat_combine_report fills them. Each writer then adds 1
//    to counts[e] on the collector once local expert e's rows have landed (cumulative across calls, slot order, no
//    go handshake), so the collector waits for calls x writers.
//  * Hash: everything the fragment depends on is in the config (hash it), the plan (a function of the mesh device and
//    the config) and the tensors' specs. No buffer address is baked in except through the returned patch list.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>

#include "flat_routed_expert_device_operation_types.hpp"
#include "flat_routed_expert_plan.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// Buffers whose addresses the flat expert's runtime args carry (re-applied on every program-cache hit).
enum class FlatAddr : uint8_t { X, Y, Counts, Regions, Ids, GateUp, Down, Done, Arena, Words, TokenIndex, Count };

struct FlatAddrPatch {
    tt::tt_metal::KernelHandle kernel;
    tt::tt_metal::CoreCoord core;
    uint32_t index;
    FlatAddr src;
    uint32_t offset;  // byte offset added to the buffer address
};

// What add_flat_routed_expert_program leaves for the caller (keep it in the cached program's shared variables).
struct FlatProgramHandles {
    std::vector<FlatAddrPatch> patches;
    // The address each source's args were last written with (0xFFFFFFFF: unknown - the next hit writes them). A hit
    // compares the source's address with it, never with an arg's current value: the framework's inferred buffer
    // bindings (apply_resolved_bindings) may already have moved an arg that equals a buffer's address, leaving that
    // source's offset args stale.
    mutable std::array<uint32_t, static_cast<size_t>(FlatAddr::Count)> last = [] {
        std::array<uint32_t, static_cast<size_t>(FlatAddr::Count)> a;
        a.fill(0xFFFFFFFFu);
        return a;
    }();
    std::vector<std::pair<tt::tt_metal::CBHandle, uint32_t>> arena_cbs;  // (CB, arena byte offset)
    // config.cmb_rt: the y writer kernels / cores whose report args the combine overlap fills, in program order (a
    // writer's index in the combine's L1 hand-off)
    std::vector<std::pair<tt::tt_metal::KernelHandle, tt::tt_metal::CoreCoord>> y_writers;
};

// The y writers' L1 hand-off args after the 4 report args (y_row_major.hpp SE_Y_L1).
constexpr uint32_t kFlatYL1Args = 8 + 5 * 32;

// Adds the flat routed expert's kernels, circular buffers and semaphores to `program`, the program of one chip of
// `mesh_device` (every chip gets the same fragment: buffers are at lockstep addresses). `tensors.output` is y.
FlatProgramHandles add_flat_routed_expert_program(
    tt::tt_metal::Program& program,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertConfig& config,
    const FlatRoutedExpertPlan& plan,
    const FlatRoutedExpertInputs& tensors);

// Re-applies every buffer address of `tensors` (a program-cache hit; only addresses differ from the build).
void override_flat_routed_expert_addresses(
    tt::tt_metal::Program& program, const FlatProgramHandles& handles, const FlatRoutedExpertInputs& tensors);

// `config` for a combine overlap: row-major bf16 y, the report args after the writers' own (cmb_rt), cumulative
// counts in slot order without a go handshake; `y_l1`: the writers hand rows to the combine's L1 instead of DRAM.
FlatRoutedExpertConfig with_combine_overlap(FlatRoutedExpertConfig config, const FlatRoutedExpertPlan& plan, bool y_l1);

// Where the y writers report (the combine overlap): the collector core's NoC coordinates (virtual), the counts
// array's L1 address there, and with config.y_l1 the L1 hand-off args (at most kFlatYL1Args; the writer index, arg 4,
// is filled per writer).
struct FlatCombineReport {
    tt::tt_metal::CoreCoord collector;
    uint32_t counts_addr = 0;
    std::vector<uint32_t> l1_args;
};
void set_flat_combine_report(
    tt::tt_metal::Program& program,
    const FlatProgramHandles& handles,
    const FlatRoutedExpertConfig& config,
    const FlatCombineReport& report);

// The defines that decide the dynamic schedule (expert_schedule.hpp: sub-block rows, expert capacity, gate/up ring
// regions, pinning): every kernel of the program is built with them, so every core builds the same schedule.
std::vector<std::pair<std::string, std::string>> flat_schedule_defines(
    const FlatRoutedExpertConfig& config, const FlatRoutedExpertPlan& plan);

// The y writers (logical cores) in program order: the down cores grouped by column width.
std::vector<tt::tt_metal::CoreCoord> flat_y_writers(const FlatRoutedExpertPlan& plan);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
