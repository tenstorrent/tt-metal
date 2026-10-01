// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_tiled_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

namespace qkv_causal_conv1d_silu_tiled {

bool is_supported_block_tiles(uint32_t block_tiles) {
    return block_tiles == 1 || block_tiles == 2 || block_tiles == 4 || block_tiles == 8;
}

uint32_t default_channel_chunk_size(uint32_t q_width, uint32_t k_width, uint32_t v_width) {
    const uint64_t channels = static_cast<uint64_t>(q_width) + k_width + v_width;
    const uint64_t channel_tiles = channels / tt::constants::TILE_WIDTH;
    uint32_t block_tiles = default_block_tiles;
    while (block_tiles > 1 && channel_tiles % block_tiles != 0) {
        block_tiles /= 2;
    }
    return block_tiles * tt::constants::TILE_WIDTH;
}

}  // namespace qkv_causal_conv1d_silu_tiled

namespace {

namespace tiled = qkv_causal_conv1d_silu_tiled;

// DRAM reads on Blackhole want 64 B aligned L1 destinations. The scratch base is only L1 aligned,
// so the reader rounds it up to 64 B; the slack pays for that.
constexpr uint32_t scratch_alignment = 64;
// One halo half = 4 face rows x 32 B. Reads start at row 28 (input) or row 0 (history), 128 B aligned.
constexpr uint32_t halo_half_bytes = 4 * 32;
// Zero source for local NoC copies (the halo when history is None).
constexpr uint32_t zeros_region_bytes = 1024;
// x_in ring depth in steps. The reader derives the depth from the DFB size and reads each input
// ring_steps - 1 steps ahead, into the chunk of the step before the current one (the halo source of
// the current step). That needs at least 3 steps; the reader has 12 NoC transaction ids for the ring.
constexpr uint32_t x_in_ring_steps = 3;
static_assert(x_in_ring_steps >= 3 && x_in_ring_steps <= 12, "x_in ring depth must be 3..12 steps");

constexpr std::string_view reader_source =
    "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
    "reader_qkv_causal_conv1d_silu_tiled.cpp";
constexpr std::string_view compute_source =
    "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
    "qkv_causal_conv1d_silu_tiled.cpp";
// fused_qk_l2_norm: taps accumulated in dest + TTI SiLU + per-head q/k L2 norm (see the kernel header).
constexpr std::string_view compute_fast_source =
    "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
    "qkv_causal_conv1d_silu_tiled_fast.cpp";
// fused_qk_l2_norm: the writer takes the fp32 q/k blocks from the out32 DFB, v from out.
constexpr std::string_view writer_qk32_source =
    "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
    "writer_qkv_causal_conv1d_silu_tiled_qk32.cpp";
constexpr std::string_view writer_source =
    "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
    "writer_qkv_causal_conv1d_silu_tiled.cpp";

uint32_t round_up_to(uint32_t value, uint32_t multiple) { return ((value + multiple - 1) / multiple) * multiple; }

// Contiguous step ranges in core order, like kda_factory_detail::distribute_prep, except that the
// cores with one extra step are the LAST ones. The reader reads DRAM on NOC_0; at kernel start all
// cores read at once, and the cores in the first grid rows get their first data last (up to
// ~15-20 us later on a P150 at T=2048). Those cores then should not also get the extra step.
kda_factory_detail::KdaPrepWorkDist distribute_steps(tt::tt_metal::CoreCoord grid, uint32_t total) {
    const uint32_t count = std::min(total, static_cast<uint32_t>(grid.x * grid.y));
    TT_FATAL(count > 0, "qkv_causal_conv1d_silu: tiled work distribution needs at least one step");
    const uint32_t base = total / count;
    const uint32_t remainder = total % count;
    kda_factory_detail::KdaPrepWorkDist distribution;
    distribution.cores.reserve(count);
    distribution.wi_start.reserve(count);
    distribution.wi_count.reserve(count);
    uint32_t offset = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t steps = base + (index >= count - remainder ? 1u : 0u);
        distribution.cores.push_back(tt::tt_metal::CoreCoord{index % grid.x, index / grid.x});
        distribution.wi_start.push_back(offset);
        distribution.wi_count.push_back(steps);
        offset += steps;
    }
    distribution.core_set = tt::tt_metal::num_cores_to_corerangeset(count, grid, true);
    return distribution;
}

// fused_qk_l2_norm: a q/k step (taps + SiLU + the per-head L2 norm epilogue, fp32 out) costs more than a
// v step (taps + SiLU, bf16 out) - see the kernel header. Equal step counts per core then leave the cores
// that only got v steps idle for a third of the kernel. Split by cost instead: contiguous step ranges in
// core order (q/k steps first, then v, as the steps are block-major), every core as light as possible
// for the same maximum cost. A q/k step has cost 100, a v step v_cost_pct. Like distribute_steps, the
// lightest range is the first one (the cores in the first grid rows get their first data last).
// Measured on a P150 die (T=1024, C=6144): ~3.65 us per q/k step and ~2.05 us per v step on top of ~24 us of
// fixed cost, i.e. a v step costs ~56% of a q/k step; any v cost in 55..61 gives the same split here (12 q/k
// steps on 84 cores, 21 v steps on 24 cores).
constexpr uint32_t v_step_cost_pct = 58;

kda_factory_detail::KdaPrepWorkDist distribute_steps_weighted(
    tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t qk_steps, uint32_t v_cost_pct) {
    constexpr uint64_t qk_cost = 100;
    const uint32_t max_cores = std::min(total, static_cast<uint32_t>(grid.x * grid.y));
    TT_FATAL(max_cores > 0, "qkv_causal_conv1d_silu: tiled work distribution needs at least one step");
    TT_FATAL(qk_steps <= total, "qkv_causal_conv1d_silu: q/k step count {} exceeds the {} steps", qk_steps, total);
    const auto step_cost = [&](uint32_t step) -> uint64_t { return step < qk_steps ? qk_cost : v_cost_pct; };
    // Ranges that fill the cores from the last step backwards, each up to `cap` cost: the ranges
    // (last core first) as (start, count). The first range is the remainder.
    const auto fill_backwards = [&](uint64_t cap, std::vector<std::pair<uint32_t, uint32_t>>& ranges) {
        ranges.clear();
        uint32_t end = total;
        while (end > 0) {
            uint32_t start = end;
            uint64_t cost = 0;
            while (start > 0 && cost + step_cost(start - 1) <= cap) {
                cost += step_cost(start - 1);
                --start;
            }
            ranges.emplace_back(start, end - start);
            end = start;
        }
    };
    uint64_t lo = std::max<uint64_t>(qk_cost, v_cost_pct);  // a range holds at least one step
    uint64_t hi = static_cast<uint64_t>(qk_steps) * qk_cost + static_cast<uint64_t>(total - qk_steps) * v_cost_pct;
    hi = std::max(hi, lo);
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    while (lo < hi) {  // smallest cap that fits in max_cores ranges
        const uint64_t mid = lo + (hi - lo) / 2;
        fill_backwards(mid, ranges);
        if (ranges.size() <= max_cores) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    fill_backwards(lo, ranges);
    TT_FATAL(ranges.size() <= max_cores, "qkv_causal_conv1d_silu: cost-weighted split needs {} cores", ranges.size());

    const uint32_t count = static_cast<uint32_t>(ranges.size());
    kda_factory_detail::KdaPrepWorkDist distribution;
    distribution.cores.reserve(count);
    distribution.wi_start.reserve(count);
    distribution.wi_count.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        const auto& [start, steps] = ranges[count - 1 - index];
        distribution.cores.push_back(tt::tt_metal::CoreCoord{index % grid.x, index / grid.x});
        distribution.wi_start.push_back(start);
        distribution.wi_count.push_back(steps);
    }
    distribution.core_set = tt::tt_metal::num_cores_to_corerangeset(count, grid, true);
    return distribution;
}

}  // namespace

QkvCausalConv1dSiluTiledPlan make_qkv_causal_conv1d_silu_tiled_plan(
    tt::tt_metal::CoreCoord grid,
    uint32_t sequence,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    bool has_history,
    bool return_conv_state,
    uint32_t tile_size,
    bool conv_state_inplace,
    bool fused_qk_l2_norm) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    constexpr std::string_view operation_name = "qkv_causal_conv1d_silu";

    TT_FATAL(grid.x > 0 && grid.y > 0, "{}: tiled plan needs a non-empty core grid", operation_name);
    TT_FATAL(tile_size > 0, "{}: tiled plan needs a positive tile size", operation_name);
    TT_FATAL(
        sequence > 0 && sequence % TILE_HEIGHT == 0, "{}: sequence must be positive and tile aligned", operation_name);
    TT_FATAL(q_width > 0 && k_width > 0 && v_width > 0, "{}: Q/K/V widths must be positive", operation_name);
    TT_FATAL(
        q_width % TILE_WIDTH == 0 && k_width % TILE_WIDTH == 0 && v_width % TILE_WIDTH == 0,
        "{}: Q/K/V widths must be tile aligned",
        operation_name);
    TT_FATAL(
        channel_chunk_size > 0 && channel_chunk_size % TILE_WIDTH == 0,
        "{}: channel_chunk_size must be positive and tile aligned",
        operation_name);
    const uint32_t block_tiles = channel_chunk_size / TILE_WIDTH;
    TT_FATAL(
        tiled::is_supported_block_tiles(block_tiles),
        "{}: TILE input needs channel_chunk_size in {{32, 64, 128, 256}} (a block of 1, 2, 4 or 8 tiles; "
        "8 tiles fill one bf16 dest half), got {}",
        operation_name,
        channel_chunk_size);

    QkvCausalConv1dSiluTiledPlan plan;
    plan.sequence = sequence;
    plan.q_width = q_width;
    plan.k_width = k_width;
    plan.v_width = v_width;
    plan.channel_chunk_size = channel_chunk_size;
    plan.block_tiles = block_tiles;
    plan.Mt = sequence / TILE_HEIGHT;
    plan.Qt = q_width / TILE_WIDTH;
    plan.Kt = k_width / TILE_WIDTH;
    plan.Vt = v_width / TILE_WIDTH;
    plan.Ct = plan.Qt + plan.Kt + plan.Vt;
    TT_FATAL(
        plan.Ct % block_tiles == 0,
        "{}: channel_chunk_size must divide Q+K+V width exactly (Ct={} tiles, B={})",
        operation_name,
        plan.Ct,
        block_tiles);
    plan.num_blocks = plan.Ct / block_tiles;
    const uint64_t num_steps = static_cast<uint64_t>(plan.num_blocks) * plan.Mt;
    TT_FATAL(
        num_steps <= std::numeric_limits<uint32_t>::max(), "{}: too many tiled steps ({})", operation_name, num_steps);
    plan.num_steps = static_cast<uint32_t>(num_steps);
    TT_FATAL(
        !conv_state_inplace || (has_history && return_conv_state),
        "{}: conv_state_inplace needs a history and return_conv_state",
        operation_name);
    plan.has_history = has_history;
    plan.return_conv_state = return_conv_state;
    plan.conv_state_inplace = conv_state_inplace;
    plan.tile_size = tile_size;

    // DFB table (design.md section 4.5). Entry = one tile.
    const uint32_t B = block_tiles;
    plan.dataflow_buffers = {
        // S_0 = the input tiles of one step: a ring of x_in_ring_steps steps (the step in compute and
        // the prefetched ones). The reader derives the ring depth from this DFB's size.
        {.name = "x_in",
         .producer = "reader",
         .consumer = "compute",
         .num_entries = x_in_ring_steps * B,
         .entry_size = tile_size},
        // S_1, S_2, S_3 of one step (3B), double-buffered.
        {.name = "shift", .producer = "reader", .consumer = "compute", .num_entries = 6 * B, .entry_size = tile_size},
        // 2 sets x 4 taps x B tiles, so a block switch overlaps compute. The reader loads faces 0-1
        // of each tap tile (row 0 = the tap); the ROW-broadcast multiply uses row 0 only, and faces
        // 2-3 of an entry are never written or read (the entries are not zero-filled).
        {.name = "weights",
         .producer = "reader",
         .consumer = "compute",
         .num_entries = 2 * tiled::tap_count * B,
         .entry_size = tile_size},
        // Tap partial sums (compute N1), double-buffered.
        {.name = "partial",
         .producer = "compute",
         .consumer = "compute",
         .num_entries = 2 * B,
         .entry_size = tile_size},
        // SiLU output of one step, 3 steps in flight.
        {.name = "out", .producer = "compute", .consumer = "writer", .num_entries = 3 * B, .entry_size = tile_size},
    };
    plan.dfb_bytes_per_core = 0;
    for (const auto& buffer : plan.dataflow_buffers) {
        plan.dfb_bytes_per_core += buffer.bytes();
    }

    // Reader-private scratchpad: [halo | zeros | state], offsets from the 64 B aligned base.
    plan.scratch_align_slack = scratch_alignment;
    plan.scratch_halo_offset = 0;
    plan.scratch_halo_bytes = 2 * halo_half_bytes * B;
    plan.scratch_zeros_offset = round_up_to(plan.scratch_halo_offset + plan.scratch_halo_bytes, scratch_alignment);
    plan.scratch_zeros_bytes = zeros_region_bytes;
    plan.scratch_state_offset = round_up_to(plan.scratch_zeros_offset + plan.scratch_zeros_bytes, scratch_alignment);
    plan.scratch_state_bytes = tile_size;
    uint32_t scratch_end = plan.scratch_state_offset + plan.scratch_state_bytes;
    if (conv_state_inplace) {
        // The reader derives the stage address as state + one tile (no extra compile-time arg), so the
        // stage must start exactly there; the state tile size keeps it 64 B aligned.
        plan.scratch_stage_offset = plan.scratch_state_offset + plan.scratch_state_bytes;
        TT_FATAL(
            plan.scratch_stage_offset % scratch_alignment == 0,
            "{}: stage offset {} is not {} B aligned",
            operation_name,
            plan.scratch_stage_offset,
            scratch_alignment);
        plan.scratch_stage_bytes = 2 * halo_half_bytes * B;
        scratch_end = plan.scratch_stage_offset + plan.scratch_stage_bytes;
    }
    plan.scratch_bytes = scratch_end + plan.scratch_align_slack;
    plan.l1_bytes_per_core = plan.dfb_bytes_per_core + plan.scratch_bytes;

    // Work split: contiguous block-major step ranges.
    plan.grid = grid;
    const uint32_t qk_blocks_tiles = plan.Qt + plan.Kt;
    if (fused_qk_l2_norm && qk_blocks_tiles % block_tiles == 0) {
        plan.work =
            distribute_steps_weighted(grid, plan.num_steps, (qk_blocks_tiles / block_tiles) * plan.Mt, v_step_cost_pct);
    } else {
        plan.work = distribute_steps(grid, plan.num_steps);
    }
    plan.min_steps_per_core = std::numeric_limits<uint32_t>::max();
    plan.max_steps_per_core = 0;
    plan.max_tap_loads_per_core = 0;
    for (size_t i = 0; i < plan.work.cores.size(); ++i) {
        const uint32_t start = plan.work.wi_start[i];
        const uint32_t count = plan.work.wi_count[i];
        plan.min_steps_per_core = std::min(plan.min_steps_per_core, count);
        plan.max_steps_per_core = std::max(plan.max_steps_per_core, count);
        if (count > 0) {
            const uint32_t units = (start + count - 1) / plan.Mt - start / plan.Mt + 1;
            plan.max_tap_loads_per_core = std::max(plan.max_tap_loads_per_core, units);
        }
    }
    const double mean_steps = static_cast<double>(plan.num_steps) / static_cast<double>(plan.num_cores());
    plan.balance = plan.max_steps_per_core > 0 ? mean_steps / plan.max_steps_per_core : 0.0;
    return plan;
}

std::string QkvCausalConv1dSiluTiledPlan::to_string() const {
    std::string text = fmt::format(
        "qkv_causal_conv1d_silu tiled program plan\n"
        "  geometry: T={} widths=({},{},{}) Mt={} Ct={} (Qt={} Kt={} Vt={}) B={} (channel_chunk_size={}) "
        "blocks={} steps={}\n"
        "  options: has_history={} return_conv_state={} conv_state_inplace={} tile_size={} B\n"
        "  L1 per core:\n",
        sequence,
        q_width,
        k_width,
        v_width,
        Mt,
        Ct,
        Qt,
        Kt,
        Vt,
        block_tiles,
        channel_chunk_size,
        num_blocks,
        num_steps,
        has_history ? 1 : 0,
        return_conv_state ? 1 : 0,
        conv_state_inplace ? 1 : 0,
        tile_size);
    for (const auto& buffer : dataflow_buffers) {
        text += fmt::format(
            "    DFB {:<8} {:>7} -> {:<7} {:>3} x {} B = {:>7} B\n",
            buffer.name,
            buffer.producer,
            buffer.consumer,
            buffer.num_entries,
            buffer.entry_size,
            buffer.bytes());
    }
    text += fmt::format(
        "    scratchpad scratch (reader only): halo @{} ({} B), zeros @{} ({} B), state @{} ({} B), "
        "stage @{} ({} B), align slack {} B = {} B\n"
        "    DFB total {} B; L1 total {} B ({:.1f} KiB) per core\n"
        "  work split: grid {}x{}, cores={}, steps/core min={} max={}, balance={:.1f}%, max tap loads/core={}",
        scratch_halo_offset,
        scratch_halo_bytes,
        scratch_zeros_offset,
        scratch_zeros_bytes,
        scratch_state_offset,
        scratch_state_bytes,
        scratch_stage_offset,
        scratch_stage_bytes,
        scratch_align_slack,
        scratch_bytes,
        dfb_bytes_per_core,
        l1_bytes_per_core,
        l1_bytes_per_core / 1024.0,
        grid.x,
        grid.y,
        num_cores(),
        min_steps_per_core,
        max_steps_per_core,
        100.0 * balance,
        max_tap_loads_per_core);
    // The ranges as runs of cores with the same step count: "cores x steps".
    text += "\n  step ranges (cores x steps):";
    for (size_t i = 0; i < work.wi_count.size();) {
        size_t j = i;
        while (j < work.wi_count.size() && work.wi_count[j] == work.wi_count[i]) {
            ++j;
        }
        text += fmt::format(" {}x{}", j - i, work.wi_count[i]);
        i = j;
    }
    return text;
}

ttnn::device_operation::ProgramArtifacts QkvCausalConv1dSiluTiledProgramFactory::create_program_artifacts(
    const QkvCausalConv1dSiluParams& attrs, const QkvCausalConv1dSiluInputs& in, std::vector<Tensor>& outputs) {
    namespace m2 = tt::tt_metal::experimental;

    const bool has_history = in.history.has_value();
    const bool return_state = attrs.return_conv_state;
    TT_FATAL(
        outputs.size() == (return_state ? 4u : 3u),
        "qkv_causal_conv1d_silu: tiled path expects {} outputs, got {}",
        return_state ? 4 : 3,
        outputs.size());

    const auto& input = in.input.mesh_tensor();
    const auto& tap0 = in.tap0.mesh_tensor();
    const auto& tap1 = in.tap1.mesh_tensor();
    const auto& tap2 = in.tap2.mesh_tensor();
    const auto& tap3 = in.tap3.mesh_tensor();
    const auto& q = outputs[0].mesh_tensor();
    const auto& k = outputs[1].mesh_tensor();
    const auto& v = outputs[2].mesh_tensor();
    const auto& device = input.device();
    const auto arch = device.arch();
    // The reader's scratch zero fills rely on the tt-1xx implementation of async_write_zeros (a
    // loopback NoC read covered by the reader's trid_setup barrier); see the reader kernel.
    TT_FATAL(
        arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE,
        "qkv_causal_conv1d_silu: the tiled (TILE input) path supports Wormhole and Blackhole only, got {}",
        arch);

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(data_format);
    const auto plan = make_qkv_causal_conv1d_silu_tiled_plan(
        device.compute_with_storage_grid_size(),
        attrs.sequence,
        attrs.q_width,
        attrs.k_width,
        attrs.v_width,
        attrs.channel_chunk_size,
        has_history,
        return_state,
        tile_size,
        attrs.conv_state_inplace,
        attrs.fused_qk_l2_norm);
    if (const char* print_plan = std::getenv("TT_KDA_QKV_CONV1D_PRINT_PLAN");
        print_plan != nullptr && print_plan[0] != '\0' && print_plan[0] != '0') {
        std::fprintf(stderr, "%s\n", plan.to_string().c_str());
    }

    const m2::KernelSpecName reader_kernel_name{"reader"};
    const m2::KernelSpecName writer_kernel_name{"writer"};
    const m2::KernelSpecName compute_kernel_name{"compute"};

    const m2::DFBSpecName x_in_dfb_name{"x_in"};
    const m2::DFBSpecName shift_dfb_name{"shift"};
    const m2::DFBSpecName weights_dfb_name{"weights"};
    const m2::DFBSpecName partial_dfb_name{"partial"};
    const m2::DFBSpecName out_dfb_name{"out"};
    const m2::ScratchpadSpecName scratch_name{"scratch"};

    const m2::TensorParamName input_tensor_name{"input"};
    const m2::TensorParamName history_tensor_name{"history"};
    const m2::TensorParamName tap0_tensor_name{"tap0"};
    const m2::TensorParamName tap1_tensor_name{"tap1"};
    const m2::TensorParamName tap2_tensor_name{"tap2"};
    const m2::TensorParamName tap3_tensor_name{"tap3"};
    const m2::TensorParamName q_tensor_name{"q"};
    const m2::TensorParamName k_tensor_name{"k"};
    const m2::TensorParamName v_tensor_name{"v"};
    const m2::TensorParamName new_state_tensor_name{"new_state"};

    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.reserve(plan.dataflow_buffers.size());
    for (const auto& buffer : plan.dataflow_buffers) {
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{buffer.name},
            .entry_size = buffer.entry_size,
            .num_entries = buffer.num_entries,
            .data_format_metadata = data_format,
        });
    }
    // fused_qk_l2_norm: per-head L2 norm of q and k in the compute epilogue (B = 4 = one head); q/k leave as
    // fp32 through out32.
    const bool qknorm = attrs.fused_qk_l2_norm;
    if (qknorm) {
        const uint32_t tile_f32 = tt::tile_size(tt::DataFormat::Float32);
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"ybuf"},
            .entry_size = tile_size,
            .num_entries = 4 * plan.block_tiles,  // the compute's 4-stage q/k pipeline keeps up to 4 steps
            .data_format_metadata = data_format,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"yq"},
            .entry_size = tile_size,
            .num_entries = 2 * plan.block_tiles,  // S1's copy of y: steps j-1 and j
            .data_format_metadata = data_format,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"ones"},
            .entry_size = tile_size,
            .num_entries = 1,
            .data_format_metadata = data_format,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"sq"},
            .entry_size = tile_f32,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float32,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"rn"},
            .entry_size = tile_f32,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float32,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out32"},
            .entry_size = tile_f32,
            .num_entries = 3 * plan.block_tiles,
            .data_format_metadata = tt::DataFormat::Float32,
        });
    }

    m2::Group<m2::TensorBinding> reader_tensors = {
        m2::TensorBinding{input_tensor_name, "input"},
        m2::TensorBinding{tap0_tensor_name, "tap0"},
        m2::TensorBinding{tap1_tensor_name, "tap1"},
        m2::TensorBinding{tap2_tensor_name, "tap2"},
        m2::TensorBinding{tap3_tensor_name, "tap3"},
    };
    if (has_history) {
        reader_tensors.push_back(m2::TensorBinding{history_tensor_name, "history"});
    }
    if (return_state) {
        reader_tensors.push_back(m2::TensorBinding{new_state_tensor_name, "new_state"});
    }

    // has_history and return_state reach the reader only as defines: the reader must drop the
    // accessors of unbound tensors, which needs the preprocessor.
    // P17_CONVOPT: QKV_CONV_OPT (bit mask of bit-exact variants; unset = the kernels' default, 0 = the previous
    // kernels) is passed from the environment to the reader and compute kernels.
    auto add_opt_define = [](m2::KernelSpec::CompilerOptions::Defines& defines) {
        if (const char* value = std::getenv("QKV_CONV_OPT")) {
            defines["QKV_CONV_OPT"] = value;
        }
    };
    m2::KernelSpec::CompilerOptions::Defines reader_defines{
        {"QKV_CONV_HAS_HISTORY", has_history ? "1" : "0"},
        {"QKV_CONV_RETURN_STATE", return_state ? "1" : "0"},
        {"QKV_CONV_STATE_INPLACE", attrs.conv_state_inplace ? "1" : "0"}};
    add_opt_define(reader_defines);

    m2::KernelSpec reader{
        .unique_id = reader_kernel_name,
        .source = std::filesystem::path(reader_source),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {
                m2::ProducerOf(x_in_dfb_name, "x_in"),
                m2::ProducerOf(shift_dfb_name, "shift"),
                m2::ProducerOf(weights_dfb_name, "weights"),
            },
        .scratchpad_bindings = {m2::ScratchpadBinding{
            .scratchpad_spec_name = scratch_name, .accessor_name = "scratch"}},
        .tensor_bindings = std::move(reader_tensors),
        .compile_time_args =
            {{"block_tiles", plan.block_tiles},
             {"Mt", plan.Mt},
             {"Ct", plan.Ct},
             {"halo_offset", plan.scratch_halo_offset},
             {"zeros_offset", plan.scratch_zeros_offset},
             {"state_offset", plan.scratch_state_offset}},
        .runtime_arg_schema = {.runtime_arg_names = {"step_start", "step_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = writer_kernel_name,
        .source = std::filesystem::path(writer_source),
        .dfb_bindings = {m2::ConsumerOf(out_dfb_name, "out")},
        .tensor_bindings =
            {
                m2::TensorBinding{q_tensor_name, "q"},
                m2::TensorBinding{k_tensor_name, "k"},
                m2::TensorBinding{v_tensor_name, "v"},
            },
        .compile_time_args =
            {{"block_tiles", plan.block_tiles}, {"Mt", plan.Mt}, {"Qt", plan.Qt}, {"Kt", plan.Kt}, {"Vt", plan.Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"step_start", "step_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // Partials in dest (see the compute kernel): only with a bf16 half-sync dest and q, k and v all in L1.
    // With DRAM outputs the faster compute lets the output writes congest the NoC (the op gets slower),
    // and an fp32 dest would not round the partials to bf16; those cases compile the partial-DFB flow.
    const auto& cfg = attrs.compute_kernel_config;
    const bool outputs_in_l1 = std::all_of(outputs.begin(), outputs.begin() + 3, [](const Tensor& output) {
        return output.memory_config().buffer_type() == tt::tt_metal::BufferType::L1;
    });
    const bool partials_in_dest = outputs_in_l1 && !cfg.fp32_dest_acc_en && !cfg.dst_full_sync_en;

    m2::KernelSpec::CompilerOptions::Defines compute_defines{
        {"QKV_CONV_PARTIALS_IN_DEST", partials_in_dest ? "1" : "0"}};
    add_opt_define(compute_defines);

    m2::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source = std::filesystem::path(qknorm ? compute_fast_source : compute_source),
        .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::ConsumerOf(x_in_dfb_name, "x_in"),
                m2::ConsumerOf(shift_dfb_name, "shift"),
                m2::ConsumerOf(weights_dfb_name, "weights"),
                m2::ProducerOf(partial_dfb_name, "partial"),
                m2::ConsumerOf(partial_dfb_name, "partial"),
                m2::ProducerOf(out_dfb_name, "out"),
            },
        .compile_time_args = {{"block_tiles", plan.block_tiles}, {"Mt", plan.Mt}},
        .runtime_arg_schema = {.runtime_arg_names = {"step_start", "step_count"}},
        .hw_config = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config),
    };

    if (qknorm) {
        for (const char* name : {"ybuf", "yq", "ones", "sq", "rn"}) {
            compute.dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{name}, name));
            compute.dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{name}, name));
        }
        const uint32_t head_dim = plan.block_tiles * tt::constants::TILE_WIDTH;
        // sq / rn are fp32 FPU operands (srcB of the row-sum matmul / the broadcast multiply).
        auto& compute_unpack_modes = m2::unpack_modes(std::get<m2::ComputeHardwareConfig>(compute.hw_config));
        compute_unpack_modes.emplace(m2::DFBSpecName{"sq"}, tt::tt_metal::UnpackMode::UnpackToSrc);
        compute_unpack_modes.emplace(m2::DFBSpecName{"rn"}, tt::tt_metal::UnpackMode::UnpackToSrc);
        compute.dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{"out32"}, "out32"));
        writer.source = std::filesystem::path(writer_qk32_source);
        writer.dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{"out32"}, "out32"));
        compute.compiler_options.defines.emplace("QKV_CONV_Q_BLOCKS", std::to_string(plan.Qt / plan.block_tiles));
        compute.compiler_options.defines.emplace(
            "QKV_CONV_QK_BLOCKS", std::to_string((plan.Qt + plan.Kt) / plan.block_tiles));
        compute.compiler_options.defines.emplace("QKV_CONV_QK_EPS", "1e-6f");
        if (attrs.qk_early_drain) {
            // Drain the epilogue pipeline when this many steps of the core's range are left (3 = measured best at
            // T=1024, C=6144: 12 q/k steps per core).
            compute.compiler_options.defines.emplace("QKV_CONV_QK_EARLY_DRAIN", "3");
        }
        compute.compiler_options.defines.emplace(
            "QKV_CONV_Q_SCALE", fmt::format("{:.9g}f", 1.0 / std::sqrt(static_cast<double>(head_dim))));
    }

    m2::KernelRunArgs reader_run_args{.kernel = reader_kernel_name};
    m2::KernelRunArgs writer_run_args{.kernel = writer_kernel_name};
    m2::KernelRunArgs compute_run_args{.kernel = compute_kernel_name};
    for (size_t i = 0; i < plan.work.cores.size(); ++i) {
        const auto& core = plan.work.cores[i];
        const uint32_t step_start = plan.work.wi_start[i];
        const uint32_t step_count = plan.work.wi_count[i];
        for (auto* kernel_args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            m2::AddRuntimeArgsForNode(
                kernel_args->runtime_arg_values, core, {{"step_start", step_start}, {"step_count", step_count}});
        }
    }

    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = input_tensor_name, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = tap0_tensor_name, .spec = tap0.tensor_spec()},
        m2::TensorParameter{.unique_id = tap1_tensor_name, .spec = tap1.tensor_spec()},
        m2::TensorParameter{.unique_id = tap2_tensor_name, .spec = tap2.tensor_spec()},
        m2::TensorParameter{.unique_id = tap3_tensor_name, .spec = tap3.tensor_spec()},
        m2::TensorParameter{.unique_id = q_tensor_name, .spec = q.tensor_spec()},
        m2::TensorParameter{.unique_id = k_tensor_name, .spec = k.tensor_spec()},
        m2::TensorParameter{.unique_id = v_tensor_name, .spec = v.tensor_spec()},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {input_tensor_name, input},
        {tap0_tensor_name, tap0},
        {tap1_tensor_name, tap1},
        {tap2_tensor_name, tap2},
        {tap3_tensor_name, tap3},
        {q_tensor_name, q},
        {k_tensor_name, k},
        {v_tensor_name, v},
    };
    if (has_history) {
        const auto& history = in.history->mesh_tensor();
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = history_tensor_name, .spec = history.tensor_spec()});
        run_args.tensor_args.emplace(history_tensor_name, m2::TensorArgument{std::cref(history)});
    }
    if (return_state) {
        const auto& new_state = outputs[3].mesh_tensor();
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = new_state_tensor_name, .spec = new_state.tensor_spec()});
        run_args.tensor_args.emplace(new_state_tensor_name, m2::TensorArgument{std::cref(new_state)});
    }

    m2::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu_tiled",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .scratchpads = {m2::ScratchpadSpec{.unique_id = scratch_name, .size_per_node = plan.scratch_bytes}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_kernel_name, writer_kernel_name, compute_kernel_name},
                    .target_nodes = plan.work.core_set,
                },
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
