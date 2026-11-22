// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <set>
#include <tracy/Tracy.hpp>
#include "flat_routed_expert_program_factory.hpp"

#include "flat_routed_expert_plan.hpp"
#include "../../common/chip_program.hpp"
#include "../../common/moe_block_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <unordered_map>

namespace ttnn::operations::bringup::moe_block::flat_routed_expert {

namespace factory_detail {
using tt::tt_metal::CoreCoord;
// The flat expert's compile-time options by name (kernels/flat_options.hpp): every kernel of the program gets all of
// them as named compile-time args; a kernel's own values (its CBs, its semaphores) are added on top
using Options = std::map<std::string, uint32_t>;

constexpr uint32_t XRD_BATCH = 2, DW_BATCH = 2;
// semaphore ids (16 created in order on every role core)
constexpr uint32_t DATA = 4, HARR = 5, GO = 6, DONE = 7, GATH = 8, XARR = 9, SFREE = 10, HFREE = 11, HSFREE = 12,
                   GATH1 = 14, GATH2 = 15;
// the gate/up cores' grant counter: the receiver counts the landing slots it frees, the weight reads
// (gate_up_weight_read.hpp) fill granted slots only (id 1 is otherwise unused on them)
constexpr uint32_t GRANT = 1;
constexpr const char* KDIR = "ttnn/ttnn/bringup/moe_block_ttnn/flat_routed_expert/device/kernels/";

// A runtime-arg list whose buffer-address words are recorded for the cache-hit patch.
struct Args {
    std::vector<uint32_t> v;
    std::vector<std::tuple<uint32_t, FlatAddr, uint32_t>> addr;
    Args& lit(uint32_t x) {
        v.push_back(x);
        return *this;
    }
    Args& lits(const std::vector<uint32_t>& values) {
        v.insert(v.end(), values.begin(), values.end());
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

tt::DataFormat w_format(bool weights_bf8) { return weights_bf8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Bfp4_b; }

// `options` with some changed (a kernel's own values)
Options with(Options options, std::initializer_list<std::pair<const std::string, uint32_t>> changes) {
    for (const auto& [name, value] : changes) {
        TT_FATAL(options.contains(name), "flat_routed_expert: no option {}", name);
        options[name] = value;
    }
    return options;
}

// `options` with values added (a kernel's own named compile-time args: the names its header lists)
Options with_added(Options options, std::initializer_list<std::pair<const std::string, uint32_t>> values) {
    for (const auto& [name, value] : values) {
        TT_FATAL(!options.contains(name), "flat_routed_expert: option {} is already set", name);
        options[name] = value;
    }
    return options;
}

// A kernel into the program descriptor: its handle is its index in the descriptor's kernels (the Program built from
// the descriptor creates the kernels in that order).
tt::tt_metal::KernelHandle add_kernel(
    tt::tt_metal::ProgramDescriptor& program, tt::tt_metal::KernelDescriptor kernel_descriptor) {
    program.kernels.push_back(std::move(kernel_descriptor));
    return static_cast<tt::tt_metal::KernelHandle>(program.kernels.size() - 1);
}

// A core's runtime args: a second call for the same core replaces its args (the Program API's SetRuntimeArgs).
void set_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel_descriptor, const CoreCoord& core, std::vector<uint32_t> runtime_args) {
    for (auto& [args_core, core_args] : kernel_descriptor.runtime_args) {
        if (args_core == core) {
            core_args = std::move(runtime_args);
            return;
        }
    }
    kernel_descriptor.runtime_args.emplace_back(core, std::move(runtime_args));
}
}  // namespace factory_detail
using namespace factory_detail;

std::vector<std::pair<std::string, uint32_t>> flat_schedule_options(
    const FlatRoutedExpertConfig& config, const FlatRoutedExpertPlan& p) {
    // pinning (expert_schedule.hpp): only a dominant expert, at least this many times the others' mean sub-blocks
    constexpr uint32_t kPinMinRatio = 8;
    return {
        {"max_experts", flat_max_experts(p)},
        {"rows_per_sub_block", p.row_tiles_per_sub_block * 32},
        {"gate_up_ring_regions", p.gate_up_weight_ring_blocks / p.gate_up_k_blocks},
        {"pin_chunk_sub_blocks", config.pin},
        {"pin_min_ratio", config.pin ? kPinMinRatio : 0u}};
}

FlatProgramHandles add_flat_routed_expert_program(
    tt::tt_metal::ProgramDescriptor& program,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertConfig& config,
    const FlatRoutedExpertPlan& p,
    const FlatRoutedExpertInputs& t) {
    using namespace tt::tt_metal;
    auto* device = &mesh_device;
    FlatProgramHandles handles;
    const auto phys = [device](const CoreCoord& c) { return device->worker_core_from_logical_core(c); };
    const auto noc_xy = [&](const CoreCoord& c) {
        const auto q = phys(c);
        return static_cast<uint32_t>((q.x << 16) | q.y);
    };
    const uint32_t E = p.experts_per_chip, row_tiles = p.row_tiles_per_sub_block,
                   pairs_per_core = p.gate_up_pairs_per_core, V = p.sub_blocks_per_chip,
                   intermediate_tiles = p.intermediate_tiles, hidden_tiles = p.hidden_tiles;
    const uint32_t w_tile = p.weight_tile_bytes, banks = p.dram_banks, rectangle_count = p.gate_up_rectangles.size();
    const uint32_t n_slots = p.gate_up_weight_slots, gate_up_core_count = p.gate_up_cores.size(),
                   down_core_count = p.down_cores.size();
    TT_FATAL(
        n_slots == banks,
        "flat_routed_expert: one gate/up weight slot a DRAM bank ({} slots, {} banks)",
        n_slots,
        banks);
    // x after the tilize and h after gate/up: bfp8 tiles, or bf16 with layout.x_bf16 / h_bf16 (the plan's sizes)
    const uint32_t x_tile = p.x_tile_bytes, h_tile = p.h_tile_bytes;
    const uint32_t x_blk = row_tiles * kKBlk, x_bytes = x_blk * x_tile;
    const uint32_t pin = config.pin;
    TT_FATAL(rectangle_count == 2, "flat_routed_expert: two gate/up rectangles");

    // ---- per-core pieces ----
    std::vector<uint32_t> head_xy, gu_xy;
    const uint32_t coord_xy = noc_xy(p.coordinator_cores[0]);
    for (uint32_t d : p.down_chain_heads) {
        head_xy.push_back(noc_xy(p.down_cores[d]));
    }
    for (uint32_t i = 0; i < gate_up_core_count; ++i) {
        gu_xy.push_back(noc_xy(p.gate_up_cores[i]));
    }
    std::vector<std::vector<uint32_t>> rect_cores(rectangle_count);
    for (uint32_t core_index = 0; core_index < gate_up_core_count; ++core_index) {
        rect_cores[p.rectangle_of(p.gate_up_cores[core_index])].push_back(core_index);
    }
    // x delivery: x read once by the helpers (the plan's two full rectangles, one helper each) and unicast snakes, no
    // multicast: the NOC0 wedge of a relay's multicast into the gate/up rectangle beside the weight and DRAM
    // traffic. Each primary unicasts its blocks into its rectangle's snake head; the gate/up cores forward them
    // along the snake (down a column, wrapping inside the rectangle, one column over, on down), each piece once its
    // successor has freed the block's slot. One snake a rectangle (2: the relay writes every block twice - GLM-5.3 M
    // 1024 238.7 vs 205.5 us / expert, multicast 211.5), 4 cut-through pieces, the forwarder on the gate/up NCRISC
    // (gate_up_x_forwarder.cpp) on NOC0 with 3 pieces in flight, the receiver (gate_up_core_reader.cpp, BRISC) on NOC1.
    const auto aligned = [&](uint32_t n) { return x_bytes % n == 0 && (x_bytes / n) % 16 == 0; };
    // (bring-up, FlatLayoutOptions.x_min_piece_bytes, DeepSeek-V4.1 P.59: > 0 = the most of 4 / 2 / 1 pieces of at
    // least that many bytes - the forwarder's 3 pieces in flight bound the snake's x rate; 0 = the source's choice)
    const uint32_t min_piece = config.layout.x_min_piece_bytes;
    const auto fits = [&](uint32_t n) { return aligned(n) && (min_piece == 0 || x_bytes / n >= min_piece); };
    const uint32_t x_pieces = fits(4) ? 4 : (fits(2) ? 2 : 1);
    constexpr uint32_t X_FWD_DEPTH = 3;
    log_info(
        tt::LogOp,
        "flat_routed_expert: x read once (helpers), unicast snakes (snakes 1 pieces {} of {} B NCRISC forwarder NOC0 "
        "depth "
        "{})",
        x_pieces,
        x_bytes / x_pieces,
        X_FWD_DEPTH);
    struct XLink {
        uint32_t pred = 0, succ0 = 0;
    };
    std::vector<XLink> xlink(gate_up_core_count);
    std::vector<uint32_t> x_head(rectangle_count, 0);  // each rectangle's snake head (gate/up index)
    for (uint32_t k = 0; k < rectangle_count; ++k) {
        const auto [first_x, last_x, first_y, last_y] = p.gate_up_rectangles[k];
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> at;
        for (uint32_t core_index : rect_cores[k]) {
            at[{p.gate_up_cores[core_index].x, p.gate_up_cores[core_index].y}] = core_index;
        }
        const uint32_t W = last_x - first_x + 1, H = last_y - first_y + 1;
        TT_FATAL(at.size() == W * H, "flat x snakes: rectangle {} is not full of gate/up cores", k);
        // the snake: from the relay's row, down a column (wrapping inside the rectangle, one NoC hop a step, NOC0:
        // east / south), one column over, on down from that row - every core forwards once
        const uint32_t relay_row = std::clamp<uint32_t>(p.relay_cores[k].y, first_y, last_y);
        std::vector<uint32_t> snake;
        uint32_t row = relay_row;
        for (uint32_t i = 0; i < W; ++i) {
            for (uint32_t r = 0; r < H; ++r) {
                const uint32_t y = first_y + (row - first_y + r) % H;
                snake.push_back(at.at({first_x + i, y}));
                if (r + 1 == H) {
                    row = y;  // the next column starts on this row
                }
            }
        }
        x_head[k] = snake.front();
        for (size_t j = 0; j < snake.size(); ++j) {
            XLink& l = xlink[snake[j]];
            l.pred = j ? noc_xy(p.gate_up_cores[snake[j - 1]]) : 0;
            l.succ0 = j + 1 < snake.size() ? noc_xy(p.gate_up_cores[snake[j + 1]]) : 0;
        }
    }
    const uint32_t ng_bytes = 4 * p.global_experts;
    Args schedule_args;
    schedule_args.a(FlatAddr::Counts)
        .a(FlatAddr::Regions)
        .lit(ng_bytes)
        .lit(1)
        .lit(1000000000)
        .a(FlatAddr::Ids)
        .lit(4 * E);

    // ---- compile-time options (kernels/flat_options.hpp) ----
    const uint32_t max_experts = flat_max_experts(p);
    const uint32_t meta_bytes = max_experts == 16 ? 512 : (max_experts == 32 ? 1024 : 2048);
    const uint32_t dyn_need = (ng_bytes + 63) / 64 * 64 + (4 * E + 63) / 64 * 64;
    uint32_t dyn_half = 512;
    while (dyn_half < dyn_need) {
        dyn_half *= 2;
    }
    const uint32_t nreg_gu = p.gate_up_weight_ring_blocks / p.gate_up_k_blocks;
    bool dn_reg = pin != 0;
    for (uint32_t d = 0; d < down_core_count && dn_reg; ++d) {
        const uint32_t down_k_tiles =
            FlatRoutedExpertPlan::down_k_block_tiles(p.down_columns_per_core[d], intermediate_tiles);
        dn_reg =
            static_cast<uint32_t>(std::nearbyint(p.down_weight_ring_experts * (intermediate_tiles / down_k_tiles))) ==
            nreg_gu * (intermediate_tiles / down_k_tiles);
    }
    // gate/up row passes: DST holds fewer row tiles of gate/up tiles than a sub-block has (4 fp32 tiles, half sync)
    const bool gu_rp = p.gate_up_row_passes && !p.gate_up_full_sync;
    // The program's options; a kernel's own (marked) are set per kernel below. Every kernel builds the schedule from
    // the same schedule options, so every core builds the same schedule.
    Options options;
    for (const auto& [name, value] : flat_schedule_options(config, p)) {
        options[name] = value;
    }
    options["down_weights_per_load"] = dn_reg ? 1 : 0;
    options["schedule_scratch_bytes"] = dyn_half;
    options["schedule_page_bytes"] = meta_bytes;
    options["local_experts"] = E;
    options["hidden_tiles"] = hidden_tiles;
    options["intermediate_tiles"] = intermediate_tiles;
    options["x_tile_bytes"] = x_tile;
    options["h_tile_bytes"] = h_tile;
    options["super_block_tiles"] = p.super_block_tiles;
    options["super_blocks_per_sub_block"] = p.super_blocks;
    options["x_block_k_tiles"] = kKBlk;
    options["x_arrived_sem"] = XARR;
    options["x_slot_free_sem"] = SFREE;
    options["x_indexed"] = t.token_index.has_value() ? 1 : 0;
    options["x_chain_pieces"] = x_pieces;
    options["x_credit_sem"] = 0;  // (an x primary relay's own)
    // the gate/up cores: the x receiver's consumed / end counts for the x forwarder (semaphores 2 / 3, free there)
    options["x_consumed_sem"] = 2;
    options["x_end_sem"] = 3;
    // gate/up weights read by the gate/up cores (gate_up_weight_read.hpp): the receiver's grant counter (GRANT) and
    // its landed-blocks count (DATA), semaphore 0 the loads-list mailbox to the NCRISC reader
    options["gate_up_block_tiles"] = p.weight_block_tiles;
    options["gate_up_weight_ring_blocks"] = p.gate_up_weight_ring_blocks;
    options["gate_up_block_bytes"] = p.weight_block_tiles * w_tile;
    options["gate_up_blocks_per_expert"] = p.gate_up_k_blocks;
    options["gate_up_reads_in_flight"] = p.gate_up_reads_in_flight;
    options["gate_up_read_pieces"] = p.gate_up_block_pieces;
    options["gate_up_mailbox_sem"] = 0;
    options["gate_up_grant_sem"] = GRANT;
    options["gate_up_data_sem"] = DATA;
    // the h exchange: the gate/up cores' slices into the down chain heads' buffers (one gather semaphore a buffer), go
    options["gate_up_core_count"] = gate_up_core_count;
    options["gate_up_tile_pairs"] = pairs_per_core;
    options["h_buffers"] = p.down_h_buffers;
    options["h_gather_sem_0"] = GATH;
    options["h_gather_sem_1"] = GATH1;
    options["h_gather_sem_2"] = GATH2;
    options["go_sem"] = GO;
    options["half_width_gate_up"] = p.half_width_gate_up ? 1 : 0;
    options["gate_up_units"] = p.units_per_core;
    options["gate_up_units_h_tiles"] = p.units_per_core ? p.h_tiles_per_core : 0;
    options["row_pass_rows"] = gu_rp ? std::max(1u, 4 / p.gate_up_weight_tiles_per_row) : 0;
    options["x_ring_blocks"] = p.x_ring_blocks;
    options["y_row_major"] = config.y_row_major ? 1 : 0;
    options["y_row_major_rows"] = 0;  // (the down kernels': their column width's)
    options["down_core_count"] = down_core_count;
    options["h_arrived_sem"] = HARR;
    options["h_slot_free_sem"] = HSFREE;
    options["down_done_sem"] = DONE;
    options["down_weight_banks"] = banks;
    options["down_columns"] = 0;  // (the down kernels': their column width's)
    options["down_k_block_tiles"] = 0;
    options["down_block_tiles"] = 0;
    options["down_weight_ring_blocks"] = 0;
    options["combine_writer_index_arg"] = 0;  // (the y writers', running with the combine)
    options["combine_report_arg"] = 0;
    options["combine_handoff_arg"] = 0;
    // h pieces per sub-block down the chains
    constexpr uint32_t H_PIECES = 32;

    // ---- kernels + runtime args ----
    // combine overlap (config.cmb_rt): a writer's own args end with its writer index (option combine_writer_index_arg),
    // then its report block (combine_report_arg: kFlatReportArgs; combine_handoff_arg: the kFlatYL1Args L1 hand-off
    // args) - per-core args, filled by set_flat_combine_report: the destinations differ by chip, and one program for
    // the mesh needs alike common args
    auto add_writer_rt = [&](KernelHandle k, const CoreCoord& c, Args args, uint32_t widx_arg) {
        if (config.cmb_rt) {
            TT_FATAL(
                args.v.size() == widx_arg && widx_arg < config.cmb_rt,
                "flat_routed_expert: {} writer args, the writer index planned at {} (cmb_rt {})",
                args.v.size(),
                widx_arg,
                config.cmb_rt);
            args.v.push_back(0);
            args.v.resize(args.v.size() + kFlatReportArgs + (config.y_l1 ? kFlatYL1Args : 0), 0u);
            handles.y_writers.emplace_back(k, c);
            handles.y_widx_arg = widx_arg;
            handles.y_report_arg = widx_arg + 1;
        }
        return args;
    };
    // the combine overlap (with_combine_overlap: row-major y, the writers report in slot order, cumulative counts)
    TT_FATAL(!config.cmb_rt || config.y_row_major, "flat_routed_expert: a combine overlap takes row-major y");
    // (a writer's per-core arg indices: the report after the writer index, the L1 hand-off after the report)
    auto writer_options = [&](const Options& base, uint32_t widx_arg) {
        if (!config.cmb_rt) {
            return base;
        }
        return with(
            base,
            {{"combine_writer_index_arg", widx_arg},
             {"combine_report_arg", widx_arg + 1},
             // (flat_with_program: y rows straight into the combine's L1)
             {"combine_handoff_arg", config.y_l1 ? widx_arg + 1 + kFlatReportArgs : 0u}});
    };
    auto add_rt = [&](KernelHandle k, const CoreCoord& c, const Args& args) {
        set_runtime_args(program.kernels[k], c, args.v);
        for (const auto& [i, s, off] : args.addr) {
            handles.patches.push_back({k, c, i, s, off});
        }
    };
    // Every data-movement kernel's positional compile-time args are the op's DRAM tensors' accessor args (a
    // TensorAccessorArgs chain is offset-based; every other compile-time arg is named), in flat_accessors.hpp's order:
    // x, y, counts, regions, expert ids, token index, the combine's y-row tables (the last two interleaved DRAM
    // placeholders when this program has none).
    std::vector<uint32_t> acc_ct;
    for (const Tensor* tensor : {&t.x, &t.output, &t.counts, &t.regions, &t.global_expert_ids}) {
        TensorAccessorArgs(*tensor->buffer()).append_to(acc_ct);
    }
    (t.token_index ? TensorAccessorArgs(*t.token_index->buffer()) : TensorAccessorArgs::create_dram_interleaved())
        .append_to(acc_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(acc_ct);
    auto add_data_movement_kernel = [&](const std::string& name,
                                        const std::vector<CoreCoord>& cores,
                                        DataMovementProcessor proc,
                                        NOC noc,
                                        const Options& kernel_options) {
        KernelDescriptor kernel_descriptor;
        kernel_descriptor.kernel_source = std::string(KDIR) + name;
        kernel_descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel_descriptor.core_ranges = rect_ranges(cores);
        kernel_descriptor.compile_time_args = acc_ct;
        kernel_descriptor.named_compile_time_args = {kernel_options.begin(), kernel_options.end()};  // (sorted by name)
        // The kernels read the args table's names; standalone they stay runtime args (off, 0). The combine's
        // runtime_args_to_dram overwrites these when it converts the kernel.
        fill_args_table_named_args(kernel_descriptor);
        kernel_descriptor.opt_level = KernelBuildOptLevel::O2;  // (DataMovementConfig's default)
        kernel_descriptor.config =
            DataMovementConfigDescriptor{.processor = proc, .noc = noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
        return add_kernel(program, std::move(kernel_descriptor));
    };
    const auto fidelity = static_cast<MathFidelity>(config.math_fidelity);
    auto add_compute_kernel = [&](const std::string& name,
                                  const std::vector<CoreCoord>& cores,
                                  const Options& kernel_options,
                                  bool fp32 = false,
                                  bool full_sync = false) {
        KernelDescriptor kernel_descriptor;
        kernel_descriptor.kernel_source = std::string(KDIR) + name;
        kernel_descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel_descriptor.core_ranges = rect_ranges(cores);
        kernel_descriptor.named_compile_time_args = {kernel_options.begin(), kernel_options.end()};  // (sorted by name)
        kernel_descriptor.opt_level = KernelBuildOptLevel::O3;  // (ComputeConfig's default)
        kernel_descriptor.config =
            ComputeConfigDescriptor{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32, .dst_full_sync_en = full_sync};
        // (bring-up, FlatLayoutOptions.weights_nk: the weight matmuls transpose their [N, K] tiles back)
        if (config.layout.weights_nk && (name == "gate_up_compute.cpp" || name == "down_compute.cpp")) {
            kernel_descriptor.defines.emplace_back("SE_W_NK", "1");
        }
        return add_kernel(program, std::move(kernel_descriptor));
    };

    // row-major y (config.y_row_major): the down compute pack-untilizes, the writers send row segments
    // (y_row_major.hpp)
    const bool y_row_major = config.y_row_major;
    // the row-major out CB (bf16 row tiles of PCD pages) takes what the down cores' arena holds after the out region's
    // start up to the CBs every arena core shares (meta 6 / counts 7 at meta_off, past the largest role; the plan's
    // 2 KB margin is kept), at most 2 row_tiles (a virtual expert ahead) and 8 (the writers' transaction ids 8..15)
    auto yrm_rows = [&](uint32_t columns) {
        const uint32_t room = p.schedule_page_offset - p.down_output_offset - 2048;
        return std::min({room / (columns * 2048), 2 * row_tiles, 8u});
    };
    // x relays: primaries (feed their snake) + helpers (read / tilize half of x each, into both primaries' rings)
    auto rl_off = [&](uint32_t idx) {
        return idx < rectangle_count ? 0 : (idx - rectangle_count) / rectangle_count + 1;
    };
    auto rl_sb = [&](uint32_t idx) {
        uint32_t n = 0;
        for (uint32_t g = 0; g < V * p.super_blocks; ++g) {
            n += g % p.relay_stride == rl_off(idx);
        }
        return n;
    };
    {
        // indexed mode: the token index address is the relay reader's last runtime arg (after schedule_args)
        const bool indexed = t.token_index.has_value();
        const uint32_t idx_arg = 3 + static_cast<uint32_t>(schedule_args.v.size());
        // the relay reads x in segments of sbt x 32 bf16 (one super-block row): a segment must not straddle two x
        // pages (they sit in different banks), so the page is a multiple of the segment or divides it
        const uint32_t x_page_bytes = t.x.logical_shape()[-1] * 2, seg_bytes = p.super_block_tiles * 64;
        TT_FATAL(
            x_page_bytes % seg_bytes == 0 || seg_bytes % x_page_bytes == 0,
            "flat_routed_expert: x page of {} B (hidden {} / x_pages_per_row {}) must be a multiple or a divisor of "
            "the {} B read segment",
            x_page_bytes,
            p.hidden,
            p.hidden / t.x.logical_shape()[-1],
            seg_bytes);
        TT_FATAL(kRmChunks % XRD_BATCH == 0, "flat_routed_expert: XRD_BATCH must divide the row-major chunks");
        // x reads on NOC1, the relays' sends on NOC0 (one RISC per NoC on a relay: dedicated-NoC mode)
        const std::vector<CoreCoord> x_helper_cores(p.relay_cores.begin() + rectangle_count, p.relay_cores.end());
        // the CB the reader fills with row-major chunks and the tilizer empties (x_row_major_cb), the CB of the tilized
        // super-blocks (x_super_block_cb)
        constexpr uint32_t X_ROW_MAJOR_CB = 0, X_SUPER_BLOCK_CB = 1;
        const Options x_reader_options = with_added(
            options,
            {{"x_row_major_cb", X_ROW_MAJOR_CB},
             {"x_page_bytes", t.x.logical_shape()[-1] * 2},
             {"row_chunks_per_read_barrier", XRD_BATCH},
             {"token_index_arg", idx_arg},
             {"x_pages_per_row", p.hidden / t.x.logical_shape()[-1]}});
        // (the helpers read / tilize all of x; the primaries only multicast it)
        const auto k_x_reader = add_data_movement_kernel(
            "x_injector_reader.cpp", x_helper_cores, DataMovementProcessor::RISCV_0, NOC::NOC_1, x_reader_options);
        add_compute_kernel(
            "x_injector_tilize_compute.cpp",
            x_helper_cores,
            with_added(options, {{"x_row_major_cb", X_ROW_MAJOR_CB}, {"x_super_block_cb", X_SUPER_BLOCK_CB}}));
        // one kernel per primary, its credits to the helpers on its own semaphore (5 / 6 there)
        std::vector<KernelHandle> k_x_injectors;
        for (uint32_t q = 0; q < rectangle_count; ++q) {
            k_x_injectors.push_back(add_data_movement_kernel(
                "x_injector_writer.cpp",
                {p.relay_cores[q]},
                DataMovementProcessor::RISCV_1,
                NOC::NOC_0,
                with(options, {{"x_credit_sem", 5 + q}})));
        }
        // one kernel per helper k: its arrivals are semaphore {4, 7}[k] on both primaries, its credits come back on 5
        // (primary 0) and 6 (primary 1)
        std::vector<KernelHandle> k_x_helpers;
        for (uint32_t k = 0; k < rectangle_count; ++k) {
            k_x_helpers.push_back(add_data_movement_kernel(
                "x_injector_helper_writer.cpp",
                {p.relay_cores[rectangle_count + k]},
                DataMovementProcessor::RISCV_1,
                NOC::NOC_0,
                with_added(
                    options,
                    {{"x_super_block_cb", X_SUPER_BLOCK_CB},
                     {"x_landing_slots", p.relay_landing_slots},
                     {"x_primary_data_sem", k ? 7u : 4u},
                     {"x_helper_credit_sem", 5}})));
        }
        const uint32_t super_block_bytes = row_tiles * p.super_block_tiles * x_tile;
        for (uint32_t idx = 0; idx < p.relay_cores.size(); ++idx) {
            const CoreCoord relay_core = p.relay_cores[idx];
            const uint32_t k = idx % rectangle_count;
            if (idx >= rectangle_count) {  // helper k: the super-blocks g % 2 == k of the whole stream
                Args reader_args;
                reader_args.a(FlatAddr::X).lit(rectangle_count).lit(k).cat(schedule_args);
                if (indexed) {
                    reader_args.a(FlatAddr::TokenIndex);
                }
                add_rt(k_x_reader, relay_core, reader_args);
            }
            if (idx < rectangle_count) {
                // the snake head and its freed word, helper 0's xy, the landing rings, their slots, helper 1's xy
                // (rectangle 0's first - helper k's ring k), then the expert_schedule.hpp args
                Args injector_args;
                injector_args.a(FlatAddr::Arena, p.x_ring_offset)
                    .lit(noc_xy(p.gate_up_cores[x_head[k]]))
                    .a(FlatAddr::Words)
                    .lit(noc_xy(p.relay_cores[rectangle_count]))
                    .a(FlatAddr::Arena, p.relay_landing_offset)
                    .lit(p.relay_landing_slots)
                    .lit(noc_xy(p.relay_cores[rectangle_count + 1]));
                injector_args.cat(schedule_args);
                add_rt(k_x_injectors[idx], relay_core, injector_args);
            } else {
                Args helper_args;
                helper_args
                    .lit(noc_xy(p.relay_cores[0]))  // (primary 0 first, ring k on both primaries)
                    .a(FlatAddr::Arena, p.relay_landing_offset + k * p.relay_landing_slots * super_block_bytes)
                    .lit(rl_sb(idx))
                    .lit(1)
                    .lit(0)
                    .cat(schedule_args);
                helper_args.lit(rectangle_count)
                    .lit(k);  // (x_injector_helper_writer.cpp: every rectangle_count-th super-block from k)
                helper_args.lit(noc_xy(p.relay_cores[1]));  // (the second primary)
                add_rt(k_x_helpers[k], relay_core, helper_args);
            }
        }
    }

    // gate/up cores
    {
        // the receiver (BRISC) on NOC1: on NOC0 the gate/up cores' h writes, the DRAM weight reads and x deliveries
        // into these same cores wedged the NoC (a relay stuck in noc_cmd_buf_ready or in its write barrier; ~1 in
        // 1e4..2e5 launches, any chip), a hardware hang the NoC probe test reproduces from the traffic alone
        // (tests/ttnn/unit_tests/operations/noc_hang_probe, op_skeleton). The receivers count their grants
        // locally; the x forwarder (NCRISC) on NOC0 reads their consumed count / end marker (options x_consumed_sem /
        // x_end_sem), one successor, X_FWD_DEPTH pieces in flight
        const Options gu_reader_options = with_added(options, {{"x_cb", 0}, {"weight_cb", 1}, {"h_local_cb", 3}});
        TT_FATAL(
            x_bytes % x_pieces == 0 && (x_bytes / x_pieces) % 16 == 0,
            "flat_routed_expert: x pieces {} of a {} B x block must be 16 B aligned",
            x_pieces,
            x_bytes);
        const KernelHandle k_gate_up_reader = add_data_movement_kernel(
            "gate_up_core_reader.cpp", p.gate_up_cores, DataMovementProcessor::RISCV_0, NOC::NOC_1, gu_reader_options);
        const KernelHandle k_x_forwarder = add_data_movement_kernel(
            "gate_up_x_forwarder.cpp",
            p.gate_up_cores,
            DataMovementProcessor::RISCV_1,
            NOC::NOC_0,
            with_added(options, {{"x_forward_depth", X_FWD_DEPTH}}));
        TT_FATAL(
            !p.half_width_gate_up || config.activation <= 3,
            "flat_routed_expert: half-width gate / up tiles take SiLU / SwiGLU-OAI / SiTU / clamped SiLU GLU only");
        const auto k_gate_up_compute =
            add_compute_kernel("gate_up_compute.cpp", p.gate_up_cores, options, true, p.gate_up_full_sync);
        // (the activation: its kind and its constants - the defaults the code stands for, routed_activation_code - as
        // named compile-time args, common/kernels/expert_activation.hpp)
        auto& activation_args = program.kernels[k_gate_up_compute].named_compile_time_args;
        activation_args.emplace_back(kExpertActivationArgName, config.activation);
        for (const auto& arg : expert_activation_constant_args(default_expert_activation(config.activation))) {
            activation_args.push_back(arg);
        }
        program.kernels[k_gate_up_reader].common_runtime_args =
            head_xy;  // (the down chain heads' xy: the same on every gate/up core)
        for (uint32_t r = 0; r < n_slots; ++r) {
            for (uint32_t j = 0; j < p.gate_up_cores_per_weight_slot; ++j) {
                const uint32_t core_index = p.gate_up_core_index(r, j);
                const CoreCoord gate_up_core = p.gate_up_cores[core_index];
                const uint32_t column_group = r * p.column_groups_per_weight_slot + j;  // (the core's column group)
                Args a;
                // the grant counter's semaphore id, column group, chain heads, GATH, h_all, head xy
                a.lit(GRANT)
                    .lit(column_group)
                    .lit(head_xy.size())
                    .lit(GATH)
                    .a(FlatAddr::Arena, p.down_h_offset)
                    .cat(schedule_args);
                // the gate/up weight reads (gate_up_weight_read.hpp, on this core's NCRISC): weights, bank, this core's
                // first block (block j of its slot's region, which is the whole of bank r's), chunk bytes, landing ring
                // (the arena's start)
                Args weight_read_args;
                weight_read_args.a(FlatAddr::GateUp)
                    .lit(r)
                    .lit(j * p.weight_block_tiles * w_tile)
                    .lit(p.column_groups_per_weight_slot * p.weight_block_tiles * w_tile)
                    .a(FlatAddr::Arena, 0);
                add_rt(k_gate_up_reader, gate_up_core, a);
                // x ring, predecessor (+ its free word's sem), successor, relay + this core's word there
                Args f;
                f.a(FlatAddr::Arena, p.x_ring_offset)
                    .lit(xlink[core_index].pred)
                    .lit(SFREE)
                    .lit(xlink[core_index].succ0)
                    .lit(noc_xy(p.relay_cores[p.rectangle_of(gate_up_core)]))
                    .a(FlatAddr::Words, 0);
                f.cat(weight_read_args);
                add_rt(k_x_forwarder, gate_up_core, f);
                set_runtime_args(
                    program.kernels[k_gate_up_compute],
                    gate_up_core,
                    std::vector<uint32_t>{(column_group * p.units_per_core) & 1u});
            }
        }
    }

    // down cores, one kernel set per column width
    std::map<uint32_t, std::vector<uint32_t>> dgroups;
    for (uint32_t d = 0; d < down_core_count; ++d) {
        dgroups[p.down_columns_per_core[d]].push_back(d);
    }
    for (const auto& [columns, down_indices] : dgroups) {
        const uint32_t down_k_tiles = FlatRoutedExpertPlan::down_k_block_tiles(columns, intermediate_tiles),
                       nblk = intermediate_tiles / down_k_tiles, slot = down_k_tiles * columns;
        const uint32_t ring = static_cast<uint32_t>(std::nearbyint(p.down_weight_ring_experts * nblk)),
                       out = row_tiles * columns;
        std::vector<CoreCoord> cores;
        for (uint32_t d : down_indices) {
            cores.push_back(p.down_cores[d]);
        }
        // row-major y written from the NCRISC on its NOC (down_weight_reader.cpp) instead of the BRISC, whose NOC
        // carries the h chain: bfp8 x + bf16 h at M 512 / 2048 / 5120 -9 / -9 / -11%, bfp8 / bfp8 -4 / -1 / -0%; +1%
        // (~0.4 us) at M <= 64. The h chain / y writes (BRISC) on NOC1, the down-weight reads (NCRISC) on NOC0.
        const bool y_nc = y_row_major;
        TT_FATAL(
            !config.cmb_rt || y_nc, "flat_routed_expert: a combine overlap writes y from the down NCRISC (row-major)");
        // the y writer's (down_weight_reader) own args: down weights, bank, rotation, the schedule args, y, column 0;
        // then its writer index
        const uint32_t w_widx = 3 + static_cast<uint32_t>(schedule_args.v.size()) + 2;
        // the column width's: the row-major out CB's row tiles, the down weight ring's blocks, the columns, the K tiles
        // and the tiles of a weight block
        const Options down_options = with(
            options,
            {{"y_row_major_rows", y_row_major ? yrm_rows(columns) : 0u},
             {"down_weight_ring_blocks", ring},
             {"down_columns", columns},
             {"down_k_block_tiles", down_k_tiles},
             {"down_block_tiles", slot}});
        const auto k_down_writer = add_data_movement_kernel(
            "down_core_writer.cpp",
            cores,
            DataMovementProcessor::RISCV_0,
            NOC::NOC_1,
            with_added(
                down_options,
                {{"h_all_cb", 2},
                 {"h_all_tiles", p.down_h_buffer_tiles},
                 {"h_pieces", H_PIECES},
                 {"out_cb", 16},
                 {"out_tiles", out}}));
        const auto k_down_weight_reader = add_data_movement_kernel(
            "down_weight_reader.cpp",
            cores,
            DataMovementProcessor::RISCV_1,
            NOC::NOC_0,
            with_added(
                writer_options(down_options, w_widx),
                {{"down_weight_cb", 1}, {"weight_tile_bytes", w_tile}, {"down_weight_reads_per_barrier", DW_BATCH}}));
        program.kernels[k_down_writer].common_runtime_args =
            gu_xy;  // (the gate/up cores' xy: the same on every down core)
        const auto k_down_compute =
            add_compute_kernel("down_compute.cpp", cores, down_options, config.down_fp32, config.down_fp32);
        for (uint32_t d : down_indices) {
            const CoreCoord down_core = p.down_cores[d];
            const uint32_t pred =
                p.down_chain_predecessor.contains(d) ? noc_xy(p.down_cores[p.down_chain_predecessor.at(d)]) : 0;
            const uint32_t succ =
                p.down_chain_successor.contains(d) ? noc_xy(p.down_cores[p.down_chain_successor.at(d)]) : 0;
            Args a;
            a.a(FlatAddr::Arena, p.down_h_offset)
                .lit(pred)
                .lit(succ)
                .lit(coord_xy)
                .lit(d == 0 ? 1 : 0)
                .lit(gate_up_core_count)
                .a(FlatAddr::Y)
                .lit(p.down_first_columns[d])
                .a(FlatAddr::Done)
                .lit(d)
                .cat(schedule_args);
            add_rt(k_down_writer, down_core, a);
            Args w;
            w.a(FlatAddr::Down).lit(d % banks).lit(d * p.down_weight_region_bytes).cat(schedule_args);
            if (y_nc) {
                w.a(FlatAddr::Y).lit(p.down_first_columns[d]);
            }
            add_rt(
                k_down_weight_reader, down_core, y_nc ? add_writer_rt(k_down_weight_reader, down_core, w, w_widx) : w);
            set_runtime_args(program.kernels[k_down_compute], down_core, std::vector<uint32_t>{0});
        }
    }

    // ---- semaphores, CBs ----
    const auto arena_cores = p.arena_cores();
    const auto all_crs = rect_ranges(arena_cores);
    // semaphore ids 0..15 on every arena core, initial value 0 (the ids the kernels take above: GRANT, DATA, GO, ...):
    // the flat expert is the first fragment of its program, no worker semaphore on these cores precedes them
    TT_FATAL(
        std::none_of(
            program.semaphores.begin(),
            program.semaphores.end(),
            [&](const SemaphoreDescriptor& semaphore) {
                return semaphore.core_type == tt::CoreType::WORKER && semaphore.core_ranges.intersects(all_crs);
            }),
        "flat_routed_expert: a semaphore already on the flat expert's cores (its ids 0..15 are fixed)");
    for (uint32_t semaphore_id = 0; semaphore_id < 16; ++semaphore_id) {
        program.semaphores.push_back(
            SemaphoreDescriptor{.id = semaphore_id, .core_ranges = all_crs, .initial_value = 0});
    }
    const Buffer& arena = *t.arena.buffer();
    auto arena_cb = [&](uint8_t idx,
                        uint32_t off,
                        uint32_t size,
                        const std::vector<CoreCoord>& cores,
                        tt::DataFormat data_format,
                        uint32_t page) {
        // (every arena-backed CB must lie inside the arena: one past its end overlaps whatever L1 sits above it)
        TT_FATAL(
            off + size <= arena.aligned_size_per_bank(),
            "flat_routed_expert: CB {} at {} + {} B runs past the {} B arena",
            idx,
            off,
            size,
            arena.aligned_size_per_bank());
        CBDescriptor circular_buffer{.total_size = size, .core_ranges = rect_ranges(cores)};
        circular_buffer.format_descriptors.push_back(
            CBFormatDescriptor{.buffer_index = idx, .data_format = data_format, .page_size = page});
        circular_buffer.buffer = const_cast<Buffer*>(&arena);
        circular_buffer.address_offset = off;
        program.cbs.push_back(std::move(circular_buffer));
    };
    const auto weight_format = w_format(config.weights_bf8);
    const auto bfp8 = tt::DataFormat::Bfp8_b;
    const auto x_format = config.layout.x_bf16 ? tt::DataFormat::Float16_b : bfp8;  // (x after the tilize)
    const auto h_format = config.layout.h_bf16 ? tt::DataFormat::Float16_b : bfp8;  // (h after gate/up)
    // the down output CB in the arena's out region (bytes = the bfp8 double block): bfp8 tiles, or with row-major y
    // bf16 row tiles of PCD pages each (as many whole row tiles as the region holds: row_tiles for row_tiles <= 16)
    auto out_cb = [&](const std::vector<CoreCoord>& cores, uint32_t columns, uint32_t bytes) {
        if (!y_row_major) {
            arena_cb(16, p.down_output_offset, bytes, cores, bfp8, kBf8Tile);
            return;
        }
        TT_FATAL(yrm_rows(columns) >= 1, "flat_routed_expert: row-major y: no room for one row tile");
        arena_cb(16, p.down_output_offset, yrm_rows(columns) * columns * 2048, cores, tt::DataFormat::Float16_b, 2048);
    };
    // (the read / tilize CBs on the helper relays only: the primaries read nothing - their arena there is free)
    const std::vector<CoreCoord> helper_relays(p.relay_cores.begin() + rectangle_count, p.relay_cores.end());
    arena_cb(0, 0, kRmChunks * 32 * p.row_segment_bytes, helper_relays, tt::DataFormat::Float16_b, p.row_segment_bytes);
    arena_cb(
        1, p.super_block_offset, kSbSlots * row_tiles * p.super_block_tiles * x_tile, helper_relays, x_format, x_tile);
    arena_cb(
        1, 0, p.gate_up_weight_ring_blocks * p.weight_block_tiles * w_tile, p.gate_up_cores, weight_format, w_tile);
    arena_cb(0, p.x_ring_offset, p.x_ring_blocks * x_bytes, p.gate_up_cores, x_format, x_tile);
    arena_cb(
        3,
        p.h_local_offset,
        p.down_h_buffers * row_tiles * p.h_tiles_per_core * h_tile,
        p.gate_up_cores,
        h_format,
        h_tile);
    arena_cb(16, p.pack_cb_offset, 2048, p.gate_up_cores, tt::DataFormat::Float16_b, 2048);
    arena_cb(2, p.down_h_offset, p.down_h_buffers * p.down_h_buffer_tiles * h_tile, p.down_cores, h_format, h_tile);
    for (const auto& [columns, down_indices] : dgroups) {
        const uint32_t down_k_tiles = FlatRoutedExpertPlan::down_k_block_tiles(columns, intermediate_tiles);
        const uint32_t ring =
            static_cast<uint32_t>(std::nearbyint(p.down_weight_ring_experts * (intermediate_tiles / down_k_tiles)));
        std::vector<CoreCoord> cores;
        for (uint32_t d : down_indices) {
            cores.push_back(p.down_cores[d]);
        }
        arena_cb(1, 0, ring * down_k_tiles * columns * w_tile, cores, weight_format, w_tile);
        out_cb(cores, columns, 2 * row_tiles * columns * kBf8Tile);
    }
    // (meta and the dynamic counts: one offset on every arena core, past every role's own bytes - the plan's sizes)
    TT_FATAL(
        meta_bytes == p.schedule_page_bytes && 4 * dyn_half == p.dynamic_counts_bytes,
        "flat_routed_expert: CB 6 / 7 sizes {} / {} B differ from the plan's {} / {} B",
        meta_bytes,
        4 * dyn_half,
        p.schedule_page_bytes,
        p.dynamic_counts_bytes);
    // (on the cores that run kernels: the idle / spare cores' arena is free - FlatRoutedExpertPlan::free_l1)
    const auto work_cores = p.work_cores();
    arena_cb(6, p.schedule_page_offset, meta_bytes, work_cores, tt::DataFormat::UInt32, meta_bytes);
    arena_cb(7, p.dynamic_counts_offset, 4 * dyn_half, work_cores, tt::DataFormat::UInt32, 4 * dyn_half);
    return handles;
}

std::array<uint32_t, static_cast<size_t>(FlatAddr::Count)> flat_source_addresses(const FlatRoutedExpertInputs& t) {
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
    return addr;
}

FlatRoutedExpertConfig with_combine_overlap(
    FlatRoutedExpertConfig config, const FlatRoutedExpertPlan& plan, bool y_l1) {
    config.y_row_major = true;
    // (the writers report local experts in slot order, with cumulative counts: combine_done.hpp)
    config.cmb_rt = 48 + static_cast<uint32_t>(plan.gate_up_cores.size()) + 2 * config.experts_per_chip;
    config.y_l1 = y_l1;
    return config;
}

void set_flat_combine_report(
    tt::tt_metal::ProgramDescriptor& program,
    const FlatProgramHandles& handles,
    const FlatRoutedExpertConfig& config,
    const FlatCombineReport& report) {
    TT_FATAL(config.cmb_rt > 0, "flat_routed_expert: set_flat_combine_report on a program built without cmb_rt");
    TT_FATAL(
        report.l1_args.size() <= kFlatYL1Args,
        "flat_routed_expert: {} L1 hand-off args > {}",
        report.l1_args.size(),
        kFlatYL1Args);
    TT_FATAL(config.y_l1 || report.l1_args.empty(), "flat_routed_expert: L1 hand-off args without y_l1");
    auto args_of = [&](tt::tt_metal::KernelHandle k, const CoreCoord& c) -> std::vector<uint32_t>& {
        for (auto& [core, a] : program.kernels[k].runtime_args) {
            if (core == c) {
                return a;
            }
        }
        TT_THROW("flat_routed_expert: writer core {} has no args", c.str());
    };
    // the report + L1 hand-off and the writer index: each writer's own args (y_report_arg on, y_widx_arg)
    uint32_t widx = 0;
    for (const auto& [k, core] : handles.y_writers) {
        auto& a = args_of(k, core);
        TT_FATAL(
            a.size() == handles.y_report_arg + kFlatReportArgs + (config.y_l1 ? kFlatYL1Args : 0),
            "flat_routed_expert: writer core {} has {} args, its report block planned at {}",
            core.str(),
            a.size(),
            handles.y_report_arg);
        const uint32_t r = handles.y_report_arg;
        a[r + 0] = static_cast<uint32_t>(report.collector.x);
        a[r + 1] = static_cast<uint32_t>(report.collector.y);
        a[r + 2] = report.counts_addr;
        a[r + 3] = 0;  // (unused)
        for (uint32_t j = 0; j < report.l1_args.size(); ++j) {
            a[r + kFlatReportArgs + j] = report.l1_args[j];
        }
        if (config.y_l1 && !report.l1_args.empty()) {
            a[handles.y_widx_arg] = widx++;
        }
    }
}

std::vector<tt::tt_metal::CoreCoord> flat_y_writers(const FlatRoutedExpertPlan& plan) {
    std::map<uint32_t, std::vector<uint32_t>> dgroups;
    for (uint32_t d = 0; d < plan.down_cores.size(); ++d) {
        dgroups[plan.down_columns_per_core[d]].push_back(d);
    }
    std::vector<tt::tt_metal::CoreCoord> out;
    for (const auto& [columns, down_indices] : dgroups) {
        for (const uint32_t d : down_indices) {
            out.push_back(plan.down_cores[d]);
        }
    }
    return out;
}

FlatRoutedExpertProgramFactory::cached_mesh_workload_t FlatRoutedExpertProgramFactory::create_mesh_workload(
    const FlatRoutedExpertParams& config,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FlatRoutedExpertInputs& t,
    Tensor& /*output*/) {
    auto& mesh_device = *t.x.device();
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& range : tensor_coords.ranges()) {
        // (every chip's program is the same: one program for the range)
        tt::tt_metal::ProgramDescriptor desc;
        const auto handles =
            add_flat_routed_expert_program(desc, mesh_device, config, *flat_routed_expert_plan(mesh_device, config), t);
        // (the args stay runtime args: every address arg is rewritten on a hit, the descriptor holds offsets only)
        ProgramAddresses addresses;
        for (const auto& patch : handles.patches) {
            addresses.args.push_back(ProgramAddresses::AddressArg{
                .kernel = patch.kernel,
                .common = false,
                .core = patch.core,
                .index = patch.index,
                .source = static_cast<uint32_t>(patch.src),
                .offset = patch.offset});
        }
        addresses.arena_cbs = arena_backed_cbs(desc, t.arena.buffer());
        tt::tt_metal::Program program(desc);
        write_program_addresses(program, addresses, flat_source_addresses(t), t.arena.buffer());
        workload.add_program(range, std::move(program));
        shared.emplace(range, std::move(addresses));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void FlatRoutedExpertProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const FlatRoutedExpertParams&, const FlatRoutedExpertInputs& t, Tensor&) {
    const auto sources = flat_source_addresses(t);
    for (auto& [range, addresses] : cached.shared_variables) {
        write_program_addresses(cached.workload.get_programs().at(range), addresses, sources, t.arena.buffer());
    }
}

}  // namespace ttnn::operations::bringup::moe_block::flat_routed_expert
