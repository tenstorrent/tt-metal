// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_dot_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <unordered_map>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/experimental/matmul_decode/device/matmul_decode_device_operation.hpp"
#include "ttnn/operations/experimental/matmul_decode/device/matmul_decode_large_k_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace fused_single_user_dot_detail {

// The dot's CBs, as in matmul_decode_large_k, whose compute kernel the dot cores run.
constexpr uint32_t kDotCbIn0 = tt::CBIndex::c_0;       // this core's K-slice of the streams
constexpr uint32_t kDotCbIn1 = tt::CBIndex::c_1;       // this core's [Kc, N] slab of fn_weight
constexpr uint32_t kDotCbPartial = tt::CBIndex::c_2;   // local partial, on cores that reduce
constexpr uint32_t kDotCbRecv = tt::CBIndex::c_3;      // children's partials, filled by the children
constexpr uint32_t kDotCbOut = tt::CBIndex::c_4;       // root only: fused_w
constexpr uint32_t kDotCbSend = tt::CBIndex::c_5;      // running sum sent to the parent (and the root's sum)
constexpr uint32_t kDotCbRmsScale = tt::CBIndex::c_6;  // root only, rms_norm: one 1x32 scale per row
// Where the root pushes fused_w, on itself and on the cores that read it: an index the hyperconnection kernels
// leave free.
constexpr uint32_t kDotCbFusedW = tt::CBIndex::c_10;

constexpr uint32_t kDotCollapseCoreCount = 8;

constexpr char kDotReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/reader_hc_dot.cpp";
constexpr char kDotWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/writer_hc_dot.cpp";
constexpr char kDotComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/compute_large_k_tree_reduce.cpp";

}  // namespace fused_single_user_dot_detail

using namespace fused_single_user_dot_detail;

FusedSingleUserDotProgramFactory::cached_program_t FusedSingleUserDotProgramFactory::create(
    const FusedSingleUserParams& operation_attributes,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using matmul_decode::build_large_k_tree;
    using matmul_decode::large_k_tree_num_levels;
    using matmul_decode::LargeKTreeNode;

    const auto& hidden_streams = tensor_args.hidden_streams;
    const auto& w = *tensor_args.fn_weight;
    auto* device = hidden_streams.device();
    const bool b_in_l1 = w.buffer()->buffer_type() == BufferType::L1;
    const CoreRangeSet dot_grid = b_in_l1 ? w.shard_spec()->grid : *operation_attributes.dot_core_grid;
    const std::vector<CoreCoord> dot_cores = corerange_to_cores(dot_grid, std::nullopt, /*row_wise=*/true);
    const uint32_t num_dot_cores = dot_cores.size();
    const CoreCoord root = dot_cores.front();
    const CoreRangeSet root_core(CoreRange(root, root));
    std::vector<CoreCoord> non_root_cores(dot_cores.begin() + 1, dot_cores.end());

    const uint32_t num_streams = operation_attributes.num_streams;
    const uint32_t d = static_cast<uint32_t>(hidden_streams.logical_shape()[-1]);
    const uint32_t k = num_streams * d;
    const uint32_t n = static_cast<uint32_t>(w.logical_shape()[-1]);
    const uint32_t kc = k / num_dot_cores;
    // One logical row of A, each partial and fused_w are 1x32 tiles: a row-major [1, W] row is W / 32 of them.
    constexpr uint32_t M_tiles = 1;
    const uint32_t Kc_tiles = kc / constants::TILE_WIDTH;
    const uint32_t N_tiles = n / constants::TILE_WIDTH;
    const bool rms_norm = operation_attributes.rms_norm_epsilon.has_value();

    // The streams, every partial and fused_w are BFLOAT16.
    const DataFormat out_format = datatype_to_dataformat_converter(hidden_streams.dtype());
    const DataFormat in1_format = datatype_to_dataformat_converter(w.dtype());
    const tt::tt_metal::Tile row_tile({1, constants::TILE_WIDTH});
    const tt::tt_metal::Tile& in1_tile = w.tensor_spec().tile();
    const uint32_t in1_tile_size = in1_tile.get_tile_size(in1_format);
    const uint32_t out_tile_size = row_tile.get_tile_size(out_format);
    const uint32_t out_num_tiles = M_tiles * N_tiles;
    // With rms_norm every partial carries one extra 1x32 tile per row holding that row's sum(A^2).
    const uint32_t block_num_tiles = out_num_tiles + (rms_norm ? M_tiles : 0);
    const uint32_t block_bytes = block_num_tiles * out_tile_size;

    const uint32_t fan_in = operation_attributes.reduce_fan_in;
    const uint32_t num_levels = large_k_tree_num_levels(num_dot_cores, fan_in);
    const std::vector<LargeKTreeNode> tree = build_large_k_tree(num_dot_cores, fan_in, num_levels);
    uint32_t max_children = 0;
    for (const auto& node : tree) {
        max_children = std::max(max_children, node.num_children);
    }

    // Half-sync DST holds 8 16-bit tiles.
    constexpr uint32_t dst_num_tiles = 8;
    const bool use_custom_mm =
        matmul_decode::is_custom_mm_kt_dim(Kc_tiles) && matmul_decode::is_custom_mm_ct_dim(N_tiles);
    if (!use_custom_mm) {
        log_warning(
            tt::LogOp,
            "fused_hyperconnection's fn dot is falling back to the general block matmul LLKs: each dot core's K slice "
            "has {} tiles, custom_mm needs an even number in [2, 256]",
            Kc_tiles);
    }
    uint32_t out_subblock_w = 1;
    for (uint32_t wdt = N_tiles; wdt >= 1; --wdt) {
        if (N_tiles % wdt == 0 && wdt <= dst_num_tiles) {
            out_subblock_w = wdt;
            break;
        }
    }

    // The cores that read fused_w: post and comb always, the collapse cores too when they compute their own pre.
    // The landing CB and its semaphore span all ten, whose reader builds the CB either way.
    const bool use_pre_mix = tensor_args.pre_mix.has_value();
    const std::vector<CoreCoord> collapse_cores =
        corerange_to_cores(hidden_streams.shard_spec()->grid, std::nullopt, /*row_wise=*/true);
    std::vector<CoreCoord> fused_w_readers = {CoreCoord{8, 0}, CoreCoord{9, 0}};
    if (!use_pre_mix) {
        fused_w_readers.insert(fused_w_readers.end(), collapse_cores.begin(), collapse_cores.end());
    }
    const CoreRangeSet hc_cores(CoreRange(CoreCoord{0, 0}, CoreCoord{9, 0}));

    Program program = CreateProgram();

    auto make_row_cb = [&](uint32_t index, uint32_t num_tiles, const CoreRangeSet& cores) {
        return CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(num_tiles * out_tile_size, {{index, out_format}})
                .set_page_size(index, out_tile_size)
                .set_tile_dims(index, row_tile));
    };
    make_row_cb(kDotCbIn0, M_tiles * Kc_tiles, dot_grid);
    CBHandle in1_cb = 0;
    {
        auto config = CircularBufferConfig(Kc_tiles * N_tiles * in1_tile_size, {{kDotCbIn1, in1_format}})
                          .set_page_size(kDotCbIn1, in1_tile_size)
                          .set_tile_dims(kDotCbIn1, in1_tile);
        if (b_in_l1) {
            config.set_globally_allocated_address(*w.buffer());
        }
        in1_cb = CreateCircularBuffer(program, dot_grid, config);
    }
    make_row_cb(kDotCbOut, out_num_tiles, root_core);
    if (max_children > 0) {
        make_row_cb(kDotCbPartial, block_num_tiles, dot_grid);
        // One CB over the whole grid so it sits at the same L1 address on every core: a child
        // addresses its slot in the parent's CB from its own copy's base.
        make_row_cb(kDotCbRecv, max_children * block_num_tiles, dot_grid);
    }
    if (rms_norm || !non_root_cores.empty()) {
        make_row_cb(kDotCbSend, block_num_tiles, rms_norm ? dot_grid : CoreRangeSet(non_root_cores));
    }
    if (rms_norm) {
        make_row_cb(kDotCbRmsScale, M_tiles, root_core);
    }
    // A whole tile, as the readers' own fused_w CB is.
    {
        const uint32_t tile_bytes = tile_size(out_format);
        CreateCircularBuffer(
            program,
            hc_cores.merge(root_core),
            CircularBufferConfig(tile_bytes, {{kDotCbFusedW, out_format}}).set_page_size(kDotCbFusedW, tile_bytes));
    }

    // Semaphore l counts the level-l children that have delivered their partial; the kernels take the level as the id.
    for (uint32_t level = 0; level < num_levels; ++level) {
        const uint32_t id = CreateSemaphore(program, dot_grid, 0);
        TT_FATAL(id == level, "fn dot tree semaphore {} got id {}", level, id);
    }
    const uint32_t fused_w_sem = CreateSemaphore(program, hc_cores.merge(root_core), 0);

    // The dot is the longest chain (its DRAM slab reads), so its kernels are created, and start, first.
    std::vector<uint32_t> reader_compile_args = {
        M_tiles * Kc_tiles,
        Kc_tiles * N_tiles,
        block_num_tiles,
        num_levels,
        kc * static_cast<uint32_t>(sizeof(uint16_t)),
        b_in_l1 ? 0u : 1u,
    };
    TensorAccessorArgs(w.buffer()).append_to(reader_compile_args);
    DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0, .compile_args = reader_compile_args};
    reader_config.named_compile_args = {{"cb_in0", kDotCbIn0}, {"cb_in1", kDotCbIn1}, {"cb_recv", kDotCbRecv}};
    const KernelHandle reader_kernel_id = CreateKernel(program, kDotReaderKernelPath, dot_grid, reader_config);

    // A parent always precedes its children in row-major order, i.e. sits left of or above them,
    // which is the direction NOC_1 routes.
    DataMovementConfig writer_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
        .compile_args = {block_num_tiles, block_bytes, out_num_tiles, out_num_tiles * out_tile_size, fused_w_sem}};
    writer_config.named_compile_args = {
        {"cb_send", kDotCbSend}, {"cb_recv", kDotCbRecv}, {"cb_out", kDotCbOut}, {"cb_fused_w", kDotCbFusedW}};
    const KernelHandle writer_kernel_id = CreateKernel(program, kDotWriterKernelPath, dot_grid, writer_config);

    ComputeConfig compute_config{
        .math_fidelity = use_custom_mm ? MathFidelity::LoFi : MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = {
            M_tiles,
            Kc_tiles,
            N_tiles,
            1,
            out_subblock_w,
            dst_num_tiles,
            std::bit_cast<uint32_t>(1.0F / static_cast<float>(k)),
            std::bit_cast<uint32_t>(operation_attributes.rms_norm_epsilon.value_or(0.0F))}};
    compute_config.named_compile_args = {
        {"cb_in0", kDotCbIn0},
        {"cb_in1", kDotCbIn1},
        {"cb_partial", kDotCbPartial},
        {"cb_recv", kDotCbRecv},
        {"cb_out", kDotCbOut},
        {"cb_send", kDotCbSend},
        {"cb_rms_scale", kDotCbRmsScale},
    };
    if (use_custom_mm) {
        compute_config.defines["USE_CUSTOM_MM"] = "1";
    }
    if (rms_norm) {
        compute_config.defines["FUSE_RMS_NORM"] = "1";
    }
    const KernelHandle compute_kernel_id = CreateKernel(program, kDotComputeKernelPath, dot_grid, compute_config);

    FusedSingleUserDotSharedVariables shared;
    shared.dot_reader_kernel_id = reader_kernel_id;
    shared.dot_cores = dot_cores;
    shared.b_in_l1 = b_in_l1;
    shared.in1_cb = in1_cb;

    // A collapse core's shard is its k / 8 K-elements, so dot core i's slice is at byte offset (i * kc) % (k / 8)
    // of collapse core (i * kc) / (k / 8).
    const uint32_t shard_k = k / kDotCollapseCoreCount;
    const uint32_t a_base = hidden_streams.buffer()->address();
    std::vector<uint32_t> reader_dests;
    for (const auto& core : fused_w_readers) {
        const CoreCoord phys = device->worker_core_from_logical_core(core);
        reader_dests.push_back(phys.x);
        reader_dests.push_back(phys.y);
    }
    for (uint32_t i = 0; i < num_dot_cores; ++i) {
        const LargeKTreeNode& node = tree[i];
        const uint32_t first_k = i * kc;
        const CoreCoord a_core = device->worker_core_from_logical_core(collapse_cores[first_k / shard_k]);
        const uint32_t a_offset = (first_k % shard_k) * static_cast<uint32_t>(sizeof(uint16_t));
        shared.a_offsets.push_back(a_offset);

        std::vector<uint32_t> reader_args = {
            static_cast<uint32_t>(a_core.x),
            static_cast<uint32_t>(a_core.y),
            a_base + a_offset,
            w.buffer()->address(),
            i * Kc_tiles * N_tiles};
        reader_args.insert(reader_args.end(), node.children_per_level.begin(), node.children_per_level.end());
        SetRuntimeArgs(program, reader_kernel_id, dot_cores[i], reader_args);

        const bool has_parent = node.parent >= 0;
        const CoreCoord parent_phys =
            has_parent ? device->worker_core_from_logical_core(dot_cores[node.parent]) : CoreCoord{0, 0};
        std::vector<uint32_t> writer_args = {
            static_cast<uint32_t>(has_parent),
            static_cast<uint32_t>(parent_phys.x),
            static_cast<uint32_t>(parent_phys.y),
            node.parent_level,
            node.slot_in_parent * block_bytes};
        if (!has_parent) {
            writer_args.push_back(static_cast<uint32_t>(fused_w_readers.size()));
            writer_args.insert(writer_args.end(), reader_dests.begin(), reader_dests.end());
        }
        SetRuntimeArgs(program, writer_kernel_id, dot_cores[i], writer_args);

        SetRuntimeArgs(
            program, compute_kernel_id, dot_cores[i], {node.num_children, static_cast<uint32_t>(!has_parent)});
    }

    shared.hc = build_single_user_hc(
        program,
        operation_attributes,
        tensor_args,
        tensor_return_value,
        FusedWSource{.pushed = true, .cb_index = kDotCbFusedW, .semaphore_id = fused_w_sem});

    log_debug(
        tt::LogOp,
        "fused_hyperconnection fn dot: cores={}, Kc_tiles={}, N_tiles={}, fan_in={}, levels={}, custom_mm={}, "
        "rms_norm={}, weight in {}",
        num_dot_cores,
        Kc_tiles,
        N_tiles,
        fan_in,
        num_levels,
        use_custom_mm,
        rms_norm,
        b_in_l1 ? "L1" : "DRAM");
    return cached_program_t{std::move(program), std::move(shared)};
}

void FusedSingleUserDotProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedSingleUserParams& /*operation_attributes*/,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    override_single_user_hc(program, shared.hc, tensor_args, tensor_return_value);

    const auto& w = *tensor_args.fn_weight;
    if (shared.b_in_l1) {
        UpdateDynamicCircularBufferAddress(program, shared.in1_cb, *w.buffer());
    }
    const uint32_t a_base = tensor_args.hidden_streams.buffer()->address();
    const uint32_t w_addr = w.buffer()->address();
    auto& reader_args = GetRuntimeArgs(program, shared.dot_reader_kernel_id);
    for (size_t i = 0; i < shared.dot_cores.size(); ++i) {
        const auto& core = shared.dot_cores[i];
        auto& args = reader_args[core.x][core.y];
        args[2] = a_base + shared.a_offsets[i];
        args[3] = w_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
