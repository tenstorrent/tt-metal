// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace tt::tt_metal;
using namespace ttnn::operations::unary;
using namespace ttnn::operations::unary::utils;
using ttnn::operations::unary::EltwiseUnaryWithParam;
using ttnn::operations::unary::UnaryOpType;


bool pack_first_op_scalars(
    const EltwiseUnaryWithParam& op, DataType input_dtype, uint32_t& packed_scalar1, uint32_t& packed_scalar2) {
    if (op.empty()) {
        return false;
    }
    switch (op.type()) {
        case UnaryOpType::WHERE_TSS:
        case UnaryOpType::MAC_TSS:
            packed_scalar1 = pack_scalar_runtime_arg(op, 0, input_dtype);
            packed_scalar2 = pack_scalar_runtime_arg(op, 1, input_dtype);
            break;
        case UnaryOpType::LOGIT: {
            const auto eps = *op.get_param_if<float>(0);
            if (eps >= 0.0f) {
                // Ensure correct clamp bounds [min(eps, 1-eps), max(eps, 1-eps)]
                auto lo = std::min(eps, 1.0f - eps);
                auto hi = std::max(eps, 1.0f - eps);
                // Pre-round the bounds to bf16 (RNE) for bf16 input: the SFPU
                // narrows the clamped value fp32->bf16 by truncation. Making
                // the bound bf16-exact host-side turns that truncating write-back into
                // a no-op.
                // Torch's bf16 boundary:
                //   * eps <= 0.5 (golden = torch.special.logit): torch quantizes
                //     eps->bf16 first, then forms 1-eps in bf16, i.e. bf16(1 - bf16(eps)).
                //   * eps > 0.5 (golden = ordered clamp on python 1-eps/eps): the
                //     bounds are bf16(1-eps)/bf16(eps) directly.
                if (input_dtype == DataType::BFLOAT16) {
                    if (eps <= 0.5f) {
                        const float eps_bf = static_cast<float>(bfloat16(eps));
                        const float one_minus_eps_bf = 1.0f - eps_bf;
                        lo = static_cast<float>(bfloat16(std::min(eps_bf, one_minus_eps_bf)));
                        hi = static_cast<float>(bfloat16(std::max(eps_bf, one_minus_eps_bf)));
                    } else {
                        lo = static_cast<float>(bfloat16(lo));
                        hi = static_cast<float>(bfloat16(hi));
                    }
                }
                packed_scalar1 = pack_scalar_runtime_arg_impl(lo, input_dtype);
                packed_scalar2 = pack_scalar_runtime_arg_impl(hi, input_dtype);
                return true;
            }
            break;
        }
        default: break;
    }
    return false;
}

bool needs_tmp0_dfb(UnaryOpType t) { return t == UnaryOpType::LOGIT; }

// ---- Resource names ----
const tt::tt_metal::experimental::KernelSpecName READER{"reader"};
const tt::tt_metal::experimental::KernelSpecName WRITER{"writer"};
const tt::tt_metal::experimental::KernelSpecName COMPUTE{"compute"};
const tt::tt_metal::experimental::DFBSpecName IN_DFB{"in"};      // legacy index c_0
const tt::tt_metal::experimental::DFBSpecName TMP0_DFB{"tmp0"};  // legacy index c_1 (LOGIT only)
const tt::tt_metal::experimental::DFBSpecName OUT_DFB{"out"};    // legacy index c_2
const tt::tt_metal::experimental::TensorParamName INPUT{"input"};
const tt::tt_metal::experimental::TensorParamName OUTPUT{"output"};

uint32_t get_shards_per_width(const ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    auto num_cores = shard_spec.grid.num_cores();
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }
    if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }
    const auto& bbox = shard_spec.grid.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (shard_spec.orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::unary {

using namespace utils;

namespace {

// Per-core runtime-arg values, as create_program_artifacts writes them.
struct CoreRtArgs {
    tt::tt_metal::CoreCoord core;
    bool noop = false;  // outside both work groups: create_program_artifacts zero-fills its args
    uint32_t in_units = 0;
    uint32_t out_units = 0;
    uint32_t start_id = 0;
    uint32_t compute_units = 0;
};

// Core-invariant ROW_MAJOR-interleaved chunk constants, reader/writer slots 3-7. All shape-derived,
// and ROW_MAJOR hashes padded_shape, so a cache hit never has to re-apply them.
struct RmChunkConstants {
    uint32_t chunks_per_row = 1;
    uint32_t input_chunk_size = 0;
    uint32_t input_last_chunk_size = 0;
    uint32_t output_chunk_size = 0;
    uint32_t output_last_chunk_size = 0;
    uint32_t rows_per_tile = 1;
    uint32_t total_rows = 0;
};

// Enumerates the per-core work split for the current tensors. create_program_artifacts and
// override_runtime_arguments both go through this, so a cache-hit patch cannot drift from the layout
// the miss path built. Cheap next to create_program_artifacts (no kernel sources, DFBs, or spec
// allocation) but deliberately not O(1): the TILE-layout hash omits shape, so the split really does
// change between hits on the same cached program.
template <typename Fn>
void enumerate_core_rt_args(
    const UnaryDeviceOperation::operation_attributes_t& operation_attributes,
    const UnaryDeviceOperation::tensor_args_t& tensor_args,
    const Tensor& output,
    const Fn& fn) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;
    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool rm_interleaved = is_row_major && !has_sharding;
    const auto& all_device_cores = operation_attributes.worker_grid;

    const auto row_major =
        has_sharding ? shard_specs->input_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;
    auto grid = has_sharding ? shard_specs->input_shard_spec.grid : CoreRangeSet{};

    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (has_sharding) {
                const auto& shard_start_coord = grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }
    const uint32_t num_cores_total =
        zero_start_grid ? compute_with_storage_grid.x * compute_with_storage_grid.y : all_device_cores.num_cores();

    const uint32_t tile_height = output.tensor_spec().tile().get_height();
    const uint32_t tile_width = output.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;

    const auto input_df = cb_dataformat_for(input.dtype());
    const auto output_df = cb_dataformat_for(output.dtype());
    const uint32_t input_tile_bytes = tile_size(input_df);
    const uint32_t output_tile_bytes = tile_size(output_df);

    const uint32_t input_page_bytes = rm_interleaved ? static_cast<uint32_t>(input.buffer()->page_size()) : 0;
    const uint32_t output_page_bytes = rm_interleaved ? static_cast<uint32_t>(output.buffer()->page_size()) : 0;
    RmChunkConstants k;
    k.chunks_per_row = rm_interleaved ? (input_page_bytes + input_tile_bytes - 1) / input_tile_bytes : 1;
    k.input_chunk_size = input_tile_bytes;
    k.input_last_chunk_size =
        rm_interleaved ? input_page_bytes - ((k.chunks_per_row - 1) * input_tile_bytes) : input_tile_bytes;
    k.output_chunk_size = output_tile_bytes;
    k.output_last_chunk_size =
        rm_interleaved ? output_page_bytes - ((k.chunks_per_row - 1) * output_tile_bytes) : output_tile_bytes;
    k.total_rows = rm_interleaved ? output.buffer()->num_pages() : 0;
    if (rm_interleaved && input_page_bytes > 0 && input_page_bytes < input_tile_bytes) {
        const uint32_t input_element_size = datum_size(input_df);
        const uint32_t row_width_elements = input_page_bytes / input_element_size;
        const uint32_t aligned_page_size = static_cast<uint32_t>(input.buffer()->aligned_page_size());
        if (input_page_bytes == aligned_page_size && row_width_elements > 0) {
            k.rows_per_tile = tile_hw / row_width_elements;
        }
    }
    const uint32_t out_num_tiles =
        rm_interleaved ? (k.total_rows + k.rows_per_tile - 1) / k.rows_per_tile : output.physical_volume() / tile_hw;
    const uint32_t oWt = output.padded_shape()[-1] / output.tensor_spec().tile().get_width();

    std::vector<CoreCoord> cores;
    if (has_sharding) {
        const CoreRangeSet& core_group_1 = grid;
        const uint32_t out_shard_height = shard_specs->output_shard_spec.shape[0] / tile_height;
        const uint32_t out_shard_width = shard_specs->output_shard_spec.shape[1] / tile_width;
        auto out_memory_layout = output.memory_config().is_sharded() ? output.memory_config().memory_layout()
                                                                     : input.memory_config().memory_layout();
        const uint32_t num_shards_per_width =
            CMAKE_UNIQUE_NAMESPACE::get_shards_per_width(shard_specs->output_shard_spec, out_memory_layout);

        auto compute_shard_pages = [&](const ShardSpec& spec,
                                       const auto& tensor) -> std::function<uint32_t(CoreCoord)> {
            if (is_row_major) {
                auto df = cb_dataformat_for(tensor.dtype());
                uint32_t ts = tile_size(df);
                uint32_t shard_bytes = spec.shape[0] * spec.shape[1] * datum_size(df);
                uint32_t pages = shard_bytes / ts;
                return [pages](CoreCoord) -> uint32_t { return pages; };
            }
            auto end_core = spec.grid.ranges().rbegin()->end_coord;
            bool rm = spec.orientation == ShardOrientation::ROW_MAJOR;
            auto mem_layout = tensor.memory_config().memory_layout();
            uint32_t sh = tt::round_up(spec.shape[0], tile_height) / tile_height;
            uint32_t sw = tt::round_up(spec.shape[1], tile_width) / tile_width;
            const auto& pshape = tensor.padded_shape();
            uint32_t D = pshape.rank() >= 5 ? pshape[-5] : 1;
            uint32_t N = pshape[-4], C = pshape[-3];
            uint32_t Ht = pshape[-2] / tile_height, Wt = pshape[-1] / tile_width;
            uint32_t unrolled_Ht = D * N * C * Ht;
            uint32_t last_h = sh - (tt::round_up(unrolled_Ht, sh) - unrolled_Ht);
            uint32_t last_w = sw - (tt::round_up(Wt, sw) - Wt);

            return [=](CoreCoord core) -> uint32_t {
                uint32_t h = sh, w = sw;
                if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
                    mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                    if (core == end_core) {
                        h = last_h;
                        w = last_w;
                    }
                } else {
                    if (rm) {
                        if (core.x == end_core.x) {
                            w = last_w;
                        }
                        if (core.y == end_core.y) {
                            h = last_h;
                        }
                    } else {
                        if (core.y == end_core.y) {
                            w = last_w;
                        }
                        if (core.x == end_core.x) {
                            h = last_h;
                        }
                    }
                }
                return h * w;
            };
        };

        auto in_shard_pages = compute_shard_pages(shard_specs->input_shard_spec, input);
        auto out_shard_pages = compute_shard_pages(shard_specs->output_shard_spec, output);

        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid.x,
                compute_with_storage_grid.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(core_group_1, all_device_cores, row_major);
        }

        for (uint32_t i = 0; i < num_cores_total; ++i) {
            const auto& core = cores[i];
            if (!core_group_1.contains(core)) {
                fn(CoreRtArgs{.core = core, .noop = true}, k);
                continue;
            }
            const uint32_t in_tiles = in_shard_pages(core);
            const uint32_t o_tiles = out_shard_pages(core);
            const uint32_t out_start_id = ((i / num_shards_per_width) * (out_shard_height * oWt)) +
                                          ((i % num_shards_per_width) * out_shard_width);
            fn(
                CoreRtArgs{
                    .core = core,
                    .in_units = in_tiles,
                    .out_units = o_tiles,
                    .start_id = out_start_id,
                    .compute_units = o_tiles},
                k);
        }
        return;
    }

    uint32_t num_tiles_per_core_group_1{}, num_tiles_per_core_group_2{};
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores;
    if (zero_start_grid) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            split_work_to_cores(compute_with_storage_grid, out_num_tiles, row_major);
        cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            split_work_to_cores(all_device_cores, out_num_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; ++i) {
        const auto& core = cores[i];
        uint32_t npc = 0;
        if (core_group_1.contains(core)) {
            npc = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            npc = num_tiles_per_core_group_2;
        } else {
            fn(CoreRtArgs{.core = core, .noop = true}, k);
            continue;
        }
        fn(
            CoreRtArgs{
                .core = core,
                .in_units = npc,
                .out_units = npc,
                .start_id = start_tile_id,
                .compute_units = rm_interleaved ? npc * k.chunks_per_row : npc},
            k);
        start_tile_id += npc;
    }
}

// The per-dispatch run args: every runtime arg on every core, plus both tensor bindings.
// create_program_artifacts and override_runtime_arguments both return this. The legacy cache-hit
// override re-applied every slot the miss path wrote, so both paths share one builder and cannot drift.
tt::tt_metal::experimental::ProgramRunArgs make_run_args(
    const UnaryDeviceOperation::operation_attributes_t& operation_attributes,
    const UnaryDeviceOperation::tensor_args_t& tensor_args,
    const Tensor& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& input = tensor_args.input;
    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool rm_interleaved = input.layout() == Layout::ROW_MAJOR && !has_sharding;

    uint32_t packed_scalar1 = 0, packed_scalar2 = 0;
    pack_first_op_scalars(operation_attributes.op_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    // Sharded readers/writers take {units, start_id}; interleaved add the five chunk fields, which are
    // zero unless ROW_MAJOR. A changed split can flip a core between noop and active, so every slot is
    // written on every core, noop cores zero-filled -- otherwise a flipped core keeps stale args.
    auto add_dm_args = [&](KernelRunArgs& kra,
                           const CoreCoord& core,
                           uint32_t num_pages,
                           uint32_t start_id,
                           const std::array<uint32_t, 5>& chunk_tail) {
        AddRuntimeArgsForNode(kra.runtime_arg_values, core, {{"num_pages", num_pages}, {"start_id", start_id}});
        if (!has_sharding) {
            AddRuntimeArgsForNode(
                kra.runtime_arg_values,
                core,
                {{"chunks_per_row", chunk_tail[0]},
                 {"chunk_size", chunk_tail[1]},
                 {"last_chunk_size", chunk_tail[2]},
                 {"rows_per_tile", chunk_tail[3]},
                 {"total_rows", chunk_tail[4]}});
        }
    };

    enumerate_core_rt_args(operation_attributes, tensor_args, output, [&](const CoreRtArgs& w, const RmChunkConstants& kc) {
        if (w.noop) {
            add_dm_args(reader_run_args, w.core, 0, 0, {});
            add_dm_args(writer_run_args, w.core, 0, 0, {});
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values,
                w.core,
                {{"num_tiles", 0}, {"packed_scalar1", 0}, {"packed_scalar2", 0}});
            return;
        }
        const std::array<uint32_t, 5> rtail =
            rm_interleaved
                ? std::array<
                      uint32_t,
                      5>{kc.chunks_per_row, kc.input_chunk_size, kc.input_last_chunk_size, kc.rows_per_tile, kc.total_rows}
                : std::array<uint32_t, 5>{};
        const std::array<uint32_t, 5> wtail =
            rm_interleaved
                ? std::array<
                      uint32_t,
                      5>{kc.chunks_per_row, kc.output_chunk_size, kc.output_last_chunk_size, kc.rows_per_tile, kc.total_rows}
                : std::array<uint32_t, 5>{};
        add_dm_args(reader_run_args, w.core, w.in_units, w.start_id, rtail);
        add_dm_args(writer_run_args, w.core, w.out_units, w.start_id, wtail);
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            w.core,
            {{"num_tiles", w.compute_units}, {"packed_scalar1", packed_scalar1}, {"packed_scalar2", packed_scalar2}});
    });

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    // Both tensors, every dispatch. The bindings carry the accessor base address and shape the legacy
    // path wrote into reader/writer arg 0 and the accessor common args, and they back the sharded-path
    // borrowed DFBs whose L1 address the legacy override re-applied.
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};
    return run_args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts UnaryDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& input = tensor_args.input;
    const auto& ops_chain = operation_attributes.op_chain;
    TT_FATAL(!ops_chain.empty(), "Unary: op_chain must not be empty");

    uint32_t packed_scalar1 = 0;
    uint32_t packed_scalar2 = 0;

    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    DataFormat dfb_data_format = cb_dataformat_for(input.dtype());
    uint32_t single_tile_size = tile_size(dfb_data_format);
    DataFormat dfb_data_format_output = cb_dataformat_for(output.dtype());
    uint32_t single_tile_size_output = tile_size(dfb_data_format_output);

    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool src_sharded = has_sharding && input.is_sharded();
    const bool dst_sharded = has_sharding && output.is_sharded();

    // For ROW_MAJOR interleaved: use tile_size DFB entries and group/chunk rows.
    // For sharded ROW_MAJOR or TILE layout: DFB entry is always tile_size.
    const bool rm_interleaved = is_row_major && !has_sharding;
    const uint32_t input_dfb_page_size = single_tile_size;
    const uint32_t output_dfb_page_size = single_tile_size_output;

    auto shard_pages = [](const tt::tt_metal::ShardSpec& spec, const Tensor& t, bool rm) -> uint32_t {
        if (rm) {
            auto df = cb_dataformat_for(t.dtype());
            uint32_t ts = tile_size(df);
            uint32_t shard_bytes = spec.shape[0] * spec.shape[1] * datum_size(df);
            TT_ASSERT(
                shard_bytes % ts == 0,
                "ROW_MAJOR shard size in bytes ({}) must be a multiple of DFB page size ({})",
                shard_bytes,
                ts);
            return shard_bytes / ts;
        }
        return spec.numel() / t.tensor_spec().tile().get_tile_hw();
    };
    const auto src_num_tiles_per_shard =
        src_sharded ? std::optional<uint32_t>(shard_pages(shard_specs->input_shard_spec, input, is_row_major))
                    : std::nullopt;
    const auto dst_num_tiles_per_shard =
        dst_sharded ? std::optional<uint32_t>(shard_pages(shard_specs->output_shard_spec, output, is_row_major))
                    : std::nullopt;

    const auto& all_device_cores = operation_attributes.worker_grid;
    const bool needs_tmp0 = needs_tmp0_dfb(ops_chain[0].type());

    const bool math_approx_mode = false;
    std::map<std::string, std::string> unary_defines = get_block_defines(ops_chain, "0", "0", input.dtype());
    add_input_dtype_defines(input.dtype(), unary_defines);
    const bool logit_clamp_enabled = pack_first_op_scalars(ops_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    const std::string compute_path = fmt::format(
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/{}",
        get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    DataFormat dfb_data_format_for_input =
        (ops_chain[0].type() == unary::UnaryOpType::BITCAST) ? dfb_data_format_output : dfb_data_format;

    // --- Dataflow Buffers ---
    // On the native-sharded path the input / output DFBs are built on the tensors' own L1 shards.
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = input_dfb_page_size,
        .num_entries = src_num_tiles_per_shard.value_or(2),
        .data_format_metadata = dfb_data_format_for_input,
        .borrowed_from = src_sharded ? std::optional<TensorParamName>(INPUT) : std::nullopt,
    });

    if (needs_tmp0) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TMP0_DFB,
            .entry_size = input_dfb_page_size,
            .num_entries = 2,
            .data_format_metadata = dfb_data_format,
        });
    }

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_dfb_page_size,
        .num_entries = dst_num_tiles_per_shard.value_or(2),
        .data_format_metadata = dfb_data_format_output,
        .borrowed_from = dst_sharded ? std::optional<TensorParamName>(OUTPUT) : std::nullopt,
    });

    // --- Tensor Parameters ---
    // The TILE-layout program hash omits shape, and so rank: one cached program legitimately serves
    // tensors of any shape and logical rank, which the legacy accessors already allowed through
    // ArgConfig::RuntimeTensorShape. The hash pins everything else a match compares exactly
    // (tensor_layout, and the sharded distribution geometry resolved from each TensorSpec).
    const TensorSpecRelaxations shape_dynamic{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec(), .relaxations = shape_dynamic};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec(), .relaxations = shape_dynamic};

    // --- Reader Kernel ---
    // Sharded readers/writers take {units, start_id}; interleaved add the five chunk fields.
    Group<std::string> dm_runtime_arg_names = {"num_pages", "start_id"};
    if (!has_sharding) {
        dm_runtime_arg_names.insert(
            dm_runtime_arg_names.end(),
            {"chunks_per_row", "chunk_size", "last_chunk_size", "rows_per_tile", "total_rows"});
    }

    const auto arch = input.device()->arch();

    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary.cpp",
        .compiler_options =
            {.defines =
                 {
                     {"SRC_SHARDED", src_sharded ? "1" : "0"},
                     {"RM_INTERLEAVED", rm_interleaved ? "1" : "0"},
                 }},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB,
            .accessor_name = "src",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = dm_runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    // --- Writer Kernel ---
    const KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary.cpp",
        .compiler_options =
            {.defines =
                 {
                     {"DST_SHARDED", dst_sharded ? "1" : "0"},
                     {"RM_INTERLEAVED", rm_interleaved ? "1" : "0"},
                 }},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "dst",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "dst",
        }},
        .runtime_arg_schema = {.runtime_arg_names = dm_runtime_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // --- Compute Kernel ---
    KernelSpec::CompileTimeArgs compute_compile_time_args;
    if (ops_chain[0].type() == UnaryOpType::HARDSWISH) {
        compute_compile_time_args.emplace("is_float32", static_cast<uint32_t>(unary_defines.contains("INP_FLOAT32")));
        compute_compile_time_args.emplace(
            "is_int",
            static_cast<uint32_t>(unary_defines.contains("INP_INT32") || unary_defines.contains("INP_UINT32")));
    } else if (ops_chain[0].type() == UnaryOpType::LOGIT) {
        compute_compile_time_args.emplace("do_clamp", static_cast<uint32_t>(logit_clamp_enabled));
    }
    compute_compile_time_args.emplace("input_data_format", static_cast<uint32_t>(dfb_data_format));

    Group<DFBBinding> compute_dfb_bindings = {DFBBinding{
        .dfb_spec_name = IN_DFB,
        .accessor_name = "in",
        .endpoint_type = DFBEndpointType::CONSUMER,
    }};
    if (needs_tmp0) {
        // The LOGIT kernel is the temporary's only toucher: it both fills and drains it.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = TMP0_DFB,
            .accessor_name = "tmp0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = TMP0_DFB,
            .accessor_name = "tmp0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = OUT_DFB,
        .accessor_name = "out",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });

    // Legacy set UnpackToDestFp32 on the input and temporary buffers under preserve_fp32_precision and
    // left them at Default otherwise. Metal 2.0 additionally requires an explicit entry for a consumed
    // Float32 buffer under a 32-bit Dest, where legacy silently defaulted; Default is UnpackToSrc.
    ComputeUnpackModes unpack_modes;
    auto set_unpack_mode = [&](const DFBSpecName& dfb, DataFormat df) {
        if (operation_attributes.preserve_fp32_precision) {
            unpack_modes.emplace(dfb, UnpackMode::UnpackToDest);
        } else if (operation_attributes.fp32_dest_acc_en && df == DataFormat::Float32) {
            unpack_modes.emplace(dfb, UnpackMode::UnpackToSrc);
        }
    };
    set_unpack_mode(IN_DFB, dfb_data_format_for_input);
    if (needs_tmp0) {
        set_unpack_mode(TMP0_DFB, dfb_data_format);
    }

    // Field values carried over from the legacy ComputeConfigDescriptor: math_fidelity = HiFi4,
    // fp32_dest_acc_en, bfp8_pack_precise, math_approx_mode = false. dst_full_sync_en was left at its
    // legacy default (false), i.e. double_buffer_dest = true, which is also the Metal 2.0 default.
    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = compute_path,
        .compiler_options =
            {
                .defines = Table<std::string, std::string>(unary_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "packed_scalar1", "packed_scalar2"}},
        .hw_config = ComputeHardwareConfig{ComputeGen1Config{
            .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
            .bfp_pack_precision_mode =
                operation_attributes.bfp8_pack_precise ? Precision::Precise : Precision::Approximate,
            .enable_32_bit_dest = operation_attributes.fp32_dest_acc_en,
            .unpack_modes = std::move(unpack_modes),
        }},
    };

    ProgramSpec spec{
        .name = "eltwise_unary",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "eltwise_unary",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_device_cores,
        }},
    };

    // --- Per-core runtime args ---
    // Work split + per-core values come from enumerate_core_rt_args, shared with
    // override_runtime_arguments so the cache-hit patch and the miss path cannot disagree.
    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = make_run_args(operation_attributes, tensor_args, output),
    };
}

tt::tt_metal::experimental::ProgramRunArgs UnaryDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The TILE-layout hash omits shape, so the work split, start ids and accessor shape args all vary
    // between hits: re-apply exactly those, through the same enumeration create_program_artifacts uses.
    // The tensor arguments refresh the accessor shape and base addresses, and the sharded-path
    // borrowed DFB addresses, which on this concept the framework does not patch by itself.
    return make_run_args(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::operations::unary
