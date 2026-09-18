// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
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
using namespace tt::tt_metal::experimental;

namespace {

// Program-scope resource names. Declared once and referenced at every use site, because both
// create_program_artifacts and override_runtime_arguments name the same kernels and tensor
// parameters and a divergence between them is a silent mis-binding.
const KernelSpecName kReaderKernel{"reader"};
const KernelSpecName kWriterKernel{"writer"};
const KernelSpecName kComputeKernel{"compute"};
const DFBSpecName kSrcDfb{"src"};
const DFBSpecName kTmp0Dfb{"tmp0"};
const DFBSpecName kDstDfb{"dst"};
const TensorParamName kSrcTensor{"src"};
const TensorParamName kDstTensor{"dst"};

// Per-core runtime-arg values, in the slot order create_program_artifacts writes them.
struct CoreRtArgs {
    tt::tt_metal::CoreCoord core;
    bool noop = false;  // outside both work groups: create_program_artifacts zero-fills its args
    uint32_t in_units = 0;
    uint32_t out_units = 0;
    uint32_t start_id = 0;
    uint32_t compute_units = 0;
};

// Core-invariant ROW_MAJOR-interleaved chunk constants, reader/writer args 3-7. All shape-derived,
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

    const auto input_df = datatype_to_dataformat_converter(input.dtype());
    const auto output_df = datatype_to_dataformat_converter(output.dtype());
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
                auto df = datatype_to_dataformat_converter(tensor.dtype());
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

// Fills the three kernels' per-node runtime-arg tables for the current tensors. Shared by
// create_program_artifacts and override_runtime_arguments for the same reason
// enumerate_core_rt_args is: the cache-hit values and the cache-miss values must be derived by
// one piece of code, or they can drift.
//
// Every argument the schema declares is written for every node -- active cores with their values,
// cores outside the work set with zeros. That is what keeps a core the split flips between active
// and no-op from retaining stale args, which is why the no-op branch writes zeros explicitly
// rather than being skipped.
void build_kernel_run_args(
    const UnaryDeviceOperation::operation_attributes_t& operation_attributes,
    const UnaryDeviceOperation::tensor_args_t& tensor_args,
    const Tensor& output,
    bool has_sharding,
    bool rm_interleaved,
    uint32_t packed_scalar1,
    uint32_t packed_scalar2,
    KernelRunArgs& reader_run_args,
    KernelRunArgs& writer_run_args,
    KernelRunArgs& compute_run_args) {
    enumerate_core_rt_args(
        operation_attributes, tensor_args, output, [&](const CoreRtArgs& w, const RmChunkConstants& kc) {
            // A no-op core leaves every CoreRtArgs count at its zero default, so the unit and
            // start-id writes below already carry the zero-fill for it.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, w.core, {{"num_pages", w.in_units}, {"start_id", w.start_id}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values, w.core, {{"num_pages", w.out_units}, {"start_id", w.start_id}});
            if (!has_sharding) {
                // The chunk constants are read only under RM_INTERLEAVED. The interleaved schema
                // carries them in both layouts and zero-fills the TILE case, so the arg set a core
                // holds never depends on which branch last wrote it.
                const bool rm = rm_interleaved && !w.noop;
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    w.core,
                    {{"chunks_per_row", rm ? kc.chunks_per_row : 0u},
                     {"chunk_size", rm ? kc.input_chunk_size : 0u},
                     {"last_chunk_size", rm ? kc.input_last_chunk_size : 0u},
                     {"rows_per_tile", rm ? kc.rows_per_tile : 0u},
                     {"total_rows", rm ? kc.total_rows : 0u}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    w.core,
                    {{"chunks_per_row", rm ? kc.chunks_per_row : 0u},
                     {"chunk_size", rm ? kc.output_chunk_size : 0u},
                     {"last_chunk_size", rm ? kc.output_last_chunk_size : 0u},
                     {"rows_per_tile", rm ? kc.rows_per_tile : 0u},
                     {"total_rows", rm ? kc.total_rows : 0u}});
            }
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values,
                w.core,
                {{"num_tiles", w.compute_units},
                 {"packed_scalar1", w.noop ? 0u : packed_scalar1},
                 {"packed_scalar2", w.noop ? 0u : packed_scalar2}});
        });
}

}  // namespace

ttnn::device_operation::ProgramArtifacts UnaryDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = operation_attributes.op_chain;
    TT_FATAL(!ops_chain.empty(), "Unary: op_chain must not be empty");

    uint32_t packed_scalar1 = 0;
    uint32_t packed_scalar2 = 0;

    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    DataFormat dfb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile_size(dfb_data_format);
    DataFormat dfb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tile_size(dfb_data_format_output);

    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool src_sharded = has_sharding && input.is_sharded();
    const bool dst_sharded = has_sharding && output.is_sharded();

    // For ROW_MAJOR interleaved: use tile_size DFB entries and group/chunk rows.
    // For sharded ROW_MAJOR or TILE layout: DFB entry is always tile_size.
    const bool rm_interleaved = is_row_major && !has_sharding;
    const uint32_t input_dfb_entry_size = single_tile_size;
    const uint32_t output_dfb_entry_size = single_tile_size_output;

    auto shard_pages = [](const tt::tt_metal::ShardSpec& spec, const Tensor& t, bool rm) -> uint32_t {
        if (rm) {
            auto df = datatype_to_dataformat_converter(t.dtype());
            uint32_t ts = tile_size(df);
            uint32_t shard_bytes = spec.shape[0] * spec.shape[1] * datum_size(df);
            TT_ASSERT(
                shard_bytes % ts == 0,
                "ROW_MAJOR shard size in bytes ({}) must be a multiple of DFB entry size ({})",
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
    const auto* device = input.device();

    const bool math_approx_mode = false;
    std::map<std::string, std::string> unary_defines = get_block_defines(ops_chain, "0", "0", input.dtype());
    add_input_dtype_defines(input.dtype(), unary_defines);
    const bool logit_clamp_enabled =
        CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(ops_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    const bool has_tmp0_dfb = CMAKE_UNIQUE_NAMESPACE::needs_tmp0_dfb(ops_chain[0].type());

    // eltwise_sfpu.cpp is lent to three external program factories that are still on the legacy
    // host API, so this port binds a Metal 2.0 fork of it that lives beside the original instead of
    // converting the original in place. Every other compute kernel here is unary-exclusive and was
    // converted where it sits.
    std::string_view compute_kernel_file = get_compute_kernel_path(ops_chain[0].type(), input.dtype());
    if (compute_kernel_file == "eltwise_sfpu.cpp") {
        compute_kernel_file = "eltwise_sfpu_metal2.cpp";
    }
    const std::string compute_path =
        fmt::format("ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/{}", compute_kernel_file);

    DataFormat dfb_data_format_for_input =
        (ops_chain[0].type() == unary::UnaryOpType::BITCAST) ? dfb_data_format_output : dfb_data_format;

    // --- Dataflow Buffers ---
    // tile_format_metadata is deliberately left unset on all three, matching the legacy CBs, which
    // set no tile either. Entry sizes come from tile_size(DataFormat), which assumes a 32x32 tile.
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = kSrcDfb,
        .entry_size = input_dfb_entry_size,
        .num_entries = src_num_tiles_per_shard.value_or(2),
        .data_format_metadata = dfb_data_format_for_input,
        // Under sharding the input DFB is a view onto the input tensor's own L1 shard, so the
        // reader only handshakes and the accessor is compiled out of the kernel entirely.
        .borrowed_from = src_sharded ? std::optional<TensorParamName>(kSrcTensor) : std::nullopt,
    });

    if (has_tmp0_dfb) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = kTmp0Dfb,
            .entry_size = input_dfb_entry_size,
            .num_entries = 2,
            .data_format_metadata = dfb_data_format,
        });
    }

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = kDstDfb,
        .entry_size = output_dfb_entry_size,
        .num_entries = dst_num_tiles_per_shard.value_or(2),
        .data_format_metadata = dfb_data_format_output,
        .borrowed_from = dst_sharded ? std::optional<TensorParamName>(kDstTensor) : std::nullopt,
    });

    // --- Tensor parameters ---
    // Both slots carry the relaxation the op's cache key requires: the TILE-layout key omits shape
    // and rank entirely, so one cache entry legitimately serves many shapes and the first hit at a
    // new shape would otherwise fail the strict TensorSpec match. match_page_size and
    // match_padded_shape_only are deliberately not set.
    constexpr TensorSpecRelaxations kDynamicShape{
        .dynamic_tensor_shape = true,
        .relax_logical_rank = true,
    };
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{
            .unique_id = kSrcTensor,
            .spec = input.tensor_spec(),
            .relaxations = kDynamicShape,
        },
        TensorParameter{
            .unique_id = kDstTensor,
            .spec = output.tensor_spec(),
            .relaxations = kDynamicShape,
        },
    };

    // --- Runtime-arg schema ---
    // Mirrors the legacy per-core slot count exactly, minus the buffer-address slot that became a
    // tensor binding: three slots when sharded, eight otherwise. The five chunk args are read only
    // under RM_INTERLEAVED but are declared in both interleaved layouts, as the legacy uniform
    // eight-slot layout did.
    Group<std::string> data_movement_arg_names = {"num_pages", "start_id"};
    if (!has_sharding) {
        data_movement_arg_names.insert(
            data_movement_arg_names.end(),
            {"chunks_per_row", "chunk_size", "last_chunk_size", "rows_per_tile", "total_rows"});
    }

    // --- Reader Kernel ---
    KernelSpec::CompilerOptions::Defines reader_defines;
    reader_defines["SRC_SHARDED"] = src_sharded ? "1" : "0";
    reader_defines["RM_INTERLEAVED"] = rm_interleaved ? "1" : "0";

    const KernelSpec reader{
        .unique_id = kReaderKernel,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = kSrcDfb,
            .accessor_name = "src",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = kSrcTensor,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = data_movement_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // --- Writer Kernel ---
    KernelSpec::CompilerOptions::Defines writer_defines;
    writer_defines["DST_SHARDED"] = dst_sharded ? "1" : "0";
    writer_defines["RM_INTERLEAVED"] = rm_interleaved ? "1" : "0";

    const KernelSpec writer{
        .unique_id = kWriterKernel,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = kDstDfb,
            .accessor_name = "dst",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = kDstTensor,
            .accessor_name = "dst",
        }},
        .runtime_arg_schema = {.runtime_arg_names = data_movement_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // --- Compute Kernel ---
    KernelSpec::CompileTimeArgs compute_compile_time_args;
    if (ops_chain[0].type() == UnaryOpType::HARDSWISH) {
        compute_compile_time_args["is_float32"] = static_cast<uint32_t>(unary_defines.contains("INP_FLOAT32"));
        compute_compile_time_args["is_int"] =
            static_cast<uint32_t>(unary_defines.contains("INP_INT32") || unary_defines.contains("INP_UINT32"));
    } else if (ops_chain[0].type() == UnaryOpType::LOGIT) {
        compute_compile_time_args["do_clamp"] = static_cast<uint32_t>(logit_clamp_enabled);
    }
    // Carried over from the legacy compile-time-arg list, where it was appended for every op type.
    // No compute kernel in this op reads it.
    compute_compile_time_args["data_format"] = static_cast<uint32_t>(dfb_data_format);

    // Legacy filled a buffer-index-keyed vector with UnpackToDestMode::Default and set the input
    // and tmp0 slots to UnpackToDestFp32 under preserve_fp32_precision. Rekeyed to DFB names, that
    // is UnpackToDest on exactly those two, and UnpackToSrc (the lowering of Default) otherwise.
    //
    // The entry is written out even where it is UnpackToSrc, rather than omitted: Metal 2.0
    // *requires* an explicit choice for a consumed Float32 DFB when the Dest register is 32 bits
    // wide, and that combination is reachable with a legacy Default (a BITCAST to FLOAT32 from a
    // non-FLOAT32 input sets fp32_dest_acc_en without setting preserve_fp32_precision). Writing
    // every consumed DFB's mode is behaviour-identical and removes the case analysis.
    //
    // tmp0's entry is gated on the DFB existing, because an entry naming a DFB the kernel does not
    // bind is rejected. Legacy set that slot unconditionally, where it was simply ignored.
    const UnpackMode consumed_unpack_mode =
        operation_attributes.preserve_fp32_precision ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc;
    ComputeUnpackModes compute_unpack_modes;
    compute_unpack_modes.emplace(kSrcDfb, consumed_unpack_mode);
    if (has_tmp0_dfb) {
        compute_unpack_modes.emplace(kTmp0Dfb, consumed_unpack_mode);
    }

    // The legacy ComputeConfigDescriptor's values, carried across one for one:
    //   math_fidelity=HiFi4 -> fpu_math_fidelity; math_approx_mode=false -> sfpu_precision_mode;
    //   fp32_dest_acc_en -> enable_32_bit_dest; bfp8_pack_precise -> bfp_pack_precision_mode.
    // dst_full_sync_en was left at its legacy default of false, which is double_buffer_dest = true
    // and already the Metal 2.0 default, so it needs no explicit setting.
    const ComputeGen1Config compute_gen1_config{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
        .bfp_pack_precision_mode = operation_attributes.bfp8_pack_precise ? Precision::Precise : Precision::Approximate,
        .enable_32_bit_dest = operation_attributes.fp32_dest_acc_en,
        .unpack_modes = std::move(compute_unpack_modes),
    };

    // tmp0 is touched by logit_kernel.cpp alone -- it packs into the buffer and copies back out of
    // it one tile at a time -- so the one compute kernel is bound as both its producer and its
    // consumer.
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = kSrcDfb,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = kDstDfb,
            .accessor_name = "output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (has_tmp0_dfb) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = kTmp0Dfb,
            .accessor_name = "tmp0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = kTmp0Dfb,
            .accessor_name = "tmp0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // O3 is explicit because the legacy compute config defaulted there while Metal 2.0's
    // kernel-type-agnostic CompilerOptions defaults to O2; leaving it unset would quietly drop a
    // level on both the compile and the link.
    const KernelSpec compute{
        .unique_id = kComputeKernel,
        .source = compute_path,
        .compiler_options =
            {.defines = KernelSpec::CompilerOptions::Defines(unary_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "packed_scalar1", "packed_scalar2"}},
        .hw_config = ComputeHardwareConfig{compute_gen1_config},
    };

    // --- Per-core runtime args ---
    // Work split + per-core values come from enumerate_core_rt_args, shared with
    // override_runtime_arguments so the cache-hit patch and the miss path cannot disagree.
    KernelRunArgs reader_run_args{.kernel = kReaderKernel};
    KernelRunArgs writer_run_args{.kernel = kWriterKernel};
    KernelRunArgs compute_run_args{.kernel = kComputeKernel};
    build_kernel_run_args(
        operation_attributes,
        tensor_args,
        output,
        has_sharding,
        rm_interleaved,
        packed_scalar1,
        packed_scalar2,
        reader_run_args,
        writer_run_args,
        compute_run_args);

    ProgramSpec spec{
        .name = "unary",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{
            .name = "unary",
            .kernels = {kReaderKernel, kWriterKernel, kComputeKernel},
            .target_nodes = all_device_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{kSrcTensor, input.mesh_tensor()}, {kDstTensor, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs UnaryDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt::tt_metal;
    // The TILE-layout hash omits shape, so the work split and start ids all vary between hits:
    // re-apply exactly those, through the same enumeration create_program_artifacts uses. The
    // accessor shape args the legacy override also had to rebuild are now carried by the tensor
    // bindings below, which the framework derives from them.
    const auto& input = tensor_args.input;
    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool rm_interleaved = input.layout() == Layout::ROW_MAJOR && !has_sharding;

    // A changed split can flip a core between noop and active, so write every arg
    // create_program_artifacts writes rather than only the ones that usually move -- otherwise a
    // flipped core keeps stale args.
    uint32_t packed_scalar1 = 0, packed_scalar2 = 0;
    CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(
        operation_attributes.op_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    KernelRunArgs reader_run_args{.kernel = kReaderKernel};
    KernelRunArgs writer_run_args{.kernel = kWriterKernel};
    KernelRunArgs compute_run_args{.kernel = kComputeKernel};
    build_kernel_run_args(
        operation_attributes,
        tensor_args,
        output,
        has_sharding,
        rm_interleaved,
        packed_scalar1,
        packed_scalar2,
        reader_run_args,
        writer_run_args,
        compute_run_args);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    // On this factory concept the framework refreshes nothing on the op's behalf, so both tensor
    // bindings are rebuilt on every dispatch. That is what the legacy override did too: it wrote
    // both buffer addresses into their arg slots and re-pointed both tensor-backed circular
    // buffers. An omitted binding would stay frozen at the cache-miss address.
    run_args.tensor_args = {{kSrcTensor, input.mesh_tensor()}, {kDstTensor, output.mesh_tensor()}};

    return run_args;
}

}  // namespace ttnn::operations::unary
