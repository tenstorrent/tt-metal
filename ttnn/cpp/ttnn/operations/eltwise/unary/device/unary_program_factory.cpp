// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace tt::tt_metal;
using namespace ttnn::operations::unary;
using namespace ttnn::operations::unary::utils;
using ttnn::operations::unary::EltwiseUnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

void apply_input_dtype_defines(DataType dtype, std::map<std::string, std::string>& defines) {
    if (dtype == DataType::FLOAT32) {
        defines["INP_FLOAT32"] = "1";
    } else if (dtype == DataType::INT32) {
        defines["INP_INT32"] = "1";
    } else if (dtype == DataType::UINT32) {
        defines["INP_UINT32"] = "1";
    } else {
        defines["INP_FLOAT"] = "1";
    }
}

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

bool needs_tmp0_cb(UnaryOpType t) { return t == UnaryOpType::LOGIT; }

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

// Per-core runtime-arg values, in the slot order create_descriptor writes them.
struct CoreRtArgs {
    tt::tt_metal::CoreCoord core;
    bool noop = false;  // outside both work groups: create_descriptor zero-fills its args
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

// Enumerates the per-core work split for the current tensors. create_descriptor and
// override_runtime_arguments both go through this, so a cache-hit patch cannot drift from the layout
// the miss path built. Cheap next to create_descriptor (no kernel sources, CBs, or descriptor
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

}  // namespace

tt::tt_metal::ProgramDescriptor UnaryDeviceOperation::ProgramFactory::create_descriptor(
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

    ProgramDescriptor desc;

    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tile_size(cb_data_format_output);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool src_sharded = has_sharding && input.is_sharded();
    const bool dst_sharded = has_sharding && output.is_sharded();

    // For ROW_MAJOR interleaved: use tile_size CB pages and group/chunk rows.
    // For sharded ROW_MAJOR or TILE layout: CB page is always tile_size.
    const bool rm_interleaved = is_row_major && !has_sharding;
    const uint32_t input_cb_page_size = single_tile_size;
    const uint32_t output_cb_page_size = single_tile_size_output;

    auto shard_pages = [](const tt::tt_metal::ShardSpec& spec, const Tensor& t, bool rm) -> uint32_t {
        if (rm) {
            auto df = datatype_to_dataformat_converter(t.dtype());
            uint32_t ts = tile_size(df);
            uint32_t shard_bytes = spec.shape[0] * spec.shape[1] * datum_size(df);
            TT_ASSERT(
                shard_bytes % ts == 0,
                "ROW_MAJOR shard size in bytes ({}) must be a multiple of CB page size ({})",
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

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    const uint32_t src0_cb_index = CBIndex::c_0;
    const uint32_t tmp0_cb_index = CBIndex::c_1;
    if (operation_attributes.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tmp0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    const bool math_approx_mode = false;
    std::map<std::string, std::string> unary_defines = get_block_defines(ops_chain, "0", "0", input.dtype());
    CMAKE_UNIQUE_NAMESPACE::apply_input_dtype_defines(input.dtype(), unary_defines);
    const bool logit_clamp_enabled =
        CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(ops_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    const std::string compute_path = fmt::format(
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/{}",
        get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    DataFormat cb_data_format_for_input =
        (ops_chain[0].type() == unary::UnaryOpType::BITCAST) ? cb_data_format_output : cb_data_format;

    // --- Circular Buffers ---
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_page_size * src_num_tiles_per_shard.value_or(2),
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format_for_input,
            .page_size = input_cb_page_size,
        }}},
        .buffer = src_sharded ? src_buffer : nullptr,
    });

    if (CMAKE_UNIQUE_NAMESPACE::needs_tmp0_cb(ops_chain[0].type())) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = input_cb_page_size * 2,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tmp0_cb_index),
                .data_format = cb_data_format,
                .page_size = input_cb_page_size,
            }}},
        });
    }

    const uint32_t output_cb_index = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_page_size * dst_num_tiles_per_shard.value_or(2),
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = output_cb_page_size,
        }}},
        .buffer = dst_sharded ? dst_buffer : nullptr,
    });

    // --- Reader Kernel ---
    std::map<std::string, std::string> reader_defines;
    reader_defines["SRC_SHARDED"] = src_sharded ? "1" : "0";
    reader_defines["RM_INTERLEAVED"] = rm_interleaved ? "1" : "0";

    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = reader_common_runtime_args;

    // --- Writer Kernel ---
    std::map<std::string, std::string> writer_defines;
    writer_defines["DST_SHARDED"] = dst_sharded ? "1" : "0";
    writer_defines["RM_INTERLEAVED"] = rm_interleaved ? "1" : "0";

    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> writer_common_runtime_args;
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = writer_common_runtime_args;

    // --- Compute Kernel ---
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    if (ops_chain[0].type() == UnaryOpType::HARDSWISH) {
        compute_desc.compile_time_args = {
            static_cast<uint32_t>(unary_defines.contains("INP_FLOAT32")),
            static_cast<uint32_t>(unary_defines.contains("INP_INT32") || unary_defines.contains("INP_UINT32")),
        };
    } else if (ops_chain[0].type() == UnaryOpType::LOGIT) {
        compute_desc.compile_time_args = {static_cast<uint32_t>(logit_clamp_enabled)};
    }
    compute_desc.compile_time_args.push_back(static_cast<uint32_t>(cb_data_format));
    compute_desc.defines = {unary_defines.begin(), unary_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
        .bfp8_pack_precise = operation_attributes.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    // --- Per-core runtime args ---
    // Work split + per-core values come from enumerate_core_rt_args, shared with
    // override_runtime_arguments so the cache-hit patch and the miss path cannot disagree.
    // Sharded readers/writers take {buffer, units, start_id}; interleaved add the five chunk fields.
    constexpr uint32_t kShardedDataMovementArgs = 3, kInterleavedDataMovementArgs = 8, kComputeArgs = 3;
    enumerate_core_rt_args(
        operation_attributes, tensor_args, output, [&](const CoreRtArgs& w, const RmChunkConstants& kc) {
            if (w.noop) {
                const uint32_t n = has_sharding ? kShardedDataMovementArgs : kInterleavedDataMovementArgs;
                reader_desc.runtime_args.emplace_back(w.core, KernelDescriptor::CoreRuntimeArgs(n, 0));
                writer_desc.runtime_args.emplace_back(w.core, KernelDescriptor::CoreRuntimeArgs(n, 0));
                compute_desc.runtime_args.emplace_back(w.core, KernelDescriptor::CoreRuntimeArgs(kComputeArgs, 0));
                return;
            }
            if (has_sharding) {
                reader_desc.emplace_runtime_args(w.core, {input.buffer(), w.in_units, w.start_id});
                writer_desc.emplace_runtime_args(w.core, {output.buffer(), w.out_units, w.start_id});
            } else if (rm_interleaved) {
                reader_desc.emplace_runtime_args(
                    w.core,
                    {input.buffer(),
                     w.in_units,
                     w.start_id,
                     kc.chunks_per_row,
                     kc.input_chunk_size,
                     kc.input_last_chunk_size,
                     kc.rows_per_tile,
                     kc.total_rows});
                writer_desc.emplace_runtime_args(
                    w.core,
                    {output.buffer(),
                     w.out_units,
                     w.start_id,
                     kc.chunks_per_row,
                     kc.output_chunk_size,
                     kc.output_last_chunk_size,
                     kc.rows_per_tile,
                     kc.total_rows});
            } else {
                reader_desc.emplace_runtime_args(w.core, {input.buffer(), w.in_units, w.start_id, 0u, 0u, 0u, 0u, 0u});
                writer_desc.emplace_runtime_args(
                    w.core, {output.buffer(), w.out_units, w.start_id, 0u, 0u, 0u, 0u, 0u});
            }
            compute_desc.runtime_args.emplace_back(
                w.core, KernelDescriptor::CoreRuntimeArgs{w.compute_units, packed_scalar1, packed_scalar2});
        });

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void UnaryDeviceOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt::tt_metal;
    // The TILE-layout hash omits shape, so the work split, start ids and accessor shape args all vary
    // between hits: re-apply exactly those, through the same enumeration create_descriptor uses.
    const auto& input = tensor_args.input;
    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool src_sharded = shard_specs.has_value() && input.is_sharded();
    const bool dst_sharded = shard_specs.has_value() && output.is_sharded();

    constexpr uint32_t kReaderKernelIdx = 0, kWriterKernelIdx = 1, kComputeKernelIdx = 2;
    const uint32_t src_addr = input.buffer()->address();
    const uint32_t dst_addr = output.buffer()->address();
    const bool has_sharding = shard_specs.has_value();
    const bool rm_interleaved = input.layout() == Layout::ROW_MAJOR && !has_sharding;

    // A changed split can flip a core between noop and active, so write every slot create_descriptor
    // writes rather than only the ones that usually move -- otherwise a flipped core keeps stale args.
    uint32_t packed_scalar1 = 0, packed_scalar2 = 0;
    CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(
        operation_attributes.op_chain[0], input.dtype(), packed_scalar1, packed_scalar2);

    enumerate_core_rt_args(
        operation_attributes, tensor_args, output, [&](const CoreRtArgs& w, const RmChunkConstants& kc) {
            auto& r = GetRuntimeArgs(program, kReaderKernelIdx, w.core);
            auto& wr = GetRuntimeArgs(program, kWriterKernelIdx, w.core);
            auto& c = GetRuntimeArgs(program, kComputeKernelIdx, w.core);
            if (w.noop) {
                for (uint32_t i = 0; i < r.size(); ++i) {
                    r[i] = 0;
                }
                for (uint32_t i = 0; i < wr.size(); ++i) {
                    wr[i] = 0;
                }
                for (uint32_t i = 0; i < c.size(); ++i) {
                    c[i] = 0;
                }
                return;
            }
            r[0] = src_addr;
            r[1] = w.in_units;
            r[2] = w.start_id;
            wr[0] = dst_addr;
            wr[1] = w.out_units;
            wr[2] = w.start_id;
            if (!has_sharding) {
                const std::array<uint32_t, 5> rtail{
                    kc.chunks_per_row, kc.input_chunk_size, kc.input_last_chunk_size, kc.rows_per_tile, kc.total_rows};
                const std::array<uint32_t, 5> wtail{
                    kc.chunks_per_row,
                    kc.output_chunk_size,
                    kc.output_last_chunk_size,
                    kc.rows_per_tile,
                    kc.total_rows};
                for (uint32_t i = 0; i < rtail.size(); ++i) {
                    r[3 + i] = rm_interleaved ? rtail[i] : 0u;
                    wr[3 + i] = rm_interleaved ? wtail[i] : 0u;
                }
            }
            c[0] = w.compute_units;
            c[1] = packed_scalar1;
            c[2] = packed_scalar2;
        });

    // The accessor's common args carry the tensor shape (ArgConfig::RuntimeTensorShape), which moves
    // with the (unhashed) shape; rebuild just those two small vectors, not the descriptor.
    std::vector<uint32_t> ct_args, common_args;
    TensorAccessorArgs(*input.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape).append_to(ct_args, common_args);
    auto& reader_common = GetCommonRuntimeArgs(program, kReaderKernelIdx);
    for (uint32_t i = 0; i < common_args.size() && i < reader_common.size(); ++i) {
        reader_common[i] = common_args[i];
    }
    ct_args.clear();
    common_args.clear();
    TensorAccessorArgs(*output.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(ct_args, common_args);
    auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelIdx);
    for (uint32_t i = 0; i < common_args.size() && i < writer_common.size(); ++i) {
        writer_common[i] = common_args[i];
    }

    // Sharded CBs are tensor-backed. CBs are matched positionally by apply_descriptor_runtime_args, so
    // mirror create_descriptor's order (src0, optional tmp0, output); a null buffer entry is skipped.
    if (src_sharded || dst_sharded) {
        ProgramDescriptor cb_addr_only;
        cb_addr_only.cbs.push_back(CBDescriptor{.buffer = src_sharded ? input.buffer() : nullptr});
        if (CMAKE_UNIQUE_NAMESPACE::needs_tmp0_cb(operation_attributes.op_chain[0].type())) {
            cb_addr_only.cbs.push_back(CBDescriptor{});
        }
        cb_addr_only.cbs.push_back(CBDescriptor{.buffer = dst_sharded ? output.buffer() : nullptr});
        apply_descriptor_runtime_args(program, cb_addr_only);  // override-rebuild-ok: cb-addr-only
    }
}

}  // namespace ttnn::operations::unary
