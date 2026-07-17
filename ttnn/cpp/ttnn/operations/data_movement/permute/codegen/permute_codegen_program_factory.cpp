// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;

namespace {
// SEQ_IDENTITY / MODE_SEQUENCED values from
// permute/codegen/kernels/sequencers.h and reader_stick_interleaved_unified.cpp — the reader is
// a shared template selected by these named CT args, not a per-op kernel.
constexpr uint32_t kModeSequenced = 4;
constexpr uint32_t kSeqIdentity = 0;
constexpr uint32_t kRmReadBatch = 4;
constexpr uint32_t kRmWriteBatch = 4;
constexpr uint32_t kBlockSize = 32;  // build_permute_rm_blocked's X_BLOCK == W_BLOCK

const char* kReaderStickSrc =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/reader_stick_interleaved_unified.cpp";
const char* kWriterRmSrc =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_interleaved.cpp";
const char* kReaderBlockedSrc =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/reader_permute_rm_blocked.cpp";
const char* kComputeBlockedSrc =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/compute_permute_xw_rm.cpp";
const char* kWriterBlockedSrc =
    "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_blocked.cpp";
}  // namespace

// Byte-identical port of an internal reference implementation's row-invariant host section:
// stick reader (SEQ_IDENTITY, MODE_SEQUENCED) with no compute + the inverse-permutation RM writer.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::RowInvariant::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    const uint32_t rank = operation_attributes.rank;
    const uint32_t aligned_stick_bytes = operation_attributes.aligned_stick_bytes;

    auto* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_group_1, rows_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, operation_attributes.num_rows);

    ProgramDescriptor desc;

    constexpr uint32_t kCbId = 0;
    const uint32_t cb_depth = std::max(kRmReadBatch, kRmWriteBatch) * 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * aligned_stick_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbId,
            .data_format = datatype_to_dataformat_converter(input_tensor.dtype()),
            .page_size = aligned_stick_bytes,
        }}},
    });

    KernelDescriptor::CompileTimeArgs reader_ct;
    TensorAccessorArgs(*input_buffer).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderStickSrc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.named_compile_time_args = {
        {"mode", kModeSequenced},
        {"cb_id", kCbId},
        {"stick_bytes", aligned_stick_bytes},
        {"aligned_page_size", aligned_stick_bytes},
        {"seq_id", kSeqIdentity},
        {"batch", kRmReadBatch},
        {"nabatch", 1},
        // reader_stick_interleaved_unified.cpp's kernel_main() is not a template, so its
        // MODE_TILEROW_PAD `else if constexpr` block is still fully name-resolved at compile time
        // even though this op only ever selects MODE_SEQUENCED — get_named_compile_time_arg_val()
        // aborts (via __builtin_unreachable in a constant expression) on any name missing from the
        // map, regardless of which branch actually runs. Values below are never read at runtime.
        {"tile_height", 32},
        {"tile_row_shift_bits", 0},
        {"num_pages_in_row", 1},
        {"unpadded_X_bytes", 0},
        {"valid_last_page_bytes", 0},
        {"page_size", 0},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {kCbId, aligned_stick_bytes};
    TensorAccessorArgs(*output_buffer).append_to(writer_ct);
    writer_ct.push_back(kRmWriteBatch);
    writer_ct.push_back(rank);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterRmSrc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    // Writer RT tail (input_shape, perm, dest_strides) is a per-op constant, identical on every
    // core — only [dst, n, start] vary.
    std::vector<uint32_t> writer_tail;
    writer_tail.reserve(3 * rank);
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.input_shape[i]);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.dims[i]);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.output_strides[i]);
    }

    const auto cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y);
    const uint32_t num_cores_group_1 = core_group_1.num_cores();
    uint32_t start_row = 0;
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t rows_this_core = i < num_cores_group_1 ? rows_per_core_group_1 : rows_per_core_group_2;

        reader_desc.emplace_runtime_args(core, {input_buffer, rows_this_core, start_row});

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(output_buffer);
        writer_rt.push_back(rows_this_core);
        writer_rt.push_back(start_row);
        writer_rt.append(writer_tail);
        writer_desc.emplace_runtime_args(core, writer_rt);

        start_row += rows_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// Byte-identical port of an internal reference implementation's W-changing host section: 32x32
// block reader -> tilize/transpose_tile/pack_untilize compute -> permuted-page scatter writer.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::BlockedGeneric::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    const uint32_t rank = operation_attributes.rank;
    const uint32_t elem_size = operation_attributes.elem_size;
    const uint32_t num_rows = operation_attributes.num_rows;
    const uint32_t x_dim = operation_attributes.dims[rank - 1];
    const uint32_t x = operation_attributes.input_shape[x_dim];
    const uint32_t w = operation_attributes.input_shape[rank - 1];
    const uint32_t x_blocks = (x + kBlockSize - 1) / kBlockSize;
    const uint32_t w_blocks = (w + kBlockSize - 1) / kBlockSize;
    const uint32_t num_blocks_total = operation_attributes.num_blocks_total;

    auto* device = input_tensor.device();
    const auto dram_alignment = input_buffer->alignment();

    const uint32_t input_cb_page_size = kBlockSize * elem_size;   // cb_0 page (one W-chunk row)
    const uint32_t output_cb_page_size = kBlockSize * elem_size;  // cb_2 page (one X row)
    const uint32_t tile_bytes = kBlockSize * kBlockSize * elem_size;

    const uint32_t in_row_bytes = w * elem_size;
    const uint32_t out_row_bytes = x * elem_size;  // output_shape[-1] == X
    const uint32_t in_page_size = tt::align(in_row_bytes, dram_alignment);
    const uint32_t out_page_size = tt::align(out_row_bytes, dram_alignment);

    // Bit-exact reinterpret: run the tilize -> transpose_tile -> pack_untilize compute as int32
    // for float32 (elem_size == 4). transpose_tile routes float32 through the matrix engine
    // (tf32 truncation) and is non-exact; the compute only moves 32-bit datums, so running the
    // identical bit pattern through the int32 datapath (bit-exact) preserves every float32 bit.
    // Only the CB DataFormat flips — reader/writer byte math and the tensors themselves are
    // untouched float32.
    const bool is_fp32 = (elem_size == 4) && (input_tensor.dtype() == DataType::FLOAT32);
    const tt::DataFormat cb_data_format =
        is_fp32 ? tt::DataFormat::Int32 : datatype_to_dataformat_converter(input_tensor.dtype());

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, blocks_per_core_group_1, blocks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_blocks_total);

    ProgramDescriptor desc;

    constexpr uint32_t kCbIn = 0;
    constexpr uint32_t kCbTilize = 1;
    constexpr uint32_t kCbOut = 2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * kBlockSize * input_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbIn,
            .data_format = cb_data_format,
            .page_size = input_cb_page_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbTilize,
            .data_format = cb_data_format,
            .page_size = tile_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * kBlockSize * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOut,
            .data_format = cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    KernelDescriptor::CompileTimeArgs reader_ct;
    TensorAccessorArgs(*input_buffer).append_to(reader_ct);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderBlockedSrc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.named_compile_time_args = {
        {"N", rank},
        {"page_size", input_cb_page_size},
        {"num_rows", num_rows},
        {"x_dim", x_dim},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", kBlockSize},
        {"w_block_size", kBlockSize},
        {"element_size", elem_size},
        {"in_page_size", in_page_size},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct;
    TensorAccessorArgs(*output_buffer).append_to(writer_ct);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterBlockedSrc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.named_compile_time_args = {
        {"N", rank},
        {"output_page_size", output_cb_page_size},
        {"num_rows", num_rows},
        {"X", x},
        {"x_dim", x_dim},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", kBlockSize},
        {"w_block_size", kBlockSize},
        {"W", w},
        {"element_size", elem_size},
        {"out_page_size", out_page_size},
    };
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeBlockedSrc;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.named_compile_time_args = {
        {"x_block_size", kBlockSize},
        {"w_block_size", kBlockSize},
    };
    // 4-byte datums (int32, and float32-as-int32) need 32-bit DEST accumulation through the
    // tilize -> transpose_tile -> pack_untilize compute.
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = (elem_size == 4)};

    // Reader/writer RT tails (input_shape / dims / output_strides) are per-op constants,
    // identical on every core — only [addr, start_block, end_block] vary.
    std::vector<uint32_t> reader_tail;
    reader_tail.reserve(2 * rank);
    std::vector<uint32_t> input_strides(PermuteCodegenDeviceOperation::MAX_DIMS, 0);
    if (rank == 1) {
        input_strides[0] = 1;
    } else {
        input_strides[rank - 1] = 1;
        input_strides[rank - 2] = 1;
        for (int32_t i = static_cast<int32_t>(rank) - 3; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * operation_attributes.input_shape[i + 1];
        }
    }
    for (uint32_t i = 0; i < rank; ++i) {
        reader_tail.push_back(operation_attributes.input_shape[i]);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        reader_tail.push_back(input_strides[i]);
    }

    std::vector<uint32_t> writer_tail;
    writer_tail.reserve(3 * rank);
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.input_shape[i]);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.dims[i]);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        writer_tail.push_back(operation_attributes.output_strides[i]);
    }

    const auto cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y);
    const uint32_t num_cores_group_1 = core_group_1.num_cores();
    uint32_t start_block = 0;
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());
    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t blocks_this_core = i < num_cores_group_1 ? blocks_per_core_group_1 : blocks_per_core_group_2;
        const uint32_t end_block = start_block + blocks_this_core;

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.push_back(input_buffer);
        reader_rt.push_back(start_block);
        reader_rt.push_back(end_block);
        reader_rt.append(reader_tail);
        reader_desc.emplace_runtime_args(core, reader_rt);

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(output_buffer);
        writer_rt.push_back(start_block);
        writer_rt.push_back(end_block);
        writer_rt.append(writer_tail);
        writer_desc.emplace_runtime_args(core, writer_rt);

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{blocks_this_core, 0, 0});

        start_block = end_block;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::data_movement
