// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "permute_codegen_device_operation.hpp"
#include "permute_codegen_program_factory.hpp"
#include "permute_codegen_supported.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace {

// get_row_strides(input_shape) — same formula as the device_operation.cpp helper, but operating on
// the attrs' fixed-width (zero-padded) array + explicit rank instead of a ttnn::Shape.
std::array<uint32_t, PermuteCodegenDeviceOperation::kMaxDims> row_strides_of(
    const std::array<uint32_t, PermuteCodegenDeviceOperation::kMaxDims>& shape, uint32_t rank) {
    std::array<uint32_t, PermuteCodegenDeviceOperation::kMaxDims> strides{};
    strides[rank - 1] = 1;
    strides[rank - 2] = 1;
    for (int i = static_cast<int>(rank) - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

}  // namespace

// Row-invariant RM permute (W unchanged): reader_stick_interleaved_unified.cpp (MODE_SEQUENCED,
// SEQ_IDENTITY) feeding a NON-sequential writer_permute_rm_interleaved.cpp that scatters each row to
// its inverse-permuted output page. No compute.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::MultiCoreRowInvariant::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    const uint32_t rank = operation_attributes.rank;
    const uint32_t elem_size = operation_attributes.elem_size;
    const uint32_t stick_bytes = operation_attributes.input_shape[rank - 1] * elem_size;
    const uint32_t num_rows = operation_attributes.num_rows;

    constexpr uint32_t kModeSequencedStick = 4;  // reader_stick_interleaved_unified.cpp MODE_SEQUENCED
    constexpr uint32_t kSeqIdentity = 0;         // sequencers.h SEQ_IDENTITY
    constexpr uint32_t kReadBatch = permute_codegen::kRmReadBatch;
    constexpr uint32_t kWriteBatch = permute_codegen::kRmWriteBatch;
    constexpr uint32_t cb_depth = permute_codegen::kRmCbSlots;

    TensorAccessorArgs src_accessor_args(*input_tensor.buffer());
    TensorAccessorArgs dst_accessor_args(*output_tensor.buffer());
    const auto src_ct = src_accessor_args.get_compile_time_args();
    const auto dst_ct = dst_accessor_args.get_compile_time_args();
    TT_FATAL(src_ct.size() == 2, "permute row-invariant reader expects an interleaved TensorAccessorArgs ABI");
    TT_FATAL(dst_ct.size() == 2, "permute row-invariant writer expects an interleaved TensorAccessorArgs ABI");
    const uint32_t source_page_size = src_ct[1];
    const uint32_t destination_page_size = dst_ct[1];

    // A slot is a whole stick, so the CB footprint scales with the input's last dim. The gate
    // rejects a stick too wide for cb_depth of them to fit in one core's L1 -- the same helper, so
    // that what is admitted here is exactly what was admitted there.
    const auto cb_budget = permute_codegen::rm_cb_budget(input_tensor, output_tensor.memory_config());
    const uint32_t cb_page_size = cb_budget.slot_stride;
    TT_FATAL(
        cb_page_size == std::max(source_page_size, destination_page_size),
        "PermuteCodegen: CB slot stride {} B disagrees with the accessors' aligned page sizes "
        "(source {} B, destination {} B)",
        cb_page_size,
        source_page_size,
        destination_page_size);
    TT_FATAL(
        cb_budget.max_slots >= cb_depth,
        "PermuteCodegen: a {} B row-major stick needs {} CB slots' worth of per-core L1 but only {} "
        "fit; supported_by_codegen() must reject this config",
        stick_bytes,
        cb_depth,
        cb_budget.max_slots);

    IDevice* device = input_tensor.device();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(device->compute_with_storage_grid_size(), num_rows);

    CBFormatDescriptor cb_fmt{
        .buffer_index = 0,
        .data_format = datatype_to_dataformat_converter(input_tensor.dtype()),
        .page_size = cb_page_size};
    CBDescriptor cb_desc{
        .total_size = cb_depth * cb_page_size, .core_ranges = all_cores, .format_descriptors = {cb_fmt}};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_stick_interleaved_unified.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = src_ct;
    reader_desc.named_compile_time_args = {
        {"mode", kModeSequencedStick},
        {"cb_id", 0},
        {"stick_bytes", stick_bytes},
        {"aligned_page_size", source_page_size},
        {"seq_id", kSeqIdentity},
        {"batch", kReadBatch},
        {"nabatch", 1},
        // Benign MODE_TILEROW_PAD defaults: kernel_main is not a template, so its `if constexpr`
        // branch is name-resolved for every mode that includes this shared reader.
        {"elem_size", elem_size},
        {"tile_height", 32},
        {"tile_row_shift_bits", 0},
        {"num_pages_in_row", 1},
        {"unpadded_X_bytes", 0},
        {"valid_last_page_bytes", 0},
        {"page_size", 0},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_interleaved.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {0, stick_bytes};
    writer_desc.compile_time_args.insert(writer_desc.compile_time_args.end(), dst_ct.begin(), dst_ct.end());
    writer_desc.compile_time_args.push_back(kWriteBatch);
    writer_desc.compile_time_args.push_back(rank);
    writer_desc.config = WriterConfigDescriptor{};

    // The output address is the only writer arg a later dispatch can change, and a per-core buffer
    // binding is re-patched once per core on every cache hit. Bind it once for the program instead.
    writer_desc.emplace_common_runtime_args({output_tensor.buffer()});

    const auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    uint32_t row_start = 0;
    for (const auto& core : cores) {
        uint32_t n;
        if (core_group_1.contains(core)) {
            n = rows_per_core_1;
        } else if (core_group_2.contains(core)) {
            n = rows_per_core_2;
        } else {
            TT_THROW("PermuteCodegen: core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(core, {input_tensor.buffer(), n, row_start});

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.reserve(2 + 3 * rank);
        writer_rt.push_back(n);
        writer_rt.push_back(row_start);
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.input_shape[i]);
        }
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.dims[i]);
        }
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.output_strides[i]);
        }
        writer_desc.emplace_runtime_args(core, writer_rt);

        row_start += n;
    }

    return ProgramDescriptor{.kernels = {reader_desc, writer_desc}, .cbs = {cb_desc}};
}

// Single-pass blocked-generic RM permute for the W-CHANGING class (dims[-1] != rank-1): reader reads
// 32x32 row-major blocks -> compute (tilize -> transpose_tile -> pack_untilize) -> writer scatters
// transposed rows to permuted pages.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::MultiCoreBlockedGeneric::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    // Fixed block edge: every CB below is sized from it, so this factory's L1 footprint does not
    // scale with any tensor dimension.
    constexpr uint32_t kBlockSize = 32;

    const uint32_t rank = operation_attributes.rank;
    const uint32_t elem_size = operation_attributes.elem_size;
    const uint32_t x_dim = operation_attributes.dims[rank - 1];
    const uint32_t X = operation_attributes.input_shape[x_dim];
    const uint32_t W = operation_attributes.input_shape[rank - 1];
    const uint32_t num_rows = operation_attributes.num_rows;
    const uint32_t x_blocks = (X + kBlockSize - 1) / kBlockSize;
    const uint32_t w_blocks = (W + kBlockSize - 1) / kBlockSize;
    const uint32_t num_blocks_total = operation_attributes.num_blocks_total;

    const uint32_t input_cb_page_size = kBlockSize * elem_size;       // cb_0 page (one W-chunk row)
    const uint32_t output_cb_page_size = kBlockSize * elem_size;      // cb_2 page (one X row)
    const uint32_t tile_bytes = kBlockSize * kBlockSize * elem_size;  // cb_1 page (one 32x32 tile)

    const uint32_t in_row_bytes = W * elem_size;
    const uint32_t out_row_bytes = X * elem_size;  // == output_shape[-1] * elem_size

    TensorAccessorArgs src_accessor_args(*input_tensor.buffer());
    TensorAccessorArgs dst_accessor_args(*output_tensor.buffer());
    const auto src_ct = src_accessor_args.get_compile_time_args();
    const auto dst_ct = dst_accessor_args.get_compile_time_args();
    TT_FATAL(src_ct.size() == 2, "permute blocked-generic reader expects an interleaved TensorAccessorArgs ABI");
    TT_FATAL(dst_ct.size() == 2, "permute blocked-generic writer expects an interleaved TensorAccessorArgs ABI");
    const uint32_t in_page_size = src_ct[1];
    const uint32_t out_page_size = dst_ct[1];
    TT_FATAL(
        in_page_size >= in_row_bytes && out_page_size >= out_row_bytes,
        "permute TensorAccessor page is smaller than its logical row: input {}<{}, output {}<{}",
        in_page_size,
        in_row_bytes,
        out_page_size,
        out_row_bytes);

    IDevice* device = input_tensor.device();
    auto [num_cores, all_cores, core_group_1, core_group_2, blocks_per_core_1, blocks_per_core_2] =
        split_work_to_cores(device->compute_with_storage_grid_size(), num_blocks_total);

    // 4-byte (int32/float32) datums need 32-bit DEST accumulation through the
    // tilize -> transpose_tile -> pack_untilize compute; bf16 (2-byte) does not.
    const bool fp32_dest_acc = (elem_size == 4);
    // transpose_tile rounds float32 through TF32 (and its UInt32 format is corrupt on Wormhole),
    // but the compute only moves 32-bit datums, so
    // float32/uint32 are carried through the CB reinterpreted as int32 to preserve every raw bit.
    // Reader/writer byte math and the tensor's own dtype are unchanged; only the CB DataFormat flips.
    const DataType input_dtype = input_tensor.dtype();
    const DataType cb_dtype =
        (input_dtype == DataType::FLOAT32 || input_dtype == DataType::UINT32) ? DataType::INT32 : input_dtype;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(cb_dtype);

    CBFormatDescriptor cb0_fmt{.buffer_index = 0, .data_format = cb_data_format, .page_size = input_cb_page_size};
    CBDescriptor cb0_desc{
        .total_size = 2 * kBlockSize * input_cb_page_size, .core_ranges = all_cores, .format_descriptors = {cb0_fmt}};

    CBFormatDescriptor cb1_fmt{.buffer_index = 1, .data_format = cb_data_format, .page_size = tile_bytes};
    CBDescriptor cb1_desc{.total_size = 2 * tile_bytes, .core_ranges = all_cores, .format_descriptors = {cb1_fmt}};

    CBFormatDescriptor cb2_fmt{.buffer_index = 2, .data_format = cb_data_format, .page_size = output_cb_page_size};
    CBDescriptor cb2_desc{
        .total_size = 2 * kBlockSize * output_cb_page_size, .core_ranges = all_cores, .format_descriptors = {cb2_fmt}};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/reader_permute_rm_blocked.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = src_ct;
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

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/writer_permute_rm_blocked.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = dst_ct;
    writer_desc.named_compile_time_args = {
        {"N", rank},
        {"output_page_size", output_cb_page_size},
        {"num_rows", num_rows},
        {"X", X},
        {"x_dim", x_dim},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", kBlockSize},
        {"w_block_size", kBlockSize},
        {"W", W},
        {"element_size", elem_size},
        {"out_page_size", out_page_size},
    };
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/codegen/kernels/compute_permute_xw_rm.cpp";
    compute_desc.core_ranges = all_cores;
    compute_desc.named_compile_time_args = {
        {"x_block_size", kBlockSize},
        {"w_block_size", kBlockSize},
    };
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc};

    const auto input_strides = row_strides_of(operation_attributes.input_shape, rank);

    const auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);
    compute_desc.runtime_args.reserve(num_cores);

    uint32_t block_start = 0;
    for (const auto& core : cores) {
        uint32_t n_blocks;
        if (core_group_1.contains(core)) {
            n_blocks = blocks_per_core_1;
        } else if (core_group_2.contains(core)) {
            n_blocks = blocks_per_core_2;
        } else {
            TT_THROW("PermuteCodegen: core not in specified core ranges");
        }
        const uint32_t block_end = block_start + n_blocks;

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(3 + 2 * rank);
        reader_rt.push_back(input_tensor.buffer());
        reader_rt.push_back(block_start);
        reader_rt.push_back(block_end);
        for (uint32_t i = 0; i < rank; i++) {
            reader_rt.push_back(operation_attributes.input_shape[i]);
        }
        for (uint32_t i = 0; i < rank; i++) {
            reader_rt.push_back(input_strides[i]);
        }
        reader_desc.emplace_runtime_args(core, reader_rt);

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.reserve(3 + 3 * rank);
        writer_rt.push_back(output_tensor.buffer());
        writer_rt.push_back(block_start);
        writer_rt.push_back(block_end);
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.input_shape[i]);
        }
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.dims[i]);
        }
        for (uint32_t i = 0; i < rank; i++) {
            writer_rt.push_back(operation_attributes.output_strides[i]);
        }
        writer_desc.emplace_runtime_args(core, writer_rt);

        compute_desc.emplace_runtime_args(core, {n_blocks, 0, 0});

        block_start = block_end;
    }

    return ProgramDescriptor{
        .kernels = {reader_desc, writer_desc, compute_desc}, .cbs = {cb0_desc, cb1_desc, cb2_desc}};
}

}  // namespace ttnn::operations::data_movement
