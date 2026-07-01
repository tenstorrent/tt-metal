// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_partial_rope_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::fused_partial_rope {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/device/kernels/dataflow/"
    "reader_fused_partial_rope_sharded.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/device/kernels/compute/"
    "fused_partial_rope.cpp";

}  // namespace

FusedPartialRopeDeviceOperation::program_factory_t FusedPartialRopeDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ShardedProgramFactory{};
}

void FusedPartialRopeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "cos must be on device");
    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "sin must be on device");
    TT_FATAL(trans_mat.storage_type() == StorageType::DEVICE, "trans_mat must be on device");

    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cos.layout() == Layout::TILE, "cos must be TILE layout");
    TT_FATAL(sin.layout() == Layout::TILE, "sin must be TILE layout");
    TT_FATAL(trans_mat.layout() == Layout::TILE, "trans_mat must be TILE layout");

    // X is the only sharded operand; cos / sin / trans_mat are DRAM-interleaved and read
    // per-core by the reader kernel.
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "input must be height-sharded (got {})",
        input.memory_config().memory_layout());
    TT_FATAL(
        cos.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "cos must be DRAM-interleaved (got {})",
        cos.memory_config().memory_layout());
    TT_FATAL(
        sin.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sin must be DRAM-interleaved (got {})",
        sin.memory_config().memory_layout());
    TT_FATAL(
        trans_mat.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "trans_mat must be DRAM-interleaved (got {})",
        trans_mat.memory_config().memory_layout());

    const auto& input_shape = input.padded_shape();
    const auto& cos_shape = cos.padded_shape();
    const auto& sin_shape = sin.padded_shape();
    const auto& trans_mat_shape = trans_mat.padded_shape();

    const uint32_t D = input_shape[-1];
    const uint32_t Rd = args.rope_dim;
    TT_FATAL(Rd > 0 && Rd <= D, "rope_dim ({}) must be in (0, D={}]", Rd, D);
    TT_FATAL(D % TILE_WIDTH == 0, "input head dim ({}) must be tile-aligned", D);
    TT_FATAL(Rd % TILE_WIDTH == 0, "rope_dim ({}) must be tile-aligned", Rd);
    TT_FATAL((D - Rd) % TILE_WIDTH == 0, "nope width (D - rope_dim = {}) must be tile-aligned", D - Rd);

    TT_FATAL(cos_shape == sin_shape, "cos and sin must have the same shape");
    TT_FATAL(cos.dtype() == sin.dtype(), "cos and sin dtype must match");
    TT_FATAL(cos_shape[-1] == Rd, "cos width ({}) must equal rope_dim ({})", cos_shape[-1], Rd);

    TT_FATAL(
        trans_mat_shape[-2] == TILE_HEIGHT && trans_mat_shape[-1] == TILE_WIDTH,
        "trans_mat must be a single [{}, {}] tile (got {})",
        TILE_HEIGHT,
        TILE_WIDTH,
        trans_mat_shape);

    // One tile-row (32 rows) per shard core. cos/sin either provide one tile-row per core
    // (per-row tables) or a single tile-row that the kernel broadcasts across every input row
    // (e.g. decode: one position shared across all heads).
    const uint32_t num_cores = input.memory_config().shard_spec().value().grid.num_cores();
    const uint32_t cos_rows_t = cos_shape[-2] / TILE_HEIGHT;
    TT_FATAL(
        cos_rows_t == num_cores || cos.logical_shape()[-2] == 1,
        "cos/sin tile-rows ({}) must equal the input shard core count ({}) or be a single row (broadcast)",
        cos_rows_t,
        num_cores);
}

FusedPartialRopeDeviceOperation::spec_return_value_t FusedPartialRopeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    // Output mirrors the input's height-sharded spec (full [.., D] shape).
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()),
            args.output_mem_config));
}

FusedPartialRopeDeviceOperation::tensor_return_value_t FusedPartialRopeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::tt_metal::ProgramDescriptor FusedPartialRopeDeviceOperation::ShardedProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;

    const tt::DataFormat input_df = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_df);
    const tt::DataFormat cos_df = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_tile_size = tt::tile_size(cos_df);
    const tt::DataFormat sin_df = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_tile_size = tt::tile_size(sin_df);
    const tt::DataFormat trans_mat_df = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_tile_size = tt::tile_size(trans_mat_df);
    const tt::DataFormat output_df = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_tile_size = tt::tile_size(output_df);

    const uint32_t D = input.padded_shape()[-1];
    const uint32_t Rd = args.rope_dim;
    const uint32_t Dt = D / TILE_WIDTH;              // full head width in tiles
    const uint32_t rope_Wt = Rd / TILE_WIDTH;        // trailing rope tiles
    const uint32_t nope_Wt = (D - Rd) / TILE_WIDTH;  // leading pass-through tiles

    // A single logical cos/sin row => broadcast that row across every input row on device
    // (e.g. one decode position shared across all heads). A multi-row table that merely fits
    // in one tile still needs its distinct per-row values, so key off the logical row count.
    const bool cos_bcast = cos.logical_shape()[-2] == 1;

    auto* device = input.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    // Kernels run on exactly the input's shard grid (one tile-row per core).
    const CoreRangeSet all_cores = input.memory_config().shard_spec().value().grid;

    tt::tt_metal::ProgramDescriptor desc;

    // Buffer-backed (globally-allocated) CBs for the resident shards.
    constexpr uint8_t in_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Dt * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb_index, .data_format = input_df, .page_size = input_tile_size}}},
        .buffer = input.buffer(),
    });
    // cos / sin are DRAM-interleaved, streamed into these CBs by the reader (not buffer-backed).
    constexpr uint8_t cos_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rope_Wt * cos_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index, .data_format = cos_df, .page_size = cos_tile_size}}},
    });
    constexpr uint8_t sin_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rope_Wt * sin_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index, .data_format = sin_df, .page_size = sin_tile_size}}},
    });
    constexpr uint8_t out_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Dt * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index, .data_format = output_df, .page_size = output_tile_size}}},
        .buffer = output.buffer(),
    });

    // Non-resident CBs: trans_mat (read from DRAM by the reader) + rotate/mul intermediates.
    constexpr uint8_t trans_mat_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * trans_mat_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index, .data_format = trans_mat_df, .page_size = trans_mat_tile_size}}},
    });
    constexpr uint8_t rotated_interm_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rope_Wt * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_interm_cb_index, .data_format = input_df, .page_size = input_tile_size}}},
    });
    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rope_Wt * cos_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index, .data_format = cos_df, .page_size = cos_tile_size}}},
    });
    constexpr uint8_t sin_interm_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rope_Wt * sin_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index, .data_format = sin_df, .page_size = sin_tile_size}}},
    });

    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();

    // Reader: streams this core's rope tile-row of cos/sin plus the (replicated) trans_mat tile,
    // all from DRAM-interleaved sources. X and the output stay resident in their L1 shards.
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)rope_Wt,
    };
    TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*trans_mat_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs compute_kernel_args = {
        (uint32_t)in_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)rotated_interm_cb_index,
        (uint32_t)cos_interm_cb_index,
        (uint32_t)sin_interm_cb_index,
        (uint32_t)out_cb_index,
        (uint32_t)Dt,
        (uint32_t)rope_Wt,
        (uint32_t)nope_Wt,
        (uint32_t)cos_bcast,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // Per-core runtime args: cos/sin/trans_mat buffer addresses (Buffer* bindings, patched on
    // cache hits) plus this core's cos/sin start tile. Core i owns input tile-row i, so it reads
    // cos/sin tiles [i*rope_Wt, (i+1)*rope_Wt) from the DRAM-interleaved tables.
    const auto& cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    reader_desc.runtime_args.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const uint32_t cos_sin_start_tile = cos_bcast ? 0 : i * rope_Wt;
        reader_desc.emplace_runtime_args(cores[i], {cos_buffer, sin_buffer, trans_mat_buffer, cos_sin_start_tile});
        compute_desc.runtime_args.emplace_back(cores[i], KernelDescriptor::CoreRuntimeArgs{});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope

namespace ttnn::prim {

ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::transformer::fused_partial_rope::FusedPartialRopeDeviceOperation;

    auto arch = input.storage_type() == StorageType::DEVICE ? input.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig out_mem_config = input.memory_config();
    if (memory_config.has_value()) {
        out_mem_config = memory_config.value();
    }

    auto attrs = OperationType::operation_attributes_t{
        .rope_dim = rope_dim,
        .output_mem_config = out_mem_config,
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input, .cos = cos, .sin = sin, .trans_mat = trans_mat};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
