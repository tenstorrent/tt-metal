// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "binary_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::binary {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    auto bcast_math = binary_op_type_to_bcast_op_math(operation_attributes.binary_op_type);
    const auto ashape = a.padded_shape();
    const auto bshape = b.has_value() ? b->padded_shape() : ttnn::Shape({1, 1});
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t NC = N * C;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    bool bnc1 = (bN * bC == 1);

    tt_metal::IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat src1_cb_data_format =
        b.has_value() ? tt_metal::datatype_to_dataformat_converter(b->dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool row_major = shard_spec.has_value() ? shard_spec->orientation == ShardOrientation::ROW_MAJOR : false;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto* src0_buffer = a.buffer();
    auto* src1_buffer = b.has_value() ? b->buffer() : nullptr;
    auto* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec->shape[0] * shard_spec->shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec->grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;
    uint32_t num_input_tiles_cb1 = src1_buffer != nullptr ? num_input_tiles : 1;
    uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb0 * src0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
        .buffer = src0_sharded ? src0_buffer : nullptr,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb1 * src1_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .buffer = output_sharded ? dst_buffer : nullptr,
    });

    // ---- Kernel compile-time args and defines ----

    std::map<std::string, std::string> reader_defines;
    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    std::map<std::string, std::string> bcast_compute_defines = bcast_op_utils::get_defines(BcastOpDim::HW, bcast_math);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }

    std::string reader_kernel_path;
    if (src1_buffer != nullptr) {
        TT_FATAL(src1_buffer->buffer_layout() == TensorMemoryLayout::INTERLEAVED, "src1_buffer must be interleaved");
        TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_hw_interleaved_partitioned.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_scalar_interleaved_partitioned.cpp";
    }

    std::map<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // ---- Reader kernel ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {bcast_compute_defines.begin(), bcast_compute_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    // ---- Per-core runtime args ----

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            reader_desc.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs(7, 0));
            compute_desc.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs{1, 1, 0});
            writer_desc.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs(3, 0));
            continue;
        }

        uint32_t arg1;
        if (src1_buffer != nullptr) {
            arg1 = src1_buffer->address();
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            arg1 = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src0_buffer->address(),             // 0
                arg1,                               // 1
                num_tensor_tiles_per_core,          // 2
                HtWt,                               // 3
                num_tiles_read / HtWt * HtWt,      // 4
                num_tiles_read % HtWt,              // 5
                bnc1 ? 0u : num_tiles_read / HtWt,  // 6
            });

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                1,                         // B
                1,                         // Ht
                num_tensor_tiles_per_core  // Wt
            });

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                dst_buffer->address(),
                num_tensor_tiles_per_core,
                num_tiles_read,
            });

        num_tiles_read += num_tensor_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::binary
