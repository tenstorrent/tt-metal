// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "binary_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::binary {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
static BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::create(
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
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    bool bnc1 = (bN * bC == 1);

    tt_metal::ProgramDescriptor program;

    tt_metal::IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat src1_cb_data_format =
        b.has_value() ? tt_metal::datatype_to_dataformat_converter(b->get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

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

    auto* cb_src0_buffer = src0_sharded ? src0_buffer : nullptr;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb0 * src0_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }},
        .buffer = cb_src0_buffer,
    });

    uint32_t num_input_tiles_cb1 = src1_buffer != nullptr ? num_input_tiles : 1;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb1 * src1_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }},
    });

    uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;
    auto* cb_output_buffer = output_sharded ? dst_buffer : nullptr;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }},
        .buffer = cb_output_buffer,
    });

    auto src0_is_dram = static_cast<uint32_t>(src0_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    auto dst_is_dram = static_cast<uint32_t>(dst_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    KernelDescriptor::Defines bcast_compute_defines = bcast_op_utils::get_defines_vec(BcastOpDim::HW, bcast_math);
    if (bnc1) {
        bcast_compute_defines.emplace_back("BCAST_SCALAR", "1");
    }

    constexpr size_t num_kernels = 3;
    program.kernels.resize(num_kernels);

    auto& binary_reader_kernel = program.kernels[0];
    if (src1_buffer != nullptr) {
        auto src1_is_dram = static_cast<uint32_t>(src1_buffer->buffer_type() == tt_metal::BufferType::DRAM);
        binary_reader_kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_hw_interleaved_partitioned.cpp";
        binary_reader_kernel.compile_time_args = {src0_is_dram, src1_is_dram};
    } else {
        binary_reader_kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_scalar_interleaved_partitioned.cpp";
        binary_reader_kernel.compile_time_args = {src0_is_dram};
    }
    binary_reader_kernel.core_ranges = {all_device_cores};
    binary_reader_kernel.config = tt_metal::ReaderConfigDescriptor{};
    if (bnc1) {
        binary_reader_kernel.defines.emplace_back("BCAST_SCALAR", "1");
    }
    if (src0_sharded) {
        binary_reader_kernel.defines.emplace_back("IN0_SHARDED", "1");
    }
    binary_reader_kernel.reserve_runtime_args();

    auto& unary_writer_kernel = program.kernels[1];
    unary_writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";
    unary_writer_kernel.core_ranges = {all_device_cores};
    unary_writer_kernel.compile_time_args = {tt::CBIndex::c_2, dst_is_dram};
    unary_writer_kernel.config = tt_metal::WriterConfigDescriptor{};
    if (output_sharded) {
        unary_writer_kernel.defines.emplace_back("OUT_SHARDED", "1");
    }
    unary_writer_kernel.reserve_runtime_args();

    auto& bcast_kernel = program.kernels[2];
    bcast_kernel.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp";
    bcast_kernel.core_ranges = {all_device_cores};
    bcast_kernel.config = tt_metal::ComputeConfigDescriptor{};
    bcast_kernel.defines = std::move(bcast_compute_defines);
    bcast_kernel.reserve_runtime_args();
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);

        auto& binary_reader_args = binary_reader_kernel.runtime_args[core.x][core.y];
        auto& bcast_args = bcast_kernel.runtime_args[core.x][core.y];
        auto& unary_writer_args = unary_writer_kernel.runtime_args[core.x][core.y];

        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            binary_reader_args = {0, 0, 0, 0, 0, 0, 0};  // 7
            bcast_args = {1, 1, 0};                      // 3
            unary_writer_args = {0, 0, 0};               // 3
            continue;
        }

        binary_reader_args = {
            src0_buffer->address(),  // 0
            0u,
            num_tensor_tiles_per_core,
            HtWt,
            num_tiles_read / HtWt * HtWt,
            num_tiles_read % HtWt,
            bnc1 ? 0 : num_tiles_read / HtWt};

        if (src1_buffer != nullptr) {
            binary_reader_args[1] = src1_buffer->address();
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            binary_reader_args[1] = packed_scalar;
        }

        bcast_args = {
            1,                         // B
            1,                         // Ht
            num_tensor_tiles_per_core  // Wt
        };

        unary_writer_args = {
            dst_buffer->address(),
            num_tensor_tiles_per_core,
            num_tiles_read,
        };

        num_tiles_read += num_tensor_tiles_per_core;
    }

    return program;
}

}  // namespace ttnn::operations::binary
