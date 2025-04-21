// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"

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

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::BroadcastWidthMultiCore::create(
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
    const auto bshape = b->padded_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC * Ht * Wt;
    uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;

    uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    tt::tt_metal::ProgramDescriptor program;

    tt_metal::IDevice* device = a.device();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b->get_dtype());
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    constexpr bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, Wt, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto src0_buffer = a.buffer();
    auto src1_buffer = b->buffer();
    auto dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }},
    });

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }},
    });

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = {all_device_cores},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }},
    });

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    constexpr size_t num_kernels = 3;
    program.kernels.resize(num_kernels);

    auto& binary_reader_kernel = program.kernels[0];
    binary_reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
        "reader_bcast_w_interleaved_input_cols_partitioned.cpp";
    binary_reader_kernel.core_ranges = {all_device_cores};
    binary_reader_kernel.compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};
    binary_reader_kernel.config = tt_metal::ReaderConfigDescriptor{};
    binary_reader_kernel.reserve_runtime_args();

    auto& unary_writer_kernel = program.kernels[1];
    unary_writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
        "writer_unary_interleaved_input_cols_batched.cpp";
    unary_writer_kernel.core_ranges = {all_device_cores};
    unary_writer_kernel.compile_time_args = {(uint32_t)dst_is_dram};
    unary_writer_kernel.config = tt_metal::WriterConfigDescriptor{};
    unary_writer_kernel.reserve_runtime_args();

    auto& bcast_kernel = program.kernels[2];
    bcast_kernel.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_w.cpp";
    bcast_kernel.core_ranges = {all_device_cores};
    bcast_kernel.defines = bcast_op_utils::get_defines_vec(BcastOpDim::W, bcast_math);
    bcast_kernel.config = tt_metal::ComputeConfigDescriptor{};
    bcast_kernel.reserve_runtime_args();

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);

        auto& binary_reader_args = binary_reader_kernel.runtime_args[core.x][core.y];
        auto& bcast_args = bcast_kernel.runtime_args[core.x][core.y];
        auto& unary_writer_args = unary_writer_kernel.runtime_args[core.x][core.y];

        uint32_t Wt_per_core;
        if (core_group_1.contains(core)) {
            Wt_per_core = Wt_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            Wt_per_core = Wt_per_core_group_2;
        } else {
            binary_reader_args = {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  // 16
            };
            bcast_args = {
                0, 0, 0  // 3
            };
            unary_writer_args = {
                0, 0, 0, 0, 0, 0, 0, 0, 0  // 9
            };
            continue;
        }
        uint32_t num_tensor_tiles_per_core = NC * Ht * Wt_per_core;
        uint32_t Wt_skip = Wt - Wt_per_core;

        binary_reader_args = {
            a.buffer()->address(),      // 0
            0,                          // 1
            0,                          // 2
            num_tensor_tiles_per_core,  // 3
            b->buffer()->address(),     // 4
            0,                          // 5
            0,                          // 6
            num_btensor_tiles,          // 7
            num_tensor_tiles_per_core,  // 8
            NC,                         // 9
            Ht,                         // 10
            Wt_per_core,                // 11
            bnc1,                       // 12
            num_Wtiles_read,            // 13
            Ht * Wt,                    // 14
            Wt_skip,                    // 15
        };

        bcast_args = {
            NC,          // B
            Ht,          // Ht
            Wt_per_core  // Wt
        };

        unary_writer_args = {
            output.buffer()->address(),
            0,
            0,
            Ht,
            Wt_per_core,
            num_Wtiles_read,
            Wt_skip,
            NC,
            Ht * Wt,
        };

        num_Wtiles_read += Wt_per_core;
    }

    return program;
}

}  // namespace ttnn::operations::binary
