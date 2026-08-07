// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_w_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

tt::tt_metal::ProgramDescriptor BcastMultiCoreWProgramFactory::create_descriptor(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t bH = bshape[-2];
    const uint32_t bW = bshape[-1];
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    const uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;

    const uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    IDevice* device = a.device();

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    constexpr bool row_major = false;
    const auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, Wt, row_major);
    (void)num_cores;
    (void)all_cores;

    const auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;

    const uint32_t src1_cb_index = 1;
    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = 2;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
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
    });

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    const std::map<std::string, std::string> bcast_defines =
        bcast_op_utils::get_defines(BcastOpDim::W, operation_attributes.math_op);

    static constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "reader_bcast_w_interleaved_input_cols_partitioned.cpp";
    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "writer_unary_interleaved_input_cols_batched.cpp";
    static constexpr const char* BCAST_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_w.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = BCAST_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {bcast_defines.begin(), bcast_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t Wt_per_core;
        if (core_group_1.contains(core)) {
            Wt_per_core = Wt_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            Wt_per_core = Wt_per_core_group_2;
        } else {
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(16, 0));
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(3, 0));
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(9, 0));
            continue;
        }
        const uint32_t num_tensor_tiles_per_core = NC * Ht * Wt_per_core;
        const uint32_t Wt_skip = Wt - Wt_per_core;

        reader_desc.emplace_runtime_args(
            core,
            {
                src0_buffer,                // 0
                0u,                         // 1
                0u,                         // 2
                num_tensor_tiles_per_core,  // 3
                src1_buffer,                // 4
                0u,                         // 5
                0u,                         // 6
                num_btensor_tiles,          // 7
                num_tensor_tiles_per_core,  // 8
                NC,                         // 9
                Ht,                         // 10
                Wt_per_core,                // 11
                bnc1,                       // 12
                num_Wtiles_read,            // 13
                Ht * Wt,                    // 14
                Wt_skip,                    // 15
            });

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                NC,          // B
                Ht,          // Ht
                Wt_per_core  // Wt
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer,
                0u,
                0u,
                Ht,
                Wt_per_core,
                num_Wtiles_read,
                Wt_skip,
                NC,
                Ht * Wt,
            });
        num_Wtiles_read += Wt_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
