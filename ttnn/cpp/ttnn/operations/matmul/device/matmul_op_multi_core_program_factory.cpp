// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn {

namespace operations {

namespace matmul {

tt::tt_metal::ProgramDescriptor matmul_multi_core(const Tensor& a, const Tensor& b, Tensor& output, bool bcast_batch) {
    tt_metal::ProgramDescriptor program;
    constexpr auto max_num_kernels = 4;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    const auto &ashape = a.get_padded_shape(), bshape = b.get_padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();
    const auto& cshape = output.get_padded_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = num_input_tiles * in0_single_tile_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
        }},
    });

    uint32_t src1_cb_index = 1;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = num_input_tiles * in1_single_tile_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
        }},
    });

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
        }},
    });

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;

    auto& reader_kernel = program.kernels[num_kernels++];
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp";
    reader_kernel.core_ranges = all_cores.ranges();
    reader_kernel.compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};
    reader_kernel.config = tt_metal::ReaderConfigDescriptor{};
    reader_kernel.reserve_runtime_args();

    auto& writer_kernel = program.kernels[num_kernels++];
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_kernel.core_ranges = all_cores.ranges();
    writer_kernel.compile_time_args = {(uint32_t)output_cb_index, (uint32_t)dst_is_dram};
    writer_kernel.config = tt_metal::WriterConfigDescriptor{};
    writer_kernel.reserve_runtime_args();

    auto& bmm_kernel = program.kernels[num_kernels++];
    bmm_kernel.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
    bmm_kernel.core_ranges = core_group_1.ranges();
    bmm_kernel.compile_time_args = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only
        // set Nt for simplicity
    bmm_kernel.config = tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .dst_full_sync_en = true,
    };

    if (!core_group_2.ranges().empty()) {
        auto& bmm_kernel_2 = program.kernels[num_kernels++];
        bmm_kernel_2.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
        bmm_kernel_2.core_ranges = core_group_2.ranges();
        bmm_kernel_2.compile_time_args = {
            1,
            1,
            Kt,
            num_output_tiles_per_core_group_2};  // bmm compute kernel the B, Mt, Nt are just 3 for loops that
                                                 // technically act as 1 large loop, so only set Nt for simplicity
        bmm_kernel_2.config = tt_metal::ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .dst_full_sync_en = true,
        };
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        reader_kernel.runtime_args[core.x][core.y] = {
            src0_addr,
            src1_addr,
            Mt,
            Kt,
            Nt,
            MtKt,
            KtNt,
            B,
            uint32_t(bcast_batch),
            num_tiles_written,
            num_output_tiles_per_core,
            MtNt};
        writer_kernel.runtime_args[core.x][core.y] = {dst_addr, num_output_tiles_per_core, num_tiles_written};
        num_tiles_written += num_output_tiles_per_core;
    }

    program.kernels.resize(num_kernels);
    return program;
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
