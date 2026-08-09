// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_stats/device/attn_res_stats_program_factory.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kAttnResStatsKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/reduction/attn_res_stats/device/kernels/";

// The row of `v` is read once and reduced twice, so the reader can only run one
// row ahead of compute.
constexpr uint32_t kRowsInFlight = 2;

// Sum of squares and dot, in the order the compute kernel packs them.
constexpr uint32_t kStatsPerRow = 2;

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResStatsProgramFactory::create_descriptor(
    const AttnResStatsParams& operation_attributes,
    const AttnResStatsInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.v.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto v_data_format = datatype_to_dataformat_converter(tensor_args.v.dtype());
    const auto v_tile_size = tt::tile_size(v_data_format);
    const auto q_data_format = datatype_to_dataformat_converter(tensor_args.q.dtype());
    const auto q_tile_size = tt::tile_size(q_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    // The transformed row is the reduce's input, so its format is what the
    // reduce unpacks; keeping it at v's format leaves the accumulation, which
    // runs in dest, as the only place precision is decided.
    const auto scratch_data_format = v_data_format;
    const auto scratch_tile_size = v_tile_size;

    // The reduce scaler must be unpackable alongside a Float32 operand.
    const auto scaler_data_format =
        v_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto scaler_tile_size = tt::tile_size(scaler_data_format);

    const auto& v_shape = tensor_args.v.padded_shape();
    const uint32_t Wt = v_shape[-1] / TILE_WIDTH;
    const uint32_t num_rows = tensor_args.v.physical_volume() / (v_shape[-1] * TILE_HEIGHT);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // Both reductions read the same resident row, so `v`, `q` and one transformed
    // copy are all live at once. That, not the tensor size, is what bounds d.
    //
    // Circular buffers can only occupy L1 above the allocator's base, so the total per
    // core overstates the budget — measuring against it accepts a d that then fails
    // during descriptor allocation instead of here.
    const uint32_t l1_available =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_required = kRowsInFlight * Wt * v_tile_size + Wt * q_tile_size + Wt * scratch_tile_size +
                                 scaler_tile_size + kStatsPerRow * output_tile_size;
    TT_FATAL(
        l1_required <= l1_available,
        "AttnResStats holds a whole row of v, q and one transformed copy in L1: d of {} needs {} B per core against {} "
        "available",
        v_shape[-1],
        l1_required,
        l1_available);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    auto [num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_rows_group_1, num_rows_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_rows, /*row_wise=*/true);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    desc.cbs.push_back(CBDescriptor{
        .total_size = kRowsInFlight * Wt * v_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = v_data_format,
            .page_size = v_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = scaler_data_format,
            .page_size = scaler_tile_size,
        }}},
    });

    // q is loop-invariant: read once per core, never popped.
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * q_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = q_data_format,
            .page_size = q_tile_size,
        }}},
    });

    // Holds v*v for one reduce and v*q for the next; the reduce drains it in
    // between, so one row's worth serves both.
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * scratch_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_6),
            .data_format = scratch_data_format,
            .page_size = scratch_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = kStatsPerRow * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = output_data_format,
            .page_size = output_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {Wt};
    TensorAccessorArgs(*tensor_args.v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.q.buffer()).append_to(reader_compile_time_args);

    // Row r holds its sum of squares at output page r and its dot a whole
    // candidate axis further on, which is the row count.
    std::vector<uint32_t> writer_compile_time_args = {num_rows};
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = std::string(kAttnResStatsKernelDir) + "reader_attn_res_stats.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = std::string(kAttnResStatsKernelDir) + "writer_attn_res_stats.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = std::string(kAttnResStatsKernelDir) + "attn_res_stats.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = {Wt};
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const v_buffer = tensor_args.v.buffer();
    auto* const q_buffer = tensor_args.q.buffer();
    auto* const output_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0, start_row = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core{i % num_cores_x, i / num_cores_x};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_kernel_desc.emplace_runtime_args(core, {v_buffer, q_buffer, num_rows_per_core, start_row});
        writer_kernel_desc.emplace_runtime_args(core, {output_buffer, num_rows_per_core, start_row});
        compute_kernel_desc.emplace_runtime_args(core, {num_rows_per_core});

        start_row += num_rows_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
