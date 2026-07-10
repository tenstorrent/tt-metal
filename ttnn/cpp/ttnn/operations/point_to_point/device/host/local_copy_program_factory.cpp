// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::point_to_point {

// Build a single-device local copy program: input_tensor -> output_tensor on one
// device, with no fabric. Reuses the op's generic unary reader/writer kernels through
// one circular buffer. Used only for the same-device (send_coord == receive_coord) case.
tt::tt_metal::ProgramDescriptor local_copy_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args, PointToPointOp::tensor_return_value_t& output_tensors) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& output_tensor = output_tensors.at(1);  // final output; intermediate is unused here

    const uint32_t num_pages = data_movement::get_num_pages(input_tensor);
    const uint32_t page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t aligned_page_size_bytes = tt::round_up(page_size_bytes, l1_alignment);
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    // Spread the pages across the device's worker grid so the copy is bounded by DRAM
    // bandwidth, not one core's NoC latency. Same kernels as the fabric path — just many
    // cores, each copying a disjoint contiguous page range.
    auto* mesh_device = dynamic_cast<MeshDevice*>(input_tensor.device());
    const CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_grid, num_pages);

    tt::tt_metal::ProgramDescriptor desc;

    // One CB shared reader -> writer. The reader kernel hardcodes CB index 0; the writer
    // kernel takes its CB index from compile-time arg 0.
    constexpr auto cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_num_pages = 2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * aligned_page_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = aligned_page_size_bytes,
        }}},
    });

    // Reader: input_tensor -> CB.
    std::vector<uint32_t> reader_ct_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // Writer: CB -> output_tensor.
    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(cb_id)};
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_unary_interleaved_start_id_gen.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    const tt::tt_metal::KernelHandle reader_kernel_id = 0;
    const tt::tt_metal::KernelHandle writer_kernel_id = 1;

    // Per core: a disjoint contiguous slice of pages. Buffer addresses go in as Buffer*
    // so the cache-hit fast path re-patches them (declarative op; see #45422). Arg layout
    // matches the unary kernels: {buffer_address, num_pages, start_id, page_size_bytes}.
    uint32_t start_id = 0;
    for (const CoreCoord& c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t core_pages = 0;
        if (core_group_1.contains(c)) {
            core_pages = num_pages_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            core_pages = num_pages_per_core_group_2;
        } else {
            continue;
        }

        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());
        reader_rt_args.push_back(core_pages);
        reader_rt_args.push_back(start_id);
        reader_rt_args.push_back(page_size_bytes);
        desc.kernels[reader_kernel_id].emplace_runtime_args(c, reader_rt_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(output_tensor.buffer());
        writer_rt_args.push_back(core_pages);
        writer_rt_args.push_back(start_id);
        writer_rt_args.push_back(page_size_bytes);
        desc.kernels[writer_kernel_id].emplace_runtime_args(c, writer_rt_args);

        start_id += core_pages;
    }

    return desc;
}

}  // namespace ttnn::operations::point_to_point
