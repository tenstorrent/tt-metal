// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_default_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
static const uint32_t max_read_size = 2048;  // max read size in bytes for reader and writer kernels

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

uint32_t get_num_stick_per_barrier(uint32_t stick_size_padded_aligned) {
    return std::max(tt::div_up(max_read_size, stick_size_padded_aligned), 1u);
}

}  // namespace

ProgramDescriptor PadRmReaderWriterMultiCoreDefaultProgramFactory::create_descriptor(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];
    uint32_t NCH_padded = H_padded * C_padded * N_padded;

    const auto& front_pad = input_tensor_start;

    auto stick_size = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_padded_front = front_pad[-1] * a.element_size();
    auto stick_size_padded_end = stick_size_padded - stick_size - stick_size_padded_front;
    uint32_t stick_size_padded_aligned = tt::align(stick_size_padded, hal::get_l1_alignment());
    uint32_t stick_size_padded_DRAM_aligned = tt::align(stick_size_padded, hal::get_dram_alignment());
    uint32_t row_major_min_bytes = 16;

    // Input page-based addressing
    uint32_t num_input_pages_in_row = 1;
    uint32_t input_accessor_page_size = stick_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_input_pages_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
        // Use the buffer's per-page size (shard width for W/B/ND sharded; full stick for height sharded).
        input_accessor_page_size = a.buffer()->aligned_page_size();
    }

    // Output page-based addressing
    uint32_t num_output_pages_in_row = 1;
    uint32_t output_accessor_page_size = stick_size_padded;
    if (output.is_sharded()) {
        uint32_t output_shard_width = output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                                                      : output.nd_shard_spec().value().shard_shape[-1];
        num_output_pages_in_row = tt::div_up(W_padded, output_shard_width);
        output_accessor_page_size = output.buffer()->aligned_page_size();
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_sticks_padded_per_core_group_1,
         num_sticks_padded_per_core_group_2] =
            sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), NCH_padded)
                                       : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);

    auto cores_in_order = corerange_to_cores(all_cores, num_cores, true);

    uint32_t src0_cb_index = tt::CBIndex::c_0;

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    ProgramDescriptor desc;

    // c_1 reused for pad-value scratch on every dispatch.
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = stick_size_padded_DRAM_aligned,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = cb_data_format,
            .page_size = stick_size_padded_DRAM_aligned,
        }}},
    });

    bool unaligned = stick_size_padded_aligned % hal::get_dram_alignment() != 0;
    if (stick_size_padded_front != 0 || unaligned) {
        uint32_t src2_cb_index = tt::CBIndex::c_2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = stick_size_padded_DRAM_aligned,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src2_cb_index),
                .data_format = cb_data_format,
                .page_size = stick_size_padded_DRAM_aligned,
            }}},
        });
    }

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)N + front_pad[-4],
        (std::uint32_t)H + front_pad[-2],
        (std::uint32_t)C + front_pad[-3],
        (std::uint32_t)stick_size,
        (std::uint32_t)N_padded,
        (std::uint32_t)H_padded,
        (std::uint32_t)C_padded,
        (std::uint32_t)stick_size_padded,
        (std::uint32_t)stick_size_padded_front,
        (std::uint32_t)stick_size_padded_end,
        (std::uint32_t)tt::div_up(stick_size_padded, 512),  // max zero size is 512B
        (std::uint32_t)(stick_size_padded % 512 == 0 ? 512 : stick_size_padded % 512),
        (std::uint32_t)not_pad_by_zero,
        (std::uint32_t)packed_pad_value,
        (std::uint32_t)row_major_min_bytes,
        (std::uint32_t)(stick_size_padded_front / row_major_min_bytes),
        (std::uint32_t)(stick_size_padded_end / row_major_min_bytes),
        (std::uint32_t)(stick_size_padded / row_major_min_bytes),
        (std::uint32_t)stick_size_padded_aligned,
        (std::uint32_t)unaligned,
        (std::uint32_t)num_input_pages_in_row,
        (std::uint32_t)input_accessor_page_size};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)stick_size_padded,
        (std::uint32_t)stick_size_padded_aligned,
        (std::uint32_t)num_output_pages_in_row,
        (std::uint32_t)output_accessor_page_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved_v2.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved_v2.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Build per-core runtime args inline (legacy path called get_runtime_args_rm()
    // which produced the same data).  Slot 0 of both reader and writer is a raw
    // buffer base address (no offset) — use Buffer* for BufferBinding so the
    // framework patches addresses on cache hits.  Idle cores (num_sticks_per_core
    // == 0) pass 0u to skip the binding since the kernel does nothing.
    // The legacy helper used input_tensor.padded_shape() for the H/C/N bounds —
    // mirror that here.  H_padded/C_padded already use the output padded shape.
    auto input_padded_shape = a.padded_shape();
    uint32_t H_in = input_padded_shape[2], C_in = input_padded_shape[1], N_in = input_padded_shape[0];
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_padded_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);

    uint32_t num_sticks_per_barrier = get_num_stick_per_barrier(stick_size_padded_aligned);

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    for (const auto& core : cores_in_order) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        KernelDescriptor::RTArgList reader_rt_args;
        KernelDescriptor::RTArgList writer_rt_args;
        if (num_sticks_per_core != 0) {
            reader_rt_args.push_back(src0_buffer);
            writer_rt_args.push_back(dst_buffer);
        } else {
            reader_rt_args.push_back(0u);
            writer_rt_args.push_back(0u);
        }
        reader_rt_args.push_back(num_sticks_per_core);
        reader_rt_args.push_back(num_sticks_per_barrier);
        reader_rt_args.push_back(curr_sticks_read * num_input_pages_in_row);
        reader_rt_args.push_back(static_cast<uint32_t>(front_pad[-4]));
        reader_rt_args.push_back(static_cast<uint32_t>(front_pad[-3]));
        reader_rt_args.push_back(static_cast<uint32_t>(front_pad[-2]));
        for (uint32_t v : start_dim_offset) {
            reader_rt_args.push_back(v);
        }

        writer_rt_args.push_back(num_sticks_per_core);
        writer_rt_args.push_back(num_sticks_per_barrier);
        writer_rt_args.push_back(curr_sticks_write * num_output_pages_in_row);

        reader_desc.emplace_runtime_args(core, reader_rt_args);
        writer_desc.emplace_runtime_args(core, writer_rt_args);

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t k = 0; k < num_sticks_per_core; ++k) {
            if ((curr_h >= front_pad[-2] and curr_h < (H_in + front_pad[-2])) and
                (curr_c >= front_pad[-3] and curr_c < (C_in + front_pad[-3])) and
                (curr_n >= front_pad[-4] and curr_n < (N_in + front_pad[-4]))) {
                curr_sticks_read++;
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};
    }

    uint32_t cb_npages = get_num_stick_per_barrier(stick_size_padded_aligned);
    const uint32_t buffer_reader_writer_async_factor = 16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_reader_writer_async_factor * cb_npages * stick_size_padded_aligned,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = stick_size_padded_aligned,
        }}},
    });

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
