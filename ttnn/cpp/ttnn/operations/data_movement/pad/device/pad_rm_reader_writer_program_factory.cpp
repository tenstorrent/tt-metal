// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

// Allocate the on-device pad-value const tensor.  Pulled out so
// create_workload_descriptor() can build it once on cache miss and park it on
// WorkloadDescriptor::buffers, deferring ~Tensor (which force-deallocates the
// device memory) until the cached workload is evicted (see #44565).
Tensor build_pad_value_const_tensor_sc(const PadInputs& tensor_args, float pad_value) {
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    auto pad_value_const_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    return Tensor(
               std::move(pad_value_const_buffer),
               ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
               DataType::BFLOAT16,
               Layout::ROW_MAJOR)
        .to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
}

ProgramDescriptor build_pad_rm_sc_program_descriptor(
    const PadParams& operation_attributes,
    const PadInputs& tensor_args,
    Tensor& output,
    Buffer* pad_value_const_buffer) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;

    auto output_shape = operation_attributes.output_padded_shape;

    uint32_t unpadded_row_size_nbytes = tensor_args.input.padded_shape()[3] * tensor_args.input.element_size();
    uint32_t padded_row_size_nbytes =
        output_shape[3] * tensor_args.input.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * tensor_args.input.element_size();

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    CoreRange cores({0, 0}, {0, 0});

    ProgramDescriptor desc;

    uint32_t cb_id = tt::CBIndex::c_0;
    uint32_t cb_npages = 16;  // multibuffering
    uint32_t cb_pagesize =
        tt::round_up(padded_row_size_nbytes, std::max(src0_buffer->alignment(), tt::constants::TILE_WIDTH));
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_npages * cb_pagesize,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = in_df,
            .page_size = cb_pagesize,
        }}},
    });

    std::vector<uint32_t> reader_ct_args = {unpadded_row_size_nbytes, padded_row_size_nbytes};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*dst_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*pad_value_const_buffer).append_to(reader_ct_args);
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

#if 0
    {
        log_debug(tt::LogOp, "src0_buffer_addr: {}", src0_buffer->address());
        log_debug(tt::LogOp, "dst_buffer_addr: {}", dst_buffer->address());
        log_debug(tt::LogOp, "a.shape[0]: {}", a.padded_shape()[0]);
        log_debug(tt::LogOp, "out.shape[0]: {}", output_shape[0]);
        log_debug(tt::LogOp, "a.shape[1]: {}", a.padded_shape()[1]);
        log_debug(tt::LogOp, "out.shape[1]: {}", output_shape[1]);
        log_debug(tt::LogOp, "a.shape[2]: {}", a.padded_shape()[2]);
        log_debug(tt::LogOp, "out.shape[2]: {}", output_shape[2]);
        log_debug(tt::LogOp, "s.shape[3]: {}", a.padded_shape()[3]);
        log_debug(tt::LogOp, "out.shape[3]: {}", output_shape[3]);
        log_debug(tt::LogOp, "unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_size_nbytes: {}", padded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug(tt::LogOp, "pad_value_const_buffer_addr: {}", pad_value_const_buffer->address());
        log_debug(tt::LogOp, "pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug(tt::LogOp, "packed_pad_value: {}", packed_pad_value);
    }
#endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    // Slots 0 (src), 1 (dst) and 13 (pad-value const) are raw buffer base addresses;
    // bind them as Buffer* so the framework patches the addresses on cache hits
    // without rebuilding the descriptor.  The const tensor is owned by
    // WorkloadDescriptor::buffers in create_workload_descriptor() so its address
    // remains valid across cache hits.
    KernelDescriptor::RTArgList reader_rt_args;
    reader_rt_args.reserve(27);
    reader_rt_args.push_back(src0_buffer);
    reader_rt_args.push_back(dst_buffer);
    reader_rt_args.push_back(static_cast<uint32_t>(a.padded_shape()[0]));
    reader_rt_args.push_back(static_cast<uint32_t>(output_shape[0]));
    reader_rt_args.push_back(static_cast<uint32_t>(a.padded_shape()[1]));
    reader_rt_args.push_back(static_cast<uint32_t>(output_shape[1]));
    reader_rt_args.push_back(static_cast<uint32_t>(a.padded_shape()[2]));
    reader_rt_args.push_back(static_cast<uint32_t>(output_shape[2]));
    reader_rt_args.push_back(static_cast<uint32_t>(a.padded_shape()[3]));
    reader_rt_args.push_back(static_cast<uint32_t>(output_shape[3]));
    reader_rt_args.push_back(unpadded_row_size_nbytes);
    reader_rt_args.push_back(padded_row_size_nbytes);
    reader_rt_args.push_back(padded_row_diff_size_nbytes);
    reader_rt_args.push_back(pad_value_const_buffer);
    reader_rt_args.push_back(pad_value_const_buffer_nbytes);
    reader_rt_args.push_back(packed_pad_value);
    reader_rt_args.push_back(start_src_stick_id);
    reader_rt_args.push_back(start_dst_stick_id);
    reader_rt_args.push_back(std::uint32_t{0});
    reader_rt_args.push_back(std::uint32_t{0});
    reader_rt_args.push_back(std::uint32_t{0});
    reader_rt_args.push_back(static_cast<uint32_t>(output_shape[2]));
    reader_rt_args.push_back(static_cast<uint32_t>(a.padded_shape()[2]));
    reader_rt_args.push_back(unpadded_row_size_nbytes);
    reader_rt_args.push_back(padded_row_size_nbytes);
    reader_rt_args.push_back(std::uint32_t{0});
    reader_rt_args.push_back(static_cast<uint32_t>(output.padded_shape()[0]));

    // Writer reads the same arg layout (slot 1 = dst addr, slot 13 = pad-value const).
    KernelDescriptor::RTArgList writer_rt_args = reader_rt_args;

    reader_desc.emplace_runtime_args(CoreCoord{0, 0}, reader_rt_args);
    writer_desc.emplace_runtime_args(CoreCoord{0, 0}, writer_rt_args);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace

WorkloadDescriptor PadRmReaderWriterProgramFactory::create_workload_descriptor(
    const PadParams& operation_attributes,
    const PadInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;

    // Build the pad-value const tensor once on cache miss and park it on the
    // WorkloadDescriptor.  Holding the SOURCE Tensor (not just a
    // shared_ptr<MeshBuffer>) is required because ~Tensor force-deallocates the
    // device memory through DeviceStorage::deallocate regardless of external
    // shared_ptr<MeshBuffer> owners (see #44565).
    Tensor pad_value_const_tensor = build_pad_value_const_tensor_sc(tensor_args, operation_attributes.pad_value);
    auto pad_value_owner = std::make_shared<Tensor>(std::move(pad_value_const_tensor));
    Buffer* pad_value_const_buffer = pad_value_owner->buffer();
    wd.buffers.push_back({pad_value_owner, pad_value_const_buffer});

    auto desc = build_pad_rm_sc_program_descriptor(
        operation_attributes, tensor_args, tensor_return_value, pad_value_const_buffer);

    auto ranges = tensor_coords.ranges();
    for (size_t i = 0; i + 1 < ranges.size(); ++i) {
        wd.programs.push_back({ranges[i], desc});
    }
    if (!ranges.empty()) {
        wd.programs.push_back({ranges.back(), std::move(desc)});
    }
    return wd;
}

}  // namespace ttnn::prim
