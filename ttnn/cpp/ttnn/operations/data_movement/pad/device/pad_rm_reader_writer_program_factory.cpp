// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

// Allocate the on-device pad-value const tensor.  Returned as an op-owned tensor in
// ProgramArtifacts::op_owned_tensors: the framework parks it at a stable address for the cached
// Program's life (allocated once on a cache miss, reused on a hit).  The kernel reads it through a
// TensorAccessor (ta::pad) exactly like the io tensors.
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

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;

    auto output_shape = operation_attributes.output_padded_shape;

    uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    Buffer* src0_buffer = a.buffer();
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};

    uint32_t cb_npages = 16;  // multibuffering
    uint32_t cb_pagesize =
        tt::round_up(padded_row_size_nbytes, std::max(src0_buffer->alignment(), tt::constants::TILE_WIDTH));
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;
    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;

    // ---- Op-owned pad-value const tensor (allocate first; bind against the vector element) ----
    std::vector<Tensor> op_owned;
    op_owned.reserve(1);
    op_owned.push_back(build_pad_value_const_tensor_sc(tensor_args, pad_value));
    const Tensor& pad_const = op_owned[0];

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_rm_reader_writer";

    // c_0 (in0): row buffer the reader fills (data + pad) and the writer drains to the output.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = cb_pagesize,
            .num_entries = cb_npages,
            .data_format_metadata = in_df},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"pad"}, .spec = pad_const.tensor_spec()},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_dims_rm_interleaved_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"pad"}, .accessor_name = "pad"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "num_total_W",
                  "num_unpadded_Z",
                  "num_total_Z",
                  "num_unpadded_Y",
                  "num_total_Y",
                  "unpadded_X_nbytes",
                  "padded_X_nbytes",
                  "padded_X_diff_nbytes",
                  "pad_value_packed",
                  "start_src_stick_id",
                  "start_src_stick_wi",
                  "start_src_stick_offset",
                  "num_local_Y",
                  "num_local_unpadded_Y",
                  "full_unpadded_X_nbytes",
                  "num_local_W"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_dims_rm_interleaved_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_total_W",
                  "num_total_Z",
                  "num_total_Y",
                  "num_total_X",
                  "padded_X_nbytes",
                  "start_dst_stick_id",
                  "start_dst_stick_wi",
                  "num_local_Y",
                  "num_local_unpadded_Y",
                  "full_padded_X_nbytes",
                  "dst_stick_offset",
                  "num_local_W"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};

    // Local DFB (in0) requires its producer (reader) and consumer (writer) to share the same
    // WorkUnitSpec.  Both run on the single core {0,0}.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "single_core",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = core_ranges},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    reader_run.runtime_arg_values.push_back(
        {m2::NodeCoord{CoreCoord{0, 0}},
         {{"num_unpadded_W", static_cast<uint32_t>(a.padded_shape()[0])},
          {"num_total_W", static_cast<uint32_t>(output_shape[0])},
          {"num_unpadded_Z", static_cast<uint32_t>(a.padded_shape()[1])},
          {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
          {"num_unpadded_Y", static_cast<uint32_t>(a.padded_shape()[2])},
          {"num_total_Y", static_cast<uint32_t>(output_shape[2])},
          {"unpadded_X_nbytes", unpadded_row_size_nbytes},
          {"padded_X_nbytes", padded_row_size_nbytes},
          {"padded_X_diff_nbytes", padded_row_diff_size_nbytes},
          {"pad_value_packed", packed_pad_value},
          {"start_src_stick_id", start_src_stick_id},
          {"start_src_stick_wi", 0u},
          {"start_src_stick_offset", 0u},
          {"num_local_Y", static_cast<uint32_t>(output_shape[2])},
          {"num_local_unpadded_Y", static_cast<uint32_t>(a.padded_shape()[2])},
          {"full_unpadded_X_nbytes", unpadded_row_size_nbytes},
          {"num_local_W", static_cast<uint32_t>(output.padded_shape()[0])}}});

    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};
    writer_run.runtime_arg_values.push_back(
        {m2::NodeCoord{CoreCoord{0, 0}},
         {{"num_total_W", static_cast<uint32_t>(output_shape[0])},
          {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
          {"num_total_Y", static_cast<uint32_t>(output_shape[2])},
          {"num_total_X", static_cast<uint32_t>(output_shape[3])},
          {"padded_X_nbytes", padded_row_size_nbytes},
          {"start_dst_stick_id", start_dst_stick_id},
          {"start_dst_stick_wi", 0u},
          {"num_local_Y", static_cast<uint32_t>(output_shape[2])},
          {"num_local_unpadded_Y", static_cast<uint32_t>(a.padded_shape()[2])},
          {"full_padded_X_nbytes", padded_row_size_nbytes},
          {"dst_stick_offset", 0u},
          {"num_local_W", static_cast<uint32_t>(output.padded_shape()[0])}}});

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, a.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
        {m2::TensorParamName{"pad"}, pad_const.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run), .op_owned_tensors = std::move(op_owned)};
}

}  // namespace ttnn::prim
