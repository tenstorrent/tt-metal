// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName RM_SC_READER{"reader"};
const KernelSpecName RM_SC_WRITER{"writer"};
const DFBSpecName RM_SC_IN0{"in0"};
const TensorParamName RM_SC_INPUT{"input"};
const TensorParamName RM_SC_OUTPUT{"output"};
const TensorParamName RM_SC_PAD_VALUE{"pad_value_const"};

// Allocate the on-device pad-value const tensor.  Pulled out so
// create_program_artifacts() can build it once on cache miss and hand its
// MeshTensor to the framework as an op-owned tensor, deferring the device
// deallocation until the cached Program is evicted (see #44565).
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

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    auto output_shape = operation_attributes.output_padded_shape;

    uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // Build the pad-value const tensor once on cache miss and release its owning MeshTensor into
    // the artifact.  The framework parks it in the cache entry, so its address stays valid for
    // every dispatch that hits this cached Program.
    std::vector<tt::tt_metal::MeshTensor> op_owned;
    op_owned.reserve(1);
    Tensor pad_value_const_tensor = build_pad_value_const_tensor_sc(tensor_args, pad_value);
    op_owned.push_back(pad_value_const_tensor.device_storage().release_mesh_tensor());
    const auto& pad_value_mesh_tensor = op_owned.back();

    const NodeCoord node{0, 0};

    uint32_t dfb_npages = 16;  // multibuffering
    uint32_t dfb_pagesize =
        tt::round_up(padded_row_size_nbytes, std::max(a.buffer()->alignment(), tt::constants::TILE_WIDTH));
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    DataflowBufferSpec in0_dfb{
        .unique_id = RM_SC_IN0,
        .entry_size = dfb_pagesize,
        .num_entries = dfb_npages,
        .data_format_metadata = in_df,
    };

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    KernelSpec reader{
        .unique_id = RM_SC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "reader_pad_dims_rm_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_SC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_SC_INPUT,
                    .accessor_name = "src",
                },
                TensorBinding{
                    .tensor_parameter_name = RM_SC_PAD_VALUE,
                    .accessor_name = "pad_value",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_unpadded_W",
                     "num_unpadded_Z",
                     "num_total_Z",
                     "unpadded_X_nbytes",
                     "padded_X_nbytes",
                     "padded_X_diff_nbytes",
                     "pad_value_packed",
                     "start_src_stick_id",
                     "start_src_stick_offset",
                     "num_local_Y",
                     "num_local_unpadded_Y",
                     "num_local_W"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    KernelSpec writer{
        .unique_id = RM_SC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "writer_pad_dims_rm_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_SC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_SC_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_total_Z",
                     "padded_X_nbytes",
                     "start_dst_stick_id",
                     "num_local_Y",
                     "dst_stick_offset",
                     "num_local_W"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;
    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;

    ProgramSpec spec{
        .name = "pad_rm_reader_writer_single_core",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in0_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = RM_SC_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = RM_SC_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = RM_SC_PAD_VALUE, .spec = pad_value_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {RM_SC_READER, RM_SC_WRITER},
                    .target_nodes = node,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = RM_SC_READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"num_unpadded_W", static_cast<uint32_t>(a.padded_shape()[0])},
                 {"num_unpadded_Z", static_cast<uint32_t>(a.padded_shape()[1])},
                 {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                 {"unpadded_X_nbytes", unpadded_row_size_nbytes},
                 {"padded_X_nbytes", padded_row_size_nbytes},
                 {"padded_X_diff_nbytes", padded_row_diff_size_nbytes},
                 {"pad_value_packed", packed_pad_value},
                 {"start_src_stick_id", start_src_stick_id},
                 {"start_src_stick_offset", std::uint32_t{0}},
                 {"num_local_Y", static_cast<uint32_t>(output_shape[2])},
                 {"num_local_unpadded_Y", static_cast<uint32_t>(a.padded_shape()[2])},
                 {"num_local_W", static_cast<uint32_t>(output.padded_shape()[0])}}),
        },
        KernelRunArgs{
            .kernel = RM_SC_WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                 {"padded_X_nbytes", padded_row_size_nbytes},
                 {"start_dst_stick_id", start_dst_stick_id},
                 {"num_local_Y", static_cast<uint32_t>(output_shape[2])},
                 {"dst_stick_offset", std::uint32_t{0}},
                 {"num_local_W", static_cast<uint32_t>(output.padded_shape()[0])}}),
        },
    };
    run_args.tensor_args = {
        {RM_SC_INPUT, TensorArgument{input_mesh_tensor}},
        {RM_SC_OUTPUT, TensorArgument{output_mesh_tensor}},
        {RM_SC_PAD_VALUE, TensorArgument{pad_value_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned),
    };
}

}  // namespace ttnn::prim
