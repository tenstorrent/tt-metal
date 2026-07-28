// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

#include <algorithm>
#include <filesystem>
#include <vector>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Allocate and fill the op-owned pad-value const tensor.  Mirrors the legacy
// build_pad_value_const_tensor_sc(): build a host tensor holding the pad value, then write it to
// an L1 interleaved device allocation.  enqueue_write_tensor() returns a sole-owner MeshTensor,
// which is what ProgramArtifacts::op_owned_tensors requires (the framework keeps it alive at a
// stable address for the cached Program; see #44565 for why sole ownership matters).
MeshTensor build_pad_value_const_mesh_tensor(const PadInputs& tensor_args, float pad_value) {
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    auto host_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    Tensor host_pad(
        std::move(host_buffer),
        ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR);
    auto& cq = device->mesh_command_queue();
    // NOTE: The const buffer is always in L1 (mirrors the legacy factory).
    const MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
    return tt::tt_metal::enqueue_write_tensor(cq, host_pad.host_tensor(), *device, mem_cfg);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: input stream (reader produces, writer consumes)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const TensorParamName PAD_TENSOR{"pad"};  // op-owned pad-value const tensor

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "reader_pad_dims_rm_interleaved_sc.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "writer_pad_dims_rm_interleaved_sc.cpp";

    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_shape = operation_attributes.output_padded_shape;

    const uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    const uint32_t padded_row_size_nbytes =
        output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");
    const uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

    // ------------------------------------------------------------------------
    // Op-owned pad-value const tensor: allocate + fill, park on the artifact,
    // then bind it like an io tensor (TensorParameter + TensorArgument).  Build
    // op_owned_tensors first so the TensorArgument references the parked element
    // (the adapter matches by pointer identity; a vector move keeps the address).
    // ------------------------------------------------------------------------
    std::vector<MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(build_pad_value_const_mesh_tensor(tensor_args, pad_value));
    const MeshTensor& pad_const = op_owned_tensors[0];

    // ------------------------------------------------------------------------
    // DataflowBuffer (legacy CB c_0): one input-row stream, multibuffered.
    // ------------------------------------------------------------------------
    const uint32_t cb_npages = 16;  // multibuffering
    const uint32_t cb_pagesize =
        tt::round_up(padded_row_size_nbytes, std::max<uint32_t>(a.buffer()->alignment(), tt::constants::TILE_WIDTH));
    const tt::DataFormat in_df = datatype_to_dataformat_converter(a.dtype());
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = cb_pagesize,
        .num_entries = cb_npages,
        .data_format_metadata = in_df,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case-1 page access).  src + pad on the reader, dst on the writer.
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};
    TensorParameter pad_param{.unique_id = PAD_TENSOR, .spec = pad_const.tensor_spec()};

    // ------------------------------------------------------------------------
    // Pad value packed exactly as the legacy factory did (preserve dtype handling).
    // ------------------------------------------------------------------------
    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    // Shape/size scalars (named RTAs), values lifted verbatim from the legacy RTA layout.
    const uint32_t num_unpadded_W = a.padded_shape()[0];
    const uint32_t num_total_W = output_shape[0];
    const uint32_t num_unpadded_Z = a.padded_shape()[1];
    const uint32_t num_total_Z = output_shape[1];
    const uint32_t num_unpadded_Y = a.padded_shape()[2];
    const uint32_t num_total_Y = output_shape[2];
    const uint32_t num_total_X = output_shape[3];
    const uint32_t num_local_Y = output_shape[2];
    const uint32_t num_local_unpadded_Y = a.padded_shape()[2];
    const uint32_t num_local_W = output.padded_shape()[0];

    // ------------------------------------------------------------------------
    // Reader: streams unpadded input rows into cb_in0, filling padding from the
    // op-owned pad-value const tensor (tensor::pad).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"},
             TensorBinding{.tensor_parameter_name = PAD_TENSOR, .accessor_name = "pad"}},
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
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // ------------------------------------------------------------------------
    // Writer: pulls padded rows from cb_in0 and writes them out page-by-page (Case-1).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
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
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // ------------------------------------------------------------------------
    // Single core (0,0): one work unit, one instance per kernel.
    // ------------------------------------------------------------------------
    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};
    const NodeCoord node{0, 0};

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
    AddRuntimeArgsForNode(
        reader_rtas,
        node,
        {
            {"num_unpadded_W", num_unpadded_W},
            {"num_total_W", num_total_W},
            {"num_unpadded_Z", num_unpadded_Z},
            {"num_total_Z", num_total_Z},
            {"num_unpadded_Y", num_unpadded_Y},
            {"num_total_Y", num_total_Y},
            {"unpadded_X_nbytes", unpadded_row_size_nbytes},
            {"padded_X_nbytes", padded_row_size_nbytes},
            {"padded_X_diff_nbytes", padded_row_diff_size_nbytes},
            {"pad_value_packed", packed_pad_value},
            {"start_src_stick_id", uint32_t{0}},
            {"start_src_stick_wi", uint32_t{0}},
            {"start_src_stick_offset", uint32_t{0}},
            {"num_local_Y", num_local_Y},
            {"num_local_unpadded_Y", num_local_unpadded_Y},
            {"full_unpadded_X_nbytes", unpadded_row_size_nbytes},
            {"num_local_W", num_local_W},
        });

    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
    AddRuntimeArgsForNode(
        writer_rtas,
        node,
        {
            {"num_total_W", num_total_W},
            {"num_total_Z", num_total_Z},
            {"num_total_Y", num_total_Y},
            {"num_total_X", num_total_X},
            {"padded_X_nbytes", padded_row_size_nbytes},
            {"start_dst_stick_id", uint32_t{0}},
            {"start_dst_stick_wi", uint32_t{0}},
            {"num_local_Y", num_local_Y},
            {"num_local_unpadded_Y", num_local_unpadded_Y},
            {"full_padded_X_nbytes", padded_row_size_nbytes},
            {"dst_stick_offset", uint32_t{0}},
            {"num_local_W", num_local_W},
        });

    WorkUnitSpec wu{
        .name = "pad_rm_reader_writer_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = core_ranges,
    };

    ProgramSpec spec{
        .name = "pad_rm_reader_writer_single_core",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec},
        .tensor_parameters = {input_param, output_param, pad_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(a.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}},
        {PAD_TENSOR, TensorArgument{std::cref(pad_const)}}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run_args), .op_owned_tensors = std::move(op_owned_tensors)};
}

}  // namespace ttnn::prim::qsr
