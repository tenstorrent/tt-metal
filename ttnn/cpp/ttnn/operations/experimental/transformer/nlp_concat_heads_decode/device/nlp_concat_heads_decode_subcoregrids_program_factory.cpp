// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_decode_subcoregrids_program_factory.hpp"
#include "nlp_concat_heads_decode_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <string>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeSubcoregridsProgramFactory::create_program_spec(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    const DFBSpecName OUT_DFB{"q_out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    constexpr const char* KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp";

    const auto& input_tensor = tensor_args.input;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto tile_w = tile_shape[1];
    const auto tile_hw = tile_shape[0] * tile_shape[1];

    const auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    const auto face_h = face_shape[0];
    const auto face_w = face_shape[1];
    const auto face_hw = face_h * face_w;

    const uint32_t head_tiles = head_dim / tile_w;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = face_w * element_size;
    const auto q_shard_spec = output.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / tile_hw;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;

    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    // NoC coordinates of the input shard cores, broadcast as common runtime varargs,
    // laid out [x0..x_{n-1}, y0..y_{n-1}] (n = in_num_cores).
    std::vector<uint32_t> noc_coords;
    noc_coords.reserve(2 * in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).x);
    }
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    };

    auto make_cta = [&](uint32_t phases_to_read) {
        return KernelSpec::CompileTimeArgs{
            {"element_size", element_size},
            {"sub_tile_line_bytes", sub_tile_line_bytes},
            {"head_size", head_size},
            {"batch", batch},
            {"head_size_num_tiles", head_tiles},
            {"phases_to_read", phases_to_read},
            {"in_num_cores", in_num_cores},
            {"face_h", static_cast<uint32_t>(face_h)},
            {"face_hw", static_cast<uint32_t>(face_hw)}};
    };

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args = make_cta(1),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    reader_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args = make_cta(2),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    writer_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.advanced_options.common_runtime_varargs = noc_coords;
    writer_run.advanced_options.common_runtime_varargs = noc_coords;

    for (uint32_t i = 0; i < num_cores; ++i) {
        // in_tile_offset_by_batch is the byte offset of head row i within the input shard. Within
        // a single 32x32 tile (= 2*face_h rows), the first face_h rows live in face 0 and the rest
        // live in face 2 (the existing formula uses (i + face_h) * sub_tile_line_bytes to land at
        // the start of face 2 directly). When padded_heads > 32 we additionally skip past
        // (i / (2*face_h)) head-tiles' worth of bytes.
        uint32_t head_tile_idx = i / (2 * face_h);
        uint32_t head_in_tile = i % (2 * face_h);
        uint32_t in_tile_offset_by_batch = (head_in_tile < face_h ? head_in_tile * sub_tile_line_bytes
                                                                  : (head_in_tile + face_h) * sub_tile_line_bytes) +
                                           head_tile_idx * head_size;

        const NodeCoord core = cores[i];
        const KernelRunArgs::RuntimeArgValues args{{"in_tile_offset_by_head", in_tile_offset_by_batch}};
        reader_run.runtime_arg_values.push_back({core, args});
        writer_run.runtime_arg_values.push_back({core, args});
    }

    WorkUnitSpec wu{
        .name = "nlp_concat_heads_decode_subcoregrids",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = q_cores,
    };

    ProgramSpec spec{
        .name = "nlp_concat_heads_decode_subcoregrids",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
