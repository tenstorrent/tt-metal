// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "nlp_concat_heads_decode_subcoregrids_program_factory.hpp"
#include "nlp_concat_heads_decode_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeSubcoregridsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t single_tile_size = tt::tile_size(data_format);
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto tile_h = tile_shape[0];
    auto tile_w = tile_shape[1];
    auto tile_hw = tile_h * tile_w;

    auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    auto face_h = face_shape[0];
    auto face_w = face_shape[1];
    auto face_hw = face_h * face_w;

    const uint32_t head_tiles = head_dim / tile_w;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = face_w * element_size;
    const auto q_shard_spec = output.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / tile_hw;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;

    // Program-scope resource names (function-local: this op's two factories share a
    // translation unit under unity builds, so no anonymous-namespace constants).
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName Q_OUT{"q_out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // The output-resident DFB: borrowed from the output tensor's L1 shard memory
    // (the backing address resolves at runtime from the OUTPUT tensor argument).
    DataflowBufferSpec q_out_dfb{
        .unique_id = Q_OUT,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT,
    };

    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    // NoC coordinate tables for the input shard cores, x block then y block; these ride the
    // kernels' runtime varargs (the kernel indexes them with a data-driven cursor).
    std::vector<uint32_t> noc_coords;
    noc_coords.reserve(2 * in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).x);
    }
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    // We parallelize the reader on risc0 and risc1 as two phases, where each risc reads half-tile of the input (Phase 1
    // reads left half-tile and Phase 2 reads right half-tile respectively)
    KernelSpec::CompileTimeArgs reader_compile_time_args{
        {"element_size", element_size},
        {"subtile_line_bytes", sub_tile_line_bytes},
        {"head_size", head_size},
        {"batch", batch},
        {"head_size_num_tiles", head_tiles},
        {"phases_to_read", 1},  // read the first phase
        {"in_num_cores", in_num_cores},
        {"face_h", face_h},
        {"face_hw", face_hw},
    };

    KernelSpec::CompileTimeArgs writer_compile_time_args = reader_compile_time_args;
    writer_compile_time_args["phases_to_read"] = 2;  // read the second phase

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp",
        // Both instances of the kernel only raw-write the output-resident DFB (no FIFO ops);
        // the PRODUCER/CONSUMER split between them is cosmetic 1P+1C to satisfy the validator.
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = Q_OUT,
            .accessor_name = "q_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .compile_time_args = std::move(reader_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = 2 * in_num_cores},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = Q_OUT,
            .accessor_name = "q_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .compile_time_args = std::move(writer_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = 2 * in_num_cores},
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

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

        const auto& core = cores[i];
        // Reader and writer instances receive identical per-core values; only the phase CTA differs.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        reader_run_args.advanced_options.runtime_varargs[core] = noc_coords;
        writer_run_args.advanced_options.runtime_varargs[core] = noc_coords;
    }

    ProgramSpec spec{
        .name = "nlp_concat_heads_decode_subcoregrids",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(q_out_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
             // OUTPUT is borrow-only: no kernel binds it, but the Q_OUT DFB borrows its memory.
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = q_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
