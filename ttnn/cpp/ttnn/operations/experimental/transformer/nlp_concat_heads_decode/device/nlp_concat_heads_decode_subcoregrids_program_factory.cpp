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
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <filesystem>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeSubcoregridsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    // Metal 2.0 named resource handles (function-local: the string values must match the sibling
    // full-grid factory's, but the C++ identifiers must not collide under unity build).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName Q_OUT{"q_out"};        // formerly CBIndex::c_16 (borrowed from output)
    const TensorParamName INPUT{"input"};    // Case 2 (raw base via get_bank_base_address)
    const TensorParamName OUTPUT{"output"};  // backs the borrowed q_out DFB

    const auto& input_tensor = tensor_args.input;
    const auto& input_mesh = input_tensor.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
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

    // Output dataflow buffer (formerly CBIndex::c_16). Borrowed from the output tensor's L1 memory;
    // both kernel instances raw-write into it via cb_q_out.get_write_ptr() + offset.
    DataflowBufferSpec q_out_dfb{
        .unique_id = Q_OUT,
        .entry_size = single_tile_size,
        .num_entries = static_cast<uint32_t>(q_num_tiles),
        .data_format_metadata = cb_data_format,
        .borrowed_from = OUTPUT,
    };

    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores);
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_x_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).x);
        noc_y_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    // The two NoC-coordinate blocks are a variable-count (input-grid-dependent) collection the kernel
    // reaches by computed index — an RTA vararg. Pack them as one per-node payload: [x-coords, y-coords].
    // The payload is identical on every output node (the input grid is fixed).
    std::vector<uint32_t> noc_coords_varargs;
    noc_coords_varargs.reserve(2 * in_num_cores);
    noc_coords_varargs.insert(noc_coords_varargs.end(), noc_x_coords.begin(), noc_x_coords.end());
    noc_coords_varargs.insert(noc_coords_varargs.end(), noc_y_coords.begin(), noc_y_coords.end());
    const uint32_t num_noc_varargs = 2 * in_num_cores;

    // Named compile-time args shared by both reader/writer kernel specializations.
    const KernelSpec::CompileTimeArgs common_cta{
        {"element_size", element_size},
        {"sub_tile_line_bytes", sub_tile_line_bytes},
        {"head_size", head_size},
        {"batch", batch},
        {"head_size_num_tiles", head_tiles},
        {"in_num_cores", in_num_cores},
        {"face_h", static_cast<uint32_t>(face_h)},
        {"face_hw", static_cast<uint32_t>(face_hw)},
    };

    const std::filesystem::path kernel_source{
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp"};

    // We parallelize the reader on risc0 and risc1 as two phases, where each risc reads half-tile of the input (Phase 1
    // reads left half-tile and Phase 2 reads right half-tile respectively). The two kernel specializations share one
    // source and differ only by PHASES_TO_READ and their (cosmetic on Gen1) DFB endpoint role — the output DFB is a
    // sync-free two-toucher work-split (1P + 1C).
    auto make_kernel =
        [&](const KernelSpecName& id, uint32_t phases_to_read, DFBEndpointType role, DataMovementHardwareConfig hw) {
            KernelSpec::CompileTimeArgs cta = common_cta;
            cta.insert({"phases_to_read", phases_to_read});
            return KernelSpec{
                .unique_id = id,
                .source = kernel_source,
                .dfb_bindings = {DFBBinding{.dfb_spec_name = Q_OUT, .accessor_name = "q_out", .endpoint_type = role}},
                .tensor_bindings =
                    {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                     TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
                .compile_time_args = cta,
                .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
                .hw_config = hw,
                .advanced_options = {.num_runtime_varargs = num_noc_varargs},
            };
        };

    KernelSpec reader = make_kernel(
        READER,
        /*phases_to_read=*/1,
        DFBEndpointType::PRODUCER,
        ttnn::create_reader_datamovement_config(device->arch()));
    KernelSpec writer = make_kernel(
        WRITER,
        /*phases_to_read=*/2,
        DFBEndpointType::CONSUMER,
        ttnn::create_writer_datamovement_config(device->arch()));

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
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
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        reader_run.advanced_options.runtime_varargs.emplace(core, noc_coords_varargs);
        writer_run.advanced_options.runtime_varargs.emplace(core, noc_coords_varargs);
    }

    ProgramSpec spec{
        .name = "nlp_concat_heads_decode_subcoregrids",
        .kernels = {reader, writer},
        .dataflow_buffers = {q_out_dfb},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = q_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT, TensorArgument{input_mesh}},
        {OUTPUT, TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
