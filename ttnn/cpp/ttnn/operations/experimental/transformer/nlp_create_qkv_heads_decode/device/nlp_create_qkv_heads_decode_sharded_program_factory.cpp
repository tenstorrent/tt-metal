// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_sharded_program_factory.hpp"

#include <filesystem>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

namespace {

// Metal 2.0 named resource handles for the sharded ProgramSpec.
// (Names prefixed to avoid Unity-build collisions with the sibling factories.)
const TensorParamName SH_INPUT{"input"};
const TensorParamName SH_BATCH_OFFSET{"batch_offset"};
// q/k/v outputs are bound as TensorParameters (written via get_bank_base_address() + offset), NOT
// borrowed-memory DFBs: a Metal 2.0 framework bug corrupts a borrowed DFB's device-side base in the
// multi-work-unit spec the !overlap layout requires. The TensorParameter base-address path is
// unaffected (same mechanism the input uses). See METAL2_PORT_REPORT.md.
const TensorParamName SH_Q_OUT{"q_out"};
const TensorParamName SH_K_OUT{"k_out"};
const TensorParamName SH_V_OUT{"v_out"};

// Distinct reader / writer batch-offset scratchpads (legacy c_15 / c_14). NOTE: the legacy factory never
// switched the writer kernel's CTA to c_14, so both reader and writer produced into c_15 and c_14 was
// dead — a pre-existing wiring bug (the subcoregrid factory switches correctly). The port applies the
// missing switch here (writer binds SH_BATCH_WRITER_DFB), making each a clean single-toucher self-loop.
// Behavior-preserving: both just read page 0 of the batch_offset tensor.
const DFBSpecName SH_BATCH_READER_DFB{"batch_offset_reader"};
const DFBSpecName SH_BATCH_WRITER_DFB{"batch_offset_writer"};

const KernelSpecName SH_Q_READER{"q_reader"};
const KernelSpecName SH_Q_WRITER{"q_writer"};
const KernelSpecName SH_K_READER{"k_reader"};
const KernelSpecName SH_K_WRITER{"k_writer"};

constexpr const char* kShardedKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
    "reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts NLPCreateQKVHeadsDecodeShardedProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& batch_offset = tensor_args.batch_offset;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;
    const auto& overlap_qk_coregrid = operation_attributes.overlap_qk_coregrid;

    const auto& input_mesh = input_tensor.mesh_tensor();
    const auto& q_mesh = output[0].mesh_tensor();
    const auto& k_mesh = output[1].mesh_tensor();
    const auto& v_mesh = output[2].mesh_tensor();

    IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;
    uint32_t batch_offset_index_stick_size = 0;

    // batch-offset scratch DFB sizing (page_size 1, total = one batch_offset tile), when provided.
    tt::DataFormat cb_batch_offset_data_format = cb_data_format;
    uint32_t single_batch_offset_tile_size = 0;
    if (batch_offset.has_value()) {
        cb_batch_offset_data_format = datatype_to_dataformat_converter(batch_offset.value().dtype());
        single_batch_offset_tile_size = tt::tile_size(cb_batch_offset_data_format);
        batch_offset_index_stick_size = batch_offset.value().buffer()->aligned_page_size();
    }

    // ----------------------------------------------------------------------------
    // Dataflow buffers: only the batch-offset scratchpads (single-toucher self-loops, allocated L1 —
    // NOT borrowed). Outputs are TensorParameters, not DFBs. Constructed unconditionally but added to
    // the spec only when a batch_offset tensor is provided.
    // ----------------------------------------------------------------------------
    DataflowBufferSpec batch_reader_dfb{
        .unique_id = SH_BATCH_READER_DFB,
        .entry_size = 1,
        .num_entries = single_batch_offset_tile_size,
        .data_format_metadata = cb_batch_offset_data_format,
    };
    DataflowBufferSpec batch_writer_dfb{
        .unique_id = SH_BATCH_WRITER_DFB,
        .entry_size = 1,
        .num_entries = single_batch_offset_tile_size,
        .data_format_metadata = cb_batch_offset_data_format,
    };

    // cores for q
    uint32_t q_num_cores = q_cores.num_cores();  // number of cores of the output
    auto q_core_grid = q_cores.bounding_box();
    uint32_t q_num_cores_x = q_core_grid.end_coord.x + 1, q_num_cores_y = q_core_grid.end_coord.y + 1;
    const auto& q_cores_vector = grid_to_cores(q_num_cores, q_num_cores_x, q_num_cores_y, true);

    // cores for k
    uint32_t k_num_cores = k_cores.num_cores();  // number of cores of the output
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores_x);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores_y);
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // Input-core NoC coords → runtime varargs, noc_x first then noc_y. Same values on every node (device
    // topology); kept per-node to mirror the legacy per-core append (a CRTA is a valid later cleanup).
    std::vector<uint32_t> noc_varargs;
    noc_varargs.reserve(in_num_cores_x + in_num_cores_y);
    noc_varargs.insert(noc_varargs.end(), noc_x_coords.begin(), noc_x_coords.end());
    noc_varargs.insert(noc_varargs.end(), noc_y_coords.begin(), noc_y_coords.end());
    const uint32_t num_varargs = in_num_cores_x + in_num_cores_y;

    // ----------------------------------------------------------------------------
    // Kernel builder. Reader/writer differ by phases_to_read; process_qv / process_k become #defines
    // gating which output tensors the kernel writes. Outputs are TensorBindings (Case 2 base pointer);
    // the batch-offset tensor (Case 1) + its scratch DFB are bound only when a batch_offset is provided.
    //   Overlapping qk coregrid: one set of q kernels also processes k and v.
    //   Non-overlapping: a second set of k kernels reads only the k heads.
    // ----------------------------------------------------------------------------
    auto make_kernel = [&](const KernelSpecName& unique_id,
                           uint32_t phases_to_read,
                           bool process_qv,
                           bool process_k,
                           const DFBSpecName& batch_dfb,
                           DataMovementHardwareConfig hw_config) {
        KernelSpec k{
            .unique_id = unique_id,
            .source = std::filesystem::path(kShardedKernel),
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = SH_INPUT, .accessor_name = "input"}},
            .compile_time_args =
                {{"element_size", element_size},
                 {"sub_tile_line_bytes", sub_tile_line_bytes},
                 {"head_size", head_size},
                 {"num_q_heads", num_q_heads},
                 {"num_kv_heads", num_kv_heads},
                 {"head_size_num_tiles", head_tiles},
                 {"phases_to_read", phases_to_read},
                 {"num_x", in_num_cores_x},
                 {"num_y", in_num_cores_y}},
            .runtime_arg_schema = {.runtime_arg_names = {"index_in_cores"}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
        KernelSpec::CompilerOptions::Defines defines;
        if (process_qv) {
            defines.insert({"PROCESS_QV", "1"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SH_Q_OUT, .accessor_name = "q_out"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SH_V_OUT, .accessor_name = "v_out"});
        }
        if (process_k) {
            defines.insert({"PROCESS_K", "1"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SH_K_OUT, .accessor_name = "k_out"});
        }
        if (batch_offset.has_value()) {
            defines.insert({"USE_BATCH_OFFSET", "1"});
            k.compile_time_args.insert({"index_stick_size", batch_offset_index_stick_size});
            k.tensor_bindings.push_back(
                TensorBinding{.tensor_parameter_name = SH_BATCH_OFFSET, .accessor_name = "batch_offset"});
            // Single-toucher scratchpad → self-loop (PRODUCER + CONSUMER on the one binding kernel).
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = batch_dfb,
                .accessor_name = "batch_offset",
                .endpoint_type = DFBEndpointType::PRODUCER});
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = batch_dfb,
                .accessor_name = "batch_offset",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        k.compiler_options.defines = std::move(defines);
        return k;
    };

    const bool process_k_on_q = overlap_qk_coregrid;  // q kernels also handle k only when overlapping

    KernelSpec q_reader = make_kernel(
        SH_Q_READER,
        /*phases_to_read=*/1,
        /*process_qv=*/true,
        /*process_k=*/process_k_on_q,
        SH_BATCH_READER_DFB,
        ttnn::create_reader_datamovement_config(device->arch()));
    KernelSpec q_writer = make_kernel(
        SH_Q_WRITER,
        /*phases_to_read=*/2,
        /*process_qv=*/true,
        /*process_k=*/process_k_on_q,
        SH_BATCH_WRITER_DFB,
        ttnn::create_writer_datamovement_config(device->arch()));

    KernelSpec k_reader;
    KernelSpec k_writer;
    if (!overlap_qk_coregrid) {
        k_reader = make_kernel(
            SH_K_READER,
            /*phases_to_read=*/1,
            /*process_qv=*/false,
            /*process_k=*/true,
            SH_BATCH_READER_DFB,
            ttnn::create_reader_datamovement_config(device->arch()));
        k_writer = make_kernel(
            SH_K_WRITER,
            /*phases_to_read=*/2,
            /*process_qv=*/false,
            /*process_k=*/true,
            SH_BATCH_WRITER_DFB,
            ttnn::create_writer_datamovement_config(device->arch()));
    }

    // ----------------------------------------------------------------------------
    // Per-node runtime args: index_in_cores (named) + the noc-coord varargs.
    // ----------------------------------------------------------------------------
    KernelRunArgs q_reader_run{.kernel = SH_Q_READER};
    KernelRunArgs q_writer_run{.kernel = SH_Q_WRITER};
    for (uint32_t i = 0; i < q_num_cores; ++i) {
        const auto& core = q_cores_vector[i];
        AddRuntimeArgsForNode(q_reader_run.runtime_arg_values, core, {{"index_in_cores", i}});
        AddRuntimeArgsForNode(q_writer_run.runtime_arg_values, core, {{"index_in_cores", i}});
        q_reader_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
        q_writer_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
    }

    KernelRunArgs k_reader_run{.kernel = SH_K_READER};
    KernelRunArgs k_writer_run{.kernel = SH_K_WRITER};
    if (!overlap_qk_coregrid) {
        for (uint32_t i = 0; i < k_num_cores; ++i) {
            const auto& core = k_cores_vector[i];
            AddRuntimeArgsForNode(k_reader_run.runtime_arg_values, core, {{"index_in_cores", i}});
            AddRuntimeArgsForNode(k_writer_run.runtime_arg_values, core, {{"index_in_cores", i}});
            k_reader_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
            k_writer_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
        }
    }

    // ----------------------------------------------------------------------------
    // Assemble spec + run-args.
    // ----------------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_decode_sharded";
    spec.kernels = {q_reader, q_writer};
    if (batch_offset.has_value()) {
        spec.dataflow_buffers = {batch_reader_dfb, batch_writer_dfb};
    }
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SH_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = SH_Q_OUT, .spec = output[0].tensor_spec()},
        TensorParameter{.unique_id = SH_K_OUT, .spec = output[1].tensor_spec()},
        TensorParameter{.unique_id = SH_V_OUT, .spec = output[2].tensor_spec()},
    };
    if (batch_offset.has_value()) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = SH_BATCH_OFFSET, .spec = batch_offset.value().tensor_spec()});
    }
    spec.work_units = {WorkUnitSpec{
        .name = "wu_q",
        .kernels = {SH_Q_READER, SH_Q_WRITER},
        .target_nodes = q_cores,
    }};

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(q_reader_run), std::move(q_writer_run)};
    run_params.tensor_args = {
        {SH_INPUT, TensorArgument{input_mesh}},
        {SH_Q_OUT, TensorArgument{q_mesh}},
        {SH_K_OUT, TensorArgument{k_mesh}},
        {SH_V_OUT, TensorArgument{v_mesh}},
    };
    if (batch_offset.has_value()) {
        run_params.tensor_args.insert({SH_BATCH_OFFSET, TensorArgument{batch_offset.value().mesh_tensor()}});
    }

    if (!overlap_qk_coregrid) {
        spec.kernels.push_back(k_reader);
        spec.kernels.push_back(k_writer);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_k",
            .kernels = {SH_K_READER, SH_K_WRITER},
            .target_nodes = k_cores,
        });
        run_params.kernel_run_args.push_back(std::move(k_reader_run));
        run_params.kernel_run_args.push_back(std::move(k_writer_run));
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::experimental::prim
