// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.hpp"

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

// Metal 2.0 named resource handles for the subcoregrid ProgramSpec.
// (Names prefixed to avoid Unity-build collisions with the sibling factories.)
const TensorParamName SG_INPUT{"input"};
const TensorParamName SG_BATCH_OFFSET{"batch_offset"};
// q/k/v outputs are bound as TensorParameters (written via get_bank_base_address() + offset), NOT
// borrowed-memory DFBs — avoids the multi-work-unit borrowed-DFB base-corruption framework bug. See
// the sharded factory comment and METAL2_PORT_REPORT.md.
const TensorParamName SG_Q_OUT{"q_out"};
const TensorParamName SG_K_OUT{"k_out"};
const TensorParamName SG_V_OUT{"v_out"};

// Distinct reader / writer batch-offset scratchpads (c_15 / c_14). This factory already switched the
// writer to c_14 in the legacy code, so both are clean single-toucher self-loops.
const DFBSpecName SG_BATCH_READER_DFB{"batch_offset_reader"};
const DFBSpecName SG_BATCH_WRITER_DFB{"batch_offset_writer"};

const KernelSpecName SG_Q_READER{"q_reader"};
const KernelSpecName SG_Q_WRITER{"q_writer"};
const KernelSpecName SG_K_READER{"k_reader"};
const KernelSpecName SG_K_WRITER{"k_writer"};

constexpr const char* kSubcoregridKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
    "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts
NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory::create_program_artifacts(
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

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = output[0].shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto k_shard_spec = output[1].shard_spec().value();
    const auto k_cores = k_shard_spec.grid;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;
    uint32_t batch_offset_index_stick_size = 0;

    tt::DataFormat cb_batch_offset_data_format = cb_data_format;
    uint32_t single_batch_offset_tile_size = 0;
    if (batch_offset.has_value()) {
        cb_batch_offset_data_format = datatype_to_dataformat_converter(batch_offset.value().dtype());
        single_batch_offset_tile_size = tt::tile_size(cb_batch_offset_data_format);
        batch_offset_index_stick_size = batch_offset.value().buffer()->aligned_page_size();
    }

    // ----------------------------------------------------------------------------
    // Dataflow buffers: only the batch-offset scratchpads (self-loops, allocated L1). Outputs are
    // TensorParameters, not DFBs. (The legacy v_shard_spec-from-output[0] Misc anomaly is moot now: the
    // v TensorParameter uses output[2].tensor_spec(), correct by construction.)
    // ----------------------------------------------------------------------------
    DataflowBufferSpec batch_reader_dfb{
        .unique_id = SG_BATCH_READER_DFB,
        .entry_size = 1,
        .num_entries = single_batch_offset_tile_size,
        .data_format_metadata = cb_batch_offset_data_format,
    };
    DataflowBufferSpec batch_writer_dfb{
        .unique_id = SG_BATCH_WRITER_DFB,
        .entry_size = 1,
        .num_entries = single_batch_offset_tile_size,
        .data_format_metadata = cb_batch_offset_data_format,
    };

    // cores for q / k
    const uint32_t q_num_cores = q_cores.num_cores();
    const auto& q_cores_vector = corerange_to_cores(q_cores, q_num_cores, true);
    const uint32_t k_num_cores = k_cores.num_cores();
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();
    auto in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords, noc_y_coords;
    noc_x_coords.reserve(in_num_cores);
    noc_y_coords.reserve(in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        auto worker_core = device->worker_core_from_logical_core(in_cores_vec[i]);
        noc_x_coords.push_back(worker_core.x);
        noc_y_coords.push_back(worker_core.y);
    }

    // Input-core NoC coords → runtime varargs, noc_x [0, in_num_cores) then noc_y [in_num_cores, 2N).
    std::vector<uint32_t> noc_varargs;
    noc_varargs.reserve(2 * in_num_cores);
    noc_varargs.insert(noc_varargs.end(), noc_x_coords.begin(), noc_x_coords.end());
    noc_varargs.insert(noc_varargs.end(), noc_y_coords.begin(), noc_y_coords.end());
    const uint32_t num_varargs = 2 * in_num_cores;

    // ----------------------------------------------------------------------------
    // Kernel builder (see the sharded factory for the shared rationale).
    // ----------------------------------------------------------------------------
    auto make_kernel = [&](const KernelSpecName& unique_id,
                           uint32_t phases_to_read,
                           bool process_qv,
                           bool process_k,
                           const DFBSpecName& batch_dfb,
                           DataMovementHardwareConfig hw_config) {
        KernelSpec k{
            .unique_id = unique_id,
            .source = std::filesystem::path(kSubcoregridKernel),
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = SG_INPUT, .accessor_name = "input"}},
            .compile_time_args =
                {{"element_size", element_size},
                 {"sub_tile_line_bytes", sub_tile_line_bytes},
                 {"head_size", head_size},
                 {"num_q_heads", num_q_heads},
                 {"num_kv_heads", num_kv_heads},
                 {"head_size_num_tiles", head_tiles},
                 {"phases_to_read", phases_to_read},
                 {"in_num_cores", in_num_cores}},
            .runtime_arg_schema = {.runtime_arg_names = {"index_in_cores"}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
        KernelSpec::CompilerOptions::Defines defines;
        if (process_qv) {
            defines.insert({"PROCESS_QV", "1"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SG_Q_OUT, .accessor_name = "q_out"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SG_V_OUT, .accessor_name = "v_out"});
        }
        if (process_k) {
            defines.insert({"PROCESS_K", "1"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SG_K_OUT, .accessor_name = "k_out"});
        }
        if (batch_offset.has_value()) {
            defines.insert({"USE_BATCH_OFFSET", "1"});
            k.compile_time_args.insert({"index_stick_size", batch_offset_index_stick_size});
            k.tensor_bindings.push_back(
                TensorBinding{.tensor_parameter_name = SG_BATCH_OFFSET, .accessor_name = "batch_offset"});
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

    const bool process_k_on_q = overlap_qk_coregrid;

    KernelSpec q_reader = make_kernel(
        SG_Q_READER,
        /*phases_to_read=*/1,
        /*process_qv=*/true,
        /*process_k=*/process_k_on_q,
        SG_BATCH_READER_DFB,
        ttnn::create_reader_datamovement_config(device->arch()));
    KernelSpec q_writer = make_kernel(
        SG_Q_WRITER,
        /*phases_to_read=*/2,
        /*process_qv=*/true,
        /*process_k=*/process_k_on_q,
        SG_BATCH_WRITER_DFB,
        ttnn::create_writer_datamovement_config(device->arch()));

    KernelSpec k_reader;
    KernelSpec k_writer;
    if (!overlap_qk_coregrid) {
        k_reader = make_kernel(
            SG_K_READER,
            /*phases_to_read=*/1,
            /*process_qv=*/false,
            /*process_k=*/true,
            SG_BATCH_READER_DFB,
            ttnn::create_reader_datamovement_config(device->arch()));
        k_writer = make_kernel(
            SG_K_WRITER,
            /*phases_to_read=*/2,
            /*process_qv=*/false,
            /*process_k=*/true,
            SG_BATCH_WRITER_DFB,
            ttnn::create_writer_datamovement_config(device->arch()));
    }

    // ----------------------------------------------------------------------------
    // Per-node runtime args.
    // ----------------------------------------------------------------------------
    KernelRunArgs q_reader_run{.kernel = SG_Q_READER};
    KernelRunArgs q_writer_run{.kernel = SG_Q_WRITER};
    for (uint32_t i = 0; i < q_num_cores; ++i) {
        const auto& core = q_cores_vector[i];
        AddRuntimeArgsForNode(q_reader_run.runtime_arg_values, core, {{"index_in_cores", i}});
        AddRuntimeArgsForNode(q_writer_run.runtime_arg_values, core, {{"index_in_cores", i}});
        q_reader_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
        q_writer_run.advanced_options.runtime_varargs.emplace(core, noc_varargs);
    }

    KernelRunArgs k_reader_run{.kernel = SG_K_READER};
    KernelRunArgs k_writer_run{.kernel = SG_K_WRITER};
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
    spec.name = "nlp_create_qkv_heads_decode_sharded_subcoregrid";
    spec.kernels = {q_reader, q_writer};
    if (batch_offset.has_value()) {
        spec.dataflow_buffers = {batch_reader_dfb, batch_writer_dfb};
    }
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SG_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = SG_Q_OUT, .spec = output[0].tensor_spec()},
        TensorParameter{.unique_id = SG_K_OUT, .spec = output[1].tensor_spec()},
        TensorParameter{.unique_id = SG_V_OUT, .spec = output[2].tensor_spec()},
    };
    if (batch_offset.has_value()) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = SG_BATCH_OFFSET, .spec = batch_offset.value().tensor_spec()});
    }
    spec.work_units = {WorkUnitSpec{
        .name = "wu_q",
        .kernels = {SG_Q_READER, SG_Q_WRITER},
        .target_nodes = q_cores,
    }};

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(q_reader_run), std::move(q_writer_run)};
    run_params.tensor_args = {
        {SG_INPUT, TensorArgument{input_mesh}},
        {SG_Q_OUT, TensorArgument{q_mesh}},
        {SG_K_OUT, TensorArgument{k_mesh}},
        {SG_V_OUT, TensorArgument{v_mesh}},
    };
    if (batch_offset.has_value()) {
        run_params.tensor_args.insert({SG_BATCH_OFFSET, TensorArgument{batch_offset.value().mesh_tensor()}});
    }

    if (!overlap_qk_coregrid) {
        spec.kernels.push_back(k_reader);
        spec.kernels.push_back(k_writer);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_k",
            .kernels = {SG_K_READER, SG_K_WRITER},
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
