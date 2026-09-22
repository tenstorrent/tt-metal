// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts
NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& batch_offset = tensor_args.batch_offset;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;
    const auto& overlap_qk_coregrid = operation_attributes.overlap_qk_coregrid;

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& q_mesh_tensor = output[0].mesh_tensor();
    const auto& k_mesh_tensor = output[1].mesh_tensor();
    const auto& v_mesh_tensor = output[2].mesh_tensor();

    const KernelSpecName Q_READER{"q_reader"};
    const KernelSpecName Q_WRITER{"q_writer"};
    const KernelSpecName K_READER{"k_reader"};
    const KernelSpecName K_WRITER{"k_writer"};
    const DFBSpecName Q_OUT{"q_out"};
    const DFBSpecName K_OUT{"k_out"};
    const DFBSpecName V_OUT{"v_out"};
    const DFBSpecName BATCH_OFFSET_READER{"batch_offset_reader"};
    const DFBSpecName BATCH_OFFSET_WRITER{"batch_offset_writer"};
    const TensorParamName QKV_IN{"qkv_in"};
    const TensorParamName Q_OUT_TENSOR{"q_out_tensor"};
    const TensorParamName K_OUT_TENSOR{"k_out_tensor"};
    const TensorParamName V_OUT_TENSOR{"v_out_tensor"};
    const TensorParamName BATCH_OFFSET_TENSOR{"batch_offset_tensor"};

    IDevice* device = input_tensor.device();

    tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t single_tile_size = tt::tile_size(data_format);

    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = output[0].shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    const auto k_shard_spec = output[1].shard_spec().value();
    const auto k_cores = k_shard_spec.grid;
    const auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;
    uint32_t batch_offset_index_stick_size = 0;

    Group<DataflowBufferSpec> dataflow_buffers;

    // Staging buffers for the batch_offset scalar, allocated only when batch_offset is provided.
    // Reader instances stage through one buffer and writer instances through the other (mirroring
    // the legacy per-RISC circular-buffer split). Each is a sync-free single-toucher per node, so
    // its owning kernel binds both the PRODUCER and CONSUMER endpoints (a Gen1-legal self-loop).
    if (batch_offset.has_value()) {
        tt::DataFormat batch_offset_data_format = datatype_to_dataformat_converter(batch_offset.value().dtype());
        uint32_t single_batch_offset_tile_size = tt::tile_size(batch_offset_data_format);
        batch_offset_index_stick_size = batch_offset.value().buffer()->aligned_page_size();

        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BATCH_OFFSET_READER,
            .entry_size = 1,
            .num_entries = single_batch_offset_tile_size,
            .data_format_metadata = batch_offset_data_format,
        });

        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BATCH_OFFSET_WRITER,
            .entry_size = 1,
            .num_entries = single_batch_offset_tile_size,
            .data_format_metadata = batch_offset_data_format,
        });
    }

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = Q_OUT,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = Q_OUT_TENSOR,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = K_OUT,
        .entry_size = single_tile_size,
        .num_entries = k_num_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = K_OUT_TENSOR,
    });

    // NOTE (preserved legacy behavior): the V buffer size is computed from Q's shard spec
    // (output[0]), while the backing memory is the V output tensor (output[2]). This is masked
    // today because num_q_heads and num_kv_heads both pad to 32, making the Q and V shard shapes
    // equal.
    const auto v_shard_spec = output[0].shard_spec().value();
    const auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = V_OUT,
        .entry_size = single_tile_size,
        .num_entries = v_num_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = V_OUT_TENSOR,
    });

    // cores for q
    const uint32_t q_num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& q_cores_vector = corerange_to_cores(q_cores, q_num_cores, true);

    // cores for k
    const uint32_t k_num_cores = k_cores.num_cores();  // number of cores of the output
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    auto in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords, noc_y_coords;
    noc_x_coords.reserve(in_num_cores);
    noc_y_coords.reserve(in_num_cores);

    for (uint32_t i = 0; i < in_num_cores; ++i) {
        auto worker_core = device->worker_core_from_logical_core(in_cores_vec[i]);
        noc_x_coords.push_back(worker_core.x);
        noc_y_coords.push_back(worker_core.y);
    }

    // In case of overlapping qk coregrid, we create a single set of kernels for q which also process k and v heads
    // from the input and write to the respective output buffers while if q and k are not overlapped, we create two
    // sets of kernels in different coregrids one set of kernels for q which also process v heads but skips k heads
    // from the input and write to the respective output buffers another set of kernels for k which reads k heads from
    // the input and write to the respective output buffers while skipping q and v heads
    //
    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2
    // of a tile respectively). The process_qv / process_k selection and the batch-offset usage gate kernel-side
    // resource bindings, so they are emitted as compile defines rather than value CTAs.
    const std::filesystem::path kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp";
    const auto arch = device->arch();

    auto make_kernel = [&](const KernelSpecName& unique_id,
                           uint32_t phases_to_read,
                           bool process_qv,
                           bool process_k,
                           DataMovementHardwareConfig hw_config) {
        // The two co-resident instances raw-write disjoint sub-tile phases of the borrowed output
        // DFBs (no FIFO ops), so the PRODUCER/CONSUMER labels are cosmetic on Gen1 — the phase-1
        // (reader-config) instance takes PRODUCER and the phase-2 (writer-config) instance
        // CONSUMER, satisfying the 1P+1C endpoint invariant.
        const bool is_writer_instance = (phases_to_read == 2);
        const auto out_role = is_writer_instance ? DFBEndpointType::CONSUMER : DFBEndpointType::PRODUCER;
        Group<DFBBinding> dfb_bindings;
        if (process_qv) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = Q_OUT,
                .accessor_name = "q_out",
                .endpoint_type = out_role,
            });
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = V_OUT,
                .accessor_name = "v_out",
                .endpoint_type = out_role,
            });
        }
        if (process_k) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = K_OUT,
                .accessor_name = "k_out",
                .endpoint_type = out_role,
            });
        }
        if (batch_offset.has_value()) {
            // Per-instance staging buffer (reader vs writer), self-looped under one shared
            // kernel-side accessor name.
            const DFBSpecName& batch_offset_dfb = is_writer_instance ? BATCH_OFFSET_WRITER : BATCH_OFFSET_READER;
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = batch_offset_dfb,
                .accessor_name = "batch_offset",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = batch_offset_dfb,
                .accessor_name = "batch_offset",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }

        KernelSpec::CompilerOptions::Defines defines;
        if (process_qv) {
            defines.emplace("PROCESS_QV", "1");
        }
        if (process_k) {
            defines.emplace("PROCESS_K", "1");
        }
        if (batch_offset.has_value()) {
            defines.emplace("USE_BATCH_OFFSET", "1");
        }

        Group<TensorBinding> tensor_bindings = {{
            .tensor_parameter_name = QKV_IN,
            .accessor_name = "qkv_in",
        }};
        if (batch_offset.has_value()) {
            tensor_bindings.push_back(TensorBinding{
                .tensor_parameter_name = BATCH_OFFSET_TENSOR,
                .accessor_name = "batch_offset_tensor",
            });
        }

        return KernelSpec{
            .unique_id = unique_id,
            .source = kernel_source,
            .compiler_options = {.defines = std::move(defines)},
            .dfb_bindings = std::move(dfb_bindings),
            .tensor_bindings = std::move(tensor_bindings),
            .compile_time_args =
                {
                    {"ELEMENT_SIZE", element_size},
                    {"SUBTILE_LINE_BYTES", sub_tile_line_bytes},
                    {"head_size", head_size},
                    {"num_q_heads", num_q_heads},
                    {"num_kv_heads", num_kv_heads},
                    {"head_size_num_tiles", head_tiles},
                    {"PHASES_TO_READ", phases_to_read},
                    {"in_num_cores", in_num_cores},
                    {"index_stick_size", batch_offset_index_stick_size},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"index_in_cores"}},
            .hw_config = std::move(hw_config),
            // The per-input-core NoC coordinate tables are genuine varargs: CTA-bounded
            // variable-count blocks the kernel indexes at runtime (noc_x table of in_num_cores
            // entries, then noc_y table of in_num_cores entries).
            .advanced_options = {.num_runtime_varargs = 2 * in_num_cores},
        };
    };

    Group<KernelSpec> kernels;
    kernels.push_back(make_kernel(
        Q_READER,
        /*phases_to_read=*/1,
        /*process_qv=*/true,
        overlap_qk_coregrid,
        create_reader_datamovement_config(arch)));
    kernels.push_back(make_kernel(
        Q_WRITER,
        /*phases_to_read=*/2,
        /*process_qv=*/true,
        overlap_qk_coregrid,
        create_writer_datamovement_config(arch)));
    if (!overlap_qk_coregrid) {
        kernels.push_back(make_kernel(
            K_READER,
            /*phases_to_read=*/1,
            /*process_qv=*/false,
            /*process_k=*/true,
            create_reader_datamovement_config(arch)));
        kernels.push_back(make_kernel(
            K_WRITER,
            /*phases_to_read=*/2,
            /*process_qv=*/false,
            /*process_k=*/true,
            create_writer_datamovement_config(arch)));
    }

    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "q",
        .kernels = {Q_READER, Q_WRITER},
        .target_nodes = q_cores,
    }};
    if (!overlap_qk_coregrid) {
        work_units.push_back(WorkUnitSpec{
            .name = "k",
            .kernels = {K_READER, K_WRITER},
            .target_nodes = k_cores,
        });
    }

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = QKV_IN, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = Q_OUT_TENSOR, .spec = output[0].tensor_spec()},
        TensorParameter{.unique_id = K_OUT_TENSOR, .spec = output[1].tensor_spec()},
        TensorParameter{.unique_id = V_OUT_TENSOR, .spec = output[2].tensor_spec()},
    };
    if (batch_offset.has_value()) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = BATCH_OFFSET_TENSOR, .spec = batch_offset.value().tensor_spec()});
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_decode_sharded_subcoregrid",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    // Every node reads the whole coordinate tables; the per-node distinction is index_in_cores.
    std::vector<uint32_t> noc_coord_varargs;
    noc_coord_varargs.reserve(2 * in_num_cores);
    noc_coord_varargs.insert(noc_coord_varargs.end(), noc_x_coords.begin(), noc_x_coords.end());
    noc_coord_varargs.insert(noc_coord_varargs.end(), noc_y_coords.begin(), noc_y_coords.end());

    auto fill_run_args = [&](KernelRunArgs& run_args_entry, const std::vector<CoreCoord>& cores, uint32_t num_cores) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = cores[i];
            AddRuntimeArgsForNode(run_args_entry.runtime_arg_values, core, {{"index_in_cores", i}});
            run_args_entry.advanced_options.runtime_varargs[core] = noc_coord_varargs;
        }
    };

    ProgramRunArgs run_args;
    {
        KernelRunArgs q_reader_run_args{.kernel = Q_READER};
        KernelRunArgs q_writer_run_args{.kernel = Q_WRITER};
        fill_run_args(q_reader_run_args, q_cores_vector, q_num_cores);
        fill_run_args(q_writer_run_args, q_cores_vector, q_num_cores);
        run_args.kernel_run_args.push_back(std::move(q_reader_run_args));
        run_args.kernel_run_args.push_back(std::move(q_writer_run_args));
    }
    if (!overlap_qk_coregrid) {
        KernelRunArgs k_reader_run_args{.kernel = K_READER};
        KernelRunArgs k_writer_run_args{.kernel = K_WRITER};
        fill_run_args(k_reader_run_args, k_cores_vector, k_num_cores);
        fill_run_args(k_writer_run_args, k_cores_vector, k_num_cores);
        run_args.kernel_run_args.push_back(std::move(k_reader_run_args));
        run_args.kernel_run_args.push_back(std::move(k_writer_run_args));
    }

    run_args.tensor_args = {
        {QKV_IN, input_mesh_tensor},
        {Q_OUT_TENSOR, q_mesh_tensor},
        {K_OUT_TENSOR, k_mesh_tensor},
        {V_OUT_TENSOR, v_mesh_tensor},
    };
    if (batch_offset.has_value()) {
        run_args.tensor_args.emplace(BATCH_OFFSET_TENSOR, TensorArgument{batch_offset.value().mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
