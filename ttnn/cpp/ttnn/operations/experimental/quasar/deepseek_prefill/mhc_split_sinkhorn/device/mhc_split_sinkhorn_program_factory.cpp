// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>

#include "mhc_split_sinkhorn_program_factory.hpp"
#include "mhc_split_sinkhorn_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim::qsr {

using namespace tt::tt_metal;

ttnn::device_operation::ProgramArtifacts MhcSplitSinkhornProgramFactory::create_program_artifacts(
    const MhcSplitSinkhornParams& operation_attributes,
    const MhcSplitSinkhornTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName MIXES_DFB{"mixes"};
    const DFBSpecName CONSTS_DFB{"consts"};  // 8 tiles
    const DFBSpecName PRE_DFB{"pre"};
    const DFBSpecName POST_DFB{"post"};
    const DFBSpecName COMB_DFB{"comb"};
    const DFBSpecName MA{"ma"};
    const DFBSpecName MB{"mb"};
    const DFBSpecName RS{"rs"};
    const DFBSpecName RECIP{"recip"};
    const DFBSpecName TMP{"tmp"};
    const TensorParamName MIXES{"mixes"};
    const TensorParamName CONSTS{"consts"};
    const TensorParamName PRE{"pre"};
    const TensorParamName POST{"post"};
    const TensorParamName COMB{"comb"};

    const tt::DataFormat df = datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    const uint32_t tile_size = tt::tile_size(df);

    const auto& mixes = tensor_args.mixes.mesh_tensor();
    const auto& consts = tensor_args.consts.mesh_tensor();
    const auto& pre = tensor_return_value.at(0).mesh_tensor();
    const auto& post = tensor_return_value.at(1).mesh_tensor();
    const auto& comb = tensor_return_value.at(2).mesh_tensor();

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/mhc_split_sinkhorn/device/kernels/";

    const auto dfb_spec = [&](const DFBSpecName& name, uint32_t n_tiles) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size,
            .num_entries = n_tiles,
            .data_format_metadata = df,
        };
    };

    // Every DFB is Float32 and compute runs with a 32-bit dest, so each one compute consumes needs an explicit unpack
    // mode; the legacy op left them all at Default, i.e. UnpackToSrc.
    const auto compute_hw_config = [&](std::initializer_list<DFBSpecName> consumed) {
        ComputeHardwareConfig config{
            .fpu_math_fidelity = MathFidelity::HiFi4,
            .enable_32_bit_dest = true,
        };
        for (const auto& name : consumed) {
            config.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
        }
        return config;
    };

    const KernelSpec::CompileTimeArgs compute_ct = {
        {"iters", operation_attributes.sinkhorn_iters},
        {"eps_bits", std::bit_cast<uint32_t>(operation_attributes.eps)},
    };

    // Compute is the only toucher of its intermediates: each is a self-loop.
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{.dfb_spec_name = MIXES_DFB, .accessor_name = "mixes", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = CONSTS_DFB, .accessor_name = "consts", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = PRE_DFB, .accessor_name = "pre", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = POST_DFB, .accessor_name = "post", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = COMB_DFB, .accessor_name = "comb", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    for (const auto& [name, accessor] : {
             std::pair{MA, "ma"},
             std::pair{MB, "mb"},
             std::pair{RS, "rs"},
             std::pair{RECIP, "recip"},
             std::pair{TMP, "tmp"},
         }) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
    }

    const Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = MIXES, .spec = mixes.tensor_spec()},
        TensorParameter{.unique_id = CONSTS, .spec = consts.tensor_spec()},
        TensorParameter{.unique_id = PRE, .spec = pre.tensor_spec()},
        TensorParameter{.unique_id = POST, .spec = post.tensor_spec()},
        TensorParameter{.unique_id = COMB, .spec = comb.tensor_spec()},
    };

    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(MIXES, mixes);
    run_args.tensor_args.emplace(CONSTS, consts);
    run_args.tensor_args.emplace(PRE, pre);
    run_args.tensor_args.emplace(POST, post);
    run_args.tensor_args.emplace(COMB, comb);

    // ---- Sharded path (#40720): mixes already resident per-core in L1; the input and output
    // DFBs are borrowed straight from the shards (zero-copy), so the reader only fetches consts and
    // there is no writer. The compute kernel is identical -- the FIFO advances through the borrowed shard.
    if (tensor_args.mixes.memory_config().is_sharded()) {
        const auto& shard = tensor_args.mixes.shard_spec().value();
        const CoreRangeSet& cores = shard.grid;
        const uint32_t tiles = shard.shape[0] / tt::constants::TILE_HEIGHT;  // token-tiles per core

        const auto borrowed_dfb_spec = [&](const DFBSpecName& name, const TensorParamName& tensor) {
            DataflowBufferSpec spec = dfb_spec(name, tiles);
            spec.borrowed_from = tensor;
            return spec;
        };
        Group<DataflowBufferSpec> dataflow_buffers = {
            borrowed_dfb_spec(MIXES_DFB, MIXES),
            dfb_spec(CONSTS_DFB, 8),  // consts stay in DRAM, read per core
            borrowed_dfb_spec(PRE_DFB, PRE),
            borrowed_dfb_spec(POST_DFB, POST),
            borrowed_dfb_spec(COMB_DFB, COMB),
            dfb_spec(MA, 2),
            dfb_spec(MB, 2),
            dfb_spec(RS, 2),
            dfb_spec(RECIP, 2),
            dfb_spec(TMP, 2),
        };

        KernelSpec reader{
            .unique_id = READER,
            .source = std::filesystem::path(kdir + "dataflow/reader_mhc_split_sinkhorn_sharded.cpp"),
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = MIXES_DFB,
                        .accessor_name = "mixes",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = CONSTS_DFB,
                        .accessor_name = "consts",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = CONSTS, .accessor_name = "consts"}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_token_tiles"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        };

        // No writer drains the outputs -- the borrowed output shards are the result -- so compute is
        // their only toucher and binds each as both PRODUCER and CONSUMER (self-loop).
        for (const auto& [name, accessor] : {
                 std::pair{PRE_DFB, "pre"},
                 std::pair{POST_DFB, "post"},
                 std::pair{COMB_DFB, "comb"},
             }) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
        }
        KernelSpec compute{
            .unique_id = COMPUTE,
            .source = std::filesystem::path(kdir + "compute/mhc_split_sinkhorn_compute.cpp"),
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = compute_ct,
            .runtime_arg_schema = {.runtime_arg_names = {"num_token_tiles"}},
            .hw_config =
                compute_hw_config({MIXES_DFB, CONSTS_DFB, PRE_DFB, POST_DFB, COMB_DFB, MA, MB, RS, RECIP, TMP}),
        };

        KernelRunArgs reader_run_args{.kernel = READER};
        KernelRunArgs compute_run_args{.kernel = COMPUTE};
        for (const CoreCoord& core : corerange_to_cores(cores)) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_token_tiles", tiles}});
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_token_tiles", tiles}});
        }
        run_args.kernel_run_args.reserve(2);
        run_args.kernel_run_args.push_back(std::move(reader_run_args));
        run_args.kernel_run_args.push_back(std::move(compute_run_args));

        ProgramSpec spec{
            .name = "mhc_split_sinkhorn",
            .kernels = {std::move(reader), std::move(compute)},
            .dataflow_buffers = std::move(dataflow_buffers),
            .tensor_parameters = tensor_parameters,
            .work_units =
                {
                    WorkUnitSpec{
                        .name = "main",
                        .kernels = {READER, COMPUTE},
                        .target_nodes = cores,
                    },
                },
        };
        return ttnn::device_operation::ProgramArtifacts{
            .spec = std::move(spec),
            .run_params = std::move(run_args),
        };
    }

    // ---- Interleaved (DRAM) path.
    // Parametrization is independent per token, so distribute the token-tiles across the grid.
    const uint32_t num_token_tiles = tensor_args.mixes.physical_volume() / tt::constants::TILE_HW;
    auto grid = tensor_args.mixes.device()->compute_with_storage_grid_size();
    // max_cores==1 pins to a single core -- the multi-core A/B baseline. It rides in the hashed
    // attributes (read from MHC_MAX_CORES at op launch), so the program cache stays correct.
    if (operation_attributes.max_cores == 1) {
        grid = CoreCoord{1, 1};
    }
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(grid, num_token_tiles);

    Group<DataflowBufferSpec> dataflow_buffers = {
        dfb_spec(MIXES_DFB, 2),
        dfb_spec(CONSTS_DFB, 8),
        dfb_spec(PRE_DFB, 2),
        dfb_spec(POST_DFB, 2),
        dfb_spec(COMB_DFB, 2),
        dfb_spec(MA, 2),
        dfb_spec(MB, 2),
        dfb_spec(RS, 2),
        dfb_spec(RECIP, 2),
        dfb_spec(TMP, 2),
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/mhc_split_sinkhorn/device/kernels/dataflow/"
            "reader_mhc_split_sinkhorn.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = MIXES_DFB,
                    .accessor_name = "mixes",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CONSTS_DFB,
                    .accessor_name = "consts",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = MIXES, .accessor_name = "mixes"},
                TensorBinding{.tensor_parameter_name = CONSTS, .accessor_name = "consts"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_token_tiles", "start_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/mhc_split_sinkhorn/device/kernels/dataflow/"
            "writer_mhc_split_sinkhorn.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = PRE_DFB, .accessor_name = "pre", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = POST_DFB,
                    .accessor_name = "post",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = COMB_DFB,
                    .accessor_name = "comb",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = PRE, .accessor_name = "pre"},
                TensorBinding{.tensor_parameter_name = POST, .accessor_name = "post"},
                TensorBinding{.tensor_parameter_name = COMB, .accessor_name = "comb"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_token_tiles", "start_tile"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/mhc_split_sinkhorn/device/kernels/compute/"
            "mhc_split_sinkhorn_compute.cpp"),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = compute_ct,
        .runtime_arg_schema = {.runtime_arg_names = {"num_token_tiles"}},
        .hw_config = compute_hw_config({MIXES_DFB, CONSTS_DFB, MA, MB, RS, RECIP, TMP}),
    };

    // Per-core work split. start_tile is the page offset into mixes/outputs; consts (8
    // tiles) are read whole on every core.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0, start_tile = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t tiles = core_group_1.contains(core) ? tiles_per_core_1 : tiles_per_core_2;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_token_tiles", tiles}, {"start_tile", start_tile}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_token_tiles", tiles}, {"start_tile", start_tile}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_token_tiles", tiles}});
        start_tile += tiles;
    }
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    ProgramSpec spec{
        .name = "mhc_split_sinkhorn",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = tensor_parameters,
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };
    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim::qsr
