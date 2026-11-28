// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include "tt-metalium/utils/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/utils/math.hpp>
#include "ttnn/operations/experimental/quasar/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;

ttnn::device_operation::ProgramArtifacts GridSampleNearestProgramFactory::create_program_artifacts(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    const bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    const bool align_corners = operation_attributes.align_corners;

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& grid_mesh_tensor = grid_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();
    const bool is_sharded = grid_tensor.is_sharded();

    // Data formats and device
    const auto [input_dfb_data_format, grid_dfb_data_format, output_dfb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));

    // Shape and dimensions
    const auto& [input_shape, grid_shape, output_shape] =
        std::tie(input_tensor.padded_shape(), grid_tensor.padded_shape(), output_tensor.padded_shape());
    const uint32_t input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid, "nearest");
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "nearest");
    tt::tt_metal::CoreRangeSet all_cores;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    std::vector<CoreCoord> logical_cores;

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        all_cores = grid_shard_spec.grid;
        num_cores = grid_shard_spec.num_cores();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    } else {
        const auto output_shard_spec = output_tensor.shard_spec().value();
        all_cores = output_shard_spec.grid;
        num_cores = output_shard_spec.num_cores();
        output_nsticks_per_core = output_shard_spec.shape[0];
        const uint32_t output_sticks_per_grid_stick =
            operation_attributes.batch_output_channels ? 1U : grid_batching_factor;
        TT_FATAL(
            output_nsticks_per_core % output_sticks_per_grid_stick == 0,
            "Nearest grid_sample output shard height must be divisible by the number of output rows per grid row ({})",
            output_sticks_per_grid_stick);
        // Each packed grid row expands to K output rows when batching along width.
        grid_nsticks_per_core = output_nsticks_per_core / output_sticks_per_grid_stick;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    }

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName GRID{"grid"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName GRID_0{"grid_0"};  // grid sticks for writer 0 (and writer 1 too, when the grid is sharded)
    const DFBSpecName GRID_1{"grid_1"};  // grid sticks for writer 1 when the grid is interleaved
    const DFBSpecName FILL{"fill"};      // a zeroed stick copied in for out-of-range samples
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName WRITER_0{"writer_0"};
    const KernelSpecName WRITER_1{"writer_1"};

    // Create DFBs
    // Sharded grid rows are stored at the buffer's aligned page size, so stride by the aligned stick size too.
    const uint32_t grid_stick_size = get_aligned_stick_size(grid_shape, grid_tensor);
    const uint32_t grid_dfb_num_pages0 = is_sharded ? grid_nsticks_per_core : 1;
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = GRID_0,
        .entry_size = grid_stick_size,
        .num_entries = grid_dfb_num_pages0,
        .data_format_metadata = grid_dfb_data_format,
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{GRID} : std::nullopt,
    });

    // Nearest mode has support for DRAM interleaved grid and sharded output
    const bool has_grid_1 = enable_split_reader && !is_sharded;
    if (has_grid_1) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = GRID_1,
            .entry_size = grid_stick_size,
            .num_entries = 1,
            .data_format_metadata = grid_dfb_data_format,
        });
    }

    const uint32_t output_dfb_page_size =
        static_cast<uint32_t>(static_cast<float>(output_shape[-1]) * output_tensor.element_size());
    const uint32_t output_dfb_pages = output_nsticks_per_core;

    // Fill DFB - single page to hold pre-filled stick
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = FILL,
        .entry_size = output_dfb_page_size,
        .num_entries = 1,
        .data_format_metadata = output_dfb_data_format,
    });

    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_dfb_page_size,
        .num_entries = output_dfb_pages,
        .data_format_metadata = output_dfb_data_format,
        .borrowed_from = OUTPUT,
    });

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);

    // Writer for nearest mode with sharded grid - matches sharded reader argument structure
    KernelSpec::CompileTimeArgs writer_compile_time_args = {
        {"input_stick_nbytes", input_stick_size},
        {"grid_stick_nbytes", grid_stick_size},
        {"input_height", input_height},
        {"input_width", input_width},
        {"grid_batching_factor", grid_batching_factor},
        {"grid_dtype", static_cast<uint32_t>(grid_tensor.dtype())},
        {"grid_hw", grid_hw},
        {"use_precomputed_grid", use_precomputed_grid ? 1U : 0U},
        {"align_corners", align_corners ? 1U : 0U},
        {"split_reader", enable_split_reader ? 1U : 0U},  // same as reader
        {"reader_id", 0U},                                // set per writer below
        {"grid_nsticks_per_core", grid_nsticks_per_core},
        {"is_sharded", is_sharded ? 1U : 0U},
        {"batch_size", input_shape[0]},
    };

    // When the grid is interleaved the writers read it page by page through the grid tensor; when it
    // is sharded they read it in place from the grid DFB (borrowed from the grid shard) instead.
    KernelSpec::CompilerOptions::Defines writer_defines;
    Group<TensorBinding> writer_tensor_bindings = {
        TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        },
    };
    if (!is_sharded) {
        writer_defines.emplace("GRID_INTERLEAVED", "1");
        writer_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = GRID,
            .accessor_name = "grid",
        });
    }
    const Group<std::string> writer_runtime_arg_names =
        is_sharded ? Group<std::string>{"global_grid_stick_start"} : Group<std::string>{"start_page_id"};

    // The two writers split the work between them (alternating grid points) on every node. The fill and
    // output DFBs, and the grid DFB when it is the shared grid shard, are touched by both without FIFO ops,
    // so writer 0 binds them as producer and writer 1 as consumer. With a single writer (or a grid DFB of
    // its own) a writer binds the DFB both ways.
    auto shared_dfb_bindings = [&](const DFBSpecName& dfb_name, const std::string& accessor, bool is_writer_0) {
        Group<DFBBinding> bindings;
        if (is_writer_0 || !enable_split_reader) {
            bindings.push_back(DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = accessor,
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        }
        if (!is_writer_0 || !enable_split_reader) {
            bindings.push_back(DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = accessor,
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        return bindings;
    };
    auto self_loop_bindings = [&](const DFBSpecName& dfb_name, const std::string& accessor) {
        return Group<DFBBinding>{
            DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = accessor,
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = dfb_name,
                .accessor_name = accessor,
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
    };
    auto append = [](Group<DFBBinding>& to, Group<DFBBinding> from) {
        for (auto& b : from) {
            to.push_back(std::move(b));
        }
    };

    std::vector<KernelSpec> kernels;
    {
        Group<DFBBinding> writer0_dfb_bindings;
        if (is_sharded) {
            append(writer0_dfb_bindings, shared_dfb_bindings(GRID_0, "grid", true));
        } else {
            append(writer0_dfb_bindings, self_loop_bindings(GRID_0, "grid"));
        }
        append(writer0_dfb_bindings, shared_dfb_bindings(FILL, "fill", true));
        append(writer0_dfb_bindings, shared_dfb_bindings(OUT_DFB, "out", true));
        auto writer0_compile_time_args = writer_compile_time_args;
        kernels.push_back(KernelSpec{
            .unique_id = WRITER_0,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                      "writer_grid_sample_nearest_sharded.cpp",
            .compiler_options = {.defines = writer_defines},
            .dfb_bindings = std::move(writer0_dfb_bindings),
            .tensor_bindings = writer_tensor_bindings,
            .compile_time_args = std::move(writer0_compile_time_args),
            .runtime_arg_schema = {.runtime_arg_names = writer_runtime_arg_names},
            // BRISC on NOC 0 (neither the reader nor the writer default).
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_0,
                            .noc = NOC::RISCV_0_default,
                        },
                },
        });
    }

    if (enable_split_reader) {
        Group<DFBBinding> writer1_dfb_bindings;
        if (is_sharded) {
            append(writer1_dfb_bindings, shared_dfb_bindings(GRID_0, "grid", false));
        } else {
            append(writer1_dfb_bindings, self_loop_bindings(GRID_1, "grid"));
        }
        append(writer1_dfb_bindings, shared_dfb_bindings(FILL, "fill", false));
        append(writer1_dfb_bindings, shared_dfb_bindings(OUT_DFB, "out", false));
        auto writer1_compile_time_args = writer_compile_time_args;
        writer1_compile_time_args["reader_id"] = 1U;
        kernels.push_back(KernelSpec{
            .unique_id = WRITER_1,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                      "writer_grid_sample_nearest_sharded.cpp",
            .compiler_options = {.defines = writer_defines},
            .dfb_bindings = std::move(writer1_dfb_bindings),
            .tensor_bindings = writer_tensor_bindings,
            .compile_time_args = std::move(writer1_compile_time_args),
            .runtime_arg_schema = {.runtime_arg_names = writer_runtime_arg_names},
            // NCRISC on NOC 1 (neither the reader nor the writer default).
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_1,
                            .noc = NOC::RISCV_1_default,
                        },
                },
        });
    }

    // Set runtime arguments
    KernelRunArgs writer0_run_args{.kernel = WRITER_0};
    KernelRunArgs writer1_run_args{.kernel = WRITER_1};
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader
            AddRuntimeArgsForNode(
                writer0_run_args.runtime_arg_values, core, {{"global_grid_stick_start", i * grid_nsticks_per_core}});
            if (enable_split_reader) {
                AddRuntimeArgsForNode(
                    writer1_run_args.runtime_arg_values,
                    core,
                    {{"global_grid_stick_start", i * grid_nsticks_per_core}});
            }
        }
    } else {
        uint32_t grid_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks = grid_nsticks_per_core;

            // Runtime arguments for interleaved reader
            AddRuntimeArgsForNode(writer0_run_args.runtime_arg_values, core, {{"start_page_id", grid_processed}});
            if (enable_split_reader) {
                // For split reader in nearest mode, second writer needs same runtime args
                // The kernel logic handles the split internally via reader_id
                AddRuntimeArgsForNode(writer1_run_args.runtime_arg_values, core, {{"start_page_id", grid_processed}});
            }

            grid_processed += grid_sticks;
        }
    }

    Group<KernelSpecName> wu_kernels = {WRITER_0};
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(writer0_run_args));
    if (enable_split_reader) {
        wu_kernels.push_back(WRITER_1);
        run_args.kernel_run_args.push_back(std::move(writer1_run_args));
    }
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {GRID, grid_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    ProgramSpec spec{
        .name = "grid_sample_nearest",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = GRID, .spec = grid_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "grid_sample_nearest",
                    .kernels = std::move(wu_kernels),
                    .target_nodes = all_cores,
                },
            },
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
