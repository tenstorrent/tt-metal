// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DataMovementGen2Config;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProducerOf;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

ttnn::device_operation::ProgramArtifacts GridSampleNearestProgramFactory::create_program_artifacts(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    const bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    const bool is_sharded = grid_tensor.is_sharded();

    const auto grid_cb_data_format = datatype_to_dataformat_converter(grid_tensor.dtype());
    const auto output_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& grid_shape = grid_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t grid_hw = grid_shape[1] * grid_shape[2];
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid, "nearest");
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "nearest");

    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores;
    uint32_t grid_nsticks_per_core;
    uint32_t output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0;
    uint32_t num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        all_cores = grid_shard_spec.grid;
        num_cores = grid_shard_spec.num_cores();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores =
            corerange_to_cores(all_cores, num_cores, grid_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        uint32_t grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];
        if (output_tensor.shard_spec().has_value()) {
            grid_nsticks = output_tensor.shard_spec().value().shape[0] * output_tensor.shard_spec().value().num_cores();
        } else {
            grid_nsticks = tt::round_up(grid_nsticks, compute_grid_size.x * compute_grid_size.y);
        }
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            split_work_to_cores(compute_grid_size, grid_nsticks);
        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;
        output_nsticks_per_core = num_sticks_1;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const TensorParamName INPUT{"input"};
    const TensorParamName GRID{"grid"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName GRID0{"grid0"};
    const DFBSpecName GRID1{"grid1"};
    const DFBSpecName FILL{"fill"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const KernelSpecName WRITER0{"writer0"};
    const KernelSpecName WRITER1{"writer1"};

    const uint32_t grid_stick_size =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const DataflowBufferSpec grid0_dfb{
        .unique_id = GRID0,
        .entry_size = grid_stick_size,
        .num_entries = is_sharded ? grid_nsticks_per_core : 1,
        .data_format_metadata = grid_cb_data_format,
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{GRID} : std::nullopt,
    };
    const std::optional<DataflowBufferSpec> grid1_dfb = enable_split_reader && !is_sharded
                                                            ? std::optional{DataflowBufferSpec{
                                                                  .unique_id = GRID1,
                                                                  .entry_size = grid_stick_size,
                                                                  .num_entries = 1,
                                                                  .data_format_metadata = grid_cb_data_format,
                                                              }}
                                                            : std::nullopt;

    const uint32_t output_cb_page_size =
        static_cast<uint32_t>(static_cast<float>(output_shape[-1]) * output_tensor.element_size());
    const DataflowBufferSpec fill_dfb{
        .unique_id = FILL,
        .entry_size = output_cb_page_size,
        .num_entries = 1,
        .data_format_metadata = output_cb_data_format,
    };
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = output_nsticks_per_core,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT,
    };

    const KernelSpec::CompileTimeArgs common_cta{
        {"input_stick_nbytes", get_aligned_stick_size(input_shape, input_tensor)},
        {"grid_stick_nbytes", grid_stick_size},
        {"input_height", input_height},
        {"input_width", input_width},
        {"grid_batching_factor", grid_batching_factor},
        {"grid_dtype", static_cast<uint32_t>(grid_tensor.dtype())},
        {"grid_hw", grid_hw},
        {"use_precomputed_grid", use_precomputed_grid ? 1U : 0U},
        {"align_corners", operation_attributes.align_corners ? 1U : 0U},
        {"split_reader", enable_split_reader ? 1U : 0U},
        {"reader_id", 0U},
        {"grid_nsticks_per_core", grid_nsticks_per_core},
        {"batch_size", input_shape[0]},
    };

    const auto make_hw_config = [&](DataMovementProcessor processor, NOC noc) -> DataMovementHardwareConfig {
        if (device->arch() == tt::ARCH::QUASAR) {
            return DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
        }
        return DataMovementGen1Config{.processor = processor, .noc = noc};
    };

    const std::string kernel_source = is_sharded ? "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                                                   "writer_grid_sample_nearest_sharded.cpp"
                                                 : "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                                                   "writer_grid_sample_nearest_interleaved.cpp";

    Group<DFBBinding> writer0_bindings;
    Group<DFBBinding> writer1_bindings;
    if (enable_split_reader) {
        if (is_sharded) {
            writer0_bindings.push_back(ProducerOf(GRID0, "grid"));
            writer1_bindings.push_back(ConsumerOf(GRID0, "grid"));
        } else {
            writer0_bindings.push_back(ProducerOf(GRID0, "grid"));
            writer0_bindings.push_back(ConsumerOf(GRID0, "grid"));
            writer1_bindings.push_back(ProducerOf(GRID1, "grid"));
            writer1_bindings.push_back(ConsumerOf(GRID1, "grid"));
        }
        writer0_bindings.push_back(ProducerOf(OUTPUT_DFB, "output"));
        writer1_bindings.push_back(ConsumerOf(OUTPUT_DFB, "output"));
        writer0_bindings.push_back(ProducerOf(FILL, "fill"));
        writer1_bindings.push_back(ConsumerOf(FILL, "fill"));
    } else {
        writer0_bindings = {
            ProducerOf(GRID0, "grid"),
            ConsumerOf(GRID0, "grid"),
            ProducerOf(OUTPUT_DFB, "output"),
            ConsumerOf(OUTPUT_DFB, "output"),
            ProducerOf(FILL, "fill"),
            ConsumerOf(FILL, "fill"),
        };
    }

    const Group<TensorBinding> tensor_bindings =
        is_sharded ? Group<TensorBinding>{TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}}
                   : Group<TensorBinding>{
                         TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                         TensorBinding{.tensor_parameter_name = GRID, .accessor_name = "grid"},
                     };
    KernelSpec writer0{
        .unique_id = WRITER0,
        .source = kernel_source,
        .dfb_bindings = std::move(writer0_bindings),
        .tensor_bindings = tensor_bindings,
        .compile_time_args = common_cta,
        .runtime_arg_schema = {.runtime_arg_names = {is_sharded ? "global_grid_stick_start" : "start_page_id"}},
        .hw_config = make_hw_config(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    std::optional<KernelSpec> writer1;
    if (enable_split_reader) {
        auto writer1_cta = common_cta;
        writer1_cta["reader_id"] = 1U;
        writer1 = KernelSpec{
            .unique_id = WRITER1,
            .source = kernel_source,
            .dfb_bindings = std::move(writer1_bindings),
            .tensor_bindings = tensor_bindings,
            .compile_time_args = std::move(writer1_cta),
            .runtime_arg_schema = {.runtime_arg_names = {is_sharded ? "global_grid_stick_start" : "start_page_id"}},
            .hw_config = make_hw_config(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
        };
    }

    KernelRunArgs writer0_run{.kernel = WRITER0};
    KernelRunArgs writer1_run{.kernel = WRITER1};
    uint32_t grid_processed = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = logical_cores[i];
        if (is_sharded) {
            const uint32_t start = i * grid_nsticks_per_core;
            AddRuntimeArgsForNode(writer0_run.runtime_arg_values, core, {{"global_grid_stick_start", start}});
            if (writer1.has_value()) {
                AddRuntimeArgsForNode(writer1_run.runtime_arg_values, core, {{"global_grid_stick_start", start}});
            }
        } else {
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            AddRuntimeArgsForNode(writer0_run.runtime_arg_values, core, {{"start_page_id", grid_processed}});
            if (writer1.has_value()) {
                AddRuntimeArgsForNode(writer1_run.runtime_arg_values, core, {{"start_page_id", grid_processed}});
            }
            grid_processed += grid_sticks;
        }
    }

    ProgramSpec spec{
        .name = "grid_sample_nearest",
        .kernels = {writer0},
        .dataflow_buffers = {grid0_dfb, fill_dfb, output_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = GRID, .spec = grid_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
    };
    if (grid1_dfb.has_value()) {
        spec.dataflow_buffers.push_back(*grid1_dfb);
    }
    Group<KernelSpecName> work_unit_kernels{WRITER0};
    if (writer1.has_value()) {
        spec.kernels.push_back(*writer1);
        work_unit_kernels.push_back(WRITER1);
    }
    spec.work_units.push_back(WorkUnitSpec{
        .name = "grid_sample_nearest", .kernels = std::move(work_unit_kernels), .target_nodes = all_cores});

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(writer0_run)},
        .tensor_args =
            {
                {INPUT, input_tensor.mesh_tensor()},
                {GRID, grid_tensor.mesh_tensor()},
                {OUTPUT, output_tensor.mesh_tensor()},
            },
    };
    if (writer1.has_value()) {
        run_args.kernel_run_args.push_back(std::move(writer1_run));
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
