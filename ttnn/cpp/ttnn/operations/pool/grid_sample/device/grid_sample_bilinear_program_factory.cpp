// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include "grid_sample_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
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

ttnn::device_operation::ProgramArtifacts GridSampleBilinearProgramFactory::create_program_artifacts(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    const bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    const bool batch_output_channels = operation_attributes.batch_output_channels;
    const bool is_sharded = grid_tensor.is_sharded();

    const auto input_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto grid_cb_data_format = datatype_to_dataformat_converter(grid_tensor.dtype());
    const auto output_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& grid_shape = grid_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t grid_hw = grid_shape[1] * grid_shape[2];
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid);
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "bilinear");

    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores;
    uint32_t grid_nsticks_per_core;
    uint32_t output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0;
    uint32_t num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;
    const uint32_t total_grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        num_cores = grid_shard_spec.num_cores();
        all_cores = grid_shard_spec.grid;
        logical_cores =
            corerange_to_cores(all_cores, num_cores, grid_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            split_work_to_cores(compute_grid_size, total_grid_nsticks);
        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const bool resolved_fp32_acc = operation_attributes.compute_kernel_config.fp32_dest_acc_en;
    const bool user_dst_full_sync = operation_attributes.compute_kernel_config.dst_full_sync_en;
    // Half-sync DEST holds four FP32 tiles, while explicit full sync holds eight. Clamp only
    // the former so FP32 accumulation remains safe without penalizing callers that requested
    // dst_full_sync_en. The compute kernel receives the same decision as a named argument.
    const bool force_4_tile_chunk = resolved_fp32_acc && !user_dst_full_sync;
    const uint32_t effective_max_tiles_per_reduction = force_4_tile_chunk ? 4U : MAX_TILES_PER_REDUCTION;
    const uint32_t in_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(input_shape[-1]) / tt::constants::TILE_WIDTH));
    const uint32_t max_tiles_per_iter = std::min<uint32_t>(in_ntiles_c, effective_max_tiles_per_reduction);
    const uint32_t in_nblocks_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(in_ntiles_c) / effective_max_tiles_per_reduction));
    const uint32_t input_chunk_nbytes =
        effective_max_tiles_per_reduction * tt::constants::TILE_WIDTH * input_tensor.element_size();
    const uint32_t input_cb_page_size = max_tiles_per_iter * tt::constants::TILE_HW * input_tensor.element_size();
    const bool last_tile_is_partial = input_shape[-1] % tt::constants::TILE_WIDTH != 0;
    // Wide channel sticks are streamed one reduction chunk at a time to bound per-core L1.
    // A short final whole-tile chunk uses a tight byte stride so the reader layout matches
    // compute's tilize-reconfiguration path. This requires !last_tile_is_partial, guaranteed by the
    // padded-width % TILE_WIDTH validation in grid_sample_device_operation.cpp, and window_size_hw
    // (REDUCTION_SIZE = 4) <= FACE_HEIGHT, which is a static property of bilinear interpolation.
    const bool last_chunk_partial =
        in_nblocks_c > 1 && in_ntiles_c % effective_max_tiles_per_reduction != 0 && !last_tile_is_partial;

    const TensorParamName INPUT{"input"};
    const TensorParamName GRID{"grid"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName GRID_DFB{"grid"};
    const DFBSpecName INPUT0{"input0"};
    const DFBSpecName INPUT1{"input1"};
    const DFBSpecName SCALAR0{"scalar0"};
    const DFBSpecName SCALAR1{"scalar1"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const KernelSpecName READER0{"reader0"};
    const KernelSpecName READER1{"reader1"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const KernelSpecName WRITER{"writer"};

    const uint32_t grid_stick_size =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const DataflowBufferSpec grid_dfb{
        .unique_id = GRID_DFB,
        .entry_size = grid_stick_size,
        .num_entries = is_sharded ? grid_nsticks_per_core : 1,
        .data_format_metadata = grid_cb_data_format,
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{GRID} : std::nullopt,
    };

    const FaceGeometry input_face_geometry{.face_r_dim = REDUCTION_SIZE, .num_faces = 2};
    const FaceGeometry scalar_face_geometry{.face_r_dim = 1, .num_faces = 2};
    const DataflowBufferSpec input0_dfb{
        .unique_id = INPUT0,
        .entry_size = input_cb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = input_face_geometry,
    };
    const std::optional<DataflowBufferSpec> input1_dfb = is_sharded && enable_split_reader
                                                             ? std::optional{DataflowBufferSpec{
                                                                   .unique_id = INPUT1,
                                                                   .entry_size = input_cb_page_size,
                                                                   .num_entries = BUFFERING_FACTOR,
                                                                   .data_format_metadata = input_cb_data_format,
                                                                   .unpack_face_geometry_metadata = input_face_geometry,
                                                               }}
                                                             : std::nullopt;

    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
    const DataflowBufferSpec scalar0_dfb{
        .unique_id = SCALAR0,
        .entry_size = scalar_cb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = scalar_face_geometry,
    };
    const std::optional<DataflowBufferSpec> scalar1_dfb =
        is_sharded && enable_split_reader ? std::optional{DataflowBufferSpec{
                                                .unique_id = SCALAR1,
                                                .entry_size = scalar_cb_page_size,
                                                .num_entries = BUFFERING_FACTOR,
                                                .data_format_metadata = input_cb_data_format,
                                                .unpack_face_geometry_metadata = scalar_face_geometry,
                                            }}
                                          : std::nullopt;

    const bool last_output_tile_is_partial = input_shape[-1] % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_output_fits_in_face =
        last_output_tile_is_partial && input_shape[-1] <= tt::constants::FACE_WIDTH;
    const FaceGeometry output_face_geometry{.face_r_dim = 1, .num_faces = single_partial_output_fits_in_face ? 1U : 2U};
    const std::optional<tt::tt_metal::Tile> output_tile =
        single_partial_output_fits_in_face ? std::optional{tt::tt_metal::Tile({1, tt::constants::FACE_WIDTH}, false)}
                                           : std::nullopt;
    const uint32_t out_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(output_shape[-1]) / tt::constants::FACE_WIDTH));
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = is_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR,
        .data_format_metadata = output_cb_data_format,
        .tile_format_metadata = output_tile,
        .unpack_face_geometry_metadata = output_face_geometry,
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    };

    const KernelSpec::CompileTimeArgs common_reader_cta{
        {"input_stick_nbytes", get_aligned_stick_size(input_shape, input_tensor)},
        {"grid_stick_nbytes", grid_stick_size},
        {"input_height", input_height},
        {"input_width", input_width},
        {"grid_batching_factor", grid_batching_factor},
        {"grid_dtype", static_cast<uint32_t>(grid_tensor.dtype())},
        {"grid_hw", grid_hw},
        {"use_precomputed_grid", use_precomputed_grid ? 1U : 0U},
        {"align_corners", operation_attributes.align_corners ? 1U : 0U},
        {"in_nblocks_c", in_nblocks_c},
        {"input_chunk_nbytes", input_chunk_nbytes},
        {"last_chunk_partial", last_chunk_partial ? 1U : 0U},
    };

    const auto make_split_hw = [&](DataMovementProcessor processor, NOC noc) -> DataMovementHardwareConfig {
        if (device->arch() == tt::ARCH::QUASAR) {
            return DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
        }
        return DataMovementGen1Config{.processor = processor, .noc = noc};
    };

    KernelSpec reader0;
    std::optional<KernelSpec> reader1;
    if (is_sharded) {
        auto reader0_cta = common_reader_cta;
        reader0_cta["input_batch"] = input_batch;
        reader0_cta["split_reader"] = enable_split_reader ? 1U : 0U;
        reader0_cta["reader_id"] = 0U;
        reader0_cta["grid_nsticks_per_core"] = grid_nsticks_per_core;
        Group<DFBBinding> reader0_bindings;
        if (enable_split_reader) {
            reader0_bindings.push_back(ProducerOf(GRID_DFB, "grid"));
        } else {
            reader0_bindings.push_back(ProducerOf(GRID_DFB, "grid"));
            reader0_bindings.push_back(ConsumerOf(GRID_DFB, "grid"));
        }
        reader0_bindings.push_back(ProducerOf(INPUT0, "input"));
        reader0_bindings.push_back(ProducerOf(SCALAR0, "scalar"));
        reader0 = KernelSpec{
            .unique_id = READER0,
            .source =
                "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp",
            .dfb_bindings = std::move(reader0_bindings),
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
            .compile_time_args = std::move(reader0_cta),
            .runtime_arg_schema = {.runtime_arg_names = {"global_grid_stick_start"}},
            .hw_config = make_split_hw(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
        };

        if (enable_split_reader) {
            auto reader1_cta = common_reader_cta;
            reader1_cta["input_batch"] = input_batch;
            reader1_cta["split_reader"] = 1U;
            reader1_cta["reader_id"] = 1U;
            reader1_cta["grid_nsticks_per_core"] = grid_nsticks_per_core;
            reader1 = KernelSpec{
                .unique_id = READER1,
                .source =
                    "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                    "reader_grid_sample_sharded.cpp",
                .dfb_bindings =
                    {
                        ConsumerOf(GRID_DFB, "grid"),
                        ProducerOf(INPUT1, "input"),
                        ProducerOf(SCALAR1, "scalar"),
                    },
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
                .compile_time_args = std::move(reader1_cta),
                .runtime_arg_schema = {.runtime_arg_names = {"global_grid_stick_start"}},
                .hw_config = make_split_hw(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
            };
        }
    } else {
        auto interleaved_reader_cta = common_reader_cta;
        interleaved_reader_cta.erase("grid_batching_factor");
        interleaved_reader_cta.erase("grid_hw");
        interleaved_reader_cta["grid_batches"] = grid_batching_factor;
        interleaved_reader_cta["output_hw_size"] = grid_hw;
        reader0 = KernelSpec{
            .unique_id = READER0,
            .source =
                "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                "reader_grid_sample_interleaved_start_id.cpp",
            .dfb_bindings =
                {
                    ProducerOf(GRID_DFB, "grid"),
                    ConsumerOf(GRID_DFB, "grid"),
                    ProducerOf(INPUT0, "input"),
                    ProducerOf(SCALAR0, "scalar"),
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                    TensorBinding{.tensor_parameter_name = GRID, .accessor_name = "grid"},
                },
            .compile_time_args = std::move(interleaved_reader_cta),
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_page_id"}},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };
    }

    const auto make_compute = [&](const KernelSpecName& name, uint32_t total_interpolations) {
        Group<DFBBinding> bindings;
        std::string source;
        if (enable_split_reader) {
            source =
                "ttnn/cpp/ttnn/operations/pool/device/kernels/compute/"
                "pool_2d_bilinear_split.cpp";
            bindings = {
                ConsumerOf(INPUT0, "input0"),
                ConsumerOf(INPUT1, "input1"),
                ConsumerOf(SCALAR0, "scalar0"),
                ConsumerOf(SCALAR1, "scalar1"),
            };
        } else {
            source = "ttnn/cpp/ttnn/operations/pool/device/kernels/compute/pool_2d_bilinear.cpp";
            bindings = {ConsumerOf(INPUT0, "input"), ConsumerOf(SCALAR0, "scalar")};
        }
        bindings.push_back(ProducerOf(OUTPUT_DFB, "output"));
        if (is_sharded) {
            bindings.push_back(ConsumerOf(OUTPUT_DFB, "output"));
        }
        return KernelSpec{
            .unique_id = name,
            .source = std::move(source),
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(bindings),
            .compile_time_args =
                {
                    {"in_ntiles_c", in_ntiles_c},
                    {"window_size_hw", REDUCTION_SIZE},
                    {"max_out_sticks_per_core", total_interpolations},
                    {"in_c", input_shape[-1]},
                    {"in_nblocks_c", in_nblocks_c},
                    {"max_sticks_for_reduction", MAX_ROWS_FOR_REDUCTION},
                    {"one_scalar_per_core", ONE_SCALAR_PER_CORE ? 1U : 0U},
                    {"force_max_tiles_per_reduction_4", force_4_tile_chunk ? 1U : 0U},
                },
            .hw_config = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
        };
    };

    std::optional<KernelSpec> compute_g1;
    std::optional<KernelSpec> compute_g2;
    if (is_sharded || core_group_1.num_cores() > 0) {
        compute_g1 = make_compute(
            COMPUTE_G1, grid_batching_factor * (is_sharded ? grid_nsticks_per_core : num_sticks_per_core_group_1));
    }
    if (!is_sharded && core_group_2.num_cores() > 0) {
        compute_g2 = make_compute(COMPUTE_G2, grid_batching_factor * num_sticks_per_core_group_2);
    }

    std::optional<KernelSpec> writer;
    if (!is_sharded) {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/pool/device/kernels/dataflow/"
                "writer_pool_stick_interleaved.cpp",
            .dfb_bindings = {ConsumerOf(OUTPUT_DFB, "output")},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args =
                {
                    {"output_stick_size", get_aligned_stick_size(output_shape, output_tensor)},
                    {"ntiles_c", out_ntiles_c},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    }

    KernelRunArgs reader0_run{.kernel = READER0};
    KernelRunArgs reader1_run{.kernel = READER1};
    KernelRunArgs writer_run{.kernel = WRITER};
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t start = i * grid_nsticks_per_core;
            AddRuntimeArgsForNode(reader0_run.runtime_arg_values, core, {{"global_grid_stick_start", start}});
            if (reader1.has_value()) {
                AddRuntimeArgsForNode(reader1_run.runtime_arg_values, core, {{"global_grid_stick_start", start}});
            }
        }
    } else {
        uint32_t grid_processed = 0;
        uint32_t output_processed = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            const uint32_t output_sticks = batch_output_channels ? grid_sticks : grid_sticks * grid_batching_factor;
            AddRuntimeArgsForNode(
                reader0_run.runtime_arg_values, core, {{"num_pages", grid_sticks}, {"start_page_id", grid_processed}});
            AddRuntimeArgsForNode(
                writer_run.runtime_arg_values,
                core,
                {{"num_sticks", output_sticks}, {"start_stick_id", output_processed}});
            grid_processed += grid_sticks;
            output_processed += output_sticks;
        }
    }

    ProgramSpec spec{
        .name = "grid_sample_bilinear",
        .kernels = {reader0},
        .dataflow_buffers = {grid_dfb, input0_dfb, scalar0_dfb, output_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = GRID, .spec = grid_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
    };
    if (input1_dfb.has_value()) {
        spec.dataflow_buffers.push_back(*input1_dfb);
    }
    if (scalar1_dfb.has_value()) {
        spec.dataflow_buffers.push_back(*scalar1_dfb);
    }
    if (reader1.has_value()) {
        spec.kernels.push_back(*reader1);
    }
    if (compute_g1.has_value()) {
        spec.kernels.push_back(*compute_g1);
        Group<KernelSpecName> kernels{READER0, COMPUTE_G1};
        if (reader1.has_value()) {
            kernels.push_back(READER1);
        }
        if (writer.has_value()) {
            kernels.push_back(WRITER);
        }
        spec.work_units.push_back(WorkUnitSpec{
            .name = "grid_sample_bilinear_g1",
            .kernels = std::move(kernels),
            .target_nodes = is_sharded ? all_cores : core_group_1,
        });
    }
    if (compute_g2.has_value()) {
        spec.kernels.push_back(*compute_g2);
        Group<KernelSpecName> kernels{READER0, COMPUTE_G2};
        if (writer.has_value()) {
            kernels.push_back(WRITER);
        }
        spec.work_units.push_back(WorkUnitSpec{
            .name = "grid_sample_bilinear_g2",
            .kernels = std::move(kernels),
            .target_nodes = core_group_2,
        });
    }
    if (writer.has_value()) {
        spec.kernels.push_back(*writer);
    }

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader0_run)},
        .tensor_args =
            {
                {INPUT, input_tensor.mesh_tensor()},
                {GRID, grid_tensor.mesh_tensor()},
                {OUTPUT, output_tensor.mesh_tensor()},
            },
    };
    if (reader1.has_value()) {
        run_args.kernel_run_args.push_back(std::move(reader1_run));
    }
    if (writer.has_value()) {
        run_args.kernel_run_args.push_back(std::move(writer_run));
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
