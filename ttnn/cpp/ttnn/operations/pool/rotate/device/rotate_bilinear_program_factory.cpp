// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::rotate {

using namespace tt;
using namespace tt::tt_metal;
using tt::tt_metal::experimental::ComputeGen1Config;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;
constexpr bool ONE_SCALAR_PER_CORE = false;

static uint16_t float_to_bfloat16(float value) {
    bfloat16 bf16_value(value);
    return std::bit_cast<uint16_t>(bf16_value);
}

ttnn::device_operation::ProgramArtifacts RotateDeviceOperation::BilinearProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t input_channels = input_shape[3];

    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    const bool is_bfloat16 = input_tensor.dtype() == DataType::BFLOAT16;
    uint32_t fill_value_bits;
    if (is_bfloat16) {
        fill_value_bits = float_to_bfloat16(operation_attributes.fill);
    } else {
        fill_value_bits = std::bit_cast<uint32_t>(operation_attributes.fill);
    }

    const uint32_t total_output_sticks = input_batch * input_height * input_width;
    const bool is_input_sharded = input_tensor.is_sharded();
    const bool is_output_sharded = output_tensor.is_sharded();

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    uint32_t input_nsticks_per_core = 0;
    uint32_t output_nsticks_per_core = 0;
    std::vector<CoreCoord> logical_cores;
    bool is_block_sharded = false;
    bool is_width_sharded = false;
    uint32_t num_cores_x = 0;

    if (is_input_sharded) {
        const auto input_shard_spec = input_tensor.shard_spec().value();
        all_cores = input_shard_spec.grid;
        num_cores = input_shard_spec.num_cores();
        input_nsticks_per_core = input_shard_spec.shape[0];
        output_nsticks_per_core =
            is_output_sharded ? output_tensor.shard_spec().value().shape[0] : input_nsticks_per_core;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        is_block_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        is_width_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        num_cores_x = input_shard_spec.grid.bounding_box().grid_size().x;
        num_sticks_per_core_group_1 = input_nsticks_per_core;
        core_group_1 = all_cores;
    } else if (is_output_sharded) {
        const auto output_shard_spec = output_tensor.shard_spec().value();
        output_nsticks_per_core = output_shard_spec.shape[0];
        num_cores = (total_output_sticks + output_nsticks_per_core - 1) / output_nsticks_per_core;
        all_cores = output_shard_spec.grid;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        num_sticks_per_core_group_1 = output_nsticks_per_core;
        core_group_1 = all_cores;
    } else {
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

        num_cores = num_cores_used;
        all_cores = all_cores_range;
        core_group_1 = core_group_1_range;
        core_group_2 = core_group_2_range;
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const bool any_sharded = is_input_sharded || is_output_sharded;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    const auto input_face_geometry = FaceGeometry{.face_r_dim = REDUCTION_SIZE, .num_faces = 2};
    const auto scalar_face_geometry = FaceGeometry{.face_r_dim = 1, .num_faces = 2};
    const bool last_output_tile_is_partial = input_channels % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_output_fits_in_face =
        last_output_tile_is_partial && input_channels <= tt::constants::FACE_WIDTH;
    const auto output_face_geometry =
        FaceGeometry{.face_r_dim = 1, .num_faces = single_partial_output_fits_in_face ? 1U : 2U};
    const std::optional<tt::tt_metal::Tile> output_tile =
        single_partial_output_fits_in_face ? std::optional{tt::tt_metal::Tile({1, tt::constants::FACE_WIDTH}, false)}
                                           : std::nullopt;

    const uint32_t fill_cb_page_size = input_stick_nbytes;

    const uint32_t in_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(input_channels) / tt::constants::TILE_WIDTH));
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * element_size;

    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);

    const uint32_t out_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(output_shape[-1]) / tt::constants::FACE_WIDTH));
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * element_size;
    const uint32_t output_cb_pages =
        any_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    const bool fill_is_zero = (fill_value_bits == 0);

    const uint32_t in_nblocks_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(in_ntiles_c) / MAX_TILES_PER_REDUCTION));

    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName FILL{"fill"};
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName SCALAR{"scalar"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const KernelSpecName WRITER{"writer"};

    const DataflowBufferSpec fill_dfb{
        .unique_id = FILL,
        .entry_size = fill_cb_page_size,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };
    const DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_cb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = input_face_geometry,
    };
    const DataflowBufferSpec scalar_dfb{
        .unique_id = SCALAR,
        .entry_size = scalar_cb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = scalar_face_geometry,
    };
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = output_cb_pages,
        .data_format_metadata = output_cb_data_format,
        .tile_format_metadata = output_tile,
        .unpack_face_geometry_metadata = output_face_geometry,
        .borrowed_from = any_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    };

    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp",
        // Gen1 compatibility: the reader both produces and consumes FILL. A Gen2 port must use scratchpad storage or
        // a LocalTensorAccessor instead of this data-movement-kernel self-loop.
        .dfb_bindings =
            {ProducerOf(INPUT_DFB, "input"),
             ProducerOf(SCALAR, "scalar"),
             ProducerOf(FILL, "fill"),
             ConsumerOf(FILL, "fill")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"input_stick_nbytes", input_stick_nbytes},
                {"input_height", input_height},
                {"input_width", input_width},
                {"input_channels", input_channels},
                {"fill_is_zero", static_cast<uint32_t>(fill_is_zero)},
                {"element_size", element_size},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks",
                  "start_stick_id",
                  "cos_angle_bits",
                  "sin_angle_bits",
                  "center_x_bits",
                  "center_y_bits",
                  "fill_value_bits"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    const auto make_compute_kernel = [&](const KernelSpecName& name, uint32_t total_interpolations) {
        Group<DFBBinding> bindings{
            ConsumerOf(INPUT_DFB, "input"), ConsumerOf(SCALAR, "scalar"), ProducerOf(OUTPUT_DFB, "output")};
        if (any_sharded) {
            bindings.push_back(ConsumerOf(OUTPUT_DFB, "output"));
        }
        return KernelSpec{
            .unique_id = name,
            .source = "ttnn/cpp/ttnn/operations/pool/device/kernels/compute/pool_2d_bilinear.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(bindings),
            .compile_time_args =
                {
                    {"in_ntiles_c", in_ntiles_c},
                    {"window_size_hw", REDUCTION_SIZE},
                    {"max_out_sticks_per_core", total_interpolations},
                    {"in_c", input_channels},
                    {"in_nblocks_c", in_nblocks_c},
                    {"max_sticks_for_reduction", MAX_ROWS_FOR_REDUCTION},
                    {"one_scalar_per_core", ONE_SCALAR_PER_CORE ? 1U : 0U},
                    {"force_max_tiles_per_reduction_4", 0U},
                },
            .hw_config =
                ComputeGen1Config{
                    .fpu_math_fidelity = MathFidelity::HiFi4,
                    .sfpu_precision_mode = Precision::Precise,
                },
        };
    };

    std::optional<KernelSpec> compute_g1;
    std::optional<KernelSpec> compute_g2;
    if (any_sharded || core_group_1.num_cores() > 0) {
        compute_g1 =
            make_compute_kernel(COMPUTE_G1, any_sharded ? output_nsticks_per_core : num_sticks_per_core_group_1);
    }
    if (!any_sharded && core_group_2.num_cores() > 0) {
        compute_g2 = make_compute_kernel(COMPUTE_G2, num_sticks_per_core_group_2);
    }

    std::optional<KernelSpec> writer;
    if (!any_sharded) {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/pool/device/kernels/dataflow/"
                "writer_pool_stick_interleaved.cpp",
            .dfb_bindings = {ConsumerOf(OUTPUT_DFB, "output")},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args =
                {
                    {"output_stick_size", input_channels * element_size},
                    {"ntiles_c", out_ntiles_c},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    }

    const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
    const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
    const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
    const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        uint32_t num_sticks;
        uint32_t start_stick_id;

        if (is_input_sharded) {
            num_sticks = input_nsticks_per_core;
            if (is_width_sharded) {
                start_stick_id = 0;
            } else if (is_block_sharded) {
                uint32_t core_y = i / num_cores_x;
                start_stick_id = core_y * input_nsticks_per_core;
            } else {
                start_stick_id = i * input_nsticks_per_core;
            }
        } else if (is_output_sharded) {
            num_sticks = output_nsticks_per_core;
            start_stick_id = sticks_processed;
        } else {
            num_sticks = core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            start_stick_id = sticks_processed;
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {
                {"num_sticks", num_sticks},
                {"start_stick_id", start_stick_id},
                {"cos_angle_bits", static_cast<uint32_t>(cos_angle_q16)},
                {"sin_angle_bits", static_cast<uint32_t>(sin_angle_q16)},
                {"center_x_bits", static_cast<uint32_t>(center_x_q16)},
                {"center_y_bits", static_cast<uint32_t>(center_y_q16)},
                {"fill_value_bits", static_cast<uint32_t>(fill_value_bits)},
            });

        if (!any_sharded && writer.has_value()) {
            AddRuntimeArgsForNode(
                writer_run.runtime_arg_values, core, {{"num_sticks", num_sticks}, {"start_stick_id", start_stick_id}});
        }

        sticks_processed += num_sticks;
    }

    ProgramSpec spec{
        .name = "rotate_bilinear",
        .kernels = {reader},
        .dataflow_buffers = {fill_dfb, input_dfb, scalar_dfb, output_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
    };
    if (compute_g1.has_value()) {
        spec.kernels.push_back(*compute_g1);
        Group<KernelSpecName> kernels{READER, COMPUTE_G1};
        if (writer.has_value()) {
            kernels.push_back(WRITER);
        }
        spec.work_units.push_back(WorkUnitSpec{
            .name = "rotate_g1",
            .kernels = std::move(kernels),
            .target_nodes = any_sharded ? all_cores : core_group_1,
        });
    }
    if (compute_g2.has_value()) {
        spec.kernels.push_back(*compute_g2);
        Group<KernelSpecName> kernels{READER, COMPUTE_G2};
        if (writer.has_value()) {
            kernels.push_back(WRITER);
        }
        spec.work_units.push_back(WorkUnitSpec{
            .name = "rotate_g2",
            .kernels = std::move(kernels),
            .target_nodes = core_group_2,
        });
    }
    if (writer.has_value()) {
        spec.kernels.push_back(*writer);
    }

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run)},
        .tensor_args = {{INPUT, input_tensor.mesh_tensor()}, {OUTPUT, output_tensor.mesh_tensor()}},
    };
    if (writer.has_value()) {
        run_args.kernel_run_args.push_back(std::move(writer_run));
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::rotate
