// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/experimental/quasar/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>
#include <ttnn/operations/experimental/quasar/grid_sample/device/grid_sample_utils.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <tt-metalium/utils/work_split.hpp>

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/utils/math.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar::rotate {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;
constexpr bool ONE_SCALAR_PER_CORE = false;

static uint16_t float_to_bfloat16(float value) {
    bfloat16 bf16_value(value);
    return std::bit_cast<uint16_t>(bf16_value);
}

// The legacy factory gave its input / scalar / output buffers an unpack face geometry
// ({face_r_dim, num_faces}). The kernel build lowered that geometry to the equivalent tile shape and
// took every unpack/pack tile parameter (face rows, face count, tile dims, tile size) from it, so
// carrying that tile as the DFB's tile metadata reproduces the same kernel metadata. Same mapping as
// the build's own conversion: tall faces stack two rows of faces, a single face is FACE_WIDTH wide.
static tt::tt_metal::Tile tile_for_face_geometry(uint32_t face_r_dim, uint32_t num_faces) {
    const uint32_t tile_height =
        face_r_dim * (num_faces > 2 ? tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = num_faces == 1 ? tt::constants::FACE_WIDTH : tt::constants::TILE_WIDTH;
    return tt::tt_metal::Tile({tile_height, tile_width});
}

ttnn::device_operation::ProgramArtifacts RotateDeviceOperation::BilinearProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    const auto input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::distributed::MeshDevice* const device = output_tensor.device();

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

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName FILL{"fill"};      // the fill stick copied in for out-of-range corners
    const DFBSpecName IN_DFB{"in"};      // reader -> compute: the 4 corner sticks of an output point
    const DFBSpecName SCALAR{"scalar"};  // reader -> compute: the bilinear weights of an output point
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};                  // any input or output sharded
    const KernelSpecName COMPUTE_GROUP_1{"compute_group_1"};  // interleaved, core group 1
    const KernelSpecName COMPUTE_GROUP_2{"compute_group_2"};  // interleaved, core group 2

    const uint32_t input_face_r_dim = REDUCTION_SIZE, input_num_faces = 2;
    const uint32_t scalar_face_r_dim = 1, scalar_num_faces = 2;
    const bool last_output_tile_is_partial = input_channels % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_output_fits_in_face =
        last_output_tile_is_partial && input_channels <= tt::constants::FACE_WIDTH;
    const uint32_t output_face_r_dim = 1, output_num_faces = single_partial_output_fits_in_face ? 1U : 2U;

    std::vector<DataflowBufferSpec> dfbs;
    const uint32_t fill_dfb_page_size = input_stick_nbytes;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = FILL,
        .entry_size = fill_dfb_page_size,
        .num_entries = 1,
        .data_format_metadata = input_dfb_data_format,
    });

    const uint32_t in_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(input_channels) / tt::constants::TILE_WIDTH));
    const uint32_t input_dfb_page_size = in_ntiles_c * tt::constants::TILE_HW * element_size;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = input_dfb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(input_face_r_dim, input_num_faces),
    });

    const uint32_t scalar_dfb_page_size = tt::tile_size(input_dfb_data_format);
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SCALAR,
        .entry_size = scalar_dfb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(scalar_face_r_dim, scalar_num_faces),
    });

    const uint32_t out_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(output_shape[-1]) / tt::constants::FACE_WIDTH));
    const uint32_t output_dfb_page_size = tt::constants::FACE_WIDTH * element_size;
    const uint32_t output_dfb_pages =
        any_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    // The legacy output buffer also carried an explicit {1, FACE_WIDTH} tile when a single partial output
    // fits in one face; that is the same tile its face geometry ({1, 1}) lowers to.
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_dfb_page_size,
        .num_entries = output_dfb_pages,
        .data_format_metadata = output_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(output_face_r_dim, output_num_faces),
        .borrowed_from = any_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    });

    const bool fill_is_zero = (fill_value_bits == 0);

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input tensor must be allocated on device for rotate operation");

    std::vector<KernelSpec> kernels;
    kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/rotate/device/kernels/dataflow/"
                  "reader_rotate_bilinear_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALAR,
                    .accessor_name = "scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The reader is the only kernel that uses the fill stick (no FIFO ops): it binds it both ways.
                DFBBinding{
                    .dfb_spec_name = FILL,
                    .accessor_name = "fill",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = FILL,
                    .accessor_name = "fill",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "input",
                },
            },
        .compile_time_args =
            {
                {"input_stick_nbytes", input_stick_nbytes},
                {"input_batch", input_batch},
                {"input_height", input_height},
                {"input_width", input_width},
                {"input_channels", input_channels},
                {"fill_is_zero", static_cast<uint32_t>(fill_is_zero)},
                {"element_size", element_size},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks",
                     "start_stick_id",
                     "cos_angle_bits",
                     "sin_angle_bits",
                     "center_x_bits",
                     "center_y_bits",
                     "fill_value_bits"},
            },
        // The legacy ReaderConfigDescriptor, i.e. the reader default.
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    const uint32_t in_nblocks_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(in_ntiles_c) / MAX_TILES_PER_REDUCTION));

    // Compute settings carried over from the legacy ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en = false,
    // math_approx_mode = false}; dst_full_sync_en was left at its default (false), so double_buffer_dest stays
    // true. No 32-bit Dest, so no unpack modes are required.
    const ComputeHardwareConfig compute_hw_config{
        .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = false,
        .double_buffer_dest = true,
        .config_1xx = ComputeHardwareConfig::Compute1XXConfig{},
    };

    // The compute kernel is the grid_sample-local Metal 2.0 copy of pool/generic's compute_pool_2d.cpp,
    // which both quasar grid_sample and quasar rotate bind. Rotate runs it without a split reader and
    // with row-major output, so only the reader-0 input / scalar DFBs and the output DFB are bound.
    auto make_compute_spec = [&](const KernelSpecName& name, uint32_t total_interpolations) {
        const auto pool_defines_map = pool::get_defines(pool::Pool2DType::AVG_POOL2D);
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN_DFB,
                .accessor_name = "in_0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = SCALAR,
                .accessor_name = "scalar_0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (any_sharded) {
            // The packed sticks land straight in the resident output shard and nothing drains them.
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        return KernelSpec{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/compute/"
                "compute_pool_2d.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(pool_defines_map),
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args =
                {
                    {"in_ntiles_c", in_ntiles_c},
                    {"window_size_hw", REDUCTION_SIZE},
                    {"split_reader", 0U},
                    {"max_out_sticks_per_core", total_interpolations},
                    {"in_c", input_channels},
                    {"in_nblocks_c", in_nblocks_c},
                    {"max_sticks_for_reduction", MAX_ROWS_FOR_REDUCTION},
                    {"one_scalar_per_core", ONE_SCALAR_PER_CORE ? 1U : 0U},
                    {"is_output_tiled", 0U},
                    {"is_output_block_format", 0U},
                    {"force_max_tiles_per_reduction_4", 0U},  // off for rotate
                },
            .hw_config = compute_hw_config,
        };
    };

    std::vector<WorkUnitSpec> work_units;
    if (any_sharded) {
        kernels.push_back(make_compute_spec(COMPUTE, output_nsticks_per_core));
        work_units.push_back(WorkUnitSpec{
            .name = "rotate_bilinear",
            .kernels = {READER, COMPUTE},
            .target_nodes = all_cores,
        });
    } else {
        if (core_group_1.num_cores() > 0) {
            kernels.push_back(make_compute_spec(COMPUTE_GROUP_1, num_sticks_per_core_group_1));
            work_units.push_back(WorkUnitSpec{
                .name = "rotate_bilinear_group_1",
                .kernels = {READER, COMPUTE_GROUP_1, WRITER},
                .target_nodes = core_group_1,
            });
        }
        if (core_group_2.num_cores() > 0) {
            kernels.push_back(make_compute_spec(COMPUTE_GROUP_2, num_sticks_per_core_group_2));
            work_units.push_back(WorkUnitSpec{
                .name = "rotate_bilinear_group_2",
                .kernels = {READER, COMPUTE_GROUP_2, WRITER},
                .target_nodes = core_group_2,
            });
        }
    }

    if (!any_sharded) {
        const uint32_t output_stick_size = input_channels * element_size;
        auto* output_buffer = output_tensor.buffer();
        TT_FATAL(output_buffer != nullptr, "Output tensor must be allocated on device for rotate operation");

        // The quasar grid_sample writer (shared with that op).
        kernels.push_back(KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                      "writer_grid_sample_interleaved.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = OUTPUT,
                        .accessor_name = "output",
                    },
                },
            .compile_time_args =
                {
                    {"output_stick_size", output_stick_size},
                    {"ntiles_c", out_ntiles_c},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_to_write", "start_stick_id"}},
            // The legacy WriterConfigDescriptor, i.e. the writer default.
            .hw_config = ttnn::create_writer_datamovement_config(),
        });
    }

    const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
    const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
    const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
    const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
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
            reader_run_args.runtime_arg_values,
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

        if (!any_sharded) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"num_sticks_to_write", num_sticks},
                    {"start_stick_id", start_stick_id},
                });
        }

        sticks_processed += num_sticks;
    }

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    if (!any_sharded) {
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    ProgramSpec spec{
        .name = "rotate_bilinear",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::quasar::rotate
