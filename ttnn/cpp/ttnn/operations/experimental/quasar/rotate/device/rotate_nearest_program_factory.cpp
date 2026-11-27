// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/experimental/quasar/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>

#include <cmath>
#include <cstdint>
#include <tt-metalium/utils/work_split.hpp>

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/utils/math.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar::rotate {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NEAREST_BUFFERING_FACTOR = 2;
constexpr uint32_t NUM_TILES_DEST = 8;
constexpr uint32_t MAX_BURST_SIZE = 5;

// Helper to convert float to bfloat16 representation using tie-to-even rounding (matches PyTorch)
static uint16_t nearest_float_to_bfloat16(float value) {
    bfloat16 bf16_value(value);
    return std::bit_cast<uint16_t>(bf16_value);
}

ttnn::device_operation::ProgramArtifacts RotateDeviceOperation::NearestProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    const bool is_sharded = input_tensor.is_sharded();

    const auto output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::distributed::MeshDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
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
        fill_value_bits = nearest_float_to_bfloat16(operation_attributes.fill);
    } else {
        fill_value_bits = std::bit_cast<uint32_t>(operation_attributes.fill);
    }
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1, core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;
    uint32_t input_nsticks_per_core = 0;
    uint32_t output_nsticks_per_core = 0;
    bool is_block_sharded = false;
    bool is_width_sharded = false;
    uint32_t num_cores_x = 0;
    uint32_t shard_width = 0;

    const bool is_nd_sharded = input_tensor.memory_config().nd_shard_spec().has_value();

    if (is_sharded && !is_nd_sharded) {
        const auto input_shard_spec = input_tensor.shard_spec().value();
        all_cores = input_shard_spec.grid;
        num_cores = input_shard_spec.num_cores();
        input_nsticks_per_core = input_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        is_block_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        is_width_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        TT_FATAL(!is_width_sharded, "Width sharding is not supported for rotate operation");
        num_cores_x = input_shard_spec.grid.bounding_box().grid_size().x;
        shard_width = input_shard_spec.shape[1];
    } else if (is_nd_sharded) {
        const auto& nd_shard_spec = input_tensor.memory_config().nd_shard_spec().value();
        all_cores = nd_shard_spec.grid;
        num_cores = nd_shard_spec.grid.num_cores();
        const auto& shard_shape = nd_shard_spec.shard_shape;
        input_nsticks_per_core = shard_shape[-3] * shard_shape[-2];
        output_nsticks_per_core = input_nsticks_per_core;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, nd_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        shard_width = shard_shape[-1];
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const bool any_sharded = is_sharded || is_nd_sharded;
    const uint32_t effective_channels = any_sharded ? shard_width : input_channels;
    const uint32_t aligned_input_stick_nbytes = any_sharded ? effective_channels * input_tensor.element_size()
                                                            : pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes = any_sharded ? effective_channels * output_tensor.element_size()
                                                             : pool::get_aligned_stick_size(input_shape, output_tensor);

    // DRAM and L1 align sticks differently: size the shared DFB page for the larger, and stride both kernels by it.
    const uint32_t output_dfb_page_size = std::max(aligned_input_stick_nbytes, aligned_output_stick_nbytes);

    const uint32_t available_l1 = NUM_TILES_DEST * tt::constants::TILE_HW * element_size;
    const uint32_t l1_for_dfb = available_l1 / NEAREST_BUFFERING_FACTOR;
    const uint32_t max_dfb_pages_from_l1 = l1_for_dfb / output_dfb_page_size;

    const uint32_t max_sticks_per_core =
        any_sharded ? input_nsticks_per_core : std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_dfb_pages = std::min(max_sticks_per_core, max_dfb_pages_from_l1);
    TT_FATAL(
        num_dfb_pages > 0,
        "Not enough L1 for even a single CB page: output_cb_page_size={} exceeds l1_for_cb={}",
        output_dfb_page_size,
        l1_for_dfb);
    const uint32_t burst_size = num_dfb_pages < MAX_BURST_SIZE ? num_dfb_pages : MAX_BURST_SIZE;
    // DFB total size must be an even multiple of burst_size (required by reserve_back/pop_front)
    num_dfb_pages = round_down(num_dfb_pages, burst_size);

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName FILL{"fill"};  // the fill stick copied in for out-of-range samples
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = FILL,
        .entry_size = output_dfb_page_size,
        .num_entries = 1,
        .data_format_metadata = output_dfb_data_format,
    });

    // The legacy factory also allocated an input buffer backed by the input shard when sharded, but no
    // kernel ever referenced it (the reader reads the input through its tensor accessor), so it has no
    // Metal 2.0 form: a DFB with no producer or consumer cannot exist. Dropping it changes nothing.

    const uint32_t output_dfb_num_pages =
        any_sharded ? output_nsticks_per_core : num_dfb_pages * NEAREST_BUFFERING_FACTOR;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_dfb_page_size,
        .num_entries = output_dfb_num_pages,
        .data_format_metadata = output_dfb_data_format,
        .borrowed_from = any_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    });

    const bool fill_is_zero = (fill_value_bits == 0);

    const uint32_t effective_stick_nbytes = any_sharded ? effective_channels * element_size : input_stick_nbytes;

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input tensor must be allocated on device for rotate operation");

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "Output tensor must be allocated on device for rotate operation");

    std::vector<KernelSpec> kernels;
    kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/rotate/device/kernels/dataflow/"
                  "reader_rotate_nearest_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
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
                {"input_stick_nbytes", output_dfb_page_size},
                {"input_batch", input_batch},
                {"input_height", input_height},
                {"input_width", input_width},
                {"input_channels", effective_channels},
                {"num_dfb_pages", num_dfb_pages},
                {"input_stick_nbytes_unaligned", effective_stick_nbytes},
                {"fill_is_zero", static_cast<uint32_t>(fill_is_zero)},
                {"burst_size", burst_size},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks",
                     "start_stick_id",
                     "cos_angle",
                     "sin_angle",
                     "center_x",
                     "center_y",
                     "fill_value_bits"},
            },
        // The legacy ReaderConfigDescriptor, i.e. the reader default.
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/rotate/device/kernels/dataflow/"
                  "writer_rotate_nearest_interleaved.cpp",
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
                {"output_stick_nbytes", aligned_output_stick_nbytes},
                {"num_dfb_pages", num_dfb_pages},
                {"burst_size", burst_size},
                {"dfb_page_size", output_dfb_page_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        // The legacy WriterConfigDescriptor, i.e. the writer default.
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    if (any_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            uint32_t start_stick_id;
            if (is_width_sharded) {
                start_stick_id = 0;
            } else if (is_block_sharded) {
                uint32_t core_y = i / num_cores_x;
                start_stick_id = core_y * input_nsticks_per_core;
            } else {
                start_stick_id = i * input_nsticks_per_core;
            }
            // An unevenly sharded tensor's last shard is only partly backed by real sticks.
            const uint32_t num_sticks = start_stick_id >= total_output_sticks
                                            ? 0
                                            : std::min(input_nsticks_per_core, total_output_sticks - start_stick_id);

            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"num_sticks", num_sticks},
                    {"start_stick_id", start_stick_id},
                    {"cos_angle", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle))},
                    {"sin_angle", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle))},
                    {"center_x", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x))},
                    {"center_y", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y))},
                    {"fill_value_bits", fill_value_bits},
                });

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"num_sticks", num_sticks},
                    {"start_stick_id", start_stick_id},
                });
        }
    } else {
        uint32_t sticks_processed = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t num_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"num_sticks", num_sticks},
                    {"start_stick_id", sticks_processed},
                    {"cos_angle", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle))},
                    {"sin_angle", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle))},
                    {"center_x", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x))},
                    {"center_y", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y))},
                    {"fill_value_bits", fill_value_bits},
                });

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"num_sticks", num_sticks},
                    {"start_stick_id", sticks_processed},
                });

            sticks_processed += num_sticks;
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    ProgramSpec spec{
        .name = "rotate_nearest",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "rotate_nearest",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::quasar::rotate
