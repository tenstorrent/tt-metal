// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::rotate {

using namespace tt;
using namespace tt::tt_metal;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

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

    const bool is_sharded = input_tensor.is_sharded();

    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

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

    const uint16_t fill_value_bf16 = nearest_float_to_bfloat16(operation_attributes.fill);
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

    const uint32_t available_l1 = NUM_TILES_DEST * tt::constants::TILE_HW * element_size;
    const uint32_t l1_for_cb = available_l1 / NEAREST_BUFFERING_FACTOR;
    const uint32_t max_cb_pages_from_l1 = l1_for_cb / aligned_input_stick_nbytes;

    const uint32_t max_sticks_per_core =
        any_sharded ? input_nsticks_per_core : std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_cb_pages = std::min(max_sticks_per_core, max_cb_pages_from_l1);
    TT_FATAL(
        num_cb_pages > 0,
        "Not enough L1 for even a single CB page: aligned_input_stick_nbytes={} exceeds l1_for_cb={}",
        aligned_input_stick_nbytes,
        l1_for_cb);
    const uint32_t burst_size = num_cb_pages < MAX_BURST_SIZE ? num_cb_pages : MAX_BURST_SIZE;
    // CB total size must be an even multiple of burst_size (required by cb_push_back/cb_pop_front API)
    num_cb_pages = round_down(num_cb_pages, burst_size);

    const uint32_t output_cb_page_size = aligned_input_stick_nbytes;
    const uint32_t output_cb_num_pages =
        any_sharded ? output_nsticks_per_core : num_cb_pages * NEAREST_BUFFERING_FACTOR;

    const bool fill_is_zero = (fill_value_bf16 == 0);

    const uint32_t effective_stick_nbytes = any_sharded ? effective_channels * element_size : input_stick_nbytes;

    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName FILL{"fill"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const DataflowBufferSpec fill_dfb{
        .unique_id = FILL,
        .entry_size = output_cb_page_size,
        .num_entries = 1,
        .data_format_metadata = output_cb_data_format,
    };
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = output_cb_num_pages,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = any_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    };

    const KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
            "reader_rotate_nearest_interleaved.cpp",
        // Gen1 compatibility: the reader both produces and consumes FILL. A Gen2 port must use scratchpad storage or
        // a LocalTensorAccessor instead of this data-movement-kernel self-loop.
        .dfb_bindings = {ProducerOf(OUTPUT_DFB, "output"), ProducerOf(FILL, "fill"), ConsumerOf(FILL, "fill")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"input_stick_nbytes", aligned_input_stick_nbytes},
                {"input_height", input_height},
                {"input_width", input_width},
                {"input_channels", effective_channels},
                {"input_stick_nbytes_unaligned", effective_stick_nbytes},
                {"fill_is_zero", static_cast<uint32_t>(fill_is_zero)},
                {"burst_size", burst_size},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks",
                  "start_stick_id",
                  "cos_angle_bits",
                  "sin_angle_bits",
                  "center_x_bits",
                  "center_y_bits",
                  "fill_value_bf16"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    const KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
            "writer_rotate_nearest_interleaved.cpp",
        .dfb_bindings = {ConsumerOf(OUTPUT_DFB, "output")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {
                {"output_stick_nbytes", aligned_output_stick_nbytes},
                {"burst_size", burst_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    const auto add_runtime_args = [&](const CoreCoord& core, uint32_t num_sticks, uint32_t start_stick_id) {
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {
                {"num_sticks", num_sticks},
                {"start_stick_id", start_stick_id},
                {"cos_angle_bits", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle))},
                {"sin_angle_bits", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle))},
                {"center_x_bits", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x))},
                {"center_y_bits", static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y))},
                {"fill_value_bf16", static_cast<uint32_t>(fill_value_bf16)},
            });
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_sticks", num_sticks}, {"start_stick_id", start_stick_id}});
    };

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

            add_runtime_args(core, input_nsticks_per_core, start_stick_id);
        }
    } else {
        uint32_t sticks_processed = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t num_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

            add_runtime_args(core, num_sticks, sticks_processed);

            sticks_processed += num_sticks;
        }
    }

    ProgramSpec spec{
        .name = "rotate_nearest",
        .kernels = {reader, writer},
        .dataflow_buffers = {fill_dfb, output_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{.name = "rotate", .kernels = {READER, WRITER}, .target_nodes = all_cores}},
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run), std::move(writer_run)},
        .tensor_args = {{INPUT, input_tensor.mesh_tensor()}, {OUTPUT, output_tensor.mesh_tensor()}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::rotate
