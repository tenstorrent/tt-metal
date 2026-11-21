// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"

#include <cstdlib>

#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace {

// Six free DM harts per cluster by default; TTNN_QSR_SLICE_THREADS overrides the count (experiments only).
uint32_t slice_num_threads_from_env() {
    constexpr uint32_t default_threads = 6;
    const char* env = std::getenv("TTNN_QSR_SLICE_THREADS");
    if (env == nullptr) {
        return default_threads;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 10);
    TT_FATAL(
        end != env && *end == '\0' && value >= 1 && value <= 6,
        "TTNN_QSR_SLICE_THREADS must be an integer in [1, 6], got '{}'",
        env);
    return static_cast<uint32_t>(value);
}

}  // namespace

void SliceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(tensor_args.input.is_allocated(), "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(
        tensor_args.input.layout() == Layout::TILE || tensor_args.input.layout() == Layout::ROW_MAJOR,
        "Input tensor layout must be TILE or ROW_MAJOR but got {}",
        tensor_args.input.layout());
    TT_FATAL(
        tensor_args.input.padded_shape().rank() == args.slice_start.rank() &&
            args.slice_start.rank() == args.slice_end.rank(),
        "Input tensor rank ({}), slice start rank ({}), and slice end rank ({}) must all be equal",
        tensor_args.input.padded_shape().rank(),
        args.slice_start.rank(),
        args.slice_end.rank());
    for (uint32_t i = 0; i < tensor_args.input.padded_shape().rank(); i++) {
        TT_FATAL(
            args.slice_start[i] < tensor_args.input.padded_shape()[i],
            "Slice start[{}] ({}) must be less than input tensor shape[{}] ({})",
            i,
            args.slice_start[i],
            i,
            tensor_args.input.padded_shape()[i]);
        TT_FATAL(
            args.slice_end[i] <= tensor_args.input.padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            args.slice_end[i],
            tensor_args.input.padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            args.slice_start[i] <= args.slice_end[i],
            "Slice start[{}] ({}) must be <= slice end[{}] ({})",
            i,
            args.slice_start[i],
            i,
            args.slice_end[i]);
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto spec_required = compute_output_specs(args, tensor_args);
        const auto& out_tensor = tensor_args.preallocated_output.value();
        // Padded, not logical: quasar::slice pads the tile-path ends up to a tile boundary before
        // calling here and views the result back down afterwards, so the caller's tensor legitimately
        // carries a smaller logical shape than the spec. That licence is tile-only; see below.
        TT_FATAL(
            out_tensor.padded_shape() == spec_required.padded_shape(),
            "The preallocated output tensor needs a padded shape of {}, however it is {}",
            spec_required.padded_shape(),
            out_tensor.padded_shape());
        // Row-major ends are passed through unpadded, and a sharded output's alignment is its shard
        // width, so padded equality does not imply logical equality. SliceRmShardedProgramFactory sizes
        // each row's copy from output.logical_shape()[-1].
        if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
            TT_FATAL(
                out_tensor.logical_shape() == spec_required.logical_shape(),
                "The preallocated output tensor needs a logical shape of {}, however it is {}",
                spec_required.logical_shape(),
                out_tensor.logical_shape());
        }
        // create_output_tensors hands the program the caller's tensor verbatim, so the destination's
        // dtype, page_config, memory_config and alignment must be what the factories were built against.
        TT_FATAL(
            out_tensor.tensor_spec().tensor_layout() == spec_required.tensor_layout(),
            "The preallocated output tensor needs a layout of {}, however it is {}",
            spec_required.tensor_layout(),
            out_tensor.tensor_spec().tensor_layout());
    }
    const auto output_spec = compute_output_specs(args, tensor_args);
    const auto& output_tensor_shape = output_spec.logical_shape();
    const auto& input = tensor_args.input;
    const uint32_t rank = input.padded_shape().rank();

    TT_FATAL(
        args.num_threads >= 1 && args.num_threads <= 6,
        "quasar.slice runs 1 to 6 data-movement threads per cluster, got {}",
        args.num_threads);
    TT_FATAL(
        args.step.rank() == rank, "Number of steps {} must match number of ends/starts {}", args.step.rank(), rank);

    // The kernel addresses both tensors by page id and moves whole rows / tiles.
    const auto& output_mem_config = tensor_args.preallocated_output.has_value()
                                        ? tensor_args.preallocated_output->memory_config()
                                        : output_spec.memory_config();
    TT_FATAL(
        slice_addresses_whole_pages(input.layout(), input.memory_config()),
        "quasar.slice cannot read a ROW_MAJOR {} input: a row is split over clusters. Unshard it first "
        "(ttnn.experimental.quasar.slice does)",
        input.memory_config().memory_layout());
    TT_FATAL(
        slice_addresses_whole_pages(input.layout(), output_mem_config),
        "quasar.slice cannot write a ROW_MAJOR {} output: a row is split over clusters. Slice to an interleaved "
        "tensor and shard that (ttnn.experimental.quasar.slice does)",
        output_mem_config.memory_layout());

    if (args.use_tensor_args) {
        TT_FATAL(input.layout() == Layout::TILE, "quasar.slice with tensor begins / ends needs a TILE input");
        TT_FATAL(tensor_args.start_tensor.has_value(), "quasar.slice with tensor begins needs the start tensor");
        const auto& start = tensor_args.start_tensor.value();
        TT_FATAL(
            start.storage_type() == StorageType::DEVICE && start.is_allocated(),
            "quasar.slice: the start tensor must be allocated on the device");
        TT_FATAL(
            start.layout() == Layout::ROW_MAJOR && start.element_size() == sizeof(uint32_t) &&
                start.logical_shape().rank() == 1 && start.logical_shape()[0] >= rank,
            "quasar.slice: the start tensor must be a 1D ROW_MAJOR tensor of at least {} 32-bit integers, got {} {} {}",
            rank,
            start.layout(),
            start.dtype(),
            start.logical_shape());
        TT_FATAL(
            !start.is_sharded() || start.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "quasar.slice: the start tensor must be interleaved or height sharded");
    }
    if (input.layout() == Layout::TILE) {
        TT_FATAL(rank >= 2, "quasar.slice: a TILE input needs at least 2 dims, got {}", rank);
        TT_FATAL(
            input.physical_volume() % TILE_HW == 0,
            "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
            input.physical_volume(),
            TILE_HW);
        if (slice_gathers_tile_elements(args, input)) {
            // Unaligned begins or strides in the last two dims: output tiles are assembled from elements.
            TT_FATAL(
                slice_tile_gather_supports(input.dtype()),
                "quasar.slice: a TILE slice with unaligned begins or strides in its last two dims gathers 16-bit "
                "or 32-bit elements and cannot address {} tiles; slice in ROW_MAJOR (ttnn.experimental.quasar.slice "
                "does)",
                input.dtype());
        } else {
            // Whole tiles move, so the slice must cover whole tiles (quasar::slice pads the ends).
            TT_FATAL(
                output_tensor_shape[-2] % TILE_HEIGHT == 0 && output_tensor_shape[-1] % TILE_WIDTH == 0,
                "quasar.slice moves whole tiles when the last two begins are tile aligned: the sliced height and "
                "width ({}, {}) must be whole tiles (ttnn.experimental.quasar.slice pads the ends)",
                output_tensor_shape[-2],
                output_tensor_shape[-1]);
        }
    }
}

SliceDeviceOperation::spec_return_value_t SliceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    ttsl::SmallVector<uint32_t> out_shape(input_tensor.logical_shape().rank());

    if (args.use_tensor_args) {
        TT_FATAL(
            args.slice_dim.has_value() && args.num_devices.has_value(),
            "slice_dim and num_devices must be provided for tensor args path");

        uint32_t slice_dimension = args.slice_dim.value();
        uint32_t num_parts = args.num_devices.value();

        for (uint32_t i = 0; i < out_shape.size(); i++) {
            out_shape[i] = input_tensor.logical_shape()[i];
        }
        TT_FATAL(
            slice_dimension < out_shape.size(),
            "slice_dim ({}) must be less than tensor rank ({})",
            slice_dimension,
            out_shape.size());

        uint32_t original_size = out_shape[slice_dimension];
        uint32_t slice_size = original_size / num_parts;

        TT_FATAL(
            original_size % num_parts == 0,
            "Input dimension {} (size={}) must be evenly divisible by num_devices ({})",
            slice_dimension,
            original_size,
            num_parts);

        out_shape[slice_dimension] = slice_size;
    } else {
        // Regular path using slice_start, slice_end, step
        auto output_dim_i = [&args](size_t i) {
            // args.step is unsigned, so this only rejects 0 (the divide-by-zero); sign is checked in quasar::slice.
            TT_FATAL(args.step[i] != 0, "Step({}) must be non-zero", i);
            return (args.slice_end[i] - args.slice_start[i] + args.step[i] - 1) / args.step[i];
        };
        for (uint32_t i = 0; i < out_shape.size(); i++) {
            out_shape[i] = output_dim_i(i);
        }
    }
    ttnn::Shape output_tensor_shape(std::move(out_shape));

    // Synthesize shard spec for sharded-no-spec output: scale from input spec when layouts match,
    // else fall back to generate_transpose_shard_spec for a fresh full-grid spec.
    auto output_mem_config = args.output_mem_config;
    if (output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value()) {
        std::optional<tt::tt_metal::ShardSpec> derived;
        if (input_tensor.is_sharded() && input_tensor.memory_config().shard_spec().has_value() &&
            input_tensor.memory_config().memory_layout() == output_mem_config.memory_layout() &&
            input_tensor.logical_shape().rank() == output_tensor_shape.rank()) {
            auto adjusted = ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape(
                *input_tensor.memory_config().shard_spec(), input_tensor.logical_shape(), output_tensor_shape);
            if (adjusted.has_value()) {
                // TILE factories require tile-aligned shards; sub-tile results from
                // adjust_shard_spec_to_shape fall through to generate_transpose_shard_spec.
                const bool tile_layout = input_tensor.layout() == Layout::TILE;
                const bool tile_aligned = adjusted->shape[0] % tt::constants::TILE_HEIGHT == 0 &&
                                          adjusted->shape[1] % tt::constants::TILE_WIDTH == 0;
                if (!tile_layout || tile_aligned) {
                    derived = std::move(adjusted);
                }
            }
        }
        if (!derived.has_value()) {
            derived = ttnn::operations::data_movement::transpose::generate_transpose_shard_spec(
                input_tensor, output_tensor_shape, output_mem_config.memory_layout());
        }
        output_mem_config =
            tt::tt_metal::MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), derived);
    }

    return tt::tt_metal::TensorSpec(
        output_tensor_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config));
}

Tensor SliceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input = tensor_args.input;
    const auto output_spec = compute_output_specs(args, tensor_args);

    return create_device_tensor(output_spec, input.device());
}

SliceDeviceOperation::program_factory_t SliceDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return SliceProgramFactory{};
}

// Port of tt-metal e517beb3f41 (#47602): the default program hash (boost::hash_combine style) has weak
// distribution for small-integer shape sequences, causing false cache hits when shapes differ but
// collide -> TT_FATAL on back-to-back slices with differing shapes. Mix in full input/output specs and
// slice params so distinct invocations get distinct keys.
ttsl::hash::hash_t SliceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto hash = tt::tt_metal::operation::hash_operation<SliceDeviceOperation>(
        operation_attributes.slice_start,
        operation_attributes.slice_end,
        operation_attributes.step,
        operation_attributes.use_tensor_args,
        operation_attributes.slice_dim,
        operation_attributes.num_devices,
        operation_attributes.output_mem_config,
        operation_attributes.sub_core_grids,
        operation_attributes.num_threads,
        tensor_args.start_tensor.has_value(),
        tensor_args.end_tensor.has_value());

    const auto& input = tensor_args.input;
    hash = ttsl::hash::hash_objects(
        hash,
        input.logical_shape().rank(),
        input.logical_shape(),
        input.padded_shape(),
        input.layout(),
        input.dtype(),
        input.memory_config());

    if (tensor_args.start_tensor.has_value()) {
        const auto& st = tensor_args.start_tensor.value();
        hash = ttsl::hash::hash_objects(
            hash,
            st.logical_shape().rank(),
            st.logical_shape(),
            st.padded_shape(),
            st.layout(),
            st.dtype(),
            st.memory_config());
    }

    // The end tensor never reaches the device (the output shape comes from slice_dim / num_devices), but
    // it stays in the key so a call keeps one cache entry per (start, end) placement, as before.
    if (tensor_args.end_tensor.has_value()) {
        const auto& et = tensor_args.end_tensor.value();
        hash = ttsl::hash::hash_objects(
            hash,
            et.logical_shape().rank(),
            et.logical_shape(),
            et.padded_shape(),
            et.layout(),
            et.dtype(),
            et.memory_config());
    }

    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    hash = ttsl::hash::hash_objects(
        hash,
        output_spec.logical_shape().rank(),
        output_spec.logical_shape(),
        output_spec.padded_shape(),
        output_spec.layout(),
        output_spec.data_type(),
        output_spec.memory_config());

    return hash;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SliceDeviceOperation::tensor_return_value_t>
SliceDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = output;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output_tensor, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

SliceDeviceOperation::tensor_return_value_t slice(
    const Tensor& input,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_tensor_args,
    std::optional<Tensor> start_tensor,
    std::optional<Tensor> end_tensor,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::prim::qsr::SliceDeviceOperation;
    const auto attributes = [&](const tt::tt_metal::MemoryConfig& mem_config) {
        return OperationType::operation_attributes_t{
            slice_start,
            slice_end,
            step,
            mem_config,
            use_tensor_args,
            slice_dim,
            num_devices,
            sub_core_grids,
            slice_num_threads_from_env()};
    };
    return ttnn::operations::data_movement::launch_with_aligned_shard_rows(
        input,
        output_mem_config,
        preallocated_output,
        [&] {
            return OperationType::compute_output_specs(
                attributes(output_mem_config),
                OperationType::tensor_args_t{input, start_tensor, end_tensor, preallocated_output});
        },
        [&](const Tensor& launch_input, const std::optional<tt::tt_metal::MemoryConfig>& explicit_mem_config) {
            return ttnn::device_operation::launch<OperationType>(
                attributes(explicit_mem_config.value_or(output_mem_config)),
                OperationType::tensor_args_t{
                    launch_input,
                    start_tensor,
                    end_tensor,
                    explicit_mem_config.has_value() ? std::nullopt : preallocated_output});
        });
}

bool slice_gathers_tile_elements(const SliceParams& args, const Tensor& input) {
    using namespace tt::constants;
    if (args.use_tensor_args || input.layout() != Layout::TILE || args.slice_start.rank() < 2) {
        return false;
    }
    return args.slice_start[-2] % TILE_HEIGHT != 0 || args.slice_start[-1] % TILE_WIDTH != 0 || args.step[-2] != 1 ||
           args.step[-1] != 1;
}

bool slice_tile_gather_supports(tt::tt_metal::DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::UINT16: return true;
        default: return false;
    }
}

bool slice_addresses_whole_pages(tt::tt_metal::Layout layout, const tt::tt_metal::MemoryConfig& memory_config) {
    if (layout == Layout::TILE || !memory_config.is_sharded()) {
        return true;
    }
    return memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
}
}  // namespace ttnn::prim::qsr
