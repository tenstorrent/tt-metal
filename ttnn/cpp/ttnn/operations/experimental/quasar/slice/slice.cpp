// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::quasar {

namespace detail {

inline bool is_rm_bw_sharded(const tt::tt_metal::MemoryConfig& mc) {
    const auto layout = mc.memory_layout();
    return mc.is_sharded() && (layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                               layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
}

// RM constraint, not TILE: `noc_async_*_sharded` derives pages_per_row from the padded shape,
// so non-tile-multiple H/W overshoots.
inline bool has_nontile_hw(const ttnn::Tensor& input) {
    const auto& s = input.logical_shape();
    if (s.rank() < 2) {
        return false;
    }
    return s[-1] % tt::constants::TILE_WIDTH != 0 || s[-2] % tt::constants::TILE_HEIGHT != 0;
}

// For B/W-sharded buffers the NOC helper splits by W only; irregular H alone is fine natively.
inline bool has_nontile_w(const ttnn::Tensor& input) {
    const auto& s = input.logical_shape();
    return s.rank() >= 1 && s[-1] % tt::constants::TILE_WIDTH != 0;
}

// Route RM sharded input through composite when native isn't safe: nontile-aligned B/W,
// B/W with non-zero width-begin, or nontile-aligned HEIGHT outside the sharded fast path.
inline bool needs_rm_composite_input(
    const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mc, bool no_step, bool width_begin_nonzero) {
    if (input.layout() != Layout::ROW_MAJOR || !input.is_sharded()) {
        return false;
    }
    if (is_rm_bw_sharded(input.memory_config())) {
        // Only W misalignment breaks the per-shard page split; irregular H is fine.
        return has_nontile_w(input) || width_begin_nonzero;
    }
    // Require a spec: no-spec HEIGHT output triggers needs_sharded_output_reshard → composite anyway.
    const bool stays_on_sharded_fast_path =
        output_mc.is_sharded() && output_mc.shard_spec().has_value() &&
        output_mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED && no_step;
    return has_nontile_hw(input) && !stays_on_sharded_fast_path;
}

// Compose RM B/W-sharded output only on nontile-aligned W (irregular H is fine natively).
inline bool needs_rm_composite_output(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mc) {
    if (input.layout() != Layout::ROW_MAJOR || !is_rm_bw_sharded(output_mc)) {
        return false;
    }
    return has_nontile_w(input);
}

// Sharded-no-spec output that can't seed from the input (not sharded, or layout differs);
// synthesize from the sliced shape.
inline bool needs_sharded_output_reshard(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mc) {
    if (!output_mc.is_sharded() || output_mc.shard_spec().has_value()) {
        return false;
    }
    if (!input.is_sharded()) {
        return true;
    }
    return input.memory_config().memory_layout() != output_mc.memory_layout();
}

}  // namespace detail

template <typename T>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const T> begins,
    ttsl::Span<const T> ends,
    ttsl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.logical_shape();
    uint32_t input_rank = input_shape.rank();
    auto input_layout = input_tensor.layout();

    if (input_rank == 0) {
        return input_tensor;
    }
    TT_FATAL(
        input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());
    TT_FATAL(begins.size() == ends.size(), "Start {} and end {} must have the same size", begins.size(), ends.size());
    TT_FATAL(
        step.size() == begins.size(),
        "Step {} must have the same size as start {} and end",
        step.size(),
        begins.size());

    bool no_step = std::ranges::all_of(step, [](uint32_t s) { return s == 1; });
    bool starts_zero = std::ranges::all_of(begins, [](uint32_t s) { return s == 0; });
    bool ends_max = true;
    for (size_t i = 0; i < ends.size(); ++i) {
        ends_max &= ends[i] == input_shape[i];
        if (!ends_max) {
            break;
        }
    }

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());
    // memory_config: original (input-compatible) config, used by to_layout in the rm_only path.
    // output_memory_config: may be rescaled below to match the sliced output dims.
    auto output_memory_config = memory_config;

    // Fill in a missing shard_spec on a sharded output: reuse source's if layouts match, else
    // synthesize from the source shape.
    auto resolve_mc = [&](const ttnn::Tensor& source) {
        auto resolved_mc = output_memory_config;
        if (resolved_mc.is_sharded() && !resolved_mc.shard_spec().has_value()) {
            const auto& in_mc = source.memory_config();
            if (in_mc.is_sharded() && in_mc.memory_layout() == resolved_mc.memory_layout() &&
                in_mc.shard_spec().has_value()) {
                resolved_mc =
                    ttnn::MemoryConfig(resolved_mc.memory_layout(), resolved_mc.buffer_type(), in_mc.shard_spec());
            } else {
                auto spec = operations::data_movement::transpose::generate_transpose_shard_spec(
                    source, source.padded_shape(), resolved_mc.memory_layout());
                resolved_mc = ttnn::MemoryConfig(resolved_mc.memory_layout(), resolved_mc.buffer_type(), spec);
            }
        }
        return resolved_mc;
    };

    // True when the next to_memory_config can land directly in the preallocated buffer: layouts
    // must match (prim::copy validates equality) and source must not alias the preallocated.
    auto can_land_in_preallocated = [&](const ttnn::Tensor& source) {
        return optional_output_tensor.has_value() && source.storage_type() == StorageType::DEVICE &&
               source.layout() == optional_output_tensor->layout() &&
               source.buffer() != optional_output_tensor->buffer();
    };
    // Safety net for paths that bypass the device op (composite / no-op / rm_only layout fixup):
    // short-circuits by buffer identity, falls back to ttnn::copy when result != preallocated.
    auto finalize_into_preallocated = [&](const ttnn::Tensor& result) -> ttnn::Tensor {
        if (!optional_output_tensor.has_value() || result.storage_type() != StorageType::DEVICE) {
            return result;
        }
        const auto& dst = optional_output_tensor.value();
        if (result.buffer() == dst.buffer()) {
            return result;
        }
        return ttnn::copy(result, dst);
    };

    auto ret_adjustment([&](const ttnn::Tensor& source) {
        if (source.storage_type() != StorageType::DEVICE) {
            return source;
        }
        // source carries the correct shard spec (input for no-op, prim::slice result otherwise).
        const auto resolved_mc = resolve_mc(source);
        const auto target = can_land_in_preallocated(source) ? optional_output_tensor : std::nullopt;
        auto tensor = ttnn::to_memory_config(source, resolved_mc, std::nullopt, target);
        tensor = ttnn::operations::experimental::quasar::to_layout(tensor, input_layout);
        return tensor;
    });

    // No-op check
    if (no_step && starts_zero && ends_max) {
        return finalize_into_preallocated(ret_adjustment(input_tensor));
    }

    // Composite hop: unshard to L1 interleaved if needed, slice, then convert to the requested mc.
    const bool width_begin_nonzero = !begins.empty() && begins.back() != 0;
    const bool rm_in_bad =
        detail::needs_rm_composite_input(input_tensor, output_memory_config, no_step, width_begin_nonzero);
    const bool rm_out_bad = detail::needs_rm_composite_output(input_tensor, output_memory_config);
    const bool out_no_spec = detail::needs_sharded_output_reshard(input_tensor, output_memory_config);
    if (rm_in_bad || rm_out_bad || out_no_spec) {
        const auto interleaved_l1 =
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
        Tensor x = rm_in_bad ? ttnn::to_memory_config(input_tensor, interleaved_l1, std::nullopt) : input_tensor;
        // Intermediate lives in L1 interleaved; to_memory_config lands the result in the caller's
        // buffer. sub_core_grids is threaded through to bound the recursive slice's work split.
        auto sliced = slice<T>(x, begins, ends, step, interleaved_l1, std::nullopt, pad_value, sub_core_grids);
        // slice preserves layout and to_memory_config doesn't change it — no trailing to_layout needed.
        // sliced is L1-interleaved so resolve_mc falls through to generate_transpose_shard_spec.
        const auto final_mc = resolve_mc(sliced);
        if (sliced.memory_config() == final_mc) {
            return finalize_into_preallocated(sliced);
        }
        const auto target = can_land_in_preallocated(sliced) ? optional_output_tensor : std::nullopt;
        return finalize_into_preallocated(ttnn::to_memory_config(sliced, final_mc, std::nullopt, target));
    }

    // Create modified vectors with wrapped indices and adjust them to match the tensor's rank
    ttsl::SmallVector<uint32_t> modified_begins(input_rank, 0);
    ttsl::SmallVector<uint32_t> modified_ends(input_rank, 0);
    ttsl::SmallVector<uint32_t> modified_step(input_rank, 1);

    // Wrap indices and adjust begins, ends, and step
    for (size_t i = 0; i < begins.size(); ++i) {
        if constexpr (std::is_signed_v<T>) {
            modified_begins[i] = operations::data_movement::wrap_index(begins[i], input_shape[i]);
            modified_ends[i] = operations::data_movement::wrap_index(ends[i], input_shape[i]);
            modified_step[i] = static_cast<uint32_t>(step[i]);
        } else {
            modified_begins[i] = begins[i];
            modified_ends[i] = ends[i];
            modified_step[i] = step[i];
        }
    }

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const ttsl::SmallVector<uint32_t>& modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    auto check_handled_tile_alignment = [&modified_begins, &input_rank, &tile_shape]() -> bool {
        return (
            modified_begins[input_rank - 1] % tile_shape[1] == 0 &&
            modified_begins[input_rank - 2] % tile_shape[0] == 0);
    };

    bool rm_only = false;
    bool one_dimensional = input_rank == 1;
    bool handled_tile_alignment = one_dimensional ? true : check_handled_tile_alignment();

    Tensor input = input_tensor;
    // Use the RM path when input isn't TILE, or TILE input has strided/1D/non-tile-aligned begins
    // (ends are padded downstream, so only begin alignment matters).
    rm_only = (input_tensor.layout() != Layout::TILE) || (!no_step || one_dimensional || !handled_tile_alignment);

    // Implicit inheritance from a sharded input: rescale the shard spec to the sliced shape so the
    // output doesn't reuse the input's (oversized) spec. Covers HEIGHT/WIDTH/BLOCK. (Issue #38016)
    if (!memory_config_arg.has_value() && !optional_output_tensor.has_value() && input_tensor.is_sharded() &&
        input_rank >= 2) {
        const auto& mem_layout = output_memory_config.memory_layout();
        if (mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
            mem_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
            mem_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const auto& shard_spec_val = output_memory_config.shard_spec().value();

            // Compute output dimensions, tile-aligned if using TILE path
            ttsl::SmallVector<uint32_t> output_dims(input_rank);
            for (size_t i = 0; i < input_rank; i++) {
                output_dims[i] = output_dim_i(i, modified_ends);
            }
            if (!rm_only) {
                output_dims[input_rank - 2] =
                    std::max(tt::round_up(output_dims[input_rank - 2], tile_shape[0]), tile_shape[0]);
                output_dims[input_rank - 1] =
                    std::max(tt::round_up(output_dims[input_rank - 1], tile_shape[1]), tile_shape[1]);
            }

            // Flatten to 2D: height = product of all dims except last, width = last dim
            uint32_t output_height = 1;
            for (size_t i = 0; i + 1 < input_rank; i++) {
                output_height *= output_dims[i];
            }
            uint32_t output_width = output_dims[input_rank - 1];

            std::array<uint32_t, 2> new_shard_shape = shard_spec_val.shape;
            if (mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
                uint32_t num_cores = shard_spec_val.num_cores();
                uint32_t new_shard_h = tt::div_up(output_height, num_cores);
                if (!rm_only) {
                    new_shard_h = std::max(tt::round_up(new_shard_h, tile_shape[0]), tile_shape[0]);
                }
                new_shard_shape = {new_shard_h, output_width};
            } else if (mem_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
                uint32_t num_cores = shard_spec_val.num_cores();
                uint32_t new_shard_w = tt::div_up(output_width, num_cores);
                if (!rm_only) {
                    new_shard_w = std::max(tt::round_up(new_shard_w, tile_shape[1]), tile_shape[1]);
                }
                new_shard_shape = {output_height, new_shard_w};
            } else {
                // BLOCK_SHARDED requires a rectangular grid; bounding_box() is valid only then.
                const auto bbox = shard_spec_val.grid.bounding_box();
                const uint32_t grid_h = bbox.end_coord.y - bbox.start_coord.y + 1;
                const uint32_t grid_w = bbox.end_coord.x - bbox.start_coord.x + 1;
                TT_FATAL(
                    shard_spec_val.num_cores() == grid_h * grid_w,
                    "BLOCK_SHARDED grid must be a full rectangle; got {} cores for {}x{} bounding box",
                    shard_spec_val.num_cores(),
                    grid_h,
                    grid_w);
                uint32_t new_shard_h = tt::div_up(output_height, grid_h);
                uint32_t new_shard_w = tt::div_up(output_width, grid_w);
                if (!rm_only) {
                    new_shard_h = std::max(tt::round_up(new_shard_h, tile_shape[0]), tile_shape[0]);
                    new_shard_w = std::max(tt::round_up(new_shard_w, tile_shape[1]), tile_shape[1]);
                }
                new_shard_shape = {new_shard_h, new_shard_w};
            }

            if (new_shard_shape != shard_spec_val.shape) {
                auto new_shard_spec =
                    tt::tt_metal::ShardSpec(shard_spec_val.grid, new_shard_shape, shard_spec_val.orientation);
                output_memory_config = MemoryConfig(
                    output_memory_config.memory_layout(), output_memory_config.buffer_type(), new_shard_spec);
            }
        }
    }

    if (rm_only) {
        if (!no_step) {
            TT_FATAL(input.dtype() != DataType::BFLOAT8_B, "Strided slice is not supported for BFLOAT8 tensors");
        }
        input =
            ttnn::operations::experimental::quasar::to_layout(input, Layout::ROW_MAJOR, std::nullopt, memory_config);
    }

    ttsl::SmallVector<uint32_t> padded_ends = modified_ends;
    if (input.layout() == Layout::TILE) {
        padded_ends[input_rank - 2] = std::max(tt::round_up(padded_ends[input_rank - 2], tile_shape[0]), tile_shape[0]);
        padded_ends[input_rank - 1] = std::max(tt::round_up(padded_ends[input_rank - 1], tile_shape[1]), tile_shape[1]);
    }

    ttsl::SmallVector<uint32_t> actual_shape_vec, final_padded_shape_vec;
    actual_shape_vec.reserve(input_rank);
    final_padded_shape_vec.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        TT_FATAL(
            modified_ends[i] >= modified_begins[i],
            "End {} must be greater than or equal to start {}",
            modified_ends[i],
            modified_begins[i]);
        auto val = output_dim_i(i, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape_vec.push_back(val);
        final_padded_shape_vec.push_back(std::max(output_dim_i(i, padded_ends), static_cast<uint32_t>(1)));
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape final_padded_shape(final_padded_shape_vec);

    if (empty) {
        TT_FATAL(
            input.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        if (optional_output_tensor.has_value()) {
            return optional_output_tensor.value();
        }
        return ttnn::empty(
            actual_shape,
            input_tensor.dtype(),
            input_tensor.layout(),
            input_tensor.device(),
            memory_config_arg.value_or(input_tensor.memory_config()));
    }
    auto res = ttnn::prim::qsr::slice(
        input,
        ttnn::Shape(modified_begins),
        ttnn::Shape(padded_ends),
        ttnn::Shape(modified_step),
        output_memory_config,
        /*use_tensor_args*/ false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        sub_core_grids,
        optional_output_tensor);
    res = ttnn::experimental::view(res, actual_shape, final_padded_shape);

    auto dim_needs_fill = [&input_shape, &actual_shape, &final_padded_shape](int i) {
        return ((actual_shape[i] != final_padded_shape[i]) && (input_shape[i] != actual_shape[i]));
    };

    if (pad_value.has_value() && (dim_needs_fill(-1) || dim_needs_fill(-2))) {
        res = ttnn::fill_implicit_tile_padding(res, pad_value.value());
    }

    // ret_adjustment may re-allocate (rm_only-from-TILE); finalize guarantees the caller's buffer.
    return finalize_into_preallocated(ret_adjustment(res));
}

template <typename T, std::size_t N>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    ttsl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    ttsl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    ttsl::Span<const T> step_vec(step.begin(), step.end());
    return slice<T>(
        input_tensor, start, end, step_vec, memory_config_arg, optional_output_tensor, pad_value, sub_core_grids);
}

template <typename T>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttsl::SmallVector<T>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(
        output_tensor_start.logical_shape().rank() == 1,
        "The start tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_start.logical_shape().rank());
    TT_FATAL(
        output_tensor_end.logical_shape().rank() == 1,
        "The end tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_end.logical_shape().rank());

    // Check if we can use the device-only tensor args path
    bool use_device_only_path = true;

    // Check if layout is supported (only TILE layout for now)
    if (input_tensor.layout() != Layout::TILE) {
        use_device_only_path = false;
    }

    // Check if step > 1 (only step=1 supported for now)
    if (step.has_value()) {
        for (auto s : step.value()) {
            if (s != 1) {
                use_device_only_path = false;
                break;
            }
        }
    }

    // Validate tensors are on device for both paths
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device for tensor args slice");

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());

    if (use_device_only_path) {
        // Validate required parameters for device-only path
        TT_FATAL(
            slice_dim.has_value() && num_devices.has_value(),
            "slice_dim and num_devices must be provided for device-only tensor args slice");

        TT_FATAL(
            output_tensor_start.storage_type() == StorageType::DEVICE,
            "Start tensor must be on device for tensor args slice");
        TT_FATAL(
            output_tensor_end.storage_type() == StorageType::DEVICE,
            "End tensor must be on device for tensor args slice");

        // Create dummy shapes for SliceDeviceOperation (will be ignored when use_tensor_args=true)
        uint32_t input_rank = input_tensor.logical_shape().rank();
        ttsl::SmallVector<uint32_t> dummy_shape(input_rank, 0);
        ttsl::SmallVector<uint32_t> dummy_step_shape(input_rank, 1);
        ttnn::Shape dummy_start(dummy_shape);
        ttnn::Shape dummy_end(dummy_shape);
        ttnn::Shape dummy_step(dummy_step_shape);

        // Use slice device operation with tensor args flag
        std::optional<Tensor> start_opt = output_tensor_start;
        std::optional<Tensor> end_opt = output_tensor_end;

        auto res = ttnn::prim::qsr::slice(
            input_tensor,
            dummy_start,
            dummy_end,
            dummy_step,
            memory_config,
            /*use_tensor_args*/ true,
            start_opt,
            end_opt,
            slice_dim,
            num_devices,
            sub_core_grids,
            optional_output_tensor);
        return res;
    }  // convert the Tensor to Vector
    std::vector<T> output_tensor_start_vector = output_tensor_start.to_vector<T>();
    std::vector<T> output_tensor_end_vector = output_tensor_end.to_vector<T>();

    // convert the Vector to Span
    ttsl::Span<const T> output_tensor_start_span(output_tensor_start_vector.data(), output_tensor_start_vector.size());
    ttsl::Span<const T> output_tensor_end_span(output_tensor_end_vector.data(), output_tensor_end_vector.size());

    // generate the step value if it is not provided
    ttsl::SmallVector<T> step_value = step.value_or(ttsl::SmallVector<T>(output_tensor_start_span.size(), 1));

    return slice<T>(
        input_tensor,
        output_tensor_start_span,
        output_tensor_end_span,
        ttsl::Span<const T>(step_value),
        memory_config_arg,
        optional_output_tensor,
        pad_value,
        sub_core_grids);
}

// Template instantiations for ttnn::slice
template ttnn::Tensor slice<int32_t>(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const int32_t> begins,
    ttsl::Span<const int32_t> ends,
    ttsl::Span<const int32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

template ttnn::Tensor slice<uint32_t>(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const uint32_t> begins,
    ttsl::Span<const uint32_t> ends,
    ttsl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Template instantiations for std::array version
template ttnn::Tensor slice<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Template instantiations for Tensor version
template ttnn::Tensor slice<uint32_t>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttsl::SmallVector<uint32_t>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::experimental::quasar
