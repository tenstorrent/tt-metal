// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <utility>
#include <enchantum/enchantum.hpp>

#include "scatter.hpp"
#include "scatter_force.hpp"

#include "device/scatter_device_operation.hpp"
#include "codegen/scatter_codegen_device_operation.hpp"
#include "codegen/scatter_codegen_program_factory.hpp"
#include "codegen/scatter_codegen_supported.hpp"

#include "slice/slice.hpp"
#include "tt_stl/small_vector.hpp"
#include "scatter/scatter_enums.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::data_movement {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr std::string_view ALLOWED_REDUCTIONS[] = {"add", "multiply"};

// validate dimension constraints before sending down to device operation working on the last dimension
// inputs are validated according to
// https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_ index_shape[...] <=
// src_shape[...]: index shape can't have any dimension longer than according dimension of source shape index_shape[d !=
// dim] <= input_shape[d != dim]: index shape must be smaller than input shape on all dimensions except the scatter one
void validate_inputs(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const int32_t& dim,
    const std::optional<std::string>& opt_reduction_string) {
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& source_shape{source_tensor.logical_shape()};
    const auto input_rank = input_shape.rank();
    const auto index_rank = index_shape.rank();
    const auto source_rank = source_shape.rank();
    const int32_t normalized_dim = (dim < 0) ? (dim + input_rank) : dim;

    TT_FATAL(
        input_rank == index_rank,
        "input_rank must be equal to index_rank (input_rank == {}, index_rank == {})",
        input_rank,
        index_rank);

    TT_FATAL(
        input_rank == source_rank,
        "input_rank must be equal to source_rank (input_rank == {}, source_rank == {})",
        input_rank,
        source_rank);

    TT_FATAL(
        dim < static_cast<int32_t>(input_rank) && -static_cast<int32_t>(input_rank) <= dim,
        "dim must follow the condition -input_rank <= dim < input_rank (dim: {}, rank: {}).",
        dim,
        static_cast<int32_t>(input_rank));

    if (opt_reduction_string.has_value()) {
        TT_FATAL(
            std::find(std::begin(ALLOWED_REDUCTIONS), std::end(ALLOWED_REDUCTIONS), *opt_reduction_string) !=
                std::end(ALLOWED_REDUCTIONS),
            "reduce must be either 'add' or 'multiply' (case-sensitive), got {}",
            *opt_reduction_string);
    }

    for (uint32_t probe_dim = 0; probe_dim < input_rank; ++probe_dim) {
        TT_FATAL(
            index_shape[probe_dim] <= source_shape[probe_dim],
            "index_shape[{}] <= source_shape[{}] == false (index_shape: {}, source_shape: {})",
            probe_dim,
            probe_dim,
            index_shape,
            source_shape);
        if (probe_dim != normalized_dim) {
            TT_FATAL(
                index_shape[probe_dim] <= input_shape[probe_dim],
                "index_shape[{}] <= input_shape[{}] == false (index_shape: {}, input_shape: {})",
                probe_dim,
                probe_dim,
                index_shape,
                input_shape);
        }
    }
}

bool is_i32(const DataType& dt) { return (dt == DataType::UINT32) || (dt == DataType::INT32); }

void check_support(
    const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& source_tensor, const int32_t& dim) {
    const auto& input_dtype = input_tensor.dtype();
    const auto& index_dtype = index_tensor.dtype();
    const auto& source_dtype = source_tensor.dtype();
    const auto& input_layout = input_tensor.layout();
    const auto& index_layout = index_tensor.layout();
    const auto& source_layout = source_tensor.layout();
    const auto& input_shape = input_tensor.logical_shape();
    const auto& index_shape = index_tensor.logical_shape();
    const auto& source_shape = source_tensor.logical_shape();
    // check if to_layout fp32 tiled precision case
    TT_FATAL(
        !(input_dtype == DataType::FLOAT32 && input_layout == Layout::TILE),
        "Scatter doesn't work for fp32 tiled tensors yet (see to_layout issue #) - input tensor is {} {}.",
        input_dtype,
        input_layout);
    TT_FATAL(
        !(source_dtype == DataType::FLOAT32 && source_layout == Layout::TILE),
        "Scatter doesn't work for fp32 tiled tensors yet (see to_layout issue #) - source tensor is {} {}.",
        source_dtype,
        source_layout);
    // check if to_layout int32 tiled row>256 garbage case
    constexpr uint32_t to_layout_int32_scatter_axis_max_length = 256;
    TT_FATAL(
        !(is_i32(input_dtype) && input_layout == Layout::TILE &&
          input_shape[dim] > to_layout_int32_scatter_axis_max_length),
        "Scatter doesn't work for int32 tensors that have scatter row longer than {} elements - input tensor is of "
        "type: {}, layout: {} and input_shape[scatter_axis] == {}",
        to_layout_int32_scatter_axis_max_length,
        enchantum::to_string(input_dtype),
        enchantum::to_string(input_layout),
        input_shape[dim]);
    TT_FATAL(
        !(is_i32(index_dtype) && index_layout == Layout::TILE &&
          index_shape[dim] > to_layout_int32_scatter_axis_max_length),
        "Scatter doesn't work for int32 tensors that have scatter row longer than {} elements - index tensor is of "
        "type: {}, layout: {} and index_shape[scatter_axis] == {}",
        to_layout_int32_scatter_axis_max_length,
        enchantum::to_string(index_dtype),
        enchantum::to_string(index_layout),
        index_shape[dim]);
    TT_FATAL(
        !(is_i32(source_dtype) && source_layout == Layout::TILE &&
          source_shape[dim] > to_layout_int32_scatter_axis_max_length),
        "Scatter doesn't work for int32 tensors that have scatter row longer than {} elements - source tensor is of "
        "type: {}, layout: {} and source_shape[scatter_axis] == {}",
        to_layout_int32_scatter_axis_max_length,
        enchantum::to_string(source_dtype),
        enchantum::to_string(source_layout),
        source_shape[dim]);
}

// The reader maps each leading axis of the input onto the same axis of the index, so the two must
// line up per axis. Prepending 1s keeps that; folding rank > 4 down to 4D does not, because each
// tensor fuses its leading dims with its own extents. See #56876.
Tensor pad_rank_up_to_4d(const Tensor& input_tensor) {
    return (input_tensor.logical_shape().rank() < 4) ? ttnn::operations::core::unsqueeze_to_4D(input_tensor)
                                                     : input_tensor;
}

// `force_row_major` is the only difference between the native and codegen legs' normalization:
// native's device operation (device/scatter_program_factory.cpp) only ever builds a ROW_MAJOR
// program, so it forces ROW_MAJOR before transposing regardless of the caller's layout; the codegen
// legs (the TILE and ROW_MAJOR program factories) each address the caller's own layout directly, so
// the codegen leg keeps it unchanged instead of paying an extra untilize.
Tensor pre_scatter_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool force_row_major,
    const std::optional<Shape>& index_shape = std::nullopt) {
    // Shape{1} is deliberately not short-circuited: this runs once per operand, and every operand
    // has to reach the kernel at the same rank (#56876). The Shape{0} arm is inert - a zero last
    // dim divides by zero in the factory whether or not this returns early (#56881).
    if (input_tensor.logical_shape() == ttnn::Shape{0}) {
        return input_tensor;
    }

    Tensor processed_tensor = input_tensor;
    // Only materialize the source-prefix slice when source is actually wider than index on some
    // axis -- when the shapes already match this is an identity slice that would still dispatch a
    // real data-movement kernel over the whole tensor for no effect.
    if (index_shape.has_value() && processed_tensor.logical_shape() != index_shape.value()) {
        const ttsl::SmallVector<uint32_t> start(index_shape->rank(), 0);
        const ttsl::SmallVector<uint32_t> steps(index_shape->rank(), 1);
        const ttsl::SmallVector<uint32_t> end(index_shape->cbegin(), index_shape->cend());
        processed_tensor = ttnn::slice(processed_tensor, start, end, steps, processed_tensor.memory_config());
    }
    // if layout is tile, convert to row-major first - this allows for minimized memory usage by transpose (no padding)
    if (force_row_major && processed_tensor.layout() != Layout::ROW_MAJOR) {
        processed_tensor = ttnn::to_layout(processed_tensor, Layout::ROW_MAJOR);
    }
    // transposing a row-major tensor here
    processed_tensor = reduction_common::perform_transpose(processed_tensor, is_dim_last_idx, dim, -1);
    processed_tensor = pad_rank_up_to_4d(processed_tensor);

    return processed_tensor;
}

Tensor post_scatter_transform_tensor(
    Tensor& output_tensor,
    const int32_t dim,
    const bool is_dim_last_idx,
    const Shape& original_logical_shape,
    const Layout& original_layout,
    const bool force_row_major) {
    const auto orig_rank = original_logical_shape.rank();

    // Only the padding applied on the way in needs undoing; rank >= 4 went through untouched.
    if (orig_rank == 1) {
        output_tensor = ttnn::reshape(output_tensor, original_logical_shape);
    } else if (orig_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
    }

    // transposing a row-major tensor here
    if (!is_dim_last_idx) {
        output_tensor = ttnn::transpose(output_tensor, dim, -1, output_tensor.memory_config());
    }

    TT_FATAL(
        output_tensor.logical_shape() == original_logical_shape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        output_tensor.logical_shape(),
        original_logical_shape);

    // The codegen leg never left the caller's own layout (pre_scatter_transform_tensor with
    // force_row_major=false), so there is nothing to restore; the native leg forced ROW_MAJOR and
    // must convert back.
    if (force_row_major && original_layout != Layout::ROW_MAJOR) {
        output_tensor = ttnn::to_layout(output_tensor, original_layout);
    }

    return output_tensor;
}

scatter::ScatterReductionType get_scatter_reduction_type_from_string(
    const std::optional<std::string>& opt_reduction_string) {
    if (!opt_reduction_string.has_value()) {
        return scatter::ScatterReductionType::INVALID;
    }
    if (*opt_reduction_string == "add") {
        return scatter::ScatterReductionType::ADD;
    }
    if (*opt_reduction_string == "multiply") {
        return scatter::ScatterReductionType::MULTIPLY;
    }
    if (*opt_reduction_string == "max" || *opt_reduction_string == "amax") {
        return scatter::ScatterReductionType::AMAX;
    }
    if (*opt_reduction_string == "min" || *opt_reduction_string == "amin") {
        return scatter::ScatterReductionType::AMIN;
    }
    return scatter::ScatterReductionType::INVALID;
}

// The codegen kernels' own reduction_mode convention (scatter_common.hpp's scatter_reduce_value):
// 0=replace, 1=add, 2=multiply. Derived from the one string parser this file has rather than
// re-parsing the string, so a reduce the parser knows but the kernels do not (amax/amin) can never be
// read as a plain overwrite. ttnn::scatter() consults this through codegen_can_serve() BEFORE
// validate_inputs() runs, so an unsupported string must fail here on its own. ScatterReductionType's
// ordinals do not match the kernel convention, so the two are never interconverted by casting.
uint32_t scatter_reduction_mode(const std::optional<std::string>& opt_reduction_string) {
    if (!opt_reduction_string.has_value()) {
        return 0;
    }
    switch (get_scatter_reduction_type_from_string(opt_reduction_string)) {
        case scatter::ScatterReductionType::ADD: return 1;
        case scatter::ScatterReductionType::MULTIPLY: return 2;
        default:
            TT_THROW(
                "scatter: reduce must be either 'add' or 'multiply' (case-sensitive), got {}", *opt_reduction_string);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::operations::data_movement::detail {

// Internal to this file. `detail` is shared across the whole data_movement library and this is a
// unity-build target, so unprefixed helper names must not have external linkage.
namespace {

namespace scatter_ns = ttnn::operations::data_movement::scatter;
using namespace ttnn::operations::data_movement::CMAKE_UNIQUE_NAMESPACE;

// Whether the codegen path can serve this call. Evaluated on the ORIGINAL (pre
// pre_scatter_transform_tensor) tensors and the caller's raw dim -- the same attributes the supported
// scope is expressed in -- before any transpose/4D-fold. Correctness and caller-controlled output
// placement only; perf demotion is a separate, routing-only question.
bool codegen_can_serve(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::string>& opt_reduction_string) {
    const auto output_mem_config = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();
    const uint32_t reduction_mode = scatter_reduction_mode(opt_reduction_string);
    return scatter_ns::supported_execution_controls(input_tensor, output_mem_config, std::nullopt) &&
           scatter_ns::supported_by_codegen(input_tensor, dim, index_tensor, source_tensor, reduction_mode);
}

// The existing native implementation, unconditionally. Callers that have already been routed here
// enter this rather than re-entering ttnn::scatter, so a call routed to native cannot be routed a
// second time and land on codegen.
Tensor scatter_native(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.logical_shape().rank();

    const int32_t normalized_dim = dim < 0 ? dim + input_tensor_rank : dim;
    TT_FATAL(
        normalized_dim >= 0 && normalized_dim < static_cast<int32_t>(input_tensor_rank),
        "scatter: dim {} is out of range for tensor rank {}",
        dim,
        input_tensor_rank);

    check_support(input_tensor, index_tensor, source_tensor, normalized_dim);
    validate_inputs(input_tensor, index_tensor, source_tensor, normalized_dim, opt_reduction_string);

    const auto& original_index_tensor_lshape = index_tensor.logical_shape();
    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    const auto original_layout = input_tensor.layout();

    const bool input_tensor_is_dim_last_idx = (normalized_dim == input_tensor_rank - 1);

    // tensors sent to the native device operation must be row-major, transposed to have the last
    // dimension as last axis, and unsqueezed to 4D if of a lower rank.
    Tensor transformed_input_tensor = pre_scatter_transform_tensor(
        input_tensor, normalized_dim, input_tensor_is_dim_last_idx, /*force_row_major=*/true);
    Tensor transformed_index_tensor = pre_scatter_transform_tensor(
        index_tensor, normalized_dim, input_tensor_is_dim_last_idx, /*force_row_major=*/true);
    Tensor transformed_source_tensor = pre_scatter_transform_tensor(
        source_tensor,
        normalized_dim,
        input_tensor_is_dim_last_idx,
        /*force_row_major=*/true,
        index_tensor.logical_shape());

    const MemoryConfig final_memory_config{
        output_memory_config.has_value() ? output_memory_config.value() : input_tensor.memory_config()};
    const auto reduction = get_scatter_reduction_type_from_string(opt_reduction_string);

    Tensor output = ttnn::prim::scatter(
        transformed_input_tensor,
        normalized_dim,
        transformed_index_tensor,
        transformed_source_tensor,
        final_memory_config,
        reduction,
        sub_core_grid);
    return post_scatter_transform_tensor(
        output,
        normalized_dim,
        input_tensor_is_dim_last_idx,
        original_input_tensor_lshape,
        original_layout,
        /*force_row_major=*/true);
}

// The generated implementation. Unlike scatter_native, this keeps the caller's own layout through the
// transpose/4D-fold sandwich (the TILE and ROW_MAJOR program factories each address that layout
// directly), so a TILE input never pays native's forced untilize -> scatter(ROW_MAJOR) -> tilize
// round trip -- except for the low-tile-row reroute below, which pays the equivalent round trip
// deliberately because the TILE factories' per-row work split leaves most cores idle at that width.
Tensor scatter_codegen_dispatch(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.logical_shape().rank();

    const int32_t normalized_dim = dim < 0 ? dim + input_tensor_rank : dim;
    TT_FATAL(
        normalized_dim >= 0 && normalized_dim < static_cast<int32_t>(input_tensor_rank),
        "scatter: dim {} is out of range for tensor rank {}",
        dim,
        input_tensor_rank);

    check_support(input_tensor, index_tensor, source_tensor, normalized_dim);
    validate_inputs(input_tensor, index_tensor, source_tensor, normalized_dim, opt_reduction_string);

    // No empty-shape passthrough here: supported_by_codegen() already rejects rank <= 0 (an empty
    // ttnn::Shape{} has rank 0), so an empty-shape call never reaches this function through the
    // auto route's codegen_can_serve() gate, and scatter_force_codegen() must TT_FATAL on it rather
    // than silently answering from host state -- a passthrough here would swallow that forced call
    // without ever invoking ttnn::prim::scatter_codegen. The passthrough itself lives in
    // scatter_native() only.
    const auto original_layout = input_tensor.layout();

    const bool input_tensor_is_dim_last_idx = (normalized_dim == input_tensor_rank - 1);

    Tensor transformed_input_tensor = pre_scatter_transform_tensor(
        input_tensor, normalized_dim, input_tensor_is_dim_last_idx, /*force_row_major=*/false);
    Tensor transformed_index_tensor = pre_scatter_transform_tensor(
        index_tensor, normalized_dim, input_tensor_is_dim_last_idx, /*force_row_major=*/false);
    Tensor transformed_source_tensor = pre_scatter_transform_tensor(
        source_tensor,
        normalized_dim,
        input_tensor_is_dim_last_idx,
        /*force_row_major=*/false,
        index_tensor.logical_shape());

    const MemoryConfig final_memory_config{
        output_memory_config.has_value() ? output_memory_config.value() : input_tensor.memory_config()};
    const uint32_t reduction_mode = scatter_reduction_mode(opt_reduction_string);

    const bool rerouted_to_row_major = ttnn::operations::data_movement::scatter::prefers_row_major_strategy(
        transformed_input_tensor,
        transformed_index_tensor,
        transformed_source_tensor,
        transformed_input_tensor.logical_shape(),
        transformed_index_tensor.logical_shape(),
        final_memory_config);
    if (rerouted_to_row_major) {
        transformed_input_tensor = ttnn::to_layout(transformed_input_tensor, Layout::ROW_MAJOR);
        transformed_index_tensor = ttnn::to_layout(transformed_index_tensor, Layout::ROW_MAJOR);
        transformed_source_tensor = ttnn::to_layout(transformed_source_tensor, Layout::ROW_MAJOR);
    }

    auto params = ttnn::prim::build_scatter_codegen_params(
        transformed_input_tensor,
        transformed_index_tensor,
        transformed_source_tensor,
        reduction_mode,
        final_memory_config,
        sub_core_grid);
    Tensor output = ttnn::prim::scatter_codegen(
        params, transformed_input_tensor, transformed_index_tensor, transformed_source_tensor, std::nullopt);
    if (rerouted_to_row_major) {
        output = ttnn::to_layout(output, Layout::TILE);
    }
    return post_scatter_transform_tensor(
        output,
        normalized_dim,
        input_tensor_is_dim_last_idx,
        original_input_tensor_lshape,
        original_layout,
        /*force_row_major=*/false);
}

}  // namespace

Tensor scatter_force_native(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return scatter_native(
        input_tensor, dim, index_tensor, source_tensor, output_memory_config, opt_reduction_string, sub_core_grid);
}

Tensor scatter_force_codegen(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    TT_FATAL(
        codegen_can_serve(input_tensor, dim, index_tensor, source_tensor, output_memory_config, opt_reduction_string),
        "scatter_force_codegen invoked for a case the codegen path does not support (requires a single shared "
        "layout -- TILE or ROW_MAJOR -- across bfloat16 input/src and an int32/uint32 index, all non-sharded, an "
        "untransposed default tile when that layout is TILE, no reduction other than add/multiply (and only on "
        "ROW_MAJOR), an unsharded output placement, and enough per-core L1 for the codegen plan). This entry never "
        "falls back to native, because a forced leg that quietly served native would make any comparison against "
        "native vacuous. Use ttnn::scatter if you want the case routed.");
    return scatter_codegen_dispatch(
        input_tensor, dim, index_tensor, source_tensor, output_memory_config, opt_reduction_string, sub_core_grid);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

// Writes all values from the tensor src into self at the indices specified in the index tensor.
// For each value in src, its output index is specified by its index in src for dimension != dim and by the
// corresponding value in index for dimension = dim. self, index and src (if it is a Tensor) should all have the same
// number of dimensions. It is also required that index.size(d) <= src.size(d) for all dimensions d, and that
// index.size(d) <= self.size(d) for all dimensions d != dim.Note that index and src do not broadcast.
Tensor scatter(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    namespace detail = ttnn::operations::data_movement::detail;
    namespace scatter_ns = ttnn::operations::data_movement::scatter;

    const MemoryConfig final_memory_config{
        output_memory_config.has_value() ? output_memory_config.value() : input_tensor.memory_config()};
    const bool use_codegen =
        detail::codegen_can_serve(
            input_tensor, dim, index_tensor, source_tensor, output_memory_config, opt_reduction_string) &&
        !scatter_ns::is_demoted(input_tensor, dim, index_tensor, source_tensor, final_memory_config);

    return use_codegen ? detail::scatter_codegen_dispatch(
                             input_tensor,
                             dim,
                             index_tensor,
                             source_tensor,
                             output_memory_config,
                             opt_reduction_string,
                             sub_core_grid)
                       : detail::scatter_native(
                             input_tensor,
                             dim,
                             index_tensor,
                             source_tensor,
                             output_memory_config,
                             opt_reduction_string,
                             sub_core_grid);
}

Tensor scatter_add(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return scatter(
        input_tensor, dim, index_tensor, source_tensor, output_memory_config, std::make_optional("add"), sub_core_grid);
}

}  // namespace ttnn
