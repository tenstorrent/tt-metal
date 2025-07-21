// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/xtensor/partition.hpp"

#include <type_traits>
#include <algorithm>

#include <tt_stl/span.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xdynamic_view.hpp>
#include <xtensor/containers/xstorage.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/views/xview.hpp>

namespace ttnn::experimental::xtensor {
namespace {

tt::stl::SmallVector<int> normalize_dims(const tt::stl::SmallVector<int>& dims, size_t tensor_dims) {
    tt::stl::SmallVector<int> normalized_dims;
    std::transform(dims.begin(), dims.end(), std::back_inserter(normalized_dims), [tensor_dims](int dim) {
        return dim < 0 ? dim + static_cast<int>(tensor_dims) : dim;
    });
    TT_FATAL(
        std::all_of(
            normalized_dims.begin(),
            normalized_dims.end(),
            [tensor_dims](int dim) { return dim >= 0 && dim < tensor_dims; }),
        "Invalid dimension index; got dims: {}, tensor dimension: {}",
        dims,
        tensor_dims);
    return normalized_dims;
}

}  // namespace

template <typename Expression>
StridedViews<Expression> chunk_ndim(
    const xt::xexpression<Expression>& expr_base,
    const tt::stl::SmallVector<int>& num_chunks,
    const tt::stl::SmallVector<int>& dims) {
    const auto& expr = expr_base.derived_cast();
    TT_FATAL(num_chunks.size() == dims.size(), "num_chunks and dims must have the same size");

    if (num_chunks.empty()) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        return StridedViews<Expression>{xt::strided_view(expr, indices)};
    }

    const auto normalized_dims = normalize_dims(dims, expr.dimension());
    auto sorted_dims = normalized_dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());
    TT_FATAL(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(), "dims must be unique");

    TT_FATAL(
        std::all_of(num_chunks.begin(), num_chunks.end(), [](int size) { return size > 0; }),
        "num_chunks must be > 0; got num_chunks: {}",
        num_chunks);

    const size_t dims_size = normalized_dims.size();
    tt::stl::SmallVector<std::vector<std::pair<int, int>>> dim_ranges;
    tt::stl::SmallVector<size_t> num_chunks_per_dim;
    for (size_t i = 0; i < dims_size; ++i) {
        const int dim = normalized_dims[i];
        const int num_chunks_along_dim = num_chunks[i];
        const int size_along_dim = static_cast<int>(expr.shape()[dim]);
        const int chunk_size = (size_along_dim + num_chunks_along_dim - 1) / num_chunks_along_dim;

        std::vector<std::pair<int, int>> ranges;
        ranges.reserve(num_chunks_along_dim);

        int start = 0;
        for (int chunk_idx = 0; chunk_idx < num_chunks_along_dim && start < size_along_dim; ++chunk_idx) {
            const int current_chunk_size = std::min(chunk_size, size_along_dim - start);
            const int end = start + current_chunk_size;
            ranges.emplace_back(start, end);
            start = end;
        }
        num_chunks_per_dim.push_back(ranges.size());
        dim_ranges.push_back(std::move(ranges));
    }

    const size_t total_chunks =
        std::accumulate(num_chunks_per_dim.begin(), num_chunks_per_dim.end(), 1, std::multiplies<size_t>());

    StridedViews<Expression> chunk_views;
    tt::stl::SmallVector<size_t> current_indices(dims_size, 0);
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        for (size_t i = 0; i < dims_size; ++i) {
            const int dim = normalized_dims[i];
            const auto [start, end] = dim_ranges[i][current_indices[i]];
            indices[dim] = xt::range(start, end);
        }

        chunk_views.push_back(xt::strided_view(expr, indices));

        for (int i = static_cast<int>(dims_size) - 1; i >= 0; --i) {
            if (++current_indices[i] < num_chunks_per_dim[i]) {
                break;
            }
            current_indices[i] = 0;
        }
    }

    return chunk_views;
}

template <typename Expression>
StridedViews<Expression> chunk(const xt::xexpression<Expression>& expr, int num_chunks, int dim) {
    return chunk_ndim(expr, {num_chunks}, {dim});
}

template <typename Expression>
XtensorAdapter<typename Expression::value_type> concat_ndim(
    const std::vector<Expression>& expressions,
    const tt::stl::SmallVector<int>& num_chunks,
    const tt::stl::SmallVector<int>& dims) {
    using DataType = typename Expression::value_type;

    TT_FATAL(num_chunks.size() == dims.size(), "num_chunks and dims must have the same size");

    if (expressions.empty()) {
        return XtensorAdapter<DataType>(std::vector<DataType>(), {0});
    }

    if (num_chunks.empty()) {
        TT_FATAL(expressions.size() == 1, "When no dims specified, must have exactly one expression");
        std::vector<DataType> data(expressions.front().begin(), expressions.front().end());
        std::vector<size_t> shape_vec(expressions.front().shape().cbegin(), expressions.front().shape().cend());
        return XtensorAdapter<DataType>(std::move(data), std::move(shape_vec));
    }

    const auto& first_expr = expressions.front();
    const auto& expected_shape = first_expr.shape();
    const size_t num_dims = first_expr.dimension();
    for (const auto& expr : expressions) {
        TT_FATAL(expr.dimension() == num_dims, "All expressions must have the same number of dimensions");
        TT_FATAL(expr.shape() == first_expr.shape(), "All expressions must have the same shape");
    }
    const auto normalized_dims = normalize_dims(dims, num_dims);
    auto sorted_dims = normalized_dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());
    TT_FATAL(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(), "dims must be unique");

    TT_FATAL(
        std::all_of(num_chunks.begin(), num_chunks.end(), [](int n) { return n > 0; }),
        "num_chunks must be > 0; got num_chunks: {}",
        num_chunks);

    const size_t expected_total = std::accumulate(num_chunks.begin(), num_chunks.end(), 1, std::multiplies<int>());
    TT_FATAL(
        expressions.size() == expected_total,
        "Number of expressions ({}) doesn't match expected ({})",
        expressions.size(),
        expected_total);

    std::vector<size_t> result_shape(expected_shape.cbegin(), expected_shape.cend());
    for (size_t i = 0; i < dims.size(); ++i) {
        const int dim = normalized_dims[i];
        result_shape[dim] *= num_chunks[i];
    }
    const size_t result_volume =
        std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    XtensorAdapter<DataType> result(std::vector<DataType>(result_volume), std::move(result_shape));

    // An optimization for concatenating along the outer dimension.
    if (normalized_dims.size() == 1) {
        // Check if all dimensions before the concat dimension have size 1.
        bool can_use_memcpy = true;
        for (int d = 0; d < normalized_dims[0]; ++d) {
            if (expected_shape[d] != 1) {
                can_use_memcpy = false;
                break;
            }
        }

        if (can_use_memcpy) {
            DataType* result_ptr = result.data().data();
            const size_t chunk_size =
                std::accumulate(expected_shape.begin(), expected_shape.end(), 1, std::multiplies<size_t>());
            size_t offset = 0;
            for (const auto& expr : expressions) {
                std::memcpy(result_ptr + offset, expr.data(), chunk_size * sizeof(DataType));
                offset += chunk_size;
            }
            return result;
        }
    }

    // Get the size of each piece along concatenation dimensions
    tt::stl::SmallVector<size_t> piece_sizes(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        piece_sizes[i] = expected_shape[normalized_dims[i]];
    }

    // Copy pieces into result in row-major order
    tt::stl::SmallVector<size_t> current_indices(dims.size(), 0);
    for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
        const auto& expr = expressions[expr_idx];

        xt::xdynamic_slice_vector indices(num_dims, xt::all());
        for (size_t i = 0; i < dims.size(); ++i) {
            const int dim = normalized_dims[i];
            const size_t offset = current_indices[i] * piece_sizes[i];
            indices[dim] = xt::range(offset, offset + piece_sizes[i]);
        }

        xt::dynamic_view(result.expr(), indices) = expr;

        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            if (++current_indices[i] < num_chunks[i]) {
                break;
            }
            current_indices[i] = 0;
        }
    }

    return result;
}

template <typename Expression>
XtensorAdapter<typename Expression::value_type> concat(const std::vector<Expression>& v, int dim) {
    return concat_ndim<Expression>(v, {v.size()}, {dim});
}

// Adaptor APIs from xtensor to ttnn::Tensor.
namespace adaptor {
namespace {

template <typename T>
Tensor concat_impl(const std::vector<Tensor>& tensors, const tt::tt_metal::TensorLayout& layout, int dim) {
    std::vector<xt::xarray<T>> xtensors;
    xtensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        xtensors.push_back(to_xtensor<T>(tensor));
    }
    xt::xarray<T> result(concat(xtensors, dim).expr());
    return from_xtensor<T>(result, TensorSpec(get_shape_from_xarray(result), layout));
}

}  // namespace
}  // namespace adaptor

Tensor concat(const std::vector<Tensor>& tensors, int dim) {
    TT_FATAL(tensors.size() > 0, "Cannot concatenate an empty list of tensors");
    const auto& reference_layout = tensors.front().tensor_spec().tensor_layout();
    switch (reference_layout.get_data_type()) {
        case tt::tt_metal::DataType::BFLOAT4_B:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::FLOAT32: return adaptor::concat_impl<float>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::BFLOAT16: return adaptor::concat_impl<bfloat16>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::INT32: return adaptor::concat_impl<int32_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT8: return adaptor::concat_impl<uint8_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT16: return adaptor::concat_impl<uint16_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT32: return adaptor::concat_impl<uint32_t>(tensors, reference_layout, dim);
        default: TT_THROW("Unsupported data type: {}", reference_layout.get_data_type());
    }
}

// Explicit instantiations for the public API.
#define EXPLICIT_INSTANTIATIONS_FOR_TYPE(T)                                                                          \
    template StridedViews<xt::xarray<T>> chunk(const xt::xexpression<xt::xarray<T>>&, int, int);                     \
    template StridedViews<xt::xarray<T>> chunk_ndim(                                                                 \
        const xt::xexpression<xt::xarray<T>>&, const tt::stl::SmallVector<int>&, const tt::stl::SmallVector<int>&);  \
    template StridedViews<AdaptedView<T>> chunk(const xt::xexpression<AdaptedView<T>>&, int, int);                   \
    template StridedViews<AdaptedView<T>> chunk_ndim(                                                                \
        const xt::xexpression<AdaptedView<T>>&, const tt::stl::SmallVector<int>&, const tt::stl::SmallVector<int>&); \
    template StridedViews<AdaptedView<const T>> chunk(const xt::xexpression<AdaptedView<const T>>&, int, int);       \
    template StridedViews<AdaptedView<const T>> chunk_ndim(                                                          \
        const xt::xexpression<AdaptedView<const T>>&,                                                                \
        const tt::stl::SmallVector<int>&,                                                                            \
        const tt::stl::SmallVector<int>&);                                                                           \
    template XtensorAdapter<T> concat(const std::vector<xt::xarray<T>>& v, int dim);                                 \
    template XtensorAdapter<T> concat_ndim(                                                                          \
        const std::vector<xt::xarray<T>>& v,                                                                         \
        const tt::stl::SmallVector<int>& num_chunks,                                                                 \
        const tt::stl::SmallVector<int>& dims);                                                                      \
    template XtensorAdapter<T> concat(const std::vector<AdaptedView<T>>& v, int dim);                                \
    template XtensorAdapter<T> concat_ndim(                                                                          \
        const std::vector<AdaptedView<T>>& v,                                                                        \
        const tt::stl::SmallVector<int>& num_chunks,                                                                 \
        const tt::stl::SmallVector<int>& dims);                                                                      \
    template XtensorAdapter<T> concat(const std::vector<AdaptedView<const T>>& v, int dim);                          \
    template XtensorAdapter<T> concat_ndim(                                                                          \
        const std::vector<AdaptedView<const T>>& v,                                                                  \
        const tt::stl::SmallVector<int>& num_chunks,                                                                 \
        const tt::stl::SmallVector<int>& dims);

EXPLICIT_INSTANTIATIONS_FOR_TYPE(bfloat16)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(float)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(double)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(int32_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint8_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint16_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint32_t)

#undef EXPLICIT_INSTANTIATIONS_FOR_TYPE

}  // namespace ttnn::experimental::xtensor
