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

template <typename T>
auto chunk_xexpression(
    const xt::xexpression<T>& expr_base, tt::stl::SmallVector<int> num_chunks, tt::stl::SmallVector<int> dims) {
    const auto& expr = expr_base.derived_cast();
    if (num_chunks.empty()) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        return StridedViews<T>{xt::strided_view(expr, indices)};
    }

    TT_FATAL(num_chunks.size() == dims.size(), "num_chunks and dims must have the same size");
    auto sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());
    TT_FATAL(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(), "dims must be unique");
    TT_FATAL(
        std::all_of(num_chunks.begin(), num_chunks.end(), [](size_t size) { return size > 0; }),
        "num_chunks must be > 0; got num_chunks: {}",
        num_chunks);
    TT_FATAL(
        std::all_of(dims.begin(), dims.end(), [&expr](size_t dim) { return dim >= 0 && dim < expr.dimension(); }),
        "invalid dimension index; got dims: {}, tensor dimension: {}",
        dims,
        expr.dimension());

    tt::stl::SmallVector<std::vector<std::pair<int, int>>> dim_ranges;
    tt::stl::SmallVector<size_t> num_chunks_per_dim;
    for (size_t i = 0; i < dims.size(); ++i) {
        int dim = dims[i];
        int num_chunks_along_dim = num_chunks[i];
        int size_along_dim = static_cast<int>(expr.shape()[dim]);
        int chunk_size = (size_along_dim + num_chunks_along_dim - 1) / num_chunks_along_dim;

        std::vector<std::pair<int, int>> ranges;
        ranges.reserve(num_chunks_along_dim);

        int start = 0;
        for (int chunk_idx = 0; chunk_idx < num_chunks_along_dim && start < size_along_dim; ++chunk_idx) {
            int current_chunk_size = std::min(chunk_size, size_along_dim - start);
            int end = start + current_chunk_size;
            ranges.emplace_back(start, end);
            start = end;
        }
        num_chunks_per_dim.push_back(ranges.size());
        dim_ranges.push_back(std::move(ranges));
    }

    const size_t total_chunks =
        std::accumulate(num_chunks_per_dim.begin(), num_chunks_per_dim.end(), 1, std::multiplies<size_t>());

    StridedViews<T> chunk_views;
    tt::stl::SmallVector<size_t> current_indices(dims.size(), 0);
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        for (size_t i = 0; i < dims.size(); ++i) {
            int dim = dims[i];
            auto [start, end] = dim_ranges[i][current_indices[i]];
            indices[dim] = xt::range(start, end);
        }

        auto chunk_view = xt::strided_view(expr, indices);
        chunk_views.push_back(chunk_view);

        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            if (++current_indices[i] < num_chunks_per_dim[i]) {
                break;
            }
            current_indices[i] = 0;
        }
    }

    return chunk_views;
}

// Helper to compute adapted types for explicit instantiations
template <typename T>
auto compute_adapted_type() -> decltype(xt::adapt(
    std::declval<T*>(), std::declval<size_t>(), xt::no_ownership(), std::declval<std::vector<size_t>>()));

template <typename T>
using AdaptedType = decltype(compute_adapted_type<T>());

}  // namespace

template <typename T>
StridedViews<T> chunk(const xt::xexpression<T>& expr, int num_chunks, int dim) {
    return chunk_xexpression(expr, {num_chunks}, {dim});
}

template <typename T>
StridedViews<T> chunk_ndim(
    const xt::xexpression<T>& expr, tt::stl::SmallVector<int> num_chunks, tt::stl::SmallVector<int> dims) {
    return chunk_xexpression(expr, num_chunks, dims);
}

// TODO: optimize concat to perform concatenation based off views.
template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim) {
    if (v.empty()) {
        return {};
    } else if (v.size() == 1) {
        return v.front();
    } else {
        // Make sure all input tensors have the same dimensions except for the concatenation dimension
        if (dim < 0) {
            dim += static_cast<int>(v.front().dimension());
        }
        TT_FATAL(
            dim >= 0 && dim < static_cast<int>(v.front().dimension()),
            "Invalid concatenation dimension {}, tensor dimension: {}",
            dim,
            v.front().dimension());

        size_t num_dims = v.front().dimension();
        auto expected_shape = v.front().shape();
        for (size_t i = 1; i < v.size(); ++i) {
            TT_FATAL(v[i].dimension() == num_dims, "All tensors must have the same number of dimensions");
            for (size_t j = 0; j < num_dims; ++j) {
                if (j != dim) {
                    TT_FATAL(
                        v[i].shape()[j] == expected_shape[j],
                        "All tensors must have the same shape except for the concatenation dimension. Dimension {} "
                        "differes, expected: {}, got: {}",
                        j,
                        expected_shape[j],
                        v[i].shape()[j]);
                }
            }
        }

        auto result_shape = v.front().shape();
        for (size_t i = 1; i < v.size(); ++i) {
            result_shape[dim] += v[i].shape()[dim];
        }
        xt::xarray<T> result;
        result.resize(result_shape);
        xt::xdynamic_slice_vector indices(num_dims, xt::all());
        size_t offset = 0;
        // TODO: Since source and destination tensors are contiguous. We can potentially optimize
        // when concatenating along the last dimension and do memcpy.
        for (size_t i = 0; i < v.size(); ++i) {
            size_t dim_size = v[i].shape()[dim];
            indices[dim] = xt::range(offset, offset + dim_size);
            auto view = xt::dynamic_view(result, indices);
            view = v[i];
            offset += dim_size;
        }
        return result;
    }
}

// Explicit instantiations for the public API.
#define EXPLICIT_INSTANTIATIONS_FOR_TYPE(T)                                                                    \
    template StridedViews<xt::xarray<T>> chunk(const xt::xexpression<xt::xarray<T>>&, int, int);               \
    template StridedViews<xt::xarray<T>> chunk_ndim(                                                           \
        const xt::xexpression<xt::xarray<T>>&, tt::stl::SmallVector<int>, tt::stl::SmallVector<int>);          \
    template StridedViews<AdaptedType<T>> chunk(const xt::xexpression<AdaptedType<T>>&, int, int);             \
    template StridedViews<AdaptedType<T>> chunk_ndim(                                                          \
        const xt::xexpression<AdaptedType<T>>&, tt::stl::SmallVector<int>, tt::stl::SmallVector<int>);         \
    template StridedViews<AdaptedType<const T>> chunk(const xt::xexpression<AdaptedType<const T>>&, int, int); \
    template StridedViews<AdaptedType<const T>> chunk_ndim(                                                    \
        const xt::xexpression<AdaptedType<const T>>&, tt::stl::SmallVector<int>, tt::stl::SmallVector<int>);   \
    template xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim);

EXPLICIT_INSTANTIATIONS_FOR_TYPE(bfloat16)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(float)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(double)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(int32_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint8_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint16_t)
EXPLICIT_INSTANTIATIONS_FOR_TYPE(uint32_t)

#undef EXPLICIT_INSTANTIATIONS_FOR_TYPE

// Adaptor APIs from xtensor to ttnn::Tensor.
namespace adaptor {
namespace {

template <typename T>
Tensor concat_impl(const std::vector<Tensor>& tensors, const tt::tt_metal::TensorLayout& layout, int dim) {
    std::vector<xt::xarray<T>> xtensors;
    for (const auto& tensor : tensors) {
        xtensors.push_back(to_xtensor<T>(tensor));
    }
    xt::xarray<T> result = concat(xtensors, dim);
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

}  // namespace ttnn::experimental::xtensor
