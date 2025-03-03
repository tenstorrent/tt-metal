// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"

namespace ttnn::experimental::xtensor {
namespace {

using ::tt::tt_metal::Tensor;

template <typename T, std::size_t... Indices>
auto vector_to_tuple_helper(const std::vector<T>& v, std::index_sequence<Indices...>) {
    return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto vector_to_tuple(const std::vector<T>& buffer) {
    TT_FATAL(buffer.size() >= N, "Buffer size {} is less than N {}", buffer.size(), N);
    return vector_to_tuple_helper(buffer, std::make_index_sequence<N>());
}

template <typename T, int N>
xt::xarray<T> concat_helper(const std::vector<xt::xarray<T>>& v, size_t axis = 0) {
    constexpr int FIXED_N = N < 2 ? 2 : N;
    TT_FATAL(N > 1, "Tuple size in concatenate must be greater than 1; got N: {}", N);
    auto tuple = vector_to_tuple<FIXED_N>(v);
    return xt::concatenate(std::move(tuple), axis);
}

template <class T, size_t... I>
consteval auto create_array_impl(std::index_sequence<I...>) {
    return std::array<xt::xarray<T> (*)(const std::vector<xt::xarray<T>>& v, size_t axis), sizeof...(I)>{
        concat_helper<T, I>...};
}

template <class T, size_t max>
consteval auto create_array() {
    return create_array_impl<T>(std::make_index_sequence<max>());
}

}  // namespace

template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& xtensor, int num_chunks, int dim) {
    TT_FATAL(num_chunks > 0, "num_chunks must be > 0; got num_chunks: {}", num_chunks);
    TT_FATAL(
        dim >= 0 && dim < xtensor.dimension(),
        "invalid dimension index; got dim: {}, tensor dimension: {}",
        dim,
        xtensor.dimension());

    const int size_along_dim = static_cast<int>(xtensor.shape()[dim]);
    TT_FATAL(
        num_chunks <= size_along_dim,
        "num_chunks cannot exceed the size of the tensor along the given dimension; got num_chunks: {}, "
        "size_along_dim: {}",
        num_chunks,
        size_along_dim);

    if (num_chunks == 1) {
        return {xtensor};
    }

    const int chunk_size = (size_along_dim + num_chunks - 1) / num_chunks;
    int remaining_size = size_along_dim;

    std::vector<xt::xarray<T>> chunks;
    chunks.reserve(static_cast<std::size_t>(num_chunks));
    int start = 0;
    int end = 0;
    for (int i = 0; i < num_chunks && end < size_along_dim; ++i) {
        int current_chunk_size = std::min(chunk_size, remaining_size);
        remaining_size -= current_chunk_size;
        end = start + current_chunk_size;

        // Build indices for slicing
        xt::xstrided_slice_vector indices(xtensor.dimension(), xt::all());
        indices[dim] = xt::range(start, end);

        auto chunk_view = xt::strided_view(xtensor, indices);

        // TODO: optimize away this copy.
        // Construct xarray from the view
        // This forces a copy of that slice into a new xarray
        chunks.push_back(xt::xarray<T>(chunk_view));
        start = end;
    }

    return chunks;
}

template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim) {
    constexpr size_t MAX_TUPLE_SIZE = 64;

    if (v.empty()) {
        return {};
    } else if (v.size() == 1) {
        return v.front();
    } else if (v.size() > MAX_TUPLE_SIZE) {
        TT_THROW(
            "Number of tensors to concatenate exceeds the maximum supported size {}; got size: {}",
            MAX_TUPLE_SIZE,
            v.size());
    } else {
        constexpr auto table = create_array<T, MAX_TUPLE_SIZE>();
        return (*table[v.size()])(v, dim);
    }
}

template xt::xarray<double> concat(const std::vector<xt::xarray<double>>& v, int dim);
template xt::xarray<float> concat(const std::vector<xt::xarray<float>>& v, int dim);
template xt::xarray<uint32_t> concat(const std::vector<xt::xarray<uint32_t>>& v, int dim);
template xt::xarray<int32_t> concat(const std::vector<xt::xarray<int32_t>>& v, int dim);

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

template <typename T>
std::vector<Tensor> chunk_impl(
    const Tensor& tensor, const tt::tt_metal::TensorLayout& layout, int num_chunks, int dim) {
    xt::xarray<T> xtensor = to_xtensor<T>(tensor);
    auto xtensor_chunks = chunk<T>(xtensor, num_chunks, dim);

    std::vector<Tensor> tensors;
    tensors.reserve(xtensor_chunks.size());
    for (const auto& c : xtensor_chunks) {
        TensorSpec chunk_spec(get_shape_from_xarray(c), layout);
        tensors.push_back(from_xtensor<T>(c, chunk_spec));
    }
    return tensors;
}

}  // namespace
}  // namespace adaptor

std::vector<Tensor> chunk(const Tensor& tensor, int num_chunks, int dim) {
    const auto& reference_layout = tensor.tensor_spec().tensor_layout();
    switch (reference_layout.get_data_type()) {
        case tt::tt_metal::DataType::BFLOAT4_B:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32:
            return adaptor::chunk_impl<float>(tensor, reference_layout, num_chunks, dim);
        case tt::tt_metal::DataType::INT32:
            return adaptor::chunk_impl<int32_t>(tensor, reference_layout, num_chunks, dim);
        case tt::tt_metal::DataType::UINT8:
            return adaptor::chunk_impl<uint8_t>(tensor, reference_layout, num_chunks, dim);
        case tt::tt_metal::DataType::UINT16:
            return adaptor::chunk_impl<uint16_t>(tensor, reference_layout, num_chunks, dim);
        case tt::tt_metal::DataType::UINT32:
            return adaptor::chunk_impl<uint32_t>(tensor, reference_layout, num_chunks, dim);
        default: TT_THROW("Unsupported data type: {}", reference_layout.get_data_type());
    }
}

Tensor concat(const std::vector<Tensor>& tensors, int dim) {
    TT_FATAL(tensors.size() > 0, "Cannot concatenate an empty list of tensors");
    const auto& reference_layout = tensors.front().tensor_spec().tensor_layout();
    switch (reference_layout.get_data_type()) {
        case tt::tt_metal::DataType::BFLOAT4_B:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32: return adaptor::concat_impl<float>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::INT32: return adaptor::concat_impl<int32_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT8: return adaptor::concat_impl<uint8_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT16: return adaptor::concat_impl<uint16_t>(tensors, reference_layout, dim);
        case tt::tt_metal::DataType::UINT32: return adaptor::concat_impl<uint32_t>(tensors, reference_layout, dim);
        default: TT_THROW("Unsupported data type: {}", reference_layout.get_data_type());
    }
}

}  // namespace ttnn::experimental::xtensor
