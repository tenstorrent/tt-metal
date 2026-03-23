// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn {

namespace operations::creation {

namespace detail {

template <typename T>
Tensor arange_impl(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout = Layout::ROW_MAJOR,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const MemoryConfig& output_mem_config = ttnn::DRAM_MEMORY_CONFIG);

template <typename T>
Tensor full_impl(
    const ttnn::Shape& shape,
    T value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);

}  // namespace detail

template <typename T>
Tensor full_impl(
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    MeshDevice* device = nullptr,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

template <typename T>
Tensor full_like_impl(
    const Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

Tensor zeros(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor ones(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

template <typename FillValueType>
    requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
Tensor full(
    const ttnn::Shape& shape,
    const FillValueType fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor empty(
    const ttnn::Shape& shape,
    const DataType& dtype,
    const Layout& layout,
    MeshDevice* device,
    const MemoryConfig& memory_config);

template <typename BufferType>
Tensor from_buffer(
    std::vector<BufferType>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

template <typename BufferType>
Tensor from_buffer(
    const std::vector<BufferType>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

template <typename FillValueType>
    requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
Tensor full_like(
    const Tensor& tensor,
    const FillValueType fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor zeros_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor ones_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor empty_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor arange(
    const int64_t start,
    const int64_t stop,
    const int64_t step = 1,
    const DataType dtype = ttnn::DataType::BFLOAT16,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const Layout layout = Layout::ROW_MAJOR);

Tensor arange(
    const int64_t stop,
    const DataType dtype = DataType::BFLOAT16,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const Layout layout = Layout::ROW_MAJOR);

}  // namespace operations::creation

using operations::creation::arange;
using operations::creation::empty;
using operations::creation::empty_like;
using operations::creation::from_buffer;
using operations::creation::full;
using operations::creation::full_like;
using operations::creation::ones;
using operations::creation::ones_like;
using operations::creation::zeros;
using operations::creation::zeros_like;

}  // namespace ttnn
