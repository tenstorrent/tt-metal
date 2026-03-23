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
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn {

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

}  // namespace ttnn
