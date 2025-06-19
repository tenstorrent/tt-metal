
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"
#include <type_traits>

#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace ttnn::operations::rand {

// the effect of instantiating a template uniform_int_distribution is undefined
// unless IntType satisfy the following concept according to 26.6.2.1 General requirements [rand.req.genl]
template <typename T>
concept IntType =
    std::is_same_v<T, short> || std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned short> || std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long>;

template <typename T>
concept RandElemType = IntType<T> or std::floating_point<T>;

template <RandElemType ElemType>
using uniform_distribution = std::conditional_t<
    std::floating_point<ElemType>,
    std::uniform_real_distribution<ElemType>,
    std::uniform_int_distribution<ElemType>>;

template <typename ElemType>
static tt::tt_metal::Tensor genRandomTensor(ElemType low, ElemType high, const ttnn::TensorSpec& spec) {
    auto total_elems = spec.padded_shape().volume();
    auto output_buffer = std::vector<ElemType>(total_elems);

    auto init_rand_elem = [low, high](auto& elem) {
        thread_local std::mt19937 rng(std::random_device{}());
        if constexpr (std::is_same_v<ElemType, ::bfloat16>) {
            uniform_distribution<float> dist(low.to_float(), high.to_float());
            elem = ElemType(dist(rng));
        } else {
            uniform_distribution<ElemType> dist(low, high);
            elem = ElemType(dist(rng));
        }
    };
    if (total_elems <= 256 * 256) {
        std::ranges::for_each(output_buffer, init_rand_elem);
        return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec);
    } else {
        tf::Executor executor;
        tf::Taskflow taskflow;

        taskflow.for_each(output_buffer.begin(), output_buffer.end(), init_rand_elem);
        executor.run(taskflow).wait();

        return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec);
    }
}

static tt::tt_metal::Tensor genRandomTensor(
    const ttnn::Shape& shape,
    const DataType dtype = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR,
    const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    ttnn::TensorSpec spec(shape, TensorLayout(dtype, PageConfig(layout), memory_config));
    switch (dtype) {
        case DataType::UINT16: return genRandomTensor(uint16_t(0), uint16_t(1), spec);
        case DataType::UINT32: return genRandomTensor(0u, 1u, spec);
        case DataType::FLOAT32: return genRandomTensor(0.0f, 1.0f, spec);
        case DataType::BFLOAT16: return genRandomTensor(::bfloat16(0.0f), ::bfloat16(1.0f), spec);
        case DataType::INT32: return genRandomTensor(int32_t(0), int32_t(1), spec);
        default: TT_THROW("Unsupported DataType!");
    };
}

Tensor Rand::invoke(QueueId queue_id, const ttnn::Shape& shape, const DataType dtype, const Layout layout) {
    return genRandomTensor(ttnn::Shape{shape}, dtype, layout);
}

Tensor Rand::invoke(
    QueueId queue_id,
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config) {
    auto output = genRandomTensor(ttnn::Shape{shape}, dtype, layout);
    return output.to_device(std::addressof(device), memory_config);
}
}  // namespace ttnn::operations::rand
