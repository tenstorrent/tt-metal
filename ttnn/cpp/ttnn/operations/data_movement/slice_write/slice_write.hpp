// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct SliceWriteOperation {
    // template <typename T>
    // static void invoke(
    //     QueueId queue_id,
    //     const ttnn::Tensor& input_tensor,
    //     const ttnn::Tensor& output_tensor,
    //     tt::stl::Span<const T> begins,
    //     tt::stl::Span<const T> ends,
    //     tt::stl::Span<const T> step);

    // template <typename T>
    // static void invoke(
    //     const ttnn::Tensor& input_tensor,
    //     const ttnn::Tensor& output_tensor,
    //     tt::stl::Span<const T> output_tensor_start,
    //     tt::stl::Span<const T> output_tensor_end,
    //     tt::stl::Span<const T> step);

    // template <typename T>
    // static void invoke(
    //     QueueId queue_id,
    //     const ttnn::Tensor& input_tensor,
    //     const ttnn::Tensor& output_tensor,
    //     const ttnn::SmallVector<T>& begins,
    //     const ttnn::SmallVector<T>& ends,
    //     const ttnn::SmallVector<T>& step) {
    //     return invoke(
    //         queue_id,
    //         input_tensor,
    //         tt::stl::Span<const T>(begins),
    //         tt::stl::Span<const T>(ends),
    //         tt::stl::Span<const T>(step));
    // }

    // template <typename T>
    // static void invoke(
    //     const ttnn::Tensor& input_tensor,
    //     const ttnn::Tensor& output_tensor,
    //     const ttnn::SmallVector<T>& begins,
    //     const ttnn::SmallVector<T>& ends,
    //     const ttnn::SmallVector<T>& step) {
    //     return invoke(
    //         input_tensor,
    //         tt::stl::Span<const T>(begins),
    //         tt::stl::Span<const T>(ends),
    //         tt::stl::Span<const T>(step));
    // }

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& output_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step);

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& output_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice_write = ttnn::
    register_operation_with_auto_launch_op<"ttnn::slice_write", ttnn::operations::data_movement::SliceWriteOperation>();

}  // namespace ttnn
