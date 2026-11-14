// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "view.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "ttnn/types.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::reshape::detail {

namespace {
// The C++ overloads of ttnn.experimental.view are reused internally by data-movement ops that
// deliberately reinterpret the last dimension on interleaved tensors, so the shape validation lives
// here, at the Python entry point only.
void validate_view(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    auto tensor_shape = tensor.logical_shape();
    TT_FATAL(
        shape.volume() == tensor.logical_volume(),
        "Invalid view, logical volumes are changing from {} to {}",
        tensor.logical_volume(),
        shape.volume());
    // The last-dim and second-last-dim invariants describe the interleaved row/tile stride; sharded
    // tensors reinterpret their shard spec (a last-dim change is valid there), and zero-volume views
    // have no data to mis-stride. Match ttnn.view's check for everything else.
    if (!tensor.is_sharded() && tensor.logical_volume() > 0) {
        TT_FATAL(
            (tensor_shape[-1] == shape[-1]),
            "The last dimension can not change in view, attempting to change last dimension from {} to {}, use reshape "
            "instead",
            tensor_shape[-1],
            shape[-1]);
        const uint32_t tile_second_dim = tensor.tensor_spec().tile().get_height();
        const uint32_t shape_second_last_dim = shape.rank() >= 2 ? shape[-2] : 1;
        const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;
        TT_FATAL(
            (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) ||                  // Its row major
                (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
                ((shape_second_last_dim % tile_second_dim == 0) &&
                 (tensor_shape_second_last_dim % tile_second_dim == 0)),
            "Invalid second last dims for TILED reshape, from {} to {}, use reshape instead\n",
            tensor_shape_second_last_dim,
            shape_second_last_dim);
    }
}

ttnn::Tensor view_checked(const ttnn::Tensor& tensor, ttsl::Span<const int32_t> shape_vector) {
    auto shape = ttnn::operations::data_movement::detail::infer_dims_for_reshape(tensor, shape_vector);
    validate_view(tensor, shape);
    return ttnn::experimental::view(tensor, shape);
}

ttnn::Tensor view_checked(const ttnn::Tensor& tensor, const ttsl::SmallVector<int32_t>& shape_vector) {
    return view_checked(tensor, ttsl::Span<const int32_t>(shape_vector.data(), shape_vector.size()));
}

ttnn::Tensor view_checked(const ttnn::Tensor& tensor, int32_t N, int32_t C, int32_t H, int32_t W) {
    ttsl::SmallVector<int32_t> shape_vec{N, C, H, W};
    return view_checked(tensor, ttsl::Span<const int32_t>(shape_vec.data(), shape_vec.size()));
}
}  // namespace

void bind_view(nb::module_& mod) {
    const auto* doc = R"doc(
        Note:

        - It is recommended to use ttnn.reshape if you are not sure which operation to use
        - If this is the functionality required for your application, it will be called by ttnn.reshape
        - The following conditions must be met for the function not to corrupt your data:
            * the last dimension must not change
            * In Layout::TILE the second last two dimensions must not change OR there is no padding on the second last dimension

        Args:
            * input_tensor: Input Tensor.
            * shape: Shape of tensor.

        Returns:
            ttnn.Tensor: the output tensor with the new shape.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((2, 1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.experimental.view(tensor, (2, 1, 1, 4))
        )doc";

    ttnn::bind_function<"view", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, int32_t, int32_t, int32_t, int32_t)>(&view_checked),
            nb::arg("input_tensor"),
            nb::arg("N"),
            nb::arg("C"),
            nb::arg("H"),
            nb::arg("W")),
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, const ttsl::SmallVector<int32_t>&)>(&view_checked),
            nb::arg("input_tensor"),
            nb::arg("shape")));
}

}  // namespace ttnn::operations::experimental::reshape::detail
