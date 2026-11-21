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

    const auto* alias_doc = R"doc(
        Zero-copy TILE view of an interleaved device tensor's buffer with a new shape and dtype.

        The view shares the input's memory and keeps it alive. The bytes are reinterpreted, not converted, so
        the view's values are meaningful only for data written through the view.

        The intended input is a paged cache of shape [num_blocks, ...]. Interleaved page p lives on bank
        p % num_banks, so a block holds the same bytes in the input and in the view only when the block's
        page count is a multiple of the bank count. alias_view therefore requires:

        * the input to have rank 3 or more, TILE layout and interleaved memory, and the input and the view
          to have at least one element;
        * the page count of one dim-0 slice of the input to be a multiple of the bank count;
        * the bytes one dim-0 slice holds on each bank to be a whole number of view pages;
        * the view to have rank 3 or more, and each of its dim-0 slices to lie inside one dim-0 slice of
          the input or to cover whole dim-0 slices of the input;
        * the view to fit inside the input's allocation on every bank.

        Then dim-0 slice i of the input and view pages [i * n, (i + 1) * n) cover the same bytes, where
        n = (pages of one input slice) * (input page size) / (view page size). Example: on 8 DRAM banks a
        bfloat16 [N, 1, 1024, 256] input (256 tiles per block) can be viewed as float32 [N, 8, 128, 128]
        (128 tiles per block). On 7 banks the same view raises.

        With check_slices=False, only the first two rules and the last rule apply. Each dim-0 slice i of
        the input then still owns bytes [i * s, (i + 1) * s) on every bank, with
        s = (pages of one input slice) / num_banks * (input page size), but a view page may straddle two
        slices. View page q holds bytes [(q // num_banks) * p, (q // num_banks + 1) * p) of bank
        q % num_banks, with p the view page size. The caller must write only the view pages that lie
        wholly inside the slice it owns. Example: a bfloat8_b [N, 1, 1024, 256] input holds 34,816 bytes
        per block per bank, 8.5 float32 pages, so a float32 view needs check_slices=False, and each
        block then holds 8 whole float32 pages per bank.

        Args:
            * input_tensor: device tensor that owns the buffer.
            * shape: shape of the view.
            * dtype: dtype of the view.
            * check_slices: if True (default), raise unless every dim-0 slice of the input maps to whole
              view pages.
        )doc";
    ttnn::bind_function<"alias_view", "ttnn.experimental.">(
        mod,
        alias_doc,
        &ttnn::experimental::alias_view,
        nb::arg("input_tensor"),
        nb::arg("shape"),
        nb::arg("dtype"),
        nb::arg("check_slices") = true);
}

}  // namespace ttnn::operations::experimental::reshape::detail
