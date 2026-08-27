// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/reduction/topk/topk.hpp"

namespace ttnn::operations::reduction::detail {

void bind_reduction_topk_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns the :attr:`k` largest or :attr:`k` smallest elements of the :attr:`input_tensor` along a given dimension :attr:`dim`.

            If :attr:`dim` is not provided, the last dimension of the :attr:`input_tensor` is used.

            If :attr:`largest` is True, the :attr:`k` largest elements are returned. Otherwise, the :attr:`k` smallest elements are returned.

            The boolean option :attr:`sorted` if True, will make sure that the returned :attr:`k` elements are sorted.

            Equivalent PyTorch code:

            .. code-block:: python

                return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted, *, output_tensor=None)

            Args:
                input_tensor (ttnn.Tensor): the input tensor. Must be on the device.
                k (number): the number of top elements to look for.
                dim (number): the dimension to reduce.
                largest (bool): whether to return the largest or the smallest elements. Defaults to `True`.
                sorted (bool): whether to return the elements in sorted order. Defaults to `True`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                output_tensor (tuple[ttnn.Tensor, ttnn.Tensor], optional): A tuple with preallocated output tensors for the values and indices. If specified, must be on the same device as :attr:`input_tensor`. Defaults to (`None`, `None`).
                sub_core_grids (ttnn.CoreRangeSet, optional): Core range set to run the operation on. Defaults to `None`.
                indices_tensor (ttnn.Tensor, optional): Input tensor containing pre-computed index values. When provided, the operation returns the labels held in this tensor for the selected elements instead of generating positional indices. It must have the same logical shape as :attr:`input_tensor`, be in TILE layout, and be UINT16, UINT32, or INT32. Its width must match the resolved output index dtype: a UINT16 tensor is rejected when 32-bit indices are required (reduced dimension above 65535, or a `float32` :attr:`input_tensor`), and a UINT32/INT32 tensor widens the output indices to 32-bit. Defaults to `None`.
                stable (bool, optional): EXPERIMENTAL, best effort only -- do not rely on this for correctness. Asks the LLK's stable bitonic network to break exact-value ties by lowest index rather than by array position. The stable network is an open issue (tenstorrent/tt-metal#33492): it can still return incorrect indices for tied values, and every stable case in the LLK test suite is currently skipped, so a caller passing `True` may get either tie-break. Only Wormhole B0 and Blackhole implement it at all; other architectures raise. Off by default. Defaults to `False`.

            Returns:
                tuple[ttnn.Tensor, ttnn.Tensor]: a tuple of (values_tensor, indices_tensor).

            Note:
                The :attr:`input_tensor` supports the following data type and layout:

                .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT8, BFLOAT16, FLOAT32
                      - TILE

                .. list-table:: index_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - UINT16, UINT32, INT32
                      - TILE

                The :attr:`output_value_tensor` will have the same data type as :attr:`input_tensor` and will be in TILE layout.
                The :attr:`output_index_tensor` will be in TILE layout. Its data type is UINT16 or UINT32 by default (chosen
                based on the reduced dimension size), widened to 32-bit by a UINT32 or INT32 :attr:`indices_tensor`, or
                matching the preallocated index tensor dtype (UINT16, UINT32, or INT32) when one is provided.

            Memory Support:
                - Interleaved: DRAM and L1

            Limitations:
                - Inputs must be located on-device.
                - The op fundamentally operates on 4D tensors with shape [N, C, H, W], and with :attr:`dim` of -1. The tensor will be manipulated as needed when this is not the case, and restored afterwards.
                - For :attr:`input_tensor`, N*C*H must be a multiple of 32
                - W is ideally ≥64. If this is not the case the op will pad the tensor to satisfy this constraint.
                - The width of :attr:`input_tensor` along :attr:`dim` should be a multiple of tile width, and will be padded to the nearest multiple of tile width if needed.
                - The padding is currently only supported for bfloat16, float32, int32, and uint32.
                - Multi-core execution is selected automatically when :attr:`k` is at most 64 and the size of :attr:`input_tensor` along :attr:`dim` is a power of two no larger than 32768. That size must normally be at least 8192; the floor drops to 1024 when the input spans at most 2 tile rows after tile padding (64 rows with the default 32x32 tile). Nothing needs to be passed to opt in — shapes outside these bounds, or qualifying shapes that do not fit the available core grid and L1 memory (which can depend on data type), automatically run on a single core with identical results.
                - On Blackhole, wide bfloat16 inputs with :attr:`largest` set and otherwise default arguments may instead be served transparently by a faster composite implementation for certain width and :attr:`k` ranges.
                - All shape validations are performed on padded shapes.
                - Sharded output memory configs are not supported for this operation.
        )doc";

    ttnn::bind_function<"topk">(
        mod,
        doc,
        &ttnn::topk,
        nb::arg("input_tensor").noconvert(),
        nb::arg("k") = 32,
        nb::arg("dim") = -1,
        nb::arg("largest") = true,
        nb::arg("sorted") = true,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("indices_tensor") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("stable") = false);
}

}  // namespace ttnn::operations::reduction::detail
