// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/reduction/sampling/sampling.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;
void bind_reduction_sampling_operation(py::module& module) {
    auto doc =
        R"doc(
            Samples from the :attr:`input_values_tensor` based on provided top-k and top-p constraints.

            This operation samples values from the :attr:`input_values_tensor` based on the provided thresholds :attr:`k` (top-k sampling)
            and :attr:`p` (top-p nucleus sampling). The operation uses the :attr:`input_indices_tensor` for indexing and applies sampling
            under the given seed for reproducibility.

            The op first converts the :attr:`input_values_tensor` into probabilities by doing a softmax.

            In top-k sampling, the op considers only the k highest-probability values from the input distribution. The remaining values are ignored, regardless of their probabilities.
            In top-p sampling, the op selects values from the input distribution such that the cumulative probability mass is less than or equal to a threshold p.
            When combining top-k and top-p sampling, the op first applies the top-k filter and then the top-p filter.

            Within this selected corpus, multinomial sampling is applied. Multinomial sampling selects values from a given distribution by comparing each probability with a randomly generated number between 0 and 1. Specifically, the operation identifies the largest cumulative probability that exceeds the random threshold.

            The op finally returns input_indices_tensor[final_index]  where final_index is the index of the largest cumulative probability > random number found in the multinomial sampling.

            Currently, this operation supports inputs and outputs with specific memory layout and data type constraints.

            Equivalent PyTorch code:
                .. code-block:: python

                    return torch.sampling(
                        input_values_tensor,
                        input_indices_tensor,
                        k=k,
                        p=p,
                        temp=temp,
                        seed=seed,
                        optional_output_tensor=optional_output_tensor,
                    )

            Note:
                This operations only supports inputs and outputs according to the following data types and layout:

                .. list-table:: input_values_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16
                        - TILE


                .. list-table:: input_indices_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - UINT32, INT32
                        - ROW_MAJOR

                .. list-table:: k
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - UINT32
                        - ROW_MAJOR

                .. list-table:: p, temp
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16
                        - ROW_MAJOR

                If no :attr:`output_tensor` is provided, the return tensor will be as follows:
                .. list-table:: output_tensor (default)
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - UINT32
                        - ROW_MAJOR

                If :attr:`output_tensor` is provided, the supported data types and layout are:
                .. list-table:: output_tensor (if provided)
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - INT32, UINT32
                        - ROW_MAJOR
                Limitations:
                - The input tensors must represent exactly `32 users` based on their shape (i.e. N*C*H = 32).
                - The last dimension of:attr:`input_values_tensor` must be padded to a multiple of 32
                - The overall shape of :attr:`input_values_tensor` must match that of :attr:`input_indices_tensor`.
                - :attr:`k`: Must contain 32 values, in the range  '(0,32]'.
                - :attr:`p`, :attr:`temp`: Must contain 32 values in the range `[0.0, 1.0]`.
                - :attr:`sub_core_grids` (if provided): number of cores must equal the number of users (which is constrained to 32).
                - All tensors use an INTERLEAVED memory layout.

            Args:
                input_values_tensor (ttnn.Tensor): The input tensor containing values to sample from.
                input_indices_tensor (ttnn.Tensor): The input tensor containing indices to assist with sampling.
                k (ttnn.Tensor): Top-k values for sampling.
                p (ttnn.Tensor): Top-p (nucleus) probabilities for sampling.
                temp (ttnn.Tensor): Temperature tensor for scaling (1/T).
                seed (int, optional): Seed for sampling randomness. Defaults to `0`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Core range set for multicore execution. Defaults to `None`.
                optional_output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: The output tensor containing sampled indices.

            Example:
                input_tensor = ttnn.rand([1, 1, 32, 64], layout=ttnn.TILE_LAYOUT, device=device)
                input_indices_tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                k_tensor = ttnn.rand([32], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                p_tensor = ttnn.rand([32], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                temp_tensor = ttnn.rand([32], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

                output = ttnn.sampling(input_tensor, input_indices_tensor, k=k_tensor, p=p_tensor, temp=temp_tensor)

        )doc";

    using OperationType = decltype(ttnn::sampling);
    bind_registered_operation(
        module,
        ttnn::sampling,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_values_tensor,
               const ttnn::Tensor& input_indices_tensor,
               const ttnn::Tensor& k,
               const ttnn::Tensor& p,
               const ttnn::Tensor& temp,
               const std::optional<uint32_t>& seed,
               const std::optional<CoreRangeSet>& sub_core_grids,
               std::optional<ttnn::Tensor> optional_output_tensor) {
                return self(
                    input_values_tensor,
                    input_indices_tensor,
                    k,
                    p,
                    temp,
                    seed,
                    sub_core_grids,
                    optional_output_tensor);
            },
            py::arg("input_values_tensor").noconvert(),
            py::arg("input_indices_tensor").noconvert(),
            py::arg("k").noconvert(),
            py::arg("p").noconvert(),
            py::arg("temp").noconvert(),
            py::kw_only(),
            py::arg("seed").noconvert() = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
