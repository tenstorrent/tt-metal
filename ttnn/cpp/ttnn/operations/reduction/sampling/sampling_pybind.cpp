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
            Samples from the input tensor based on provided top-k and top-p constraints.

            This operation samples values from the input tensor `input_values_tensor` based on the provided thresholds `k` (top-k sampling)
            and `p` (top-p nucleus sampling). The operation uses the `input_indices_tensor` for indexing and applies sampling
            under the given seed for reproducibility.

            The op first converts the input_values_tensor into probabilities by doing a softmax.

            In top-k sampling, the op considers only the k highest-probability values from the input distribution. The remaining values are ignored, regardless of their probabilities.
            In top-p sampling, the op selects values from the input distribution such that the cumulative probability mass is less than or equal to a threshold p.
            When combining top-k and top-p sampling, the op first applies the top-k filter and then the top-p filter.

            Within this selected corpus, multinomial sampling is applied. Multinomial sampling selects values from a given distribution by comparing each probability with a randomly generated number between 0 and 1. Specifically, the operation identifies the largest cumulative probability that exceeds the random threshold.

            The op finally returns input_indices_tensor[final_index]  where final_index is the index of the largest cumulative probability > random number found in the multinomial sampling.

            Currently, this operation supports inputs and outputs with specific memory layout and data type constraints.

            Constraints:
                - `input_values_tensor`:
                    - Must have `BFLOAT16` data type.
                    - Must have `TILE` layout.
                    - Must have `INTERLEAVED` memory layout.
                    - Must be padded to a multiple of 32 on the last dim
                - `input_indices_tensor`:
                    - Must have `UINT32` or `INT32` data type.
                    - Must have `ROW_MAJOR` layout.
                    - Must have the same shape as `input_values_tensor`.
                - The input tensors must represent exactly `32 users` (based on their shape).
                - `k`: All values in the list must be >0 and ≤ 32.
                - `p`: All values in the list must be in the range `[0.0, 1.0]`.
                - Output tensor (if provided):
                    - Must have `UINT32` or `INT32` data type.
                    - Must have `INTERLEAVED` memory layout.

            Equivalent PyTorch code:

            .. code-block:: python

                return torch.sampling(
                    input_values_tensor,
                    input_indices_tensor,
                    k=k,
                    p=p,
                    seed=seed,
                    optional_output_tensor=optional_output_tensor,
                    queue_id=queue_id
                )

            Args:
                input_values_tensor (ttnn.Tensor): The input tensor containing values to sample from.
                input_indices_tensor (ttnn.Tensor): The input tensor containing indices to assist with sampling.
                k (List[int]): Top-k values for sampling.
                p (List[float]): Top-p (nucleus) probabilities for sampling.
                seed (int, optional): Seed for sampling randomness. Defaults to `0`.
                optional_output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                queue_id (int, optional): Command queue ID for execution. Defaults to `0`.

            Returns:
                ttnn.Tensor: The output tensor containing sampled indices.
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
               const std::vector<uint16_t>& k,
               const std::vector<float>& p,
               const uint32_t seed,
               const std::optional<CoreRangeSet>& sub_core_grids,
               std::optional<ttnn::Tensor> optional_output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_values_tensor,
                    input_indices_tensor,
                    k,
                    p,
                    seed,
                    sub_core_grids,
                    optional_output_tensor);
            },
            py::arg("input_values_tensor").noconvert(),
            py::arg("input_indices_tensor").noconvert(),
            py::kw_only(),
            py::arg("k").noconvert(),
            py::arg("p").noconvert(),
            py::arg("seed").noconvert() = 0,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::reduction::detail
