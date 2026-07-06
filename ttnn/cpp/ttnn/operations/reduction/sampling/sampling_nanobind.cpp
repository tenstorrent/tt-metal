// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/reduction/sampling/sampling.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_sampling_operation(nb::module_& mod) {
    const auto* doc =
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

        Args:
            input_values_tensor (ttnn.Tensor): The input tensor containing values to sample from.
            input_indices_tensor (ttnn.Tensor): The input tensor containing indices to assist with sampling.
            k (ttnn.Tensor): Top-k values for sampling.
            p (ttnn.Tensor): Top-p (nucleus) probabilities for sampling.
            temp (ttnn.Tensor): Temperature tensor for scaling (1/T).
            seed (int, optional): Seed for sampling randomness. Defaults to `0`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Core range set for multicore execution. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Note:
            This operations only supports inputs and outputs according to the following data types and layout:

            .. list-table:: input_values_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

            .. list-table:: input_indices_tensor, k
                :header-rows: 1

                * - dtype
                  - layout
                * - UINT32 (only on Wormhole/Blackhole), INT32
                  - ROW_MAJOR

            .. list-table:: p, temp
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - ROW_MAJOR

            .. list-table:: output_tensor (optional)
                :header-rows: 1

                * - dtype
                  - layout
                * - UINT32 (only on Wormhole/Blackhole), INT32
                  - ROW_MAJOR

            On Wormhole and Blackhole, both UINT32 and INT32 are supported for
            :attr:`input_indices_tensor`, :attr:`k`, and the output tensor. Otherwise (e.g. on
            Quasar, which does not support UINT32/UINT16 tile formats), only INT32 is supported.

            When :attr:`output_tensor` is not provided, the default output dtype is architecture-dependent:
            UINT32 on Wormhole/Blackhole, and INT32 otherwise.

        Returns:
            ttnn.Tensor: The output tensor containing sampled indices.

        Memory Support:
            - Interleaved: DRAM and L1

        Limitations:
            - Inputs must be 4D tensors with shape [N, C, H, W], and must be located on the device.
            - Input dims 0 and 1 (``N``, ``C``) must equal 1
            - Input dim 2 (``H``) represents the number of users, which must be in
              the range ``[1, 32]``. The op runs one core per user, so ``num_users`` cores are used.
            - The last dimension of:attr:`input_values_tensor` (``W``) must be padded to a multiple of 32
            - The number of tiles along the last dimension, ``Wt = W / 32``, must be a power of 2
              (i.e. ``W`` must be a power-of-2 multiple of 32: 32, 64, 128, 256, ...). The internal
              top-k stage uses a bitonic merge tree that assumes a power-of-2 tile count; a
              non-power-of-2 ``Wt`` is rejected. Pad ``W`` up to the next power-of-2 multiple of 32
              (e.g. with ``-inf`` values and dummy indices) if needed.
            - The overall shape of :attr:`input_values_tensor` must match that of :attr:`input_indices_tensor`.
            - :attr:`k`: Must be a 1D tensor of shape ``[num_users]`` (one value per user), in the range '(0,32]'.
            - :attr:`p`, :attr:`temp`: Must be 1D tensors of shape ``[num_users]`` (one value per user);
              :attr:`p` values must be in the range `[0.0, 1.0]`.
            - :attr:`sub_core_grids` (if provided): must supply at least ``num_users`` cores (1 to 32);
              only the first ``num_users`` cores are used and any extras are ignored.
        )doc";

    ttnn::bind_function<"sampling">(
        mod,
        doc,
        &ttnn::sampling,
        nb::arg("input_values_tensor").noconvert(),
        nb::arg("input_indices_tensor").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("p").noconvert(),
        nb::arg("temp").noconvert(),
        nb::kw_only(),
        nb::arg("seed") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("war_semaphore") = nb::none(),
        nb::arg("war_sem_drain_core") = nb::none());
}

}  // namespace ttnn::operations::reduction::detail
