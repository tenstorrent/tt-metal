// // SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// //
// // SPDX-License-Identifier: Apache-2.0

#include "roll_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/roll/roll.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename data_movement_operation_t>
void bind_roll_op(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int>& shifts,
               const ttnn::SmallVector<int>& dim) -> ttnn::Tensor { return self(input_tensor, shifts, dim); },
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim")},

        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int shifts, const int dim)
                -> ttnn::Tensor { return self(input_tensor, shifts, dim); },
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim")},

        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int shifts)
                -> ttnn::Tensor { return self(input_tensor, shifts); },
            nb::arg("input_tensor"),
            nb::arg("shifts")});
}

}  // namespace

void bind_roll(nb::module_& mod) {
    bind_roll_op(
        mod,
        ttnn::roll,
        R"doc(
        Performs circular shifting of elements along the specified dimension(s).

        Args:
            input_tensor: A tensor whose elements will be rolled.
            shifts: The number of places by which elements are shifted. Can be an integer or a list of integers (one per dimension).
            dim: The dimension(s) along which to roll. If shifts is a list, then dim must be a list of the same length as shifts.
        )doc");
}

}  // namespace ttnn::operations::data_movement
