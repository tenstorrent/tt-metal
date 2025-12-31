// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "fill_rm.hpp"
#include "ttnn-nanobind/decorators.hpp"


namespace ttnn::operations::data_movement {
namespace {

void bind_fill_rm_op(nb::module_& mod) {
    const char* doc = R"doc(
            Generates an NCHW row-major tensor and fill it with high values up to
            hOnes, wOnes in each HW tile with the rest padded with high values. So
            for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:

            .. code-block::

                +------------> W
                | hi hi lo
                | lo lo lo
                |
                v H

            H, W are expected to be multiples of 32.

            The 'any' Tensor arg is only used to pass the device and resulting
            tensor dtype.

            val_hi/lo are expected to be floats.

            Args:
                N (number): Batch count of output tensor.
                C (number): Channel count of output tensor.
                H (number): Height count of output tensor.
                W (number): Width count of output tensor.
                hOnes (number): Height of high values region.
                wOnes (number): Width of high values region.
                any (ttnn.tensor): Any input tensor with desired device and data types for output tensor.
                val_hi (number): High value to use.
                val_lo (number): Low value to use.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc";

    mod.def(
        "fill_rm",
        &ttnn::fill_rm,
        doc,
        nb::arg("N"),
        nb::arg("C"),
        nb::arg("H"),
        nb::arg("W"),
        nb::arg("hOnes"),
        nb::arg("wOnes"),
        nb::arg("any"),
        nb::arg("val_hi"),
        nb::arg("val_lo"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

void bind_fill_ones_rm_op(nb::module_& mod) {
    const char* doc = R"doc(
            Same as ``fill_rm``, but ``val_hi`` is set to ``1`` and ``val_lo`` is
            ``0``.

            Args:
                N (number): Batch count of output tensor.
                C (number): Channel count of output tensor.
                H (number): Height count of output tensor.
                W (number): Width count of output tensor.
                hOnes (number): Height of high values region.
                wOnes (number): Width of high values region.
                any (ttnn.tensor): Any input tensor with desired device and data types for output tensor.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::fill_ones_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("N"),
            nb::arg("C"),
            nb::arg("H"),
            nb::arg("W"),
            nb::arg("hOnes"),
            nb::arg("wOnes"),
            nb::arg("any"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace

void bind_fill_rm(nb::module_& mod) {
    bind_fill_rm_op(mod);
    bind_fill_ones_rm_op(mod);
}

}  // namespace ttnn::operations::data_movement
