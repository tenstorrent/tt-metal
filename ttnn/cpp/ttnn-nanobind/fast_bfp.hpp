// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::fast_bfp {
namespace nb = nanobind;

// Python bindings for tt::tt_metal::bfp_fast_pack (ttnn._ttnn.fast_bfp).
void py_module(nb::module_& mod);

// `view` is a dense CPU tensor (16- or 32-bit elements) whose last two dims are swapped, i.e. the transpose of a
// row-major contiguous tensor. Returns a row-major contiguous copy (same shape and dtype) made with a blocked,
// multithreaded transpose into huge-page backed memory.
nb::ndarray<nb::array_api> transposed_copy_of_view(const nb::ndarray<nb::array_api>& view);

}  // namespace ttnn::fast_bfp
