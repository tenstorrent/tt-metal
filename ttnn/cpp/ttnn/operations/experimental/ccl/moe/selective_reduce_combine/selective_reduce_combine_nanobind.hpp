// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

<<<<<<<< HEAD:ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata_nanobind.hpp
namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_all_to_all_dispatch_metadata(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
========
namespace ttnn::operations::experimental::ccl::moe {
namespace nb = nanobind;
void bind_selective_reduce_combine(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl::moe
>>>>>>>> bf37eacc71 (docs, move to experimental namespace, test fixes, clean up):ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/selective_reduce_combine_nanobind.hpp
