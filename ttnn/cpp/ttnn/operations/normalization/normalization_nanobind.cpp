// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "normalization_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "softmax/softmax_nanobind.hpp"
#include "layernorm/layernorm_nanobind.hpp"
#include "rmsnorm/rmsnorm_nanobind.hpp"
#include "groupnorm/groupnorm_nanobind.hpp"
#include "layernorm_distributed/layernorm_distributed_nanobind.hpp"
#include "rmsnorm_distributed/rmsnorm_distributed_nanobind.hpp"
#include "batch_norm/batch_norm_nanobind.hpp"

namespace ttnn::operations::normalization {

void py_module(nb::module_& mod) {
    detail::bind_normalization_softmax(mod);
    detail::bind_normalization_layernorm(mod);
    detail::bind_normalization_rms_norm(mod);
    detail::bind_normalization_group_norm(mod);
    detail::bind_normalization_layernorm_distributed(mod);
    detail::bind_normalization_rms_norm_distributed(mod);
    detail::bind_batch_norm_operation(mod);
}

}  // namespace ttnn::operations::normalization
