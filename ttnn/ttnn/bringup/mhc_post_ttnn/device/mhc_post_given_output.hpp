// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

// output_tensor / stats_output / stats_scratch (deepseek_v41_flash_d_p P.16, default absent = the op allocates): the
// caller's buffer, which must have exactly the spec the op would allocate and must not be one of the inputs.
inline Tensor given_or_new(
    const std::optional<Tensor>& given,
    const tt::tt_metal::TensorSpec& spec,
    const Tensor& like,
    std::initializer_list<const Tensor*> inputs,
    const char* name) {
    if (!given.has_value()) {
        return create_device_tensor(spec, like.device());
    }
    TT_FATAL(
        given->storage_type() == ttnn::StorageType::DEVICE && given->is_allocated(),
        "mhc_post: {} must be an allocated device tensor",
        name);
    TT_FATAL(given->tensor_spec() == spec, "mhc_post: {} spec {} is not the op's {}", name, given->tensor_spec(), spec);
    for (const Tensor* x : inputs) {
        TT_FATAL(given->buffer()->address() != x->buffer()->address(), "mhc_post: {} must not be an input", name);
    }
    return *given;
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
