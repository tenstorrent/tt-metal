// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DropoutParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;

    // Specifies the seed for the dropout operation.
    // If `use_per_device_seed` is true, the seed is offset by device ID across devices in a mesh.
    uint32_t seed = 0;
    bool use_per_device_seed = false;

    const float prob = 0.0f;
    const float scale = 1.0f;

    // `seed` is re-applied via override_runtime_arguments, so it is excluded from the program hash
    // (calls differing only in seed cache-hit). `prob`/`scale` are baked as compile-time args and
    // `use_per_device_seed` selects the program factory, so all three are structural and kept.
    static constexpr auto attribute_names =
        std::forward_as_tuple("output_dtype", "output_memory_config", "use_per_device_seed", "prob", "scale");
    auto attribute_values() const {
        return std::forward_as_tuple(output_dtype, output_memory_config, use_per_device_seed, prob, scale);
    }
};

// tensor_args must stay a plain reflectable aggregate: the device-operation framework walks it
// structurally to discover the Tensor leaves (output-spec counting, buffer extraction). Compile-time
// attributes here would both be ambiguous against the Reflectable visitor and hide the Tensors, so
// the input is hashed in full via its TensorSpec (correct and stricter than the old volume-only
// hash; the seed is still excluded via DropoutParams + override_runtime_arguments).
struct DropoutInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
