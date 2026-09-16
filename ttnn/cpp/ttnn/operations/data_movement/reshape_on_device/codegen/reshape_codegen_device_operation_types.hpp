// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Split out from the device-operation / program-factory headers so they can include each other's
// types without a circular #include: the device-operation header needs the factory type for its
// `program_factory_t` variant, and the factory needs `operation_attributes_t`/`tensor_args_t`.
struct ReshapeCodegenParams {
    ttnn::Shape output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ReshapeCodegenInputs {
    Tensor input;
};

}  // namespace ttnn::prim
