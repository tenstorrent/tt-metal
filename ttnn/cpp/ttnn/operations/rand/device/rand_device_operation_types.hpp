// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt_stl/small_vector.hpp>

namespace ttnn::operations::rand {

struct RandOperationAttributes {
    const ttnn::Shape shape;
    DataType dtype;
    Layout layout;
    const MemoryConfig memory_config;
    MeshDevice* device;
    const float from;
    const float to;
    uint32_t seed;
    ttsl::SmallVector<bool> mesh_dim_is_sharded;

    // Program identity (the program-cache key). seed/from/to are DELIBERATELY OMITTED here so that
    // calls differing only in those values cache-HIT instead of recompiling; they are re-applied on
    // every dispatch as Metal 2.0 per-enqueue run args (see RandProgramFactory::create_program_run_args).
    // `device` must be FIRST: rand has no input tensor, so the framework discovers the mesh device via
    // get_first_object_of_type over attribute_values(), and its tuple path only inspects element 0.
    static constexpr auto attribute_names =
        std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
    auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
};

struct RandTensorArgs {};

}  // namespace ttnn::operations::rand
