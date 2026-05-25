// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <tt_stl/reflection.hpp>

#include "fusion_dispatch_op_types.hpp"

namespace ttnn::operations::experimental::fusion {

struct FusionDispatchOpDeviceOperation {
    using operation_attributes_t = fusion_dispatch_operation_attributes_t;
    using tensor_args_t = fusion_dispatch_tensor_args_t;
    using spec_return_value_t = fusion_dispatch_spec_return_value_t;
    using tensor_return_value_t = fusion_dispatch_tensor_return_value_t;

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // The operation_attributes_t holds a MeshPrograms vector of
    // (MeshCoordinateRange, ProgramDescriptor) pairs.  In practice Python emits a
    // single full-mesh entry; we return that single descriptor here.  Callers
    // (see fusion_dispatch_op_nanobind.cpp) pre-patch each descriptor's runtime
    // args and CB buffers in-place before invoking the primitive, so on cache
    // miss the freshly-built Program is correct, and on cache hit the framework's
    // slow path (apply_descriptor_runtime_args) bulk-copies the patched runtime
    // args + CB addresses into the cached Program.
    //
    // Note: not taking ``mesh_dispatch_coordinate`` makes the framework iterate
    // ``tensor_coords.ranges()`` rather than ``tensor_coords.coords()``, which
    // preserves the original "one Program per MeshCoordinateRange" behavior of
    // the legacy factory for the single-range case.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::fusion

namespace ttnn::prim {
ttnn::operations::experimental::fusion::fusion_dispatch_tensor_return_value_t fusion_dispatch_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::experimental::fusion::fusion_dispatch_operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
