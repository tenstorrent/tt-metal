// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "matmul_decode_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

// Descriptor-facing (nanobind-friendly) mirror of MatmulDecodeDeviceOperation, matching the
// pattern ttnn::prim::MatmulDeviceOperation / MatmulParams / MatmulInputs use for the plain
// matmul descriptor (see ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation_types.hpp
// and matmul_device_operation.hpp). The real MatmulDecodeDeviceOperation::tensor_args_t holds
// `const Tensor&` members, which nanobind cannot own across a Python call; this by-value
// adapter is what models/experimental/ops/descriptors/matmul_decode.py binds against.
namespace ttnn::prim {

// Same fields as MatmulDecodeDeviceOperation::operation_attributes_t, by value, so nanobind can
// default-construct and field-assign it from Python the way MatmulParams is used today.
struct MatmulDecodeParams {
    int M = 0;
    int N = 0;
    int K = 0;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config = std::nullopt;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    bool partial_width_sharded = false;
    int batch = 1;
    int b_blocks = 1;
    int n_blocks = 1;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    uint32_t global_cb_k_blocks = 1;
    // Fused-weight path: where this op's weight lives inside a larger height-sharded weight
    // tensor (see packed_weight_spec.hpp). Mutually exclusive with global_cb.
    std::optional<ttnn::operations::experimental::matmul_decode::PackedWeightSpec> packed_weight = std::nullopt;
    bool all_gather = false;
    uint32_t ring_size = 1;
};

struct MatmulDecodeInputs {
    Tensor input_tensor_a;
    Tensor input_tensor_b;
};

// Descriptor-only facade: adapts MatmulDecodeParams/MatmulDecodeInputs to the real
// ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation's
// operation_attributes_t / tensor_args_t (which carry references, not values) so
// select_program_factory / compute_output_specs / compute_program_hash and each program
// factory's create_descriptor can be reached from the by-value Python-facing types.
struct MatmulDecodeDeviceOperation {
    using operation_attributes_t = MatmulDecodeParams;
    using tensor_args_t = MatmulDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // Named (not "compute_program_hash") for the same reason
    // ttnn::prim::MatmulDeviceOperation::compute_descriptor_program_hash is: this facade must
    // not be mistaken by the device-operation framework for a custom-program-cache hash. Reached
    // from Python under the pybind name "compute_program_hash", matching the plain matmul
    // descriptor's convention.
    static ttsl::hash::hash_t compute_descriptor_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

// Python-facing program-factory selector: mirrors matmul_select_program_factory. Returns which
// of FullWidthSharded / PartialWidthSharded / BatchedWidthSharded the real device operation would
// pick for this (attributes, tensor_args) pair, so the Python descriptor can call the matching
// factory's create_descriptor directly.
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::program_factory_t
matmul_decode_select_program_factory(
    const MatmulDecodeParams& operation_attributes, const MatmulDecodeInputs& tensor_args);

// Per-factory create_descriptor adapters (by-value params/inputs -> the real device
// operation's create_descriptor, which wants the internal reference-holding types).
tt::tt_metal::ProgramDescriptor matmul_decode_full_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);

tt::tt_metal::ProgramDescriptor matmul_decode_partial_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes, const MatmulDecodeInputs& tensor_args, Tensor& tensor_return_value);

tt::tt_metal::ProgramDescriptor matmul_decode_batched_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes, const MatmulDecodeInputs& tensor_args, Tensor& tensor_return_value);

}  // namespace ttnn::prim
