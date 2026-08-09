// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/generated/schemes/registry.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct Lwt1DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    MemoryConfig output_memory_config;
};

struct Lwt1DInputs {
    const Tensor& input;
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_outputs;
};

struct Lwt1DDeviceOperation {
    using operation_attributes_t = Lwt1DParams;
    using tensor_args_t = Lwt1DInputs;
    using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    struct ProgramFactory {
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct Ilwt1DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    uint32_t original_length;
    MemoryConfig output_memory_config;
};

struct Ilwt1DInputs {
    const Tensor& approximation;
    const Tensor& detail;
    const std::optional<Tensor>& preallocated_output;
};

struct Ilwt1DDeviceOperation {
    using operation_attributes_t = Ilwt1DParams;
    using tensor_args_t = Ilwt1DInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::tuple<Tensor, Tensor> lwt(
    const Tensor& input,
    operations::wavelet::SchemeId scheme_id,
    operations::wavelet::BoundaryMode boundary_mode,
    const MemoryConfig& output_memory_config,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_outputs = std::nullopt);

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    operations::wavelet::SchemeId scheme_id,
    operations::wavelet::BoundaryMode boundary_mode,
    uint32_t original_length,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
