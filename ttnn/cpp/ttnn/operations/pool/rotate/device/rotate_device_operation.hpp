// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::rotate {

struct RotateDeviceOperation {
    struct operation_attributes_t {
        const float angle;
        const std::optional<std::tuple<float, float>> center;
        const float fill;
        const bool expand;
        const std::string interpolation_mode;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct NearestProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    struct BilinearProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<NearestProgramFactory, BilinearProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // The reader kernel (kernel index 0 in both program factories) bakes the angle-DERIVED values
    // (cos_angle_q16, sin_angle_q16, center_x_q16, center_y_q16, fill) into reader runtime args [3..7].
    // compute_program_hash deliberately EXCLUDES angle/center/fill (only memory_config, interpolation_mode,
    // input shape and dtype participate), so a program-cache HIT with a different angle/center/fill would
    // otherwise re-use the previously baked values and produce a WRONG result. The tensor base addresses
    // (reader arg [0], writer arg [0]) ride on patchable Buffer* rt-arg bindings handled by the framework,
    // but these scalar args cannot be expressed that way -- so we re-derive them EXACTLY as the factory
    // does and re-apply them on every cache hit here. Declaring this also opts the op into the descriptor
    // fast-path (no create_descriptor() rebuild on a hit). Mirrors the move / pool pattern.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::rotate

namespace ttnn::prim {
ttnn::Tensor rotate(
    const Tensor& input,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::string& interpolation_mode,
    const std::optional<MemoryConfig>& memory_config);
}  // namespace ttnn::prim
