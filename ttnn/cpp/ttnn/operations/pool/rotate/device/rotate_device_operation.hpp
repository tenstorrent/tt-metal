// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include <tt-metalium/core_coord.hpp>
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

// Per-core runtime layout for one rotate dispatch, derived purely from (operation_attributes, input,
// output). This is the SINGLE SOURCE OF TRUTH shared by both program factories' create_descriptor()
// (cache miss) and get_dynamic_runtime_args() (cache-hit re-apply): the factories build reader/writer
// runtime args from it, and get_dynamic_runtime_args() re-emits the angle-derived reader args from the
// same values. compute_program_hash() excludes angle/center/fill, so those scalar args must be
// re-applied on every cache hit (the tensor base addresses ride on patchable Buffer* bindings instead).
struct RotatePerCoreArgs {
    // Work cores in factory iteration order. Reader runtime args [1]=num_sticks, [2]=start_stick_id are
    // per-core; the angle-derived slots [3..7] are identical for every core and stored once below.
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> num_sticks;      // indexed by position in `cores`
    std::vector<uint32_t> start_stick_id;  // indexed by position in `cores`

    // Angle/center/fill-derived reader args (Q16.16 fixed point, fill as raw bits). Constant across cores.
    uint32_t cos_angle_q16 = 0;
    uint32_t sin_angle_q16 = 0;
    uint32_t center_x_q16 = 0;
    uint32_t center_y_q16 = 0;
    uint32_t fill_value_bits = 0;

    // Reader is kernel index 0 in both factories. These are the reader runtime-arg slots holding the
    // angle-derived values above; arg [0] (input buffer base address) rides on a patchable Buffer* and
    // is NOT listed here. Used by get_dynamic_runtime_args() to re-apply the hash-excluded scalars.
    static constexpr uint32_t kReaderKernelIdx = 0;
    static constexpr uint32_t kCosArgIdx = 3;
    static constexpr uint32_t kSinArgIdx = 4;
    static constexpr uint32_t kCenterXArgIdx = 5;
    static constexpr uint32_t kCenterYArgIdx = 6;
    static constexpr uint32_t kFillArgIdx = 7;
};

// Derives the shared per-core layout. `is_bilinear` selects the work-core derivation (and fill-bit
// encoding) that matches the corresponding program factory.
RotatePerCoreArgs compute_rotate_per_core_args(
    const RotateDeviceOperation::operation_attributes_t& operation_attributes,
    const Tensor& input,
    const Tensor& output,
    bool is_bilinear);

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
