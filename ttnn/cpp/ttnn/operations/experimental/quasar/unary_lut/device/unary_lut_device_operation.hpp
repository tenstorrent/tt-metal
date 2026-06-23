// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::quasar::unary_lut {

// Per-activation LUT configuration baked into the compute kernel at JIT time (as -D
// defines emitted by the program factory). When absent, the kernel falls back to its
// compile-time default (the deg-2 / 4-seg sigmoid). This is what generalizes the op
// into a GENERIC DFB eltwise flow over (activation, eval_method): the Python driver
// parses a fitter coefficient CSV into a LutConfig and the same op/kernel evaluates
// any activation.
//
// `data` layout (matches unary_lut_sfpu.h LUT_DATA):
//   POLY (eval_method=0):     [b0..bS (num_segments+1 boundaries),
//                              per segment (poly_degree+1) Horner coeffs c0..cN]
//   RATIONAL (eval_method=1): [b0..bS,
//                              per segment (num_degree+1) numerator n0..nN
//                                        + (den_degree+1) denominator d0..dM]
struct LutConfig {
    uint32_t eval_method = 0;  // 0 = POLY_CASCADE, 1 = RATIONAL
    uint32_t poly_degree = 2;
    uint32_t num_segments = 4;
    uint32_t num_degree = 0;  // RATIONAL numerator degree
    uint32_t den_degree = 0;  // RATIONAL denominator degree
    std::vector<float> data;  // boundaries + per-segment coefficients (see layout above)

    bool operator==(const LutConfig&) const = default;
};

// Minimal UNARY piecewise-LUT activation device op, Metal 2.0 / DataflowBuffer path
// only. The unary analog of binary_ng's DFB slice: a single input DFB (in0) + an
// output DFB (out), with an embedded piecewise-polynomial LUT SFPU evaluation as the
// compute (instead of ADD). Supports exactly the fully-sharded (height/block L1),
// no-broadcast, TILE 32x32, bf16 slice — the smallest config that proves the
// unary-LUT DFB path produces correct output on craq-sim Quasar.
struct UnaryLutDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        const CoreRangeSet worker_grid;
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
        std::optional<LutConfig> lut_config;  // per-activation LUT (nullopt => kernel default sigmoid)

        ttsl::hash::hash_t to_hash() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> output_tensor;
    };

    // Metal 2.0 / DataflowBuffer factory (the only factory: this op exists to prove
    // the unary-LUT DFB path). See unary_lut_metal_v2_factory.cpp.
    struct ProgramFactoryMetalV2 {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactoryMetalV2>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::experimental::quasar::unary_lut

namespace ttnn::prim::qsr {

ttnn::operations::experimental::quasar::unary_lut::UnaryLutDeviceOperation::tensor_return_value_t unary_lut(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    const std::optional<ttnn::operations::experimental::quasar::unary_lut::LutConfig>& lut_config = std::nullopt);

}  // namespace ttnn::prim::qsr
