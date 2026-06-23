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

    // ---- Range reduction (RR). Mirrors the tt-llk generic-LUT build header contract
    // (LUT_RR_* defines). The Python driver parses range_reduction_method (+ params)
    // from the fitter CSV METADATA into these fields; the factory bakes them into the
    // compute kernel as -D defines so reduce-then-poly-then-reconstruct runs uniformly,
    // with NO per-activation special-casing. rr_method == 0 (none) => byte-identical to
    // the no-RR path. Method codes (match the kernel's LUT_RR_METHOD contract):
    //   0 none, 1 log, 2 exp (Cody-Waite), 3 cbrt, 4 expalu_exp2, 5 expalu_log2,
    //   6 expalu_pow, 7 trig (sin/cos), 8 tan, 9 newton_root (standalone seed+Newton).
    uint32_t rr_method = 0;
    float rr_log_ln2 = 1.0f;                   // method 1
    float rr_exp_mult = 1.4426950408889634f;   // method 2
    float rr_exp_const = 0.6931471805599453f;  // method 2
    float rr_scale0 = 1.0f;                    // methods 3/6 scale table
    float rr_scale1 = 1.0f;
    float rr_scale2 = 1.0f;
    float rr_exp2_mult = 1.0f;           // method 4
    uint32_t rr_compose = 0;             // method 4: 0 none, 1 sigmoid, 2 minus_one
    float rr_log2_scale = 1.0f;          // method 5
    uint32_t rr_log2_basis_mminus1 = 0;  // method 5
    float rr_input_offset = 0.0f;        // method 5 (log1p)
    uint32_t rr_pow_n = 2;               // method 6
    uint32_t rr_pow_recip = 0;           // method 6 (rsqrt)

    // ---- Newton-root (rr_method == 9). STANDALONE magic-seed + Newton/Householder
    // evaluator (sqrt / rsqrt / cbrt); bypasses the segment cascade. Parsed by the
    // driver from the fitter CSV newton_root_* METADATA, baked into the kernel as the
    // LUT_NR_* defines. Defaults are the sqrt seed (harmless when rr_method != 9).
    uint32_t nr_magic = 0x5f1110a0u;  // sqrt magic seed; rsqrt uses 0x5f3759df
    float nr_c1 = 2.2825186f;
    float nr_c2 = 2.2533049f;
    uint32_t nr_iters = 2;
    uint32_t nr_n = 2;           // root order (2 = sqrt/rsqrt)
    uint32_t nr_reciprocal = 0;  // 0 = sqrt, 1 = rsqrt

    // ---- Asymptotic factoring. The tail segments of some deployed picks are fit as
    // f(x) = dominant(x) * correction(x) (is_asymptotic=True in the fitter CSV): the CSV
    // stores the CORRECTION polynomial as the segment's ordinary Horner coeffs, and the
    // TRUE value is dominant(x) times the Horner result. The driver parses the per-segment
    // is_asymptotic column into a bitmask (bit SEG => segment SEG is asymptotic) and the
    // shared dominant_factor class into dom_class; the factory bakes them as the
    // LUT_ASYM_MASK / LUT_DOMINANT_CLASS defines so the kernel multiplies by dominant(x)
    // for the flagged segments. dom_class == 0 (default) => no factoring; tail segments are
    // never dropped. Class codes mirror precision/eval.py DOMINANT_FACTORS (see kernel).
    uint32_t asym_mask = 0;  // bitmask over segments; bit SEG set => segment SEG is asymptotic
    uint32_t dom_class = 0;  // dominant-factor class code (0 = none)

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
