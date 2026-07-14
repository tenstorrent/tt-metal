// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"
namespace ttnn::operations::binary_ng {

enum class SubtileBroadcastType {
    NONE,         // both tensors have equal tile dimensions (H & W)
    SCALAR_A,     // a is a scalar (H = 1, W = 1)
    SCALAR_B,     // b is a scalar (H = 1, W = 1)
    ROW_A_COL_B,  // a has a single tile row, b has a single tile column
    ROW_B_COL_A,  // b has a single tile row, a has a single tile column
    ROW_A,        // a has a single tile row, b is full
    ROW_B,        // b has a single tile row, a is full
    COL_A,        // a has a single tile column, b is full
    COL_B,        // b has a single tile column, a is full
};

SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w);

struct BinaryNgDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> post_activations;
        std::optional<unary::ScalarVariant> scalar;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        std::optional<DataType> dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        std::optional<CoreRangeSet> sub_core_grids;
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
        SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;
        bool is_sfpu = false;
        bool is_quant_op = false;
        bool is_where_op = false;
        float rtol = 0.0f;
        float atol = 0.0f;
        bool equal_nan = false;
        Layout input_layout_a = Layout::TILE;
        Layout input_layout_b = Layout::TILE;
        Layout output_layout = Layout::TILE;

        ttsl::hash::hash_t to_hash() const;
        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c);
    };

    using program_factory_t = std::variant<ProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // compute_program_hash EXCLUDES the tensor volume, so one cached program is reused across
    // differently-shaped calls.  All shape-/work-split-dependent per-core runtime args are therefore
    // re-applied on every cache hit here.  Mirrors create_descriptor()'s shared builder.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& c,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);

    // Opt in to the in-place output_tensor program-cache fast path (#48928: in-place residual add):
    // an output_tensor carried in tensor_args that aliases input_a is treated as a safe in-place
    // alias instead of bailing to slow-path rebuild (drives resolve_bindings'
    // allow_inplace_output_tensor_alias via the adapter). Safe here because get_dynamic_runtime_args()
    // above re-derives EVERY per-core arg for the current tensors on each cache hit, so the shared
    // cached program stays correct for a differently-shaped/-allocated in-place call. Ops without a
    // complete get_dynamic (unary/ternary/moreh_*) must NOT set this — they keep bailing (see #49573).
    //
    // POLICY: this flag IS the correctness argument and nothing verifies it — the UNSAFE_ prefix is a
    // deliberate hazard marker at the declaration site, not a normal tuning knob. Set it true ONLY with
    // an accompanying in-place cache-hit regression test that varies shape/allocation across the hit and
    // asserts a single cache entry + PCC (test_binary_ng_program_cache.py: the interleaved cross-shape
    // and sharded-readdress cases). No test → do not opt in. (Removed once get_dynamic is made complete
    // for all in-place ops, or a debug parity check lands, or descriptors move to Metal 2.0.)
    static constexpr bool UNSAFE_optin_inplace_program_cache_alias = true;
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::prim {

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    float rtol = 0.0f,
    float atol = 0.0f,
    bool equal_nan = false);

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    ttnn::operations::unary::ScalarVariant scalar,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

}  // namespace ttnn::prim
