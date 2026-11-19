// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

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
    using spec_return_value_t = tt::tt_metal::TensorSpec;
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
        // The scalar operand is the mathematical left-hand side, so the compute kernel
        // evaluates op(scalar, tensor). Only ever set on the scalar path, where the tensor
        // occupies slot a and the scalar slot b regardless of operand order.
        bool scalar_is_lhs = false;
        bool is_sfpu = false;
        bool is_quant_op = false;
        bool is_where_op = false;
        float rtol = 0.0f;
        float atol = 0.0f;
        bool equal_nan = false;
        Layout input_layout_a = Layout::TILE;
        Layout input_layout_b = Layout::TILE;
        Layout output_layout = Layout::TILE;
        std::optional<std::uint32_t> a_shard_volume;
        std::optional<std::uint32_t> b_shard_volume;
        std::optional<std::uint32_t> c_shard_volume;
        // Sharded output's shape in pages on the accessor path. The inputs' equivalent rides in
        // tensor_args_t::to_hash(); the output has no Tensor at hash time, so it is carried here.
        std::optional<tt::tt_metal::Shape> c_tensor_shape_in_pages;
        // Parameters of the op being run, for ops that take any; empty for ops that do not.
        std::optional<binary::BinaryOpParams> op_params;

        DataType get_dtype() const;

        // Program-cache attributes. Runtime scalar/tolerance values are excluded; they are patched as runtime args.
        // Not keyed, on purpose: scalar, rtol, atol (runtime args), input_dtype (a's dtype is keyed through
        // OperandKey), sub_device_id (worker_grid carries its effect). post_activations is keyed except for
        // where/quant ops, which read their params (zero point, where scalar) as runtime args; equal_nan only
        // matters to ISCLOSE and op_params only to BIAS_GELU. A field added to this struct must be listed here
        // or in that note -- see the static_assert in binary_ng_device_operation.cpp.
        static constexpr auto attribute_names = std::make_tuple(
            "binary_op_type",
            "lhs_activations",
            "rhs_activations",
            "post_activations",
            "memory_config",
            "dtype",
            "compute_kernel_config",
            "sub_core_grids",
            // core_ranges of every CB and kernel. Depends on the device's sub-device layout, which
            // nothing else here carries and which does not clear the cache when swapped. Same set on a
            // single sub-device, so no extra entries there.
            "worker_grid",
            "subtile_broadcast_type",
            "scalar_is_lhs",
            "is_sfpu",
            "is_quant_op",
            "is_where_op",
            "input_layout_a",
            "input_layout_b",
            "output_layout",
            "equal_nan",
            "a_shard_volume",
            "b_shard_volume",
            "c_shard_volume",
            "c_tensor_shape_in_pages",
            "op_params");

        auto attribute_values() const {
            return std::make_tuple(
                binary_op_type,
                lhs_activations,
                rhs_activations,
                (is_where_op || is_quant_op) ? ttsl::SmallVector<unary::EltwiseUnaryWithParam>{} : post_activations,
                memory_config,
                get_dtype(),
                compute_kernel_config,
                sub_core_grids,
                worker_grid,
                subtile_broadcast_type,
                scalar_is_lhs,
                is_sfpu,
                is_quant_op,
                is_where_op,
                input_layout_a,
                input_layout_b,
                output_layout,
                binary_op_type == BinaryOpType::ISCLOSE ? equal_nan : false,
                a_shard_volume,
                b_shard_volume,
                c_shard_volume,
                c_tensor_shape_in_pages,
                binary_op_type == BinaryOpType::BIAS_GELU ? op_params : std::optional<binary::BinaryOpParams>{});
        }
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };

    // Shape-blind program key of one tensor operand: omits the logical shape by design, so differently-shaped
    // interleaved calls share one cache entry (per-core args are re-derived on a hit).
    struct OperandKey {
        DataType dtype;
        tt::tt_metal::MemoryConfig memory_config;
        tt::tt_metal::Alignment alignment;
        tt::tt_metal::Tile tile;
        // Sharded only: the accessor bakes the shape in pages, which is not rewritten on a hit.
        std::optional<tt::tt_metal::Shape> shard_shape_in_pages;
    };

    // Everything the cached program depends on. The framework derives both the program hash and the exact
    // canonical cache key from this tuple (the tensor part used to be a lossy 64-bit to_hash leaf in the key).
    using program_hash_inputs_t = std::tuple<
        const operation_attributes_t&,  // keyed through attribute_names / attribute_values
        OperandKey,                     // a
        std::optional<OperandKey>,      // b; nullopt on the scalar path
        DataType>;                      // output dtype (fp32 DEST); a's dtype when there is no output tensor
    static program_hash_inputs_t compute_program_hash_inputs(const operation_attributes_t&, const tensor_args_t&);

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c);

        // Cache-hit re-apply of all per-dispatch state (per-core args + tensor-backed CB/buffer
        // addresses), since compute_program_hash excludes the tensor volume. Correct by construction —
        // re-derives from the same shared builder create_descriptor() uses, no address inference. See the .cpp.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    bool scalar_is_lhs = false);

}  // namespace ttnn::prim
