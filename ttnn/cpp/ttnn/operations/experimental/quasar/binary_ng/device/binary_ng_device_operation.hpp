// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tuple>
#include <tt-metalium/program_descriptors.hpp>
namespace ttnn::operations::experimental::quasar::binary_ng {

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
        bool is_sfpu = false;
        bool is_quant_op = false;
        bool is_where_op = false;
        float rtol = 0.0f;
        float atol = 0.0f;
        bool equal_nan = false;
        Layout input_layout_a = Layout::TILE;
        Layout input_layout_b = Layout::TILE;
        Layout output_layout = Layout::TILE;

        // `worker_grid` is hashed because get_worker_grid resolves it from MUTABLE device state, so
        // unhashed a cache hit could reuse a program placed on a different core set. `sub_device_id`
        // stays out -- already covered via `sub_core_grids`.
        static constexpr auto attribute_names = std::forward_as_tuple(
            "binary_op_type",
            "lhs_activations",
            "rhs_activations",
            "post_activations",
            "memory_config",
            "dtype",
            "compute_kernel_config",
            "sub_core_grids",
            "worker_grid",
            "subtile_broadcast_type",
            "is_sfpu",
            "is_quant_op",
            "is_where_op",
            "input_layout_a",
            "input_layout_b",
            "output_layout",
            "equal_nan",
            "scalar",
            "rtol",
            "atol");
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
                is_sfpu,
                is_quant_op,
                is_where_op,
                input_layout_a,
                input_layout_b,
                output_layout,
                equal_nan,
                scalar,
                rtol,
                atol);
        }
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

    // Metal 2.0 / DataflowBuffer factory. Emits a ProgramSpec + ProgramRunArgs (instead of a
    // ProgramDescriptor) for the slice that matches_metal_v2_slice() admits: ADD, no broadcast,
    // TILE 32x32, fully height/block-sharded L1, bf8/bf16, optional fused RELU, in-place (the
    // ResNet50 residual config). The DFB path is arch-portable: CB-backed on Wormhole/Blackhole,
    // overlay-backed on Quasar. `select_program_factory` routes a matching op here; every other case
    // falls through to `ProgramFactory`. See binary_ng_metal_v2_factory.cpp.
    struct ProgramFactoryMetalV2 {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c);
    };

    // Quasar-native DFB factory. Starts as a mechanical copy of ProgramFactoryMetalV2 so later commits
    // read as a diff of what "native" means (>2 of 6 DM cores, >1 of 4 Tensix). Opt-in via
    // TTNN_QSR_NATIVE, so the metal_v2 path stays live as the A/B reference arm.
    // Must stay a stateless literal type -- the framework default-constructs it into static storage.
    struct ProgramFactoryQuasarNative {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c);
    };

    // APPEND-ONLY: the index is persisted per cache entry, so reordering mis-resolves existing entries.
    using program_factory_t = std::variant<ProgramFactory, ProgramFactoryMetalV2, ProgramFactoryQuasarNative>;

    // Returns ProgramFactoryQuasarNative{} when the op matches the (opt-in) Quasar-native slice,
    // ProgramFactoryMetalV2{} when it matches the Metal 2.0 slice, and ProgramFactory{} otherwise.
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // True iff (attributes, tensor_args) match the Metal 2.0 factory's supported slice. Shared by
    // select_program_factory.
    static bool matches_metal_v2_slice(const operation_attributes_t&, const tensor_args_t&);

    // Strict subset of matches_metal_v2_slice, plus TTNN_QSR_NATIVE. Checked first; its leading test is
    // the cached env bool, so the non-native hot-path cost is a load and a branch.
    static bool matches_quasar_native_slice(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::experimental::quasar::binary_ng

namespace ttnn::prim::qsr {

ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    ttnn::operations::experimental::quasar::binary_ng::BinaryOpType binary_op_type,
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

ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    ttnn::operations::unary::ScalarVariant scalar,
    ttnn::operations::experimental::quasar::binary_ng::BinaryOpType binary_op_type,
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

}  // namespace ttnn::prim::qsr
