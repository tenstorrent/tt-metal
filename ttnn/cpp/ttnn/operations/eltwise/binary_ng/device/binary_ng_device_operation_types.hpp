#pragma once

#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

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

struct BinaryNgParams {
    binary::BinaryOpType binary_op_type;
    ttnn::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations;
    ttnn::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations;
    ttnn::SmallVector<unary::EltwiseUnaryWithParam> post_activations;
    std::optional<unary::ScalarVariant> scalar;
    tt::tt_metal::MemoryConfig memory_config;
    DataType input_dtype;
    std::optional<DataType> dtype;
    const CoreRangeSet worker_grid;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    std::optional<CoreRangeSet> sub_core_grids;
    SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;
    bool is_sfpu = false;
    bool is_quant_op = false;
    bool is_where_op = false;

    ttsl::hash::hash_t to_hash() const;
    DataType get_dtype() const;
};

struct BinaryNgInputs {
    const Tensor& input_tensor_a;
    std::optional<Tensor> input_tensor_b;
    std::optional<Tensor> output_tensor;
};

}  // namespace ttnn::operations::binary_ng
