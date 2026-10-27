// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ternary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp"

using namespace tt::tt_metal;
namespace {
// Helper function to add broadcast defines for dataflow kernels
void add_broadcast_defines(
    std::map<std::string, std::string>& defines, bool pred_is_bcast, bool true_is_bcast, bool false_is_bcast) {
    defines["SRC_BCAST_A"] = pred_is_bcast ? "1" : "0";   // Predicate tensor (CB0)
    defines["SRC_BCAST_B"] = true_is_bcast ? "1" : "0";   // True tensor (CB1)
    defines["SRC_BCAST_C"] = false_is_bcast ? "1" : "0";  // False tensor (CB2)
}

// Helper function to get sharding status for tensors
std::tuple<bool, bool, bool> get_tensor_sharding_status(
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor) {
    bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
    bool value_true_sharded = value_true_tensor.has_value() && value_true_tensor->memory_config().is_sharded();
    bool value_false_sharded = value_false_tensor.has_value() && value_false_tensor->memory_config().is_sharded();
    return {predicate_sharded, value_true_sharded, value_false_sharded};
}

// Helper function to detect broadcast patterns for different variants and broadcast types
void detect_broadcasts(
    bool& pred_is_bcast,
    bool& true_is_bcast,
    bool& false_is_bcast,
    ttnn::operations::ternary::TernaryVariant variant,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor) {
    auto pred_shape = predicate_tensor.logical_shape();

    using TernaryBroadcastType = ttnn::operations::ternary::TernaryBroadcastType;
    using TernaryVariant = ttnn::operations::ternary::TernaryVariant;

    if (broadcast_type == TernaryBroadcastType::OUTER_BCAST || broadcast_type == TernaryBroadcastType::NONE) {
        pred_is_bcast = true_is_bcast = false_is_bcast = false;
    } else if (broadcast_type == TernaryBroadcastType::COL_BCAST) {
        // Column broadcast detection
        auto pred_w = pred_shape[pred_shape.rank() - 1];

        if (variant == TernaryVariant::TTS) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto true_w = true_shape[true_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && true_w > 1);
            true_is_bcast = (true_w == 1 && pred_w > 1);
            false_is_bcast = false;  // False is scalar for TTS
        } else if (variant == TernaryVariant::TST) {
            auto false_shape = value_false_tensor.value().logical_shape();
            auto false_w = false_shape[false_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && false_w > 1);
            false_is_bcast = (false_w == 1 && pred_w > 1);
            true_is_bcast = false;  // True is scalar for TST
        } else {                    // TTT
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto true_w = true_shape[true_shape.rank() - 1];
            auto false_w = false_shape[false_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && (true_w > 1 || false_w > 1));
            true_is_bcast = (true_w == 1 && (pred_w > 1 || false_w > 1));
            false_is_bcast = (false_w == 1 && (pred_w > 1 || true_w > 1));
        }
    } else if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        // Row broadcast detection
        auto pred_h = pred_shape[pred_shape.rank() - 2];

        if (variant == TernaryVariant::TTS) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto true_h = true_shape[true_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && true_h > 1);
            true_is_bcast = (true_h == 1 && pred_h > 1);
            false_is_bcast = false;  // False is scalar for TTS
        } else if (variant == TernaryVariant::TST) {
            auto false_shape = value_false_tensor.value().logical_shape();
            auto false_h = false_shape[false_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && false_h > 1);
            true_is_bcast = false;  // True is scalar for TST
            false_is_bcast = (false_h == 1 && pred_h > 1);
        } else {  // TTT
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto true_h = true_shape[true_shape.rank() - 2];
            auto false_h = false_shape[false_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && (true_h > 1 || false_h > 1));
            true_is_bcast = (true_h == 1 && (pred_h > 1 || false_h > 1));
            false_is_bcast = (false_h == 1 && (pred_h > 1 || true_h > 1));
        }
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_BCAST) {
        // Scalar broadcast detection (H and W dimensions must be broadcast for TTT)
        auto true_shape = value_true_tensor.value().logical_shape();
        auto false_shape = value_false_tensor.value().logical_shape();

        auto pred_w = pred_shape[-1], pred_h = pred_shape[-2];
        auto true_w = true_shape[-1], true_h = true_shape[-2];
        auto false_w = false_shape[-1], false_h = false_shape[-2];

        auto max_h = std::max({pred_h, true_h, false_h});
        auto max_w = std::max({pred_w, true_w, false_w});

        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool true_row_bcast = (true_h == 1 && max_h > 1);
        bool false_row_bcast = (false_h == 1 && max_h > 1);

        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool true_col_bcast = (true_w == 1 && max_w > 1);
        bool false_col_bcast = (false_w == 1 && max_w > 1);

        pred_is_bcast = (pred_row_bcast && pred_col_bcast);
        true_is_bcast = (true_row_bcast && true_col_bcast);
        false_is_bcast = (false_row_bcast && false_col_bcast);
    } else if (broadcast_type == TernaryBroadcastType::ROW_COL_BCAST) {
        // Mixed row+col broadcast: is_bcast = true for col-broadcast (W=1) tensors,
        // which are pushed before the tw loop and waited on outside the compute freq loop.
        auto pred_w = pred_shape[-1];

        if (variant == TernaryVariant::TTT) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto true_w = true_shape[-1];
            auto false_w = false_shape[-1];
            auto max_w = std::max({pred_w, true_w, false_w});
            pred_is_bcast = (pred_w == 1 && max_w > 1);
            true_is_bcast = (true_w == 1 && max_w > 1);
            false_is_bcast = (false_w == 1 && max_w > 1);
        } else if (variant == TernaryVariant::TTS) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto true_w = true_shape[-1];
            auto max_w = std::max(pred_w, true_w);
            pred_is_bcast = (pred_w == 1 && max_w > 1);
            true_is_bcast = (true_w == 1 && max_w > 1);
            false_is_bcast = false;
        } else if (variant == TernaryVariant::TST) {
            auto false_shape = value_false_tensor.value().logical_shape();
            auto false_w = false_shape[-1];
            auto max_w = std::max(pred_w, false_w);
            pred_is_bcast = (pred_w == 1 && max_w > 1);
            true_is_bcast = false;
            false_is_bcast = (false_w == 1 && max_w > 1);
        }
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST) {
        // Scalar broadcast detection (H and W dimensions of condition tensor must be broadcast for TTS/TST)
        pred_is_bcast = true;
        true_is_bcast = false;
        false_is_bcast = false;
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) {
        // Scalar broadcast detection (H and W dimensions of true/false tensor must be broadcast for TTS/TST)
        pred_is_bcast = false;
        true_is_bcast = true;
        false_is_bcast = true;
    }
}

// Helper function to set up reader defines based on variant and broadcast type
void setup_reader_defines(
    std::map<std::string, std::string>& reader_defines,
    ttnn::operations::ternary::TernaryVariant variant,
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor,
    bool pred_is_bcast,
    bool true_is_bcast,
    bool false_is_bcast,
    bool has_sharding) {
    using TernaryVariant = ttnn::operations::ternary::TernaryVariant;

    // Get sharding status for all tensors
    auto [predicate_sharded, value_true_sharded, value_false_sharded] =
        get_tensor_sharding_status(predicate_tensor, value_true_tensor, value_false_tensor);

    // Set up dataflow defines first
    if (variant == TernaryVariant::TTT) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype(), value_false_tensor.value().dtype());
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TTS) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype());
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TST) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_false_tensor.value().dtype());
        // For TST the two tensors are predicate and false_tensors and are SRC_A and SRC_B respectively
        add_broadcast_defines(reader_defines, pred_is_bcast, false_is_bcast, true_is_bcast);
    }

    // Set sharding defines after dataflow defines
    // SRC_SHARDED_A always maps to CB0 (predicate)
    // SRC_SHARDED_B maps to CB1: value_true for TTT/TTS, value_false for TST
    // SRC_SHARDED_C maps to CB2: value_false for TTT (unused for TTS/TST)
    // SRC_SHARDED_* is 0 for unequal shaped sharded inputs, since has_sharding and is_native_L1_sharding are both false
    reader_defines["SRC_SHARDED_A"] = (predicate_sharded && has_sharding) ? "1" : "0";
    if (variant == TernaryVariant::TST) {
        reader_defines["SRC_SHARDED_B"] = (value_false_sharded && has_sharding) ? "1" : "0";
    } else {
        reader_defines["SRC_SHARDED_B"] = (value_true_sharded && has_sharding) ? "1" : "0";
    }
    reader_defines["SRC_SHARDED_C"] = (value_false_sharded && has_sharding) ? "1" : "0";
}

namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;
using namespace tt::tt_metal::experimental;

// The ternary program's tensor parameters, DFBs and kernels. The kernels name each DFB and tensor binding after the
// operand slot it holds, the same slots the A / B / C letters of the SRC_SHARDED_*, SRC_BCAST_* and BCAST_* defines
// refer to:
//   in_a    - the first tensor (the predicate for where, input_a for the other ops)
//   in_b    - the second tensor operand: value_true for TTT and TTS, value_false for TST
//   in_c    - the third tensor operand: value_false (TTT only)
//   out     - the output
//   bcast_a / bcast_b / bcast_c - row-broadcast copies of in_a / in_b / in_c, made by the LLK row-broadcast compute
//             kernels
const TensorParamName PREDICATE_TENSOR{"predicate"};
const TensorParamName VALUE_TRUE_TENSOR{"value_true"};
const TensorParamName VALUE_FALSE_TENSOR{"value_false"};
const TensorParamName OUTPUT_TENSOR{"output"};

const DFBSpecName DFB_IN_A{"in_a"};
const DFBSpecName DFB_IN_B{"in_b"};
const DFBSpecName DFB_IN_C{"in_c"};
const DFBSpecName DFB_OUT{"out"};
const DFBSpecName DFB_BCAST_A{"bcast_a"};
const DFBSpecName DFB_BCAST_B{"bcast_b"};
const DFBSpecName DFB_BCAST_C{"bcast_c"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL{"compute"};

bool is_llk_bcast(
    const ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::DataType a_dtype,
    const ttnn::DataType b_dtype,
    const ttnn::DataType c_dtype,
    const ttnn::DataType output_dtype) {
    // The LLK row broadcast (MOVB2D) corrupts bf16 sources in an fp32 DEST, so a non-bf16
    // output (which enables fp32 DEST) uses the reader-side row broadcast instead.
    if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        if (a_dtype == ttnn::DataType::BFLOAT16 && b_dtype == ttnn::DataType::BFLOAT16 &&
            c_dtype == ttnn::DataType::BFLOAT16 && output_dtype == ttnn::DataType::BFLOAT16) {
            return true;
        }
    }
    return false;
}

void overwrite_compute_kernel_name_and_defines(
    ttnn::operations::ternary::KernelName& kernel_name,
    const ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::operations::ternary::TernaryOpType op_type) {
    if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        if (op_type == TernaryOpType::ADDCMUL || op_type == TernaryOpType::ADDCDIV) {
            kernel_name = KernelName::ComputeRowBcastAddcOp;
        } else {
            kernel_name = KernelName::ComputeRowBcastTTT;
        }
    }
}

// Get operation-specific compute kernel defines
std::map<std::string, std::string> get_ternary_compute_defines(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes, ttnn::DataType dtype) {
    return get_compute_defines(operation_attributes.ternary_op_type, dtype);
}

// Helper function to get shape dimensions
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {
        shape.rank() >= 5 ? shape[-5] : 1,
        shape[-4],
        shape[-3],
        shape[-2] / tile.get_height(),
        shape[-1] / tile.get_width()};
}

// Helper function to get memory layout
TensorMemoryLayout get_memory_layout(
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor,
    const ttnn::Tensor& output) {
    if (output.memory_config().is_sharded()) {
        return output.memory_config().memory_layout();
    }

    if (predicate_tensor.memory_config().is_sharded()) {
        return predicate_tensor.memory_config().memory_layout();
    }

    if (value_true_tensor.has_value() && value_true_tensor->memory_config().is_sharded()) {
        return value_true_tensor->memory_config().memory_layout();
    }

    if (value_false_tensor.has_value() && value_false_tensor->memory_config().is_sharded()) {
        return value_false_tensor->memory_config().memory_layout();
    }

    return TensorMemoryLayout::INTERLEAVED;
}

// Get number of shards per width dimension
uint32_t get_shards_per_width(const tt::tt_metal::ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    auto num_cores = shard_spec.grid.num_cores();
    if (memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }

    if (memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }

    const auto& bbox = shard_spec.grid.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (shard_spec.orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

std::optional<AllShardSpecs> get_shard_specs(
    const tt::tt_metal::TensorSpec& predicate_spec,
    const std::optional<tt::tt_metal::TensorSpec>& true_spec,
    const std::optional<tt::tt_metal::TensorSpec>& false_spec,
    const tt::tt_metal::TensorSpec& output_spec) {
    bool predicate_sharded = predicate_spec.memory_config().is_sharded();
    bool true_sharded = true_spec.has_value() && true_spec->memory_config().is_sharded();
    bool false_sharded = false_spec.has_value() && false_spec->memory_config().is_sharded();
    bool output_sharded = output_spec.memory_config().is_sharded();

    if (!predicate_sharded && !true_sharded && !false_sharded && !output_sharded) {
        return std::nullopt;
    }

    if (!is_native_L1_sharding(predicate_spec, true_spec, false_spec, output_spec.memory_config()) ||
        is_uneven(output_spec)) {
        return std::nullopt;
    }

    const auto& predicate_shape = predicate_spec.padded_shape();
    const auto& output_shape = output_spec.padded_shape();

    TT_FATAL(get_shard_spec(output_spec).has_value(), "Output must have a shard spec");
    const auto& output_shard = *get_shard_spec(output_spec);

    auto derive_shard_spec = [&](const std::optional<tt::tt_metal::TensorSpec>& spec,
                                 bool sharded) -> tt::tt_metal::ShardSpec {
        if (sharded) {
            return *get_shard_spec(*spec);
        }
        if (spec.has_value()) {
            return adjust_to_shape(output_shard, output_shape, spec->padded_shape());
        }
        return output_shard;
    };

    return AllShardSpecs{
        predicate_sharded ? *get_shard_spec(predicate_spec)
                          : adjust_to_shape(output_shard, output_shape, predicate_shape),
        derive_shard_spec(true_spec, true_sharded),
        derive_shard_spec(false_spec, false_sharded),
        output_shard};
}

// ShardShapeGenerator class
class ShardShapeGenerator {
    CoreCoord end_core;
    bool row_major{};
    TensorMemoryLayout memory_layout{TensorMemoryLayout::INTERLEAVED};
    std::array<uint32_t, 2> shard_shape{};
    std::array<uint32_t, 2> last_shard_shape{};

public:
    ShardShapeGenerator() = default;

    ShardShapeGenerator(const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Tensor& tensor) :
        // core ranges are sorted, so the last one is indeed the last core
        end_core(shard_spec.grid.ranges().rbegin()->end_coord),
        row_major(shard_spec.orientation == ShardOrientation::ROW_MAJOR),
        memory_layout(tensor.memory_config().memory_layout()) {
        auto tile_height = tensor.tensor_spec().tile().get_height();
        auto tile_width = tensor.tensor_spec().tile().get_width();

        shard_shape = {
            tt::round_up(shard_spec.shape[0], tile_height) / tile_height,
            tt::round_up(shard_spec.shape[1], tile_width) / tile_width};

        TT_FATAL(
            shard_shape[0] != 0 and shard_shape[1] != 0,
            "Shard shape must not contain zero dimensions but got {{{}, {}}}",
            shard_shape[0],
            shard_shape[1]);

        const auto [D, N, C, Ht, Wt] = get_shape_dims(tensor);
        const auto unrolled_Ht = D * N * C * Ht;
        last_shard_shape = {
            shard_shape[0] - (tt::round_up(unrolled_Ht, shard_shape[0]) - unrolled_Ht),
            shard_shape[1] - (tt::round_up(Wt, shard_shape[1]) - Wt),
        };
    }
    std::array<uint32_t, 2> operator()(CoreCoord core) const {
        const unsigned majorDim = row_major ? 1 : 0;
        const unsigned minorDim = row_major ? 0 : 1;
        auto current_shape = shard_shape;
        if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            if (core == end_core) {
                current_shape[majorDim] = last_shard_shape[majorDim];
                current_shape[minorDim] = last_shard_shape[minorDim];
            }
        } else if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            // For BLOCK_SHARDED, edges can have uneven shards
            if (row_major) {
                if (core.x == end_core.x) {
                    current_shape[1] = last_shard_shape[1];  // width
                }
                if (core.y == end_core.y) {
                    current_shape[0] = last_shard_shape[0];  // height
                }
            } else {  // col_major
                if (core.y == end_core.y) {
                    current_shape[1] = last_shard_shape[1];  // width
                }
                if (core.x == end_core.x) {
                    current_shape[0] = last_shard_shape[0];  // height
                }
            }
        }
        return current_shape;
    }
};

// For rank > 5 dims will be collapsed into a single dim
uint32_t extract_nD_dims(const ttnn::Tensor& x, const int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        // A lower-rank operand has no dims beyond its own rank; they broadcast as 1.
        const int rank = std::min<int>(out_rank, shape.rank());
        for (int i = -6; i >= -rank; --i) {
            auto dim = shape[i];
            nD_dim *= dim;
        }
    }
    return nD_dim;
}

// Helper struct to hold tensor dimensions
struct TensorDimensions {
    uint32_t D = 1, N = 1, C = 1, Ht = 1, Wt = 1, ND = 1;
    uint32_t num_tiles = 0;
};

// Helper function to extract tensor dimensions
TensorDimensions extract_tensor_dimensions(const ttnn::Tensor& tensor, int out_rank, uint32_t tile_h, uint32_t tile_w) {
    TensorDimensions dims;
    const auto& shape = tensor.padded_shape();
    dims.ND = extract_nD_dims(tensor, out_rank);
    dims.D = shape.rank() >= 5 ? shape[-5] : 1;
    dims.N = shape[-4];
    dims.C = shape[-3];
    dims.Ht = shape[-2] / tile_h;
    dims.Wt = shape[-1] / tile_w;

    return dims;
}

// Helper function to calculate strides for broadcast operations
struct Strides {
    uint32_t nD_stride = 0, d_stride = 0, n_stride = 0, c_stride = 0;
};

Strides calculate_strides(const TensorDimensions& dims) {
    Strides strides;
    strides.nD_stride = dims.Ht * dims.Wt * dims.C * dims.N * dims.D * (dims.ND > 1);
    strides.d_stride = dims.Ht * dims.Wt * dims.C * dims.N * (dims.D > 1);
    strides.n_stride = dims.Ht * dims.Wt * dims.C * (dims.N > 1);
    strides.c_stride = dims.Ht * dims.Wt * (dims.C > 1);
    return strides;
}

// Helper function to set up reader runtime arguments and dimensions for TTS/TST variants
template <size_t num_reader_args>
void setup_ts_reader_args_and_dims(
    std::array<uint32_t, num_reader_args>& reader_runtime_args,
    const ttnn::Tensor& predicate_tensor,
    const ttnn::Tensor& tensor_operand,  // true tensor for TTS, false tensor for TST
    const ttnn::Tensor& output,
    uint32_t num_tiles_per_core,
    uint32_t start_tile_id,
    uint32_t& a_num_tiles,
    uint32_t& tensor_num_tiles,  // number of tiles for the tensor operand
    uint32_t& c_current_shard_width,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    bool has_sharding,
    int out_rank,
    uint32_t tile_h,
    uint32_t tile_w) {
    auto output_dims = extract_tensor_dimensions(output, out_rank, tile_h, tile_w);
    auto pred_dims = extract_tensor_dimensions(predicate_tensor, out_rank, tile_h, tile_w);
    auto tensor_dims = extract_tensor_dimensions(tensor_operand, out_rank, tile_h, tile_w);

    auto pred_strides = calculate_strides(pred_dims);
    auto tensor_strides = calculate_strides(tensor_dims);

    // Only set tile counts if sharding is enabled
    if (has_sharding) {
        a_num_tiles = pred_dims.Ht * pred_dims.Wt;           // predicate tiles per core
        tensor_num_tiles = tensor_dims.Ht * tensor_dims.Wt;  // tensor operand tiles per core
        c_current_shard_width = output_dims.Wt;
    }
    // Standard first 2 arguments. The tensor base addresses are not runtime args: the reader binds the tensors.
    reader_runtime_args[0] = num_tiles_per_core;  // 0: num_tiles (per core)
    reader_runtime_args[1] = start_tile_id;       // 1: start_id
    // Extended broadcast arguments
    if (broadcast_type != ttnn::operations::ternary::TernaryBroadcastType::NONE) {
        reader_runtime_args[2] = pred_strides.nD_stride;  // 2: nD_stride
        reader_runtime_args[3] = pred_strides.d_stride;   // 3: d_stride
        reader_runtime_args[4] = pred_strides.n_stride;   // 4: n_stride
        reader_runtime_args[5] = pred_strides.c_stride;   // 5: c_stride
        reader_runtime_args[6] = output_dims.D;           // 6: D
        reader_runtime_args[7] = output_dims.N;           // 7: N
        reader_runtime_args[8] = output_dims.C;           // 8: C
        reader_runtime_args[9] = output_dims.Ht;          // 9: Ht
        reader_runtime_args[10] = output_dims.Wt;         // 10: Wt
        reader_runtime_args[11] = output_dims.ND;         // 11: cND
        // Both TTS and TST: tensor operand strides in 12-16, scalar operand zeros in 17-21
        reader_runtime_args[12] = tensor_strides.nD_stride;  // 12: tensor_operand_nD_stride
        reader_runtime_args[13] = tensor_strides.d_stride;   // 13: tensor_operand_d_stride
        reader_runtime_args[14] = tensor_strides.n_stride;   // 14: tensor_operand_n_stride
        reader_runtime_args[15] = tensor_strides.c_stride;   // 15: tensor_operand_c_stride
        reader_runtime_args[16] = tensor_num_tiles;          // 16: tensor_operand_num_tiles
        // Scalar operand has no strides
        reader_runtime_args[17] = 0u;  // 17: scalar_operand_nD_stride
        reader_runtime_args[18] = 0u;  // 18: scalar_operand_d_stride
        reader_runtime_args[19] = 0u;  // 19: scalar_operand_n_stride
        reader_runtime_args[20] = 0u;  // 20: scalar_operand_c_stride
        reader_runtime_args[21] = 0u;  // 21: scalar_operand_num_tiles

        reader_runtime_args[22] = c_current_shard_width;  // 22: dst_shard_width
        reader_runtime_args[23] = a_num_tiles;            // 23: src_num_tiles (predicate)
    }
}

// Shared work-split so create_program_artifacts() (cache miss + cache-hit re-apply) partitions the same cores.
struct TernaryCorePartition {
    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores_total = 0;
    bool has_sharding = false;
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    uint32_t num_tiles_per_core_group_1 = 0;
    uint32_t num_tiles_per_core_group_2 = 0;
    bool row_major = true;
};

TernaryCorePartition compute_core_partition(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    TernaryDeviceOperation::tensor_return_value_t& output) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    TernaryCorePartition p;

    // Get shard specs early
    const auto shard_specs = get_shard_specs(
        predicate_tensor.tensor_spec(),
        value_true_tensor.has_value() ? value_true_tensor->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{},
        value_false_tensor.has_value() ? value_false_tensor->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{},
        output.tensor_spec());
    p.has_sharding = shard_specs.has_value();
    auto grid = p.has_sharding ? shard_specs->predicate_shard_spec.grid : CoreRangeSet{};

    p.row_major = p.has_sharding ? shard_specs->predicate_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    const auto& all_device_cores = operation_attributes.worker_grid;
    // A multi-range worker grid would leave cores past its first range without args.
    if (grid.size() == 1 && all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (p.has_sharding) {
                const auto& shard_start_coord = grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    p.zero_start_grid = true;
                    p.compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                p.zero_start_grid = true;
                p.compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }
    p.num_cores_total = p.zero_start_grid ? p.compute_with_storage_grid.x * p.compute_with_storage_grid.y
                                          : all_device_cores.num_cores();

    uint32_t num_cores;
    CoreRangeSet all_cores;
    if (p.has_sharding) {
        p.core_group_1 = grid;
        if (p.zero_start_grid) {
            auto bbox = p.core_group_1.bounding_box();
            p.cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                p.compute_with_storage_grid.x,
                p.compute_with_storage_grid.y,
                p.row_major);
        } else {
            p.cores = grid_to_cores_with_noop(p.core_group_1, all_device_cores, p.row_major);
        }
    } else if (p.zero_start_grid) {
        std::tie(
            num_cores,
            all_cores,
            p.core_group_1,
            p.core_group_2,
            p.num_tiles_per_core_group_1,
            p.num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(p.compute_with_storage_grid, num_output_tiles, p.row_major);
        p.cores =
            grid_to_cores(p.num_cores_total, p.compute_with_storage_grid.x, p.compute_with_storage_grid.y, p.row_major);
    } else {
        std::tie(
            num_cores,
            all_cores,
            p.core_group_1,
            p.core_group_2,
            p.num_tiles_per_core_group_1,
            p.num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, num_output_tiles, p.row_major);
        p.cores = corerange_to_cores(all_device_cores, {}, p.row_major);
    }
    return p;
}

// Single source of truth for packing the (hash-excluded, dynamic) compute-kernel scalar arg.
uint32_t pack_compute_scalar_arg(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes, DataType output_dtype) {
    TernaryVariant variant = operation_attributes.ternary_variant;
    if (variant == TernaryVariant::TTS) {
        return pack_scalar_runtime_arg(operation_attributes.scalar_input_b.value(), output_dtype);
    }
    if (variant == TernaryVariant::TST || ((operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL ||
                                            operation_attributes.ternary_op_type == TernaryOpType::ADDCDIV) &&
                                           operation_attributes.scalar_input_a.has_value())) {
        return pack_scalar_runtime_arg(operation_attributes.scalar_input_a.value(), output_dtype);
    }
    return 0u;
}

// Per-kernel runtime-arg counts.  Both the cache-miss and cache-hit paths fill exactly this many
// slots for every core.
constexpr size_t kNumReaderArgs = 24;
constexpr size_t kNumWriterArgs = 9;
constexpr size_t kNumComputeArgs = 4;

// The runtime-arg name of each slot of a core's row, in row order. The kernels read their args by
// these names.
constexpr std::array<const char*, kNumReaderArgs> kReaderArgNames = {
    "num_tiles",
    "start_id",
    "nD_stride",
    "d_stride",
    "n_stride",
    "c_stride",
    "D",
    "N",
    "C",
    "Ht",
    "Wt",
    "cND",
    "nD_stride_b",
    "d_stride_b",
    "n_stride_b",
    "c_stride_b",
    "src_num_tiles_b",
    "nD_stride_c",
    "d_stride_c",
    "n_stride_c",
    "c_stride_c",
    "src_num_tiles_c",
    "dst_shard_width",
    "src_num_tiles",
};
constexpr std::array<const char*, kNumWriterArgs> kWriterArgNames = {
    "num_tiles",
    "start_id",
    "dst_shard_width",
    "D",
    "N",
    "C",
    "Ht",
    "Wt",
    "cND",
};
constexpr std::array<const char*, kNumComputeArgs> kComputeArgNames = {
    "num_tiles",
    "tile_freq",
    "tile_start",
    "scalar_arg",
};

// Per-core runtime args for every core in the partition, in one place so the cache-miss and
// cache-hit paths cannot drift apart.
//
// Args are stored FLAT and row-major -- core i's reader args occupy
// [i * kNumReaderArgs, (i + 1) * kNumReaderArgs).  make_kernel_run_args() turns the rows into the
// named runtime args of each kernel.  No row holds a tensor base address: the kernels bind the
// tensors, and the framework supplies the addresses from the tensor arguments.
struct TernaryPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> reader;   // flat, kNumReaderArgs per core
    std::vector<uint32_t> writer;   // flat, kNumWriterArgs per core
    std::vector<uint32_t> compute;  // flat, kNumComputeArgs per core
};

// SINGLE SOURCE OF TRUTH for ternary per-core runtime args.  Run by BOTH create_program_artifacts()
// (cache miss) and TernaryProgramFactory::override_runtime_arguments() (cache hit).
//
// compute_program_hash coarsens each input tensor to its padded VOLUME, so one cached program is
// shared across calls whose shapes differ in their factorization -- e.g. a predicate of [4,1,32,32]
// and one of [1,4,32,32] hash identically.  Those two need different reader strides (n_stride /
// c_stride are functions of the individual dims, not of their product), and on a cache hit the
// descriptor is NOT rebuilt.  So every shape-/work-split-dependent per-core arg (strides, D/N/C/Ht/Wt,
// per-core tile counts, c_start_id, freq/counter, packed scalar, ...) would otherwise stay frozen at
// the first-miss shape and silently corrupt the result (issue #54235).  override_runtime_arguments()
// re-runs THIS builder for the current tensors and re-applies every arg.
//
// The work-core partition itself shifts with the output volume (a core can flip between work and
// noop across hits), so the builder emits args for ALL num_cores_total cores -- noop cores get
// zero-filled lists sized to match the work-core layout -- and the caller re-applies every slot
// and every tensor argument.  Nothing is left frozen however the partition moves.
TernaryPerCoreArgs build_per_core_runtime_args(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    TernaryDeviceOperation::tensor_return_value_t& output,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryVariant variant = operation_attributes.ternary_variant;

    TernaryPerCoreArgs result;

    auto partition = compute_core_partition(operation_attributes, tensor_args, output);
    const auto& cores = partition.cores;
    const auto& core_group_1 = partition.core_group_1;
    const auto& core_group_2 = partition.core_group_2;
    const uint32_t num_cores_total = partition.num_cores_total;
    const bool has_sharding = partition.has_sharding;
    const uint32_t num_tiles_per_core_group_1 = partition.num_tiles_per_core_group_1;
    const uint32_t num_tiles_per_core_group_2 = partition.num_tiles_per_core_group_2;

    const uint32_t tile_height = output.tensor_spec().tile().get_height();
    const uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t c_shard_height{}, c_shard_width{}, num_shards_per_width{};

    ShardShapeGenerator predicate_shard_shape_generator;
    ShardShapeGenerator true_shard_shape_generator;
    ShardShapeGenerator false_shard_shape_generator;
    ShardShapeGenerator output_shard_shape_generator;

    if (has_sharding) {
        // Re-derive the shard specs only for the per-core shard-shape generators (the core list /
        // work-split partition was already produced by compute_core_partition above).
        const auto shard_specs = get_shard_specs(
            predicate_tensor.tensor_spec(),
            value_true_tensor.has_value() ? value_true_tensor->tensor_spec()
                                          : std::optional<tt::tt_metal::TensorSpec>{},
            value_false_tensor.has_value() ? value_false_tensor->tensor_spec()
                                           : std::optional<tt::tt_metal::TensorSpec>{},
            output.tensor_spec());
        predicate_shard_shape_generator = ShardShapeGenerator(shard_specs->predicate_shard_spec, predicate_tensor);
        if (value_true_tensor.has_value()) {
            true_shard_shape_generator = ShardShapeGenerator(shard_specs->true_shard_spec, *value_true_tensor);
        }
        if (value_false_tensor.has_value()) {
            false_shard_shape_generator = ShardShapeGenerator(shard_specs->false_shard_spec, *value_false_tensor);
        }
        output_shard_shape_generator = ShardShapeGenerator(shard_specs->output_shard_spec, output);
        c_shard_height = shard_specs->output_shard_spec.shape[0] / tile_height;
        c_shard_width = shard_specs->output_shard_spec.shape[1] / tile_width;
        num_shards_per_width = get_shards_per_width(
            shard_specs->output_shard_spec,
            get_memory_layout(predicate_tensor, value_true_tensor, value_false_tensor, output));
    }
    constexpr size_t num_reader_args = kNumReaderArgs;
    constexpr size_t num_writer_args = kNumWriterArgs;
    constexpr size_t num_kernel_args = kNumComputeArgs;

    result.cores.reserve(num_cores_total);
    result.reader.reserve(num_cores_total * num_reader_args);
    result.writer.reserve(num_cores_total * num_writer_args);
    result.compute.reserve(num_cores_total * num_kernel_args);

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            result.cores.push_back(core);
            result.reader.resize(result.reader.size() + num_reader_args, 0u);
            result.writer.resize(result.writer.size() + num_writer_args, 0u);
            result.compute.resize(result.compute.size() + num_kernel_args, 0u);
            continue;
        }

        result.cores.push_back(core);

        // Declare variables common to all variants
        uint32_t a_num_tiles = 0, b_num_tiles = 0, f_num_tiles = 0, c_current_shard_width = 0;
        const auto out_rank = output.logical_shape().rank();
        const auto& tile = output.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();
        auto output_dims = extract_tensor_dimensions(output, out_rank, tile_h, tile_w);

        uint32_t c_start_id = 0;
        if (has_sharding) {
            auto output_shard_shape = output_shard_shape_generator(core);
            num_tiles_per_core = output_shard_shape[0] * output_shard_shape[1];  // actual
            c_current_shard_width = output_shard_shape[1];                       // actual
            auto predicate_shard_shape = predicate_shard_shape_generator(core);
            a_num_tiles = predicate_shard_shape[0] * predicate_shard_shape[1];  // actual
            if (value_true_tensor.has_value()) {
                auto true_shard_shape = true_shard_shape_generator(core);
                b_num_tiles = true_shard_shape[0] * true_shard_shape[1];  // actual
            }
            if (value_false_tensor.has_value()) {
                auto false_shard_shape = false_shard_shape_generator(core);
                f_num_tiles = false_shard_shape[0] * false_shard_shape[1];  // actual
            }
            c_start_id = (i / num_shards_per_width) * (c_shard_height * output_dims.Wt) +
                         (i % num_shards_per_width) * c_shard_width;
        } else {
            c_start_id = start_tile_id;
        }

        // Set reader runtime arguments based on variant
        if (variant == ttnn::operations::ternary::TernaryVariant::TTS ||
            variant == ttnn::operations::ternary::TernaryVariant::TST) {
            // Get the appropriate tensor operand
            const auto& tensor_operand = (variant == ttnn::operations::ternary::TernaryVariant::TTS)
                                             ? value_true_tensor.value()
                                             : value_false_tensor.value();
            uint32_t& tensor_num_tiles_ref =
                (variant == ttnn::operations::ternary::TernaryVariant::TTS) ? b_num_tiles : f_num_tiles;

            std::array<uint32_t, num_reader_args> reader_runtime_args{};
            setup_ts_reader_args_and_dims<num_reader_args>(
                reader_runtime_args,
                predicate_tensor,
                tensor_operand,
                output,
                num_tiles_per_core,
                c_start_id,
                a_num_tiles,
                tensor_num_tiles_ref,
                c_current_shard_width,
                broadcast_type,
                has_sharding,
                out_rank,
                tile_h,
                tile_w);
            result.reader.insert(result.reader.end(), reader_runtime_args.begin(), reader_runtime_args.end());
        } else if (variant == TernaryVariant::TTT) {
            auto pred_dims = extract_tensor_dimensions(predicate_tensor, out_rank, tile_h, tile_w);
            auto true_dims = extract_tensor_dimensions(value_true_tensor.value(), out_rank, tile_h, tile_w);
            auto false_dims = extract_tensor_dimensions(value_false_tensor.value(), out_rank, tile_h, tile_w);
            auto pred_strides = calculate_strides(pred_dims);
            auto true_strides = calculate_strides(true_dims);
            auto false_strides = calculate_strides(false_dims);
            // Tile counts are already calculated above if has_sharding
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized
            reader_runtime_args[0] = num_tiles_per_core;                  // 0: num_tiles (per core)
            reader_runtime_args[1] = c_start_id;                          // 1: start_id
            // Set arguments for both broadcast and no-bcast cases (same arguments needed for width sharding)
            reader_runtime_args[2] = pred_strides.nD_stride;    // 2: nD_stride
            reader_runtime_args[3] = pred_strides.d_stride;     // 3: d_stride
            reader_runtime_args[4] = pred_strides.n_stride;     // 4: n_stride
            reader_runtime_args[5] = pred_strides.c_stride;     // 5: c_stride
            reader_runtime_args[6] = output_dims.D;             // 6: D
            reader_runtime_args[7] = output_dims.N;             // 7: N
            reader_runtime_args[8] = output_dims.C;             // 8: C
            reader_runtime_args[9] = output_dims.Ht;            // 9: Ht
            reader_runtime_args[10] = output_dims.Wt;           // 10: Wt
            reader_runtime_args[11] = output_dims.ND;           // 11: cND
            reader_runtime_args[12] = true_strides.nD_stride;   // 12: true_nD_stride
            reader_runtime_args[13] = true_strides.d_stride;    // 13: true_d_stride
            reader_runtime_args[14] = true_strides.n_stride;    // 14: true_n_stride
            reader_runtime_args[15] = true_strides.c_stride;    // 15: true_c_stride
            reader_runtime_args[16] = b_num_tiles;              // 16: true_num_tiles
            reader_runtime_args[17] = false_strides.nD_stride;  // 17: false_nD_stride
            reader_runtime_args[18] = false_strides.d_stride;   // 18: false_d_stride
            reader_runtime_args[19] = false_strides.n_stride;   // 19: false_n_stride
            reader_runtime_args[20] = false_strides.c_stride;   // 20: false_c_stride
            reader_runtime_args[21] = f_num_tiles;              // 21: false_num_tiles
            reader_runtime_args[22] = c_current_shard_width;    // 22: dst_shard_width
            reader_runtime_args[23] = a_num_tiles;              // 23: src_num_tiles (predicate)

            result.reader.insert(result.reader.end(), reader_runtime_args.begin(), reader_runtime_args.end());
        } else {
            TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
        }

        // Writer runtime args.  The output base address is not among them: the writer binds the
        // output tensor.  The framework allows the output buffer to alias an input (in-place ops).
        const std::array<uint32_t, num_writer_args> writer_args = {
            num_tiles_per_core,     // 0: num_tiles
            c_start_id,             // 1: start_id
            c_current_shard_width,  // 2: dst_shard_width
            output_dims.D,          // 3: D
            output_dims.N,          // 4: N
            output_dims.C,          // 5: C
            output_dims.Ht,         // 6: Ht
            output_dims.Wt,         // 7: Wt
            output_dims.ND,         // 8: cND
        };
        result.writer.insert(result.writer.end(), writer_args.begin(), writer_args.end());

        // Compute runtime args.  scalar_arg is the packed scalar_input_a/scalar_input_b; it is
        // EXCLUDED from compute_program_hash and therefore DYNAMIC -- like every other arg here it is
        // re-derived by this builder on each cache hit. Packed via the shared single-source-of-truth helper.
        const uint32_t scalar_arg = pack_compute_scalar_arg(operation_attributes, output.dtype());
        auto [freq, counter] = [&] {
            switch (broadcast_type) {
                case TernaryBroadcastType::ROW_COL_BCAST:
                case TernaryBroadcastType::COL_BCAST: {
                    uint32_t start_t = start_tile_id % (output_dims.Ht * output_dims.Wt);
                    uint32_t start_tw = start_t % output_dims.Wt;
                    return std::pair{output_dims.Wt, start_tw};
                }
                case TernaryBroadcastType::SCALAR_BCAST:
                case TernaryBroadcastType::SCALAR_A_BCAST:
                case TernaryBroadcastType::SCALAR_B_BCAST: {
                    uint32_t HtWt = output_dims.Ht * output_dims.Wt;
                    uint32_t start_t = start_tile_id % HtWt;
                    return std::pair{HtWt, start_t};
                }
                case TernaryBroadcastType::NONE:
                case TernaryBroadcastType::OUTER_BCAST:
                case TernaryBroadcastType::ROW_BCAST: return std::pair{0u, 0u};
                default: __builtin_unreachable();
            }
        }();
        const std::array<uint32_t, num_kernel_args> compute_args = {num_tiles_per_core, freq, counter, scalar_arg};
        result.compute.insert(result.compute.end(), compute_args.begin(), compute_args.end());

        start_tile_id += num_tiles_per_core;
    }

    TT_FATAL(
        result.reader.size() == result.cores.size() * num_reader_args &&
            result.writer.size() == result.cores.size() * num_writer_args &&
            result.compute.size() == result.cores.size() * num_kernel_args,
        "ternary per-core arg rows are ragged: {} cores but {}/{}/{} reader/writer/compute args",
        result.cores.size(),
        result.reader.size(),
        result.writer.size(),
        result.compute.size());

    return result;
}

// Sets one named runtime arg per slot of a kernel's rows: slot k of core i's row becomes arg names[k]
// on core i.
template <size_t N>
void add_runtime_args(
    KernelRunArgs& kernel_run_args,
    const std::array<const char*, N>& names,
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& rows) {
    for (size_t slot = 0; slot < N; ++slot) {
        auto& per_node = kernel_run_args.runtime_arg_values[names[slot]];
        for (size_t i = 0; i < cores.size(); ++i) {
            per_node[cores[i]] = rows[(i * N) + slot];
        }
    }
}

// Turns the builder's per-core rows into the named runtime args of the three kernels, for every core.
// Shared by the cache-miss and cache-hit paths.
Group<KernelRunArgs> make_kernel_run_args(const TernaryPerCoreArgs& per_core) {
    KernelRunArgs reader{.kernel = READER_KERNEL};
    KernelRunArgs writer{.kernel = WRITER_KERNEL};
    KernelRunArgs compute{.kernel = COMPUTE_KERNEL};
    add_runtime_args(reader, kReaderArgNames, per_core.cores, per_core.reader);
    add_runtime_args(writer, kWriterArgNames, per_core.cores, per_core.writer);
    add_runtime_args(compute, kComputeArgNames, per_core.cores, per_core.compute);
    return {std::move(reader), std::move(writer), std::move(compute)};
}

// One tensor argument per tensor parameter of the variant, each referencing the op's own io tensor.
// Shared by the cache-miss and cache-hit paths.
Table<TensorParamName, TensorArgument> make_tensor_args(
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    const TernaryDeviceOperation::tensor_return_value_t& output) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    Table<TensorParamName, TensorArgument> args{
        {PREDICATE_TENSOR, predicate_tensor.mesh_tensor()},
        {OUTPUT_TENSOR, output.mesh_tensor()},
    };
    if (value_true_tensor.has_value()) {
        args.emplace(VALUE_TRUE_TENSOR, value_true_tensor->mesh_tensor());
    }
    if (value_false_tensor.has_value()) {
        args.emplace(VALUE_FALSE_TENSOR, value_false_tensor->mesh_tensor());
    }
    return args;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::ternary {

std::optional<AllShardVolumes> get_shard_volumes(
    const tt::tt_metal::TensorSpec& predicate_spec,
    const std::optional<tt::tt_metal::TensorSpec>& true_spec,
    const std::optional<tt::tt_metal::TensorSpec>& false_spec,
    const tt::tt_metal::TensorSpec& output_spec) {
    const auto shard_specs =
        CMAKE_UNIQUE_NAMESPACE::get_shard_specs(predicate_spec, true_spec, false_spec, output_spec);

    if (not shard_specs.has_value()) {
        return std::nullopt;
    }

    const auto predicate_sharded = predicate_spec.memory_config().is_sharded();
    const auto true_sharded = true_spec.has_value() and true_spec->memory_config().is_sharded();
    const auto false_sharded = false_spec.has_value() and false_spec->memory_config().is_sharded();
    const auto output_sharded = output_spec.memory_config().is_sharded();
    const auto tile_hw = output_spec.tile().get_tile_hw();

    return AllShardVolumes{
        .predicate_shard_volume =
            predicate_sharded ? shard_specs->predicate_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .true_shard_volume =
            true_sharded ? shard_specs->true_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .false_shard_volume =
            false_sharded ? shard_specs->false_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .output_shard_volume =
            output_sharded ? shard_specs->output_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
    };
}

ttnn::device_operation::ProgramArtifacts TernaryDeviceOperation::TernaryProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryVariant variant = operation_attributes.ternary_variant;
    TernaryBroadcastType broadcast_type = operation_attributes.broadcast_type;

    // Use TernaryKernelConfig to get the appropriate kernel names
    TernaryKernelConfig kernel_config(operation_attributes.ternary_op_type, variant, broadcast_type);
    auto reader_kernel = kernel_config.reader_kernel;
    auto compute_kernel = kernel_config.compute_kernel;
    auto writer_kernel = kernel_config.writer_kernel;

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    /*  The where_llk uses UINT16 instrn set for bfloat16 inputs.
        If the bfloat16 inputs' DFBs are set to UINT16 dataformat this will enable us to get 'NaN' in the outputs even
       for bfloat16 dtype. We need to test the impact of this hack on the composite ops that use where op and on the
       models, since bfloat16 packs NaN as inf as this is an expected behaviour and a known HW limitation of bfloat16
       dtype. Ex: Force the dataformat of all the bfloat16 inputs' DFBs to be of UINT16 dataformat
        (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());
        datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype()); */

    // Handle data formats based on variant and tensor availability
    DataFormat value_true_data_format, value_false_data_format;
    if (variant == TernaryVariant::TTS) {
        // TTS: only value_true tensor exists
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        value_false_data_format = value_true_data_format;
    } else if (variant == TernaryVariant::TST) {
        // TST: only value_false tensor exists
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        value_true_data_format = value_false_data_format;
    } else {
        // TTT: both tensors exist
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
    }
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());

    uint32_t predicate_single_tile_size = tt::tile_size(predicate_data_format);
    uint32_t value_true_single_tile_size = tt::tile_size(value_true_data_format);
    uint32_t value_false_single_tile_size = tt::tile_size(value_false_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    // Get shard volumes (using tt::tt_metal::TensorSpec)
    const auto shard_volumes = get_shard_volumes(
        predicate_tensor.tensor_spec(),
        value_true_tensor.has_value() ? value_true_tensor->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{},
        value_false_tensor.has_value() ? value_false_tensor->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{},
        output.tensor_spec());
    const bool has_sharding = shard_volumes.has_value();
    const auto predicate_sharded = has_sharding and shard_volumes->predicate_shard_volume.has_value();
    const auto true_sharded = has_sharding and shard_volumes->true_shard_volume.has_value();
    const auto false_sharded = has_sharding and shard_volumes->false_shard_volume.has_value();
    const auto output_sharded = has_sharding and shard_volumes->output_shard_volume.has_value();
    const auto predicate_num_tiles_per_shard = has_sharding ? shard_volumes->predicate_shard_volume : std::nullopt;
    const auto true_num_tiles_per_shard = has_sharding ? shard_volumes->true_shard_volume : std::nullopt;
    const auto false_num_tiles_per_shard = has_sharding ? shard_volumes->false_shard_volume : std::nullopt;
    const auto output_num_tiles_per_shard = has_sharding ? shard_volumes->output_shard_volume : std::nullopt;

    const auto& all_device_cores = operation_attributes.worker_grid;

    constexpr uint32_t num_tiles_per_dfb = 2;

    // ---- Tensor parameters ----
    // Every tensor parameter is declared with dynamic_tensor_shape and relax_logical_rank. The reasons:
    //  - compute_program_hash keys each tensor on its logical H and W, not on its leading dims or its rank.
    //    Calls such as [4, 1, 32, 32] and [1, 4, 32, 32], or [1, 32, 64] and [1, 1, 32, 64], therefore share
    //    one cached program. The second call of each pair brings a different logical shape, and in the
    //    second pair a different rank.
    //  - Nothing that depends on the leading dims is baked into the program. override_runtime_arguments
    //    re-derives every per-core runtime arg on every dispatch.
    //  - A sharded tensor's accessor reads its shape in pages from a runtime word. The rest of its shard
    //    geometry is fixed in the program, and compute_program_hash pins that geometry for every tensor.
    //  - The op accepts TILE layout only, in every slot, and the key hashes each slot's dtype, memory config,
    //    alignment and tile. The parts of a TensorSpec that no relaxation frees are pinned by the key as well.
    // Without the two flags, the second call of such a pair would fail the TensorSpec check on its cache hit.
    constexpr TensorSpecRelaxations relaxations{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{
        .unique_id = PREDICATE_TENSOR,
        .spec = predicate_tensor.mesh_tensor().tensor_spec(),
        .relaxations = relaxations,
    });
    if (value_true_tensor.has_value()) {
        tensor_parameters.push_back(TensorParameter{
            .unique_id = VALUE_TRUE_TENSOR,
            .spec = value_true_tensor->mesh_tensor().tensor_spec(),
            .relaxations = relaxations,
        });
    }
    if (value_false_tensor.has_value()) {
        tensor_parameters.push_back(TensorParameter{
            .unique_id = VALUE_FALSE_TENSOR,
            .spec = value_false_tensor->mesh_tensor().tensor_spec(),
            .relaxations = relaxations,
        });
    }
    tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output.mesh_tensor().tensor_spec(),
        .relaxations = relaxations,
    });

    // ---- Dataflow Buffers ----
    // A natively sharded tensor's DFB is borrowed from the tensor: it holds the tensor's resident shard.
    Group<DataflowBufferSpec> dataflow_buffers;

    // Input buffers - Create predicate DFB (always in_a)
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DFB_IN_A,
        .entry_size = predicate_single_tile_size,
        .num_entries = predicate_num_tiles_per_shard.value_or(num_tiles_per_dfb),
        .data_format_metadata = predicate_data_format,
        .borrowed_from = predicate_sharded ? std::optional{PREDICATE_TENSOR} : std::nullopt,
    });

    // Create in_b based on variant - this is the primary tensor DFB
    if (variant == TernaryVariant::TTS) {
        // TTS: in_b = value_true tensor (value_false is scalar)
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFB_IN_B,
            .entry_size = value_true_single_tile_size,
            .num_entries = true_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_true_data_format,
            .borrowed_from = true_sharded ? std::optional{VALUE_TRUE_TENSOR} : std::nullopt,
        });
    } else if (variant == TernaryVariant::TST) {
        // TST: in_b = value_false tensor (value_true is scalar)
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFB_IN_B,
            .entry_size = value_false_single_tile_size,
            .num_entries = false_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_false_data_format,
            .borrowed_from = false_sharded ? std::optional{VALUE_FALSE_TENSOR} : std::nullopt,
        });
    } else if (variant == TernaryVariant::TTT) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFB_IN_B,
            .entry_size = value_true_single_tile_size,
            .num_entries = true_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_true_data_format,
            .borrowed_from = true_sharded ? std::optional{VALUE_TRUE_TENSOR} : std::nullopt,
        });

        // Create DFB for value_false (using actual false tensor)
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFB_IN_C,
            .entry_size = value_false_single_tile_size,
            .num_entries = false_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_false_data_format,
            .borrowed_from = false_sharded ? std::optional{VALUE_FALSE_TENSOR} : std::nullopt,
        });
    } else {
        TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
    }

    // Output buffer - use out for all cases now
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DFB_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_num_tiles_per_shard.value_or(num_tiles_per_dfb),
        .data_format_metadata = output_data_format,
        .borrowed_from = output_sharded ? std::optional{OUTPUT_TENSOR} : std::nullopt,
    });

    // BROADCAST DETECTION - Common for both reader and compute kernels
    bool pred_is_bcast = false, true_is_bcast = false, false_is_bcast = false;
    detect_broadcasts(
        pred_is_bcast,
        true_is_bcast,
        false_is_bcast,
        variant,
        broadcast_type,
        predicate_tensor,
        value_true_tensor,
        value_false_tensor);

    // READER KERNEL - Use kernel path from utils
    std::map<std::string, std::string> reader_defines;
    setup_reader_defines(
        reader_defines,
        variant,
        predicate_tensor,
        value_true_tensor,
        value_false_tensor,
        pred_is_bcast,
        true_is_bcast,
        false_is_bcast,
        has_sharding);

    // For mixed row+col broadcast: add per-tensor row-bcast and scalar defines for the reader
    if (broadcast_type == TernaryBroadcastType::ROW_COL_BCAST) {
        auto pred_shape = predicate_tensor.logical_shape();
        auto pred_h = pred_shape[-2];
        auto pred_w = pred_shape[-1];

        if (variant == TernaryVariant::TTT) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto max_h = std::max({pred_h, true_shape[-2], false_shape[-2]});
            auto max_w = std::max({pred_w, true_shape[-1], false_shape[-1]});

            bool pred_h1 = (pred_h == 1 && max_h > 1);
            bool pred_w1 = (pred_w == 1 && max_w > 1);
            bool true_h1 = (true_shape[-2] == 1 && max_h > 1);
            bool true_w1 = (true_shape[-1] == 1 && max_w > 1);
            bool false_h1 = (false_shape[-2] == 1 && max_h > 1);
            bool false_w1 = (false_shape[-1] == 1 && max_w > 1);

            reader_defines["SRC_ROW_BCAST_A"] = (pred_h1 && !pred_w1) ? "1" : "0";
            reader_defines["SRC_ROW_BCAST_B"] = (true_h1 && !true_w1) ? "1" : "0";
            reader_defines["SRC_ROW_BCAST_C"] = (false_h1 && !false_w1) ? "1" : "0";
            reader_defines["SRC_SCALAR_A"] = (pred_h1 && pred_w1) ? "1" : "0";
            reader_defines["SRC_SCALAR_B"] = (true_h1 && true_w1) ? "1" : "0";
            reader_defines["SRC_SCALAR_C"] = (false_h1 && false_w1) ? "1" : "0";
        } else {
            const auto& tensor_op =
                (variant == TernaryVariant::TTS) ? value_true_tensor.value() : value_false_tensor.value();
            auto tensor_shape = tensor_op.logical_shape();
            auto max_h = std::max(pred_h, tensor_shape[-2]);
            auto max_w = std::max(pred_w, tensor_shape[-1]);

            bool pred_h1 = (pred_h == 1 && max_h > 1);
            bool pred_w1 = (pred_w == 1 && max_w > 1);
            bool tensor_h1 = (tensor_shape[-2] == 1 && max_h > 1);
            bool tensor_w1 = (tensor_shape[-1] == 1 && max_w > 1);

            reader_defines["SRC_ROW_BCAST_A"] = (pred_h1 && !pred_w1) ? "1" : "0";
            reader_defines["SRC_SCALAR_A"] = (pred_h1 && pred_w1) ? "1" : "0";
            reader_defines["SRC_ROW_BCAST_CB1"] = (tensor_h1 && !tensor_w1) ? "1" : "0";
            reader_defines["SRC_SCALAR_CB1"] = (tensor_h1 && tensor_w1) ? "1" : "0";
        }
    }

    std::map<std::string, std::string> kernel_defines;
    if (variant == TernaryVariant::TTT) {
        if (CMAKE_UNIQUE_NAMESPACE::is_llk_bcast(
                broadcast_type,
                predicate_tensor.dtype(),
                value_true_tensor.value().dtype(),
                value_false_tensor.value().dtype(),
                output.dtype())) {
            CMAKE_UNIQUE_NAMESPACE::overwrite_compute_kernel_name_and_defines(
                compute_kernel, broadcast_type, operation_attributes.ternary_op_type);
            reader_defines["BCAST_LLK"] = "1";
        } else {
            reader_defines["BCAST_LLK"] = "0";
        }
    }

    // ---- Reader Kernel ----
    bool is_fpu = false;

    if (operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL ||
        operation_attributes.ternary_op_type == TernaryOpType::ADDCDIV) {
        is_fpu = (predicate_tensor.dtype() == value_true_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() == value_false_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() != DataType::FLOAT32 && predicate_tensor.dtype() != DataType::INT32 &&
                  predicate_tensor.dtype() != DataType::UINT32);
    }

    // The reader produces in_a and in_b, and in_c for TTT. An operand whose SRC_SHARDED_* define is 1 is natively
    // sharded: its DFB is borrowed from the tensor, the reader only publishes the resident shard, and the kernel
    // compiles that operand's accessor out. Every other operand is read through a tensor accessor. The reader
    // therefore binds an operand's tensor exactly when its SRC_SHARDED_* define is 0, read back from the defines
    // the kernel is built with.
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = DFB_IN_A,
            .accessor_name = "in_a",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_IN_B,
            .accessor_name = "in_b",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<TensorBinding> reader_tensor_bindings;
    if (reader_defines.at("SRC_SHARDED_A") == "0") {
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = PREDICATE_TENSOR,
            .accessor_name = "in_a",
        });
    }
    if (reader_defines.at("SRC_SHARDED_B") == "0") {
        // in_b holds value_true for TTS and TTT, and value_false for TST
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = (variant == TernaryVariant::TST) ? VALUE_FALSE_TENSOR : VALUE_TRUE_TENSOR,
            .accessor_name = "in_b",
        });
    }
    if (variant == TernaryVariant::TTT) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_IN_C,
            .accessor_name = "in_c",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        if (reader_defines.at("SRC_SHARDED_C") == "0") {
            reader_tensor_bindings.push_back(TensorBinding{
                .tensor_parameter_name = VALUE_FALSE_TENSOR,
                .accessor_name = "in_c",
            });
        }
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = get_kernel_file_path(reader_kernel, is_fpu),
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"has_sharding", static_cast<uint32_t>(has_sharding)}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = Group<std::string>(kReaderArgNames.begin(), kReaderArgNames.end()),
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // ---- Writer Kernel ----
    std::map<std::string, std::string> writer_defines;
    writer_defines["DST_SHARDED"] = (output_sharded && has_sharding) ? "1" : "0";

    // With DST_SHARDED=1 the output is natively sharded: compute packs straight into the resident output shard, and
    // the writer's body is compiled out. The writer then touches neither out nor the output tensor. It binds them
    // only when DST_SHARDED is 0.
    const bool dst_sharded = writer_defines.at("DST_SHARDED") == "1";
    Group<DFBBinding> writer_dfb_bindings;
    Group<TensorBinding> writer_tensor_bindings;
    if (!dst_sharded) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = OUTPUT_TENSOR,
            .accessor_name = "out",
        });
    }

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = get_kernel_file_path(writer_kernel, is_fpu),
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args = {{"has_sharding", static_cast<uint32_t>(has_sharding)}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = Group<std::string>(kWriterArgNames.begin(), kWriterArgNames.end()),
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // ---- Compute Kernel ----
    // Any 32-bit operand needs a 32-bit Dest: a Float32 input DFB is unpacked with UnpackToDest below,
    // and that lands corrupted in a Dest configured for 16-bit datums.
    auto is_32bit_format = [](tt::DataFormat format) {
        return format == tt::DataFormat::UInt32 || format == tt::DataFormat::Int32 || format == tt::DataFormat::Float32;
    };
    bool fp32_dest_acc_en = is_32bit_format(output_data_format) || is_32bit_format(predicate_data_format) ||
                            is_32bit_format(value_true_data_format) || is_32bit_format(value_false_data_format);

    // Unpack modes are keyed by DFB. A Float32 DFB is unpacked straight to Dest (UnpackToDest); every other DFB
    // takes the default unpack-to-SrcA/B path, which needs no entry.
    ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    const auto unpack_to_dest_if_float32 = [&unpack_modes](const DFBSpecName& dfb, DataType dtype) {
        if (dtype == DataType::FLOAT32) {
            unpack_modes.emplace(dfb, UnpackMode::UnpackToDest);
        }
    };

    // in_a is always predicate
    unpack_to_dest_if_float32(DFB_IN_A, predicate_tensor.dtype());

    // in_b assignment depends on variant
    if (variant == TernaryVariant::TTS) {
        // TTS: in_b = value_true tensor
        unpack_to_dest_if_float32(DFB_IN_B, value_true_tensor.value().dtype());
    } else if (variant == TernaryVariant::TST) {
        // TST: in_b = value_false tensor
        unpack_to_dest_if_float32(DFB_IN_B, value_false_tensor.value().dtype());
    } else {
        // TTT: in_b = value_true tensor, in_c = value_false tensor (including column broadcast)
        unpack_to_dest_if_float32(DFB_IN_B, value_true_tensor.value().dtype());
        unpack_to_dest_if_float32(DFB_IN_C, value_false_tensor.value().dtype());
    }

    // Output DFB depends on variant: c_2 for binary_ng compatibility (TTT col bcast), c_3 for other cases
    unpack_to_dest_if_float32(DFB_OUT, output.dtype());

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    // All variants use the same compile args now
    uint32_t scalar_is_true_value = (variant == TernaryVariant::TST) ? 1 : 0;

    // Add binary_ng style defines for TTT column broadcast case
    if (variant == TernaryVariant::TTT) {
        // 3-tensor broadcast configuration - set defines for each tensor independently
        kernel_defines["BCAST_A"] = pred_is_bcast ? "1" : "0";
        kernel_defines["BCAST_B"] = true_is_bcast ? "1" : "0";
        kernel_defines["BCAST_C"] = false_is_bcast ? "1" : "0";
    } else if (
        (variant == TernaryVariant::TTS || variant == TernaryVariant::TST) &&
        (broadcast_type == TernaryBroadcastType::COL_BCAST || broadcast_type == TernaryBroadcastType::ROW_COL_BCAST)) {
        // Unified TTS/TST column and mixed broadcast configuration
        kernel_defines["BCAST_A"] = pred_is_bcast ? "1" : "0";
        if (variant == TernaryVariant::TTS) {
            kernel_defines["BCAST_B"] = true_is_bcast ? "1" : "0";
            kernel_defines["BCAST_C"] = "0";  // False is scalar for TTS
        } else {                              // TST
            kernel_defines["BCAST_B"] = "0";  // True is scalar for TST
            kernel_defines["BCAST_C"] = false_is_bcast ? "1" : "0";
        }
    }
    if ((variant == TernaryVariant::TTS || variant == TernaryVariant::TST) &&
        (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST ||
         broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST)) {
        // Unified TTS/TST scalar broadcast configuration
        kernel_defines["BCAST_A"] = (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST) ? "1" : "0";
        if (variant == TernaryVariant::TTS) {
            kernel_defines["BCAST_B"] = (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
            kernel_defines["BCAST_C"] = "0";  // False is scalar for TTS
        } else {                              // TST
            kernel_defines["BCAST_B"] = "0";  // True is scalar for TST
            kernel_defines["BCAST_C"] = (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
        }
    }
    // The SFPU variant must match the Dest layout: Float16_b variants access only the low 16 bits of each Dest word,
    // so once another operand has put Dest in 32-bit mode a 16-bit predicate selects the Float32 variant. That holds
    // for every floating-point predicate, and for where with any 16-bit predicate.
    auto sfpu_dtype = predicate_tensor.dtype();
    if (fp32_dest_acc_en && !is_32bit_format(predicate_data_format) &&
        (is_floating_point(predicate_tensor.dtype()) || operation_attributes.ternary_op_type == TernaryOpType::WHERE)) {
        sfpu_dtype = DataType::FLOAT32;
    }
    auto op_defines = CMAKE_UNIQUE_NAMESPACE::get_ternary_compute_defines(operation_attributes, sfpu_dtype);
    for (const auto& [key, value] : op_defines) {
        kernel_defines[key] = value;
    }

    // Add common fill defines
    kernel_defines["FILL_LLK"] = "fill_tile";
    if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["FILL_LLK"] = "fill_tile_int<DataFormat::Int32>";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else if (predicate_tensor.dtype() == DataType::UINT32) {
        kernel_defines["FILL_LLK"] = "fill_tile_int<DataFormat::UInt32>";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else {
        kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
    }

    const bool use_addcmul_int_kernel =
        operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL &&
        (operation_attributes.dtype == DataType::INT32 || operation_attributes.dtype == DataType::UINT32);
    if (use_addcmul_int_kernel) {
        for (const auto& [k, v] : get_addcmul_int_kernel_defines(operation_attributes.dtype.value())) {
            kernel_defines[k] = v;
        }
    }
    std::string compute_kernel_path = use_addcmul_int_kernel ? override_addcmul_compute_kernel(compute_kernel)
                                                             : get_kernel_file_path(compute_kernel, is_fpu);

    // The compute kernel consumes in_a and in_b, and in_c for TTT, and produces out. When the output is natively
    // sharded, nothing drains out (see the writer above). Compute is then the only kernel that touches out, and it
    // also binds out as consumer.
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = DFB_IN_A,
            .accessor_name = "in_a",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_IN_B,
            .accessor_name = "in_b",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    if (variant == TernaryVariant::TTT) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_IN_C,
            .accessor_name = "in_c",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_OUT,
        .accessor_name = "out",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (dst_sharded) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // The two LLK row-broadcast compute kernels (ComputeRowBcastTTT, and ComputeRowBcastAddcOp on the FPU path)
    // expand each row-broadcast operand into a staging DFB with unary_bcast, then copy from that DFB: bcast_a,
    // bcast_b and bcast_c stage in_a, in_b and in_c. Compute both fills and drains a staging DFB. It therefore binds
    // each one as producer and consumer. A staging DFB exists only for an operand whose BCAST_* define is 1, and the
    // kernels reference it only under that define.
    const bool compute_stages_llk_row_bcast =
        !use_addcmul_int_kernel && (compute_kernel == KernelName::ComputeRowBcastTTT ||
                                    (compute_kernel == KernelName::ComputeRowBcastAddcOp && is_fpu));
    const auto add_llk_row_bcast_dfb =
        [&](const DFBSpecName& dfb, const char* accessor_name, uint32_t tile_size, DataFormat data_format) {
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = dfb,
                .entry_size = tile_size,
                .num_entries = num_tiles_per_dfb,
                .data_format_metadata = data_format,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = dfb,
                .accessor_name = accessor_name,
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = dfb,
                .accessor_name = accessor_name,
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        };
    if (compute_stages_llk_row_bcast) {
        if (kernel_defines.at("BCAST_A") == "1") {
            add_llk_row_bcast_dfb(DFB_BCAST_A, "bcast_a", predicate_single_tile_size, predicate_data_format);
        }
        if (kernel_defines.at("BCAST_B") == "1") {
            add_llk_row_bcast_dfb(DFB_BCAST_B, "bcast_b", value_true_single_tile_size, value_true_data_format);
        }
        if (kernel_defines.at("BCAST_C") == "1") {
            add_llk_row_bcast_dfb(DFB_BCAST_C, "bcast_c", value_false_single_tile_size, value_false_data_format);
        }
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = compute_kernel_path,
        // O3 is the level a compute kernel that sets none builds at; KernelSpec defaults every kernel to O2.
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"num_tiles_per_cycle", num_tiles_per_cycle},
                {"scalar_is_true", scalar_is_true_value},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = Group<std::string>(kComputeArgNames.begin(), kComputeArgNames.end()),
            },
        .hw_config =
            ComputeHardwareConfig{
                .enable_32_bit_dest = fp32_dest_acc_en,
                .unpack_modes = std::move(unpack_modes),
            },
    };

    // ---- Per-core runtime args ----
    // Built via the shared single-source-of-truth builder so create_program_artifacts() (cache miss, here)
    // and override_runtime_arguments() (cache hit) stay byte-identical.
    //
    // The tensor base addresses are not runtime args: each kernel binds the tensors it reads or writes, and the
    // framework supplies their addresses from the tensor arguments.
    auto per_core = build_per_core_runtime_args(operation_attributes, tensor_args, output, broadcast_type);

    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader_spec));
    kernels.push_back(std::move(writer_spec));
    kernels.push_back(std::move(compute_spec));

    return ttnn::device_operation::ProgramArtifacts{
        .spec =
            ProgramSpec{
                .name = "ternary",
                .kernels = std::move(kernels),
                .dataflow_buffers = std::move(dataflow_buffers),
                .tensor_parameters = std::move(tensor_parameters),
                .work_units =
                    {
                        WorkUnitSpec{
                            .name = "ternary",
                            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
                            .target_nodes = all_device_cores,
                        },
                    },
            },
        .run_params =
            ProgramRunArgs{
                .kernel_run_args = make_kernel_run_args(per_core),
                .tensor_args = make_tensor_args(tensor_args, output),
            },
    };
}

tt::tt_metal::experimental::ProgramRunArgs TernaryDeviceOperation::TernaryProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-apply ALL per-dispatch state to the cached program on a program-cache hit.  No spec
    // rebuild and no kernel recompile -- but every per-core arg IS re-derived. We re-derive them from the SAME
    // shared builder create_program_artifacts() uses, so the two paths stay byte-identical by construction.
    //
    // Every tensor argument is supplied as well. That refreshes each tensor binding's base address, and it
    // re-points each borrowed (natively sharded) DFB at the current buffer. Each DFB is matched to its tensor
    // by tensor parameter, which is what keeps an in-place alias correct: out always tracks the output
    // tensor, even when the output shares its buffer with an input.
    //
    // A tensor binding also carries its accessor's runtime words, and supplying the tensor re-emits them
    // together with the address.  Re-emitting them changes nothing.  They are a pure function of state that
    // compute_program_hash already pins, and a cache hit therefore cannot find them stale.  Two cases:
    //
    //   Interleaved -- vacuous.  An interleaved TILE accessor has no runtime words.
    //
    //   Sharded -- the words are BufferDistributionSpec::tensor_shape_in_pages(), the shard-collapsed
    //   view of the tensor (leading dims flattened into the height), never the dim factorization.  So
    //   they are a function of (padded volume, H, W) -- all three hashed.
    //   ND sharding cannot reach this path: get_shard_specs() reads memory_config().shard_spec(), the
    //   2D spec, so an ND-sharded tensor never takes it.
    //
    // If ternary ever gains ND sharding, tensor_shape_in_pages() would carry the individual dims and
    // this reasoning lapses -- the words would then change between hits, and the tensor arguments
    // supplied here are what keeps them current.
    auto per_core = CMAKE_UNIQUE_NAMESPACE::build_per_core_runtime_args(
        operation_attributes, tensor_args, output, operation_attributes.broadcast_type);

    return tt::tt_metal::experimental::ProgramRunArgs{
        .kernel_run_args = CMAKE_UNIQUE_NAMESPACE::make_kernel_run_args(per_core),
        .tensor_args = CMAKE_UNIQUE_NAMESPACE::make_tensor_args(tensor_args, output),
    };
}

}  // namespace ttnn::operations::ternary
