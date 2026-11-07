// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ternary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace {
// Helper function to add broadcast defines for dataflow kernels
void add_broadcast_defines(
    std::map<std::string, std::string>& defines, bool pred_is_bcast, bool true_is_bcast, bool false_is_bcast) {
    defines["SRC_BCAST_A"] = pred_is_bcast ? "1" : "0";   // Predicate tensor (in_a)
    defines["SRC_BCAST_B"] = true_is_bcast ? "1" : "0";   // True tensor (in_b)
    defines["SRC_BCAST_C"] = false_is_bcast ? "1" : "0";  // False tensor (in_c)
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
    // SRC_SHARDED_A always maps to in_a (predicate)
    // SRC_SHARDED_B maps to in_b: value_true for TTT/TTS, value_false for TST
    // SRC_SHARDED_C maps to in_c: value_false for TTT (unused for TTS/TST)
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
    // Slots 0-2 used to hold the operand base addresses. The tensor bindings carry them now, and the
    // slots stay zero. Slot 2 was always 0 here: TTS / TST have no third tensor.
    reader_runtime_args[3] = num_tiles_per_core;  // 3: num_tiles (per core)
    reader_runtime_args[4] = start_tile_id;       // 4: start_id
    // Extended broadcast arguments
    if (broadcast_type != ttnn::operations::ternary::TernaryBroadcastType::NONE) {
        reader_runtime_args[5] = pred_strides.nD_stride;  // 5: nD_stride
        reader_runtime_args[6] = pred_strides.d_stride;   // 6: d_stride
        reader_runtime_args[7] = pred_strides.n_stride;   // 7: n_stride
        reader_runtime_args[8] = pred_strides.c_stride;   // 8: c_stride
        reader_runtime_args[9] = output_dims.D;           // 9: D
        reader_runtime_args[10] = output_dims.N;          // 10: N
        reader_runtime_args[11] = output_dims.C;          // 11: C
        reader_runtime_args[12] = output_dims.Ht;         // 12: Ht
        reader_runtime_args[13] = output_dims.Wt;         // 13: Wt
        reader_runtime_args[14] = output_dims.ND;         // 14: cND
        // Both TTS and TST: tensor operand strides in 15-19, scalar operand zeros in 20-24
        reader_runtime_args[15] = tensor_strides.nD_stride;  // 15: tensor_operand_nD_stride
        reader_runtime_args[16] = tensor_strides.d_stride;   // 16: tensor_operand_d_stride
        reader_runtime_args[17] = tensor_strides.n_stride;   // 17: tensor_operand_n_stride
        reader_runtime_args[18] = tensor_strides.c_stride;   // 18: tensor_operand_c_stride
        reader_runtime_args[19] = tensor_num_tiles;          // 19: tensor_operand_num_tiles
        // Scalar operand has no strides
        reader_runtime_args[20] = 0u;  // 20: scalar_operand_nD_stride
        reader_runtime_args[21] = 0u;  // 21: scalar_operand_d_stride
        reader_runtime_args[22] = 0u;  // 22: scalar_operand_n_stride
        reader_runtime_args[23] = 0u;  // 23: scalar_operand_c_stride
        reader_runtime_args[24] = 0u;  // 24: scalar_operand_num_tiles

        reader_runtime_args[25] = c_current_shard_width;  // 25: dst_shard_width
        reader_runtime_args[26] = a_num_tiles;            // 26: src_num_tiles (predicate)
    }
}

// Shared work-split so create_descriptor() (cache miss + cache-hit re-apply) partitions the same cores.
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
// slots for every core, which is what lets the hit path copy a fixed-width row per core.
constexpr size_t kNumReaderArgs = 27;
constexpr size_t kNumWriterArgs = 11;
constexpr size_t kNumComputeArgs = 4;

// Per-core runtime args for every core in the partition, in one place so the cache-miss and
// cache-hit paths cannot drift apart.
//
// Args are stored FLAT and row-major -- core i's reader args occupy
// [i * kNumReaderArgs, (i + 1) * kNumReaderArgs).  Each slot keeps its legacy position, and the
// k*ArgNames tables below give the named runtime arg each slot feeds.  The slots that held buffer
// base addresses have no name: the tensor bindings carry those addresses instead.
struct TernaryPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> reader;   // flat, kNumReaderArgs per core; noop cores are zero-filled
    std::vector<uint32_t> writer;   // flat, kNumWriterArgs per core
    std::vector<uint32_t> compute;  // flat, kNumComputeArgs per core

    // Start of core i's row within each flat array.
    const uint32_t* reader_row(size_t i) const { return reader.data() + i * kNumReaderArgs; }
    const uint32_t* writer_row(size_t i) const { return writer.data() + i * kNumWriterArgs; }
    const uint32_t* compute_row(size_t i) const { return compute.data() + i * kNumComputeArgs; }
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
// zero-filled lists sized to match the work-core layout -- and the caller re-applies every named arg
// and every tensor binding, so nothing is left frozen however the partition moves.
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
            // Slots 0-2 used to hold the three operand base addresses. The tensor bindings carry them
            // now, and the slots stay zero.
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized
            reader_runtime_args[3] = num_tiles_per_core;                  // 3: num_tiles (per core)
            reader_runtime_args[4] = c_start_id;                          // 4: start_id
            // Set arguments for both broadcast and no-bcast cases (same arguments needed for width sharding)
            reader_runtime_args[5] = pred_strides.nD_stride;    // 5: nD_stride
            reader_runtime_args[6] = pred_strides.d_stride;     // 6: d_stride
            reader_runtime_args[7] = pred_strides.n_stride;     // 7: n_stride
            reader_runtime_args[8] = pred_strides.c_stride;     // 8: c_stride
            reader_runtime_args[9] = output_dims.D;             // 9: D
            reader_runtime_args[10] = output_dims.N;            // 10: N
            reader_runtime_args[11] = output_dims.C;            // 11: C
            reader_runtime_args[12] = output_dims.Ht;           // 12: Ht
            reader_runtime_args[13] = output_dims.Wt;           // 13: Wt
            reader_runtime_args[14] = output_dims.ND;           // 14: cND
            reader_runtime_args[15] = true_strides.nD_stride;   // 15: true_nD_stride
            reader_runtime_args[16] = true_strides.d_stride;    // 16: true_d_stride
            reader_runtime_args[17] = true_strides.n_stride;    // 17: true_n_stride
            reader_runtime_args[18] = true_strides.c_stride;    // 18: true_c_stride
            reader_runtime_args[19] = b_num_tiles;              // 19: true_num_tiles
            reader_runtime_args[20] = false_strides.nD_stride;  // 20: false_nD_stride
            reader_runtime_args[21] = false_strides.d_stride;   // 21: false_d_stride
            reader_runtime_args[22] = false_strides.n_stride;   // 22: false_n_stride
            reader_runtime_args[23] = false_strides.c_stride;   // 23: false_c_stride
            reader_runtime_args[24] = f_num_tiles;              // 24: false_num_tiles
            reader_runtime_args[25] = c_current_shard_width;    // 25: dst_shard_width
            reader_runtime_args[26] = a_num_tiles;              // 26: src_num_tiles (predicate)

            result.reader.insert(result.reader.end(), reader_runtime_args.begin(), reader_runtime_args.end());
        } else {
            TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
        }

        // Writer runtime args.  Slot 0 used to hold the output base address. The "dst" tensor
        // binding carries it now, and the slot stays zero.  The framework allows the output buffer
        // to alias an input (in-place ops).
        const std::array<uint32_t, num_writer_args> writer_args = {
            0u,                     // 0: dst_addr (now the "dst" tensor binding)
            num_tiles_per_core,     // 1: num_tiles
            c_start_id,             // 2: start_id
            c_current_shard_width,  // 3: dst_shard_width
            output_dims.D,          // 4: D
            output_dims.N,          // 5: N
            output_dims.C,          // 6: C
            output_dims.Ht,         // 7: Ht
            output_dims.Wt,         // 8: Wt
            output_dims.ND,         // 9: cND
            0u,                     // 10: padding
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

// Names of the program's kernels, dataflow buffers (DFBs) and tensor parameters.
//
// The DFB names, and the reader's tensor accessor names, follow the operand slots rather than the
// operand roles.  TTS and TST share their kernel sources, and the second slot holds value_true for
// TTT / TTS but value_false for TST.  Each DFB's accessor name, the dfb::<name> its kernels use, is
// the same string as its spec name.
const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL{"compute"};

const DFBSpecName IN_A_DFB{"in_a"};        // predicate
const DFBSpecName IN_B_DFB{"in_b"};        // value_true (TTT, TTS) or value_false (TST)
const DFBSpecName IN_C_DFB{"in_c"};        // value_false (TTT)
const DFBSpecName BCAST_A_DFB{"bcast_a"};  // LLK row broadcast of in_a
const DFBSpecName BCAST_B_DFB{"bcast_b"};  // LLK row broadcast of in_b
const DFBSpecName BCAST_C_DFB{"bcast_c"};  // LLK row broadcast of in_c
const DFBSpecName OUT_DFB{"out"};

const TensorParamName PREDICATE_TENSOR{"predicate"};
const TensorParamName VALUE_TRUE_TENSOR{"value_true"};
const TensorParamName VALUE_FALSE_TENSOR{"value_false"};
const TensorParamName OUTPUT_TENSOR{"output"};

// Runtime-arg names, indexed by the row slot build_per_core_runtime_args fills.  A slot with no
// name is not passed to the kernel.  Reader slots 0-2 and writer slot 0 held buffer base addresses,
// which the tensor bindings carry now.  Writer slot 10 was padding that no kernel reads.
//
// Every reader source shares one table, the way every reader used to receive the same 27-word row.
// Each source reads only the names it needs.
constexpr std::array<const char*, kNumReaderArgs> kReaderArgNames = {
    nullptr,            // 0: src0_addr -> tensor::src_a
    nullptr,            // 1: src1_addr -> tensor::src_b
    nullptr,            // 2: src2_addr -> tensor::src_c (TTT); always 0 for TTS / TST
    "num_tiles",        // 3
    "start_id",         // 4
    "nD_stride",        // 5
    "d_stride",         // 6
    "n_stride",         // 7
    "c_stride",         // 8
    "D",                // 9
    "N",                // 10
    "C",                // 11
    "Ht",               // 12
    "Wt",               // 13
    "cND",              // 14
    "nD_stride_b",      // 15
    "d_stride_b",       // 16
    "n_stride_b",       // 17
    "c_stride_b",       // 18
    "src_b_num_tiles",  // 19
    "nD_stride_c",      // 20
    "d_stride_c",       // 21
    "n_stride_c",       // 22
    "c_stride_c",       // 23
    "src_c_num_tiles",  // 24
    "dst_shard_width",  // 25
    "src_a_num_tiles",  // 26
};
constexpr std::array<const char*, kNumWriterArgs> kWriterArgNames = {
    nullptr,            // 0: dst_addr -> tensor::dst
    "dst_num_tiles",    // 1
    "start_tile_id",    // 2
    "dst_shard_width",  // 3
    "D",                // 4
    "N",                // 5
    "C",                // 6
    "Ht",               // 7
    "Wt",               // 8
    "cND",              // 9
    nullptr,            // 10: padding
};
constexpr std::array<const char*, kNumComputeArgs> kComputeArgNames = {
    "num_tiles",   // 0
    "tile_freq",   // 1
    "tile_start",  // 2
    "scalar_arg",  // 3
};

// A kernel's runtime-arg schema: the named slots of its table, in slot order.
template <size_t N>
Group<std::string> runtime_arg_names(const std::array<const char*, N>& names) {
    Group<std::string> result;
    for (const char* name : names) {
        if (name != nullptr) {
            result.emplace_back(name);
        }
    }
    return result;
}

// Builds the complete ProgramRunArgs: every named runtime arg of the three kernels on every core of
// the partition, and a TensorArgument for every TensorParameter.  create_program_artifacts() (cache
// miss) and override_runtime_arguments() (cache hit) both return it.  That keeps the two paths
// identical, the way both legacy paths applied the same builder rows.
ProgramRunArgs build_program_run_args(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    TernaryDeviceOperation::tensor_return_value_t& output,
    const TernaryPerCoreArgs& per_core) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    const TernaryVariant variant = operation_attributes.ternary_variant;

    ProgramRunArgs run_args;

    // Every TensorParameter gets its current tensor.  These arguments replace the buffer addresses
    // the legacy rows carried, and the re-pointing of the tensor-backed buffers: a DFB borrowed
    // from a tensor takes its address from that tensor's argument.
    run_args.tensor_args.emplace(PREDICATE_TENSOR, predicate_tensor.mesh_tensor());
    if (variant == TernaryVariant::TTT || variant == TernaryVariant::TTS) {
        run_args.tensor_args.emplace(VALUE_TRUE_TENSOR, value_true_tensor.value().mesh_tensor());
    }
    if (variant == TernaryVariant::TTT || variant == TernaryVariant::TST) {
        run_args.tensor_args.emplace(VALUE_FALSE_TENSOR, value_false_tensor.value().mesh_tensor());
    }
    run_args.tensor_args.emplace(OUTPUT_TENSOR, output.mesh_tensor());

    auto add_kernel_run_args =
        [&]<size_t N>(
            const KernelSpecName& kernel, const std::array<const char*, N>& names, const std::vector<uint32_t>& rows) {
            KernelRunArgs kernel_run_args{.kernel = kernel};
            for (size_t slot = 0; slot < N; ++slot) {
                if (names[slot] == nullptr) {
                    continue;
                }
                auto& values = kernel_run_args.runtime_arg_values[names[slot]];
                for (size_t i = 0; i < per_core.cores.size(); ++i) {
                    values[per_core.cores[i]] = rows[(i * N) + slot];
                }
            }
            run_args.kernel_run_args.push_back(std::move(kernel_run_args));
        };
    add_kernel_run_args(READER_KERNEL, kReaderArgNames, per_core.reader);
    add_kernel_run_args(WRITER_KERNEL, kWriterArgNames, per_core.writer);
    add_kernel_run_args(COMPUTE_KERNEL, kComputeArgNames, per_core.compute);

    return run_args;
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

    ProgramSpec spec{.name = "ternary"};

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

    // Each TensorParameter accepts a tensor whose logical shape and rank differ from the ones the
    // program was built for.  The program cache needs this.  compute_program_hash hashes only the
    // logical height and width of each input, besides its dtype, memory config, alignment and tile.
    // One cached program therefore serves inputs that differ in their outer dims or in their rank,
    // for example [4,1,32,32] and [1,4,32,32], or [1,32,64] and [1,1,32,64].  The kernels tolerate
    // it.  They read every tensor tile by tile through its accessor, and every shape-dependent kernel
    // arg is a runtime arg that the builder re-derives on every dispatch.
    const TensorSpecRelaxations relaxations{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = PREDICATE_TENSOR,
        .spec = predicate_tensor.tensor_spec(),
        .relaxations = relaxations,
    });
    if (variant == TernaryVariant::TTT || variant == TernaryVariant::TTS) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = VALUE_TRUE_TENSOR,
            .spec = value_true_tensor.value().tensor_spec(),
            .relaxations = relaxations,
        });
    }
    if (variant == TernaryVariant::TTT || variant == TernaryVariant::TST) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = VALUE_FALSE_TENSOR,
            .spec = value_false_tensor.value().tensor_spec(),
            .relaxations = relaxations,
        });
    }
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output.tensor_spec(),
        .relaxations = relaxations,
    });

    // ---- Dataflow Buffers ----
    // A DFB on a natively sharded tensor is borrowed from that tensor: it uses the tensor's L1
    // shard as its storage, and follows the tensor's address on every dispatch.

    // Input buffers - Create predicate DFB (always in_a)
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_A_DFB,
        .entry_size = predicate_single_tile_size,
        .num_entries = predicate_num_tiles_per_shard.value_or(num_tiles_per_dfb),
        .data_format_metadata = predicate_data_format,
        .borrowed_from = predicate_sharded ? std::optional{PREDICATE_TENSOR} : std::nullopt,
    });

    // Create in_b based on variant - this is the primary tensor DFB
    if (variant == TernaryVariant::TTS) {
        // TTS: in_b = value_true tensor (value_false is scalar)
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_B_DFB,
            .entry_size = value_true_single_tile_size,
            .num_entries = true_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_true_data_format,
            .borrowed_from = true_sharded ? std::optional{VALUE_TRUE_TENSOR} : std::nullopt,
        });
    } else if (variant == TernaryVariant::TST) {
        // TST: in_b = value_false tensor (value_true is scalar)
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_B_DFB,
            .entry_size = value_false_single_tile_size,
            .num_entries = false_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_false_data_format,
            .borrowed_from = false_sharded ? std::optional{VALUE_FALSE_TENSOR} : std::nullopt,
        });
    } else if (variant == TernaryVariant::TTT) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_B_DFB,
            .entry_size = value_true_single_tile_size,
            .num_entries = true_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_true_data_format,
            .borrowed_from = true_sharded ? std::optional{VALUE_TRUE_TENSOR} : std::nullopt,
        });

        // Create DFB for value_false (using actual false tensor)
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_C_DFB,
            .entry_size = value_false_single_tile_size,
            .num_entries = false_num_tiles_per_shard.value_or(num_tiles_per_dfb),
            .data_format_metadata = value_false_data_format,
            .borrowed_from = false_sharded ? std::optional{VALUE_FALSE_TENSOR} : std::nullopt,
        });
    } else {
        TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
    }

    // Output buffer - use out for all cases now
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
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

    // The LLK row-broadcast compute kernels stage each row-broadcast operand in a DFB of its own.
    // The kernel broadcasts the operand's first row into that DFB, then reads the tile back from it.
    // A staging DFB exists only when one of those kernels runs and that operand is row-broadcast
    // (BCAST_A / BCAST_B / BCAST_C, which the kernels test before they touch the DFB).
    Group<std::pair<DFBSpecName, std::string>> llk_bcast_dfbs;  // {spec name, accessor name}
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

            auto add_llk_bcast_dfb = [&](const DFBSpecName& name, uint32_t tile_size, DataFormat data_format) {
                spec.dataflow_buffers.push_back(DataflowBufferSpec{
                    .unique_id = name,
                    .entry_size = tile_size,
                    .num_entries = num_tiles_per_dfb,
                    .data_format_metadata = data_format,
                });
                llk_bcast_dfbs.emplace_back(name, name.get());
            };
            if (pred_is_bcast) {
                add_llk_bcast_dfb(BCAST_A_DFB, predicate_single_tile_size, predicate_data_format);
            }
            if (true_is_bcast) {
                add_llk_bcast_dfb(BCAST_B_DFB, value_true_single_tile_size, value_true_data_format);
            }
            if (false_is_bcast) {
                add_llk_bcast_dfb(BCAST_C_DFB, value_false_single_tile_size, value_false_data_format);
            }
        } else {
            reader_defines["BCAST_LLK"] = "0";
        }
    }

    // ---- Reader Kernel ----
    // The reader produces in_a / in_b (/ in_c) and reads the operand tensors through src_a / src_b
    // (/ src_c).  A reader reads a natively sharded operand without its accessor (SRC_SHARDED_*),
    // but every tensor is bound anyway, the way every accessor's args used to be passed.
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN_A_DFB,
            .accessor_name = "in_a",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = IN_B_DFB,
            .accessor_name = "in_b",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{
            .tensor_parameter_name = PREDICATE_TENSOR,
            .accessor_name = "src_a",
        },
    };

    if (variant == TernaryVariant::TTS) {
        // TTS: in_a = predicate, in_b = value_true tensor
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = VALUE_TRUE_TENSOR,
            .accessor_name = "src_b",
        });
    } else if (variant == TernaryVariant::TST) {
        // TST: in_a = predicate, in_b = value_false tensor
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = VALUE_FALSE_TENSOR,
            .accessor_name = "src_b",
        });
    } else if (variant == TernaryVariant::TTT) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = IN_C_DFB,
            .accessor_name = "in_c",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = VALUE_TRUE_TENSOR,
            .accessor_name = "src_b",
        });
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = VALUE_FALSE_TENSOR,
            .accessor_name = "src_c",
        });
    }

    bool is_fpu = false;

    if (operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL ||
        operation_attributes.ternary_op_type == TernaryOpType::ADDCDIV) {
        is_fpu = (predicate_tensor.dtype() == value_true_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() == value_false_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() != DataType::FLOAT32 && predicate_tensor.dtype() != DataType::INT32 &&
                  predicate_tensor.dtype() != DataType::UINT32);
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = get_kernel_file_path(reader_kernel, is_fpu),
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"has_sharding", static_cast<uint32_t>(has_sharding)}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names(kReaderArgNames)},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // ---- Writer Kernel ----
    // The writer drains out into the output tensor through dst.  A natively sharded output
    // (DST_SHARDED) needs no writer: the compute kernel packs straight into the output's shard, and
    // the writer body is compiled out.  The writer then does not bind out.
    const bool dst_sharded = output_sharded && has_sharding;
    std::map<std::string, std::string> writer_defines;
    writer_defines["DST_SHARDED"] = dst_sharded ? "1" : "0";

    Group<DFBBinding> writer_dfb_bindings;
    if (!dst_sharded) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = get_kernel_file_path(writer_kernel, is_fpu),
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_TENSOR,
                    .accessor_name = "dst",
                },
            },
        .compile_time_args = {{"has_sharding", static_cast<uint32_t>(has_sharding)}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names(kWriterArgNames)},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // ---- Compute Kernel ----
    // Any 32-bit operand needs a 32-bit Dest: a 32-bit input DFB is unpacked with UnpackToDest below,
    // and that lands corrupted in a Dest configured for 16-bit datums.
    auto is_32bit_format = [](tt::DataFormat format) {
        return format == tt::DataFormat::UInt32 || format == tt::DataFormat::Int32 || format == tt::DataFormat::Float32;
    };
    bool fp32_dest_acc_en = is_32bit_format(output_data_format) || is_32bit_format(predicate_data_format) ||
                            is_32bit_format(value_true_data_format) || is_32bit_format(value_false_data_format);

    // A 32-bit DFB (Float32, Int32 or UInt32) unpacks straight to Dest (UnpackToDest).  Every other DFB
    // has no entry and takes the default path through SrcA / SrcB.
    auto is_32bit_dtype = [](DataType dtype) {
        return dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32;
    };
    ComputeHardwareConfig::ComputeUnpackModes unpack_modes;

    // in_a is always predicate
    if (is_32bit_dtype(predicate_tensor.dtype())) {
        unpack_modes.emplace(IN_A_DFB, UnpackMode::UnpackToDest);
    }

    // in_b assignment depends on variant
    if (variant == TernaryVariant::TTS) {
        // TTS: in_b = value_true tensor
        if (is_32bit_dtype(value_true_tensor.value().dtype())) {
            unpack_modes.emplace(IN_B_DFB, UnpackMode::UnpackToDest);
        }
    } else if (variant == TernaryVariant::TST) {
        // TST: in_b = value_false tensor
        if (is_32bit_dtype(value_false_tensor.value().dtype())) {
            unpack_modes.emplace(IN_B_DFB, UnpackMode::UnpackToDest);
        }
    } else {
        // TTT: in_b = value_true tensor, in_c = value_false tensor (including column broadcast)
        if (is_32bit_dtype(value_true_tensor.value().dtype())) {
            unpack_modes.emplace(IN_B_DFB, UnpackMode::UnpackToDest);
        }
        if (is_32bit_dtype(value_false_tensor.value().dtype())) {
            unpack_modes.emplace(IN_C_DFB, UnpackMode::UnpackToDest);
        }
    }

    // Output DFB
    if (is_32bit_dtype(output.dtype())) {
        unpack_modes.emplace(OUT_DFB, UnpackMode::UnpackToDest);
    }

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

    // The compute kernel consumes the operand DFBs and produces out.  With a natively sharded output
    // nothing drains out: the packer writes straight into the output's shard.  The compute kernel is
    // then the only kernel that touches out, and it binds out as both producer and consumer.  It
    // binds each staging DFB of the LLK row broadcast the same way, since it fills and reads both.
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN_A_DFB,
            .accessor_name = "in_a",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = IN_B_DFB,
            .accessor_name = "in_b",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    if (variant == TernaryVariant::TTT) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = IN_C_DFB,
            .accessor_name = "in_c",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = OUT_DFB,
        .accessor_name = "out",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (dst_sharded) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    for (const auto& [name, accessor_name] : llk_bcast_dfbs) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = accessor_name,
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = accessor_name,
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = compute_kernel_path,
        // A compute kernel built from a ComputeConfigDescriptor defaulted to O3.  A KernelSpec defaults
        // to O2.  The legacy level is therefore set here.
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
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names(kComputeArgNames)},
        // The legacy config set only these two fields.  Every other field keeps its default, and each
        // default equals the legacy default.
        .hw_config =
            ComputeHardwareConfig{
                .enable_32_bit_dest = fp32_dest_acc_en,
                .unpack_modes = std::move(unpack_modes),
            },
    };

    spec.kernels.push_back(std::move(reader_spec));
    spec.kernels.push_back(std::move(writer_spec));
    spec.kernels.push_back(std::move(compute_spec));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "ternary",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_device_cores,
    });

    // ---- Per-core runtime args and tensor arguments ----
    // Built via the shared single-source-of-truth builder so create_program_artifacts() (cache miss,
    // here) and override_runtime_arguments() (cache hit) stay byte-identical.
    auto per_core = build_per_core_runtime_args(operation_attributes, tensor_args, output, broadcast_type);
    auto run_args = build_program_run_args(operation_attributes, tensor_args, output, per_core);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ProgramRunArgs TernaryDeviceOperation::TernaryProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-apply ALL per-dispatch state to the cached program on a program-cache hit.  No program
    // rebuild and no kernel recompile -- but every per-core arg IS re-derived. We re-derive them from the SAME
    // shared builder create_program_artifacts() uses, so the two paths stay byte-identical by construction.
    //
    // On this factory concept the framework refreshes nothing by itself: it applies exactly the
    // ProgramRunArgs returned here.  They carry every named runtime arg on every core, and a
    // TensorArgument for every TensorParameter.  A DFB borrowed from a tensor follows that tensor's
    // argument to the tensor's current shard.  Each borrowed DFB names its own TensorParameter.  An
    // output that aliases an input (an in-place call) therefore still lands on the output's DFB.
    //
    // The legacy override left the tensor accessors' args alone.  The framework now manages those
    // args, and refreshes them from the tensor arguments.
    using namespace CMAKE_UNIQUE_NAMESPACE;

    auto per_core =
        build_per_core_runtime_args(operation_attributes, tensor_args, output, operation_attributes.broadcast_type);

    // UpdateProgramRunArgs keeps the old value of any arg it is not given.  The cached kernels run on
    // every core of the worker grid, and a core can switch between work and noop across hits.  The
    // builder must therefore give every one of those cores a full set of args.
    TT_FATAL(
        per_core.cores.size() == operation_attributes.worker_grid.num_cores(),
        "ternary override: the builder produced runtime args for {} cores, but the cached kernels run on {} cores",
        per_core.cores.size(),
        operation_attributes.worker_grid.num_cores());

    return build_program_run_args(operation_attributes, tensor_args, output, per_core);
}

}  // namespace ttnn::operations::ternary
