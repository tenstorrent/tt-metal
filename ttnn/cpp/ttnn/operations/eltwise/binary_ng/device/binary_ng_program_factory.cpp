// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/circular_buffer.hpp>

#include <algorithm>
#include <variant>
#include <vector>
using namespace tt::tt_metal;
using ttnn::Tensor;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::binary_ng;

// For rank > 5 dims will be collapsed into a single dim
uint32_t extract_nD_dims(const Tensor& x, const int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        for (int i = -6; i >= -out_rank; --i) {
            auto dim = shape[i];
            nD_dim *= dim;
        }
    }
    return nD_dim;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {
        shape.rank() >= 5 ? shape[-5] : 1,
        shape[-4],
        shape[-3],
        tt::div_up(shape[-2], tile.get_height()),
        tt::div_up(shape[-1], tile.get_width())};
}

std::tuple<uint32_t, uint32_t> calculate_compute_kernel_args(
    SubtileBroadcastType broadcast_type, uint32_t start_tile_id, uint32_t Ht, uint32_t Wt) {
    uint32_t start_t = start_tile_id % (Ht * Wt);
    uint32_t start_tw = start_t % Wt;

    switch (broadcast_type) {
        case SubtileBroadcastType::NONE:
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B: return {1, 0};
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::SCALAR_B: return {Ht * Wt, start_t};
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::ROW_B_COL_A:
        case SubtileBroadcastType::COL_B:
        case SubtileBroadcastType::ROW_A_COL_B: return {Wt, start_tw};
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

TensorMemoryLayout get_memory_layout(const Tensor& a, const std::optional<Tensor>& b, const Tensor& c) {
    if (!b.has_value()) {
        return TensorMemoryLayout::INTERLEAVED;
    }
    // c is first preferred
    if (c.memory_config().is_sharded()) {
        return c.memory_config().memory_layout();
    }

    if (a.memory_config().is_sharded()) {
        return a.memory_config().memory_layout();
    }
    if (b.has_value() && b->memory_config().is_sharded()) {
        return b->memory_config().memory_layout();
    }

    return TensorMemoryLayout::INTERLEAVED;
}

std::optional<AllShardSpecs> get_shard_specs(
    const tt::tt_metal::TensorSpec& a,
    const std::optional<tt::tt_metal::TensorSpec>& b,
    const tt::tt_metal::TensorSpec& c) {
    bool a_sharded = a.memory_config().is_sharded();
    bool b_sharded = b.has_value() && b->memory_config().is_sharded();
    bool c_sharded = c.memory_config().is_sharded();

    if ((!a_sharded && !b_sharded) && !c_sharded) {
        return std::nullopt;
    }

    if (!is_native_L1_sharding(a, b, c.memory_config())) {
        return std::nullopt;
    }

    // If the output is unevenly sharded, only allow when all tensors share the same shard spec
    // (each core sees identical tile counts for a, b, c -- no deadlock risk).
    if (is_uneven(c)) {
        bool all_specs_match =
            b.has_value() && a_sharded && b_sharded && c_sharded && a.memory_config().shard_spec().has_value() &&
            b->memory_config().shard_spec().has_value() && c.memory_config().shard_spec().has_value() &&
            *a.memory_config().shard_spec() == *b->memory_config().shard_spec() &&
            *a.memory_config().shard_spec() == *c.memory_config().shard_spec();
        if (!all_specs_match) {
            return std::nullopt;
        }
    }

    const auto& a_shape = a.padded_shape();
    auto b_shape = b.has_value() ? b->padded_shape() : ttnn::Shape{1, 1};
    const auto& c_shape = c.padded_shape();

    TT_FATAL(get_shard_spec(c).has_value(), "C must have a shard spec");
    return AllShardSpecs{
        a_sharded ? *get_shard_spec(a) : adjust_to_shape(*get_shard_spec(c), c_shape, a_shape),
        b_sharded ? *get_shard_spec(*b) : adjust_to_shape(*get_shard_spec(c), c_shape, b_shape),
        *get_shard_spec(c)};
}

bool should_use_row_major_path(
    const BinaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const std::optional<Tensor>& b,
    const bool has_sharding) {
    if (operation_attributes.input_layout_a != Layout::ROW_MAJOR ||
        operation_attributes.output_layout != Layout::ROW_MAJOR || has_sharding) {
        return false;
    }
    if (b.has_value()) {
        return operation_attributes.input_layout_b == Layout::ROW_MAJOR;
    }
    return true;
}

uint32_t get_shards_per_width(const ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
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

class ShardShapeGenerator {
    CoreCoord end_core;
    bool row_major{};
    TensorMemoryLayout memory_layout{TensorMemoryLayout::INTERLEAVED};
    std::array<uint32_t, 2> shard_shape{};
    std::array<uint32_t, 2> last_shard_shape{};

public:
    ShardShapeGenerator() = default;

    ShardShapeGenerator(const ShardSpec& shard_spec, const Tensor& tensor) :
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

KernelName get_reader_kernel_name_and_defines(
    const SubtileBroadcastType subtile_broadcast_type, std::map<std::string, std::string>& reader_defines) {
    if (subtile_broadcast_type == SubtileBroadcastType::NONE) {
        return KernelName::ReaderNoBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::ROW_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B ? "1" : "0";
        return KernelName::ReaderRowBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::COL_B ? "1" : "0";
        return KernelName::ReaderColBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B) {
        reader_defines["SRC_BCAST_COL"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_ROW_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        return KernelName::ReaderRowBColABcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ? "1" : "0";
        return KernelName::ReaderScalarBcastNg;
    }
    TT_FATAL(false, "Unsupported subtile broadcast type {}", static_cast<int>(subtile_broadcast_type));
}

KernelName get_reader_rm_kernel_name_and_defines(
    const SubtileBroadcastType subtile_broadcast_type,
    const bool has_rhs_tensor,
    std::map<std::string, std::string>& reader_defines) {
    if (!has_rhs_tensor) {
        return KernelName::ReaderRmScalarOpNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::NONE) {
        return KernelName::ReaderRmNoBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::ROW_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B ? "1" : "0";
        return KernelName::ReaderRmRowBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::COL_B ? "1" : "0";
        return KernelName::ReaderRmColBcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B) {
        reader_defines["SRC_BCAST_ROW_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        return KernelName::ReaderRmRowBColABcastNg;
    }
    if (subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ? "1" : "0";
        return KernelName::ReaderRmScalarBcastNg;
    }
    TT_FATAL(false, "Unsupported row-major subtile broadcast type {}", static_cast<int>(subtile_broadcast_type));
}

void overwrite_compute_kernel_name_and_defines(
    KernelName& kernel_name,
    const SubtileBroadcastType subtile_broadcast_type,
    std::map<std::string, std::string>& compute_defines,
    bool is_where_op) {
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B) {
        compute_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::ROW_A ? "1" : "0";
        compute_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B ? "1" : "0";
        kernel_name = KernelName::ComputeRowBcastNg;
    } else if (
        subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A) {
        kernel_name = KernelName::ComputeRowColBcastNg;
    } else if (
        not is_where_op && (subtile_broadcast_type == SubtileBroadcastType::COL_A ||
                            subtile_broadcast_type == SubtileBroadcastType::COL_B)) {
        compute_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::COL_A ? "1" : "0";
        compute_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::COL_B ? "1" : "0";
        kernel_name = KernelName::ComputeColBcastNg;
    } else if (
        not is_where_op && (subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
                            subtile_broadcast_type == SubtileBroadcastType::SCALAR_B)) {
        compute_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ? "1" : "0";
        compute_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ? "1" : "0";
        kernel_name = KernelName::ComputeScalarBcastNg;
    }
}

// Returns true on the (arch, broadcast, dtype) tuple that hangs the LLK
// `unary_bcast` path on Blackhole: COL bcast + BFLOAT16 input + fp32_dest_acc_en.
bool hits_bh_col_bcast_bf16_to_fp32_hang(
    SubtileBroadcastType subtile_broadcast_type, DataType a_dtype, DataType b_dtype, bool fp32_dest_acc_en) {
    const bool is_col_bcast =
        subtile_broadcast_type == SubtileBroadcastType::COL_A || subtile_broadcast_type == SubtileBroadcastType::COL_B;
    const bool has_bf16_input = a_dtype == DataType::BFLOAT16 || b_dtype == DataType::BFLOAT16;
    return tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE && is_col_bcast && fp32_dest_acc_en && has_bf16_input;
}

bool is_llk_bcast(
    const SubtileBroadcastType subtile_broadcast_type,
    const DataType a_dtype,
    const DataType b_dtype,
    const bool fp32_dest_acc_en) {
    if (hits_bh_col_bcast_bf16_to_fp32_hang(subtile_broadcast_type, a_dtype, b_dtype, fp32_dest_acc_en)) {
        return false;
    }

    auto all_match = [&](DataType dt) { return a_dtype == dt && b_dtype == dt; };

    if (subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_B ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        if (all_match(DataType::BFLOAT16) || all_match(DataType::BFLOAT8_B) || all_match(DataType::BFLOAT4_B)) {
            return true;
        }
        if (all_match(DataType::FLOAT32) || all_match(DataType::INT32) || all_match(DataType::UINT32) ||
            all_match(DataType::UINT16)) {
            tt::ARCH arch = tt::tt_metal::hal::get_arch();
            if (arch == tt::ARCH::WORMHOLE_B0) {
                return true;
            }
        }
    }

    return false;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::binary_ng {

std::optional<AllShardVolumes> get_shard_volumes(
    const tt::tt_metal::TensorSpec& a,
    const std::optional<tt::tt_metal::TensorSpec>& b,
    const tt::tt_metal::TensorSpec& c) {
    const auto shard_specs = CMAKE_UNIQUE_NAMESPACE::get_shard_specs(a, b, c);

    if (not shard_specs.has_value()) {
        return std::nullopt;
    }

    const auto a_sharded = a.memory_config().is_sharded();
    const auto b_sharded = b.has_value() and b->memory_config().is_sharded();
    const auto c_sharded = c.memory_config().is_sharded();
    const auto tile_hw = c.tile().get_tile_hw();

    return AllShardVolumes{
        .a_shard_volume = a_sharded ? shard_specs->a_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .b_shard_volume = b_sharded ? shard_specs->b_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .c_shard_volume = c_sharded ? shard_specs->c_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
    };
}

namespace {

// Per-core runtime-arg lists for every core in the worker grid (work AND noop cores), in the same
// order create_descriptor() populates them.  Each arg list is a vector of variants: Buffer* entries
// are buffer base addresses (bindings), uint32_t entries are plain values.
struct BinaryNgPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<std::vector<std::variant<uint32_t, Buffer*>>> reader;
    std::vector<std::vector<std::variant<uint32_t, Buffer*>>> writer;
    std::vector<std::vector<std::variant<uint32_t, Buffer*>>> compute;
};

// SINGLE SOURCE OF TRUTH for binary_ng per-core runtime args.  Run by BOTH create_descriptor()
// (cache miss) and BinaryNgDeviceOperation::override_runtime_arguments() (cache hit).
//
// binary_ng's compute_program_hash intentionally EXCLUDES the tensor volume, so one cached program
// is shared across differently-shaped calls.  On a cache hit the descriptor is NOT rebuilt, so every
// shape/work-split-dependent per-core arg (c_start_id, per-core tile counts, strides, D/N/C/Ht/Wt,
// compute_tiles, freq/counter, packed scalar, ...) would otherwise stay frozen at the first-miss
// shape and corrupt results.  override_runtime_arguments() re-runs THIS builder for the current
// tensors and re-applies each arg every dispatch.  Because the work-core set itself changes with
// volume (a core can flip between work and noop across hits), the builder emits args for ALL
// num_cores_total cores -- noop cores get zero-filled lists sized to match the work-core layout -- and
// override_runtime_arguments re-applies every slot (buffer slots via their current address), so nothing
// is left frozen regardless of how the partition shifts.
BinaryNgPerCoreArgs build_per_core_runtime_args(
    const BinaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const Tensor& a,
    const std::optional<Tensor>& b,
    const Tensor& c) {
    using namespace tt;
    using namespace tt::tt_metal;

    BinaryNgPerCoreArgs result;

    const auto& all_device_cores = operation_attributes.worker_grid;
    const auto op_type = operation_attributes.binary_op_type;

    const auto out_rank = c.logical_shape().rank();
    auto aND = CMAKE_UNIQUE_NAMESPACE::extract_nD_dims(a, out_rank);
    auto bND = b.has_value() ? CMAKE_UNIQUE_NAMESPACE::extract_nD_dims(*b, out_rank) : 1;
    auto cND = CMAKE_UNIQUE_NAMESPACE::extract_nD_dims(c, out_rank);
    const auto aHt_r = a.padded_shape()[-2];
    const auto aWt_r = a.padded_shape()[-1];
    const auto bHt_r = b.has_value() ? b->padded_shape()[-2] : 0;
    const auto bWt_r = b.has_value() ? b->padded_shape()[-1] : 0;
    const auto cHt_r = c.padded_shape()[-2];
    const auto cWt_r = c.padded_shape()[-1];

    const auto [aD, aN, aC, aHt, aWt] = CMAKE_UNIQUE_NAMESPACE::get_shape_dims(a);
    const auto [bD, bN, bC, bHt, bWt] =
        b.has_value() ? CMAKE_UNIQUE_NAMESPACE::get_shape_dims(*b) : std::tuple{1u, 1u, 1u, 1u, 1u};
    const auto [cD, cN, cC, cHt, cWt] = CMAKE_UNIQUE_NAMESPACE::get_shape_dims(c);

    const auto shard_specs = CMAKE_UNIQUE_NAMESPACE::get_shard_specs(
        a.tensor_spec(), b.has_value() ? b->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{}, c.tensor_spec());
    const bool rt_has_sharding = shard_specs.has_value();
    auto grid = rt_has_sharding ? shard_specs->a_shard_spec.grid : CoreRangeSet{};

    const auto row_major =
        rt_has_sharding ? shard_specs->a_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;

    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    if (grid.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (rt_has_sharding) {
                const auto& shard_start_coord = grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }
    const uint32_t num_cores_total =
        zero_start_grid ? compute_with_storage_grid.x * compute_with_storage_grid.y : all_device_cores.num_cores();

    uint32_t num_tiles_per_core_group_1{}, num_tiles_per_core_group_2{};
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores;
    std::vector<CoreCoord> cores;

    const bool row_major_inputs =
        CMAKE_UNIQUE_NAMESPACE::should_use_row_major_path(operation_attributes, b, rt_has_sharding);
    const uint32_t a_alignment = a.buffer()->alignment();
    const uint32_t b_alignment = b.has_value() ? b->buffer()->alignment() : a_alignment;
    const uint32_t c_alignment = c.buffer()->alignment();

    const uint32_t tile_height = c.tensor_spec().tile().get_height();
    const uint32_t tile_width = c.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;

    uint32_t rt_c_num_tiles;
    uint32_t num_rows_per_tile = 0;
    uint32_t row_blocks_per_channel = 1;
    uint32_t tiles_per_row_width = 1;
    uint32_t common_row_width_elements = 0;
    uint32_t reader_stride_size_bytes = 0;
    uint32_t writer_stride_size_bytes = 0;

    if (row_major_inputs) {
        const uint32_t c_aligned_page_size = c.buffer()->aligned_page_size();
        const uint32_t a_aligned_page_size = a.buffer()->aligned_page_size();
        const uint32_t b_aligned_page_size = b.has_value() ? b->buffer()->aligned_page_size() : a_aligned_page_size;

        const uint32_t c_row_width_elements_aligned = c_aligned_page_size / c.element_size();
        const uint32_t a_row_width_elements_aligned = a_aligned_page_size / a.element_size();
        const uint32_t b_row_width_elements_aligned =
            b.has_value() ? (b_aligned_page_size / b->element_size()) : a_row_width_elements_aligned;

        common_row_width_elements = c_row_width_elements_aligned;
        if (aWt_r == cWt_r) {
            common_row_width_elements = std::min(common_row_width_elements, a_row_width_elements_aligned);
        }
        if (b.has_value() && bWt_r == cWt_r) {
            common_row_width_elements = std::min(common_row_width_elements, b_row_width_elements_aligned);
        }
        common_row_width_elements = std::max<uint32_t>(1u, common_row_width_elements);

        num_rows_per_tile = std::max<uint32_t>(1u, tile_hw / common_row_width_elements);
        const bool aligned_for_a =
            (aWt_r == cWt_r) ? ((common_row_width_elements * a.element_size()) == a_aligned_page_size) : true;
        const bool aligned_for_b = (b.has_value() && bWt_r == cWt_r)
                                       ? ((common_row_width_elements * b->element_size()) == b_aligned_page_size)
                                       : true;
        const bool aligned_for_c = (common_row_width_elements * c.element_size()) == c_aligned_page_size;
        if (!aligned_for_a || !aligned_for_b || !aligned_for_c) {
            num_rows_per_tile = 1;
        }

        row_blocks_per_channel = tt::div_up(cHt_r, num_rows_per_tile);
        const uint32_t total_row_blocks = cND * cD * cN * cC * row_blocks_per_channel;
        tiles_per_row_width = tt::div_up(common_row_width_elements, tile_hw);
        const uint32_t a_tile_bytes = tile_hw * a.element_size();
        const uint32_t a_row_width_bytes = common_row_width_elements * a.element_size();
        reader_stride_size_bytes =
            (a_row_width_bytes > a_tile_bytes) ? a_tile_bytes : tt::round_up(a_row_width_bytes, a_alignment);
        const uint32_t c_tile_bytes = tile_hw * c.element_size();
        const uint32_t c_row_width_bytes = common_row_width_elements * c.element_size();
        writer_stride_size_bytes =
            (c_row_width_bytes > c_tile_bytes) ? c_tile_bytes : tt::round_up(c_row_width_bytes, c_alignment);
        rt_c_num_tiles = total_row_blocks;
    } else {
        rt_c_num_tiles = c.physical_volume() / tile_hw;
    }

    uint32_t c_shard_height{}, c_shard_width{}, num_shards_per_width{};

    CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator a_shard_shape_generator;
    CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator b_shard_shape_generator;
    CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator c_shard_shape_generator;

    bool all_same_shard_spec = rt_has_sharding && a.memory_config().is_sharded() && b.has_value() &&
                               b->memory_config().is_sharded() && c.memory_config().is_sharded() &&
                               shard_specs->a_shard_spec == shard_specs->b_shard_spec &&
                               shard_specs->a_shard_spec == shard_specs->c_shard_spec;

    if (rt_has_sharding) {
        core_group_1 = grid;
        a_shard_shape_generator = CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator(shard_specs->a_shard_spec, a);
        if (b.has_value()) {
            b_shard_shape_generator = CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator(shard_specs->b_shard_spec, *b);
        }
        c_shard_shape_generator = CMAKE_UNIQUE_NAMESPACE::ShardShapeGenerator(shard_specs->c_shard_spec, c);
        c_shard_height = shard_specs->c_shard_spec.shape[0] / tile_height;
        c_shard_width = shard_specs->c_shard_spec.shape[1] / tile_width;
        num_shards_per_width = CMAKE_UNIQUE_NAMESPACE::get_shards_per_width(
            shard_specs->c_shard_spec, CMAKE_UNIQUE_NAMESPACE::get_memory_layout(a, b, c));

        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid.x,
                compute_with_storage_grid.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(core_group_1, all_device_cores, row_major);
        }
    } else if (zero_start_grid) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid, rt_c_num_tiles, row_major);
        cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, rt_c_num_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }

    result.cores.reserve(num_cores_total);
    result.reader.reserve(num_cores_total);
    result.writer.reserve(num_cores_total);
    result.compute.reserve(num_cores_total);

    uint32_t current_block = 0;
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        std::vector<std::variant<uint32_t, Buffer*>> reader_runtime_args;
        std::vector<std::variant<uint32_t, Buffer*>> writer_runtime_args;
        std::vector<std::variant<uint32_t, Buffer*>> compute_runtime_args;

        uint32_t a_num_tiles = 0;
        uint32_t b_num_tiles = 0;
        uint32_t c_num_tiles_core = 0;
        if (core_group_1.contains(core)) {
            c_num_tiles_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            c_num_tiles_core = num_tiles_per_core_group_2;
        } else {
            // Noop core: zero-filled runtime args, sized to match the active kernel variant so unused
            // cores neither inflate the per-kernel max runtime-arg allocation nor change slot count when a
            // core flips between noop and work across differently-shaped cache hits.
            const size_t reader_len = row_major_inputs ? 26 : 21;
            const size_t writer_len = row_major_inputs ? 14 : (b.has_value() ? 11 : 12);
            const size_t compute_len = (op_type == BinaryOpType::ISCLOSE) ? 5 : 4;
            reader_runtime_args.assign(reader_len, std::variant<uint32_t, Buffer*>{uint32_t{0}});
            writer_runtime_args.assign(writer_len, std::variant<uint32_t, Buffer*>{uint32_t{0}});
            compute_runtime_args.assign(compute_len, std::variant<uint32_t, Buffer*>{uint32_t{0}});
            result.cores.push_back(core);
            result.reader.push_back(std::move(reader_runtime_args));
            result.writer.push_back(std::move(writer_runtime_args));
            result.compute.push_back(std::move(compute_runtime_args));
            continue;
        }

        uint32_t c_start_id = 0;
        uint32_t c_current_shard_width = 0;
        if (rt_has_sharding) {
            if (all_same_shard_spec) {
                c_num_tiles_core = c_shard_height * c_shard_width;
                c_current_shard_width = c_shard_width;
                a_num_tiles = c_shard_height * c_shard_width;
            } else {
                auto c_shard_shape = c_shard_shape_generator(core);
                c_num_tiles_core = c_shard_shape[0] * c_shard_shape[1];
                c_current_shard_width = c_shard_shape[1];
                auto a_shard_shape = a_shard_shape_generator(core);
                a_num_tiles = a_shard_shape[0] * a_shard_shape[1];
            }
            c_start_id =
                (i / num_shards_per_width) * (c_shard_height * cWt) + (i % num_shards_per_width) * c_shard_width;
        } else {
            c_start_id = start_tile_id;
        }

        const bool rt_is_quant_op = operation_attributes.is_quant_op;
        TT_FATAL(
            rt_is_quant_op ==
                ((operation_attributes.post_activations.size() == 1) &&
                 (operation_attributes.post_activations[0].type() == ttnn::operations::unary::UnaryOpType::ZERO_POINT)),
            "Quantization op needs to exactly one zero-point value as a post activation");
        const uint32_t quantization_zero_point =
            rt_is_quant_op ? std::bit_cast<uint32_t>(
                                 operation_attributes.post_activations[0].get_param_if<float>(0).value_or(0.0f))
                           : 0u;
        uint32_t compute_scalar_value = quantization_zero_point;

        uint32_t compute_tiles = row_major_inputs ? (c_num_tiles_core * tiles_per_row_width) : c_num_tiles_core;

        uint32_t packed_scalar_for_reader = 0u;
        if (b.has_value()) {
            if (rt_has_sharding) {
                if (all_same_shard_spec) {
                    b_num_tiles = c_shard_height * c_shard_width;
                } else {
                    auto b_shard_shape = b_shard_shape_generator(core);
                    b_num_tiles = b_shard_shape[0] * b_shard_shape[1];
                }
            }
            if (row_major_inputs) {
                writer_runtime_args = {
                    c.buffer(),
                    common_row_width_elements,
                    c_num_tiles_core,
                    cD,
                    cN,
                    cC,
                    cHt_r,
                    cND,
                    current_block,
                    num_rows_per_tile,
                    static_cast<uint32_t>(c.buffer()->aligned_page_size()),
                    c_alignment,
                    tiles_per_row_width,
                    writer_stride_size_bytes};
            } else {
                writer_runtime_args = {
                    c.buffer(), c_start_id, c_num_tiles_core, c_current_shard_width, cD, cN, cC, cHt, cWt, cND, 0u};
            }

            auto [freq, counter] = CMAKE_UNIQUE_NAMESPACE::calculate_compute_kernel_args(
                operation_attributes.subtile_broadcast_type, c_start_id, cHt, cWt);
            if (operation_attributes.binary_op_type == BinaryOpType::WHERE_TTS ||
                operation_attributes.binary_op_type == BinaryOpType::WHERE_TST) {
                compute_scalar_value = pack_scalar_runtime_arg(
                    operation_attributes.scalar.value(), b.has_value() ? b->dtype() : a.dtype(), false);
            }
            if (row_major_inputs) {
                freq = 1;
                counter = 0;
            }
            if (operation_attributes.binary_op_type == BinaryOpType::ISCLOSE) {
                compute_runtime_args = {
                    compute_tiles,
                    freq,
                    counter,
                    // rtol and atol are float variables
                    std::bit_cast<uint32_t>(operation_attributes.rtol),
                    std::bit_cast<uint32_t>(operation_attributes.atol)};
            } else {
                compute_runtime_args = {compute_tiles, freq, counter, compute_scalar_value};
            }
        } else {
            const auto scalar = *operation_attributes.scalar;
            const auto packed_scalar = pack_scalar_runtime_arg(scalar, a.dtype(), rt_is_quant_op);
            packed_scalar_for_reader = packed_scalar;
            if (row_major_inputs) {
                writer_runtime_args = {
                    c.buffer(),
                    common_row_width_elements,
                    c_num_tiles_core,
                    cD,
                    cN,
                    cC,
                    cHt_r,
                    cND,
                    current_block,
                    num_rows_per_tile,
                    static_cast<uint32_t>(c.buffer()->aligned_page_size()),
                    c_alignment,
                    tiles_per_row_width,
                    writer_stride_size_bytes};
            } else {
                writer_runtime_args = {
                    packed_scalar,
                    c.buffer(),
                    c_start_id,
                    c_num_tiles_core,
                    c_current_shard_width,
                    cD,
                    cN,
                    cC,
                    cHt,
                    cWt,
                    cND,
                    0u};
            }

            compute_runtime_args = {compute_tiles, 0u, 0u, compute_scalar_value};
        }

        if (row_major_inputs) {
            const std::variant<uint32_t, Buffer*> b_addr =
                b.has_value() ? std::variant<uint32_t, Buffer*>{b->buffer()} : std::variant<uint32_t, Buffer*>{0u};
            const uint32_t b_page_size = b.has_value() ? static_cast<uint32_t>(b->buffer()->aligned_page_size())
                                                       : static_cast<uint32_t>(a.buffer()->aligned_page_size());
            const uint32_t bD_arg = b.has_value() ? bD : 1u;
            const uint32_t bN_arg = b.has_value() ? bN : 1u;
            const uint32_t bC_arg = b.has_value() ? bC : 1u;
            const uint32_t bHt_r_arg = b.has_value() ? bHt_r : 1u;
            reader_runtime_args = {
                a.buffer(),
                c_num_tiles_core,
                aD,
                aN,
                aC,
                aHt_r,
                aND,
                b_addr,
                bD_arg,
                bN_arg,
                bC_arg,
                bHt_r_arg,
                bND,
                cHt_r,
                cC,
                cND,
                current_block,
                num_rows_per_tile,
                common_row_width_elements,
                static_cast<uint32_t>(a.buffer()->aligned_page_size()),
                b_page_size,
                a_alignment,
                b_alignment,
                tiles_per_row_width,
                reader_stride_size_bytes,
                packed_scalar_for_reader};
        } else {
            reader_runtime_args = {
                a.buffer(),
                c_start_id,
                a_num_tiles,
                c_num_tiles_core,
                c_current_shard_width,
                aHt * aWt * aC * aN * aD * (aND > 1),
                aHt * aWt * aC * aN * (aD > 1),
                aHt * aWt * aC * (aN > 1),
                aHt * aWt * (aC > 1),
                cD,
                cN,
                cC,
                cHt,
                cWt,
                cND,
                b.has_value() ? std::variant<uint32_t, Buffer*>{b->buffer()} : std::variant<uint32_t, Buffer*>{0u},
                bHt * bWt * bC * bN * bD * (bND > 1),
                bHt * bWt * bC * bN * (bD > 1),
                bHt * bWt * bC * (bN > 1),
                bHt * bWt * (bC > 1),
                b_num_tiles,
            };
        }

        result.cores.push_back(core);
        result.reader.push_back(std::move(reader_runtime_args));
        result.writer.push_back(std::move(writer_runtime_args));
        result.compute.push_back(std::move(compute_runtime_args));

        start_tile_id += c_num_tiles_core;
        if (row_major_inputs) {
            current_block += c_num_tiles_core;
        }
    }

    return result;
}

}  // namespace

// Implements c = a op b
tt::tt_metal::ProgramDescriptor BinaryNgDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    using namespace tt;
    using namespace tt::tt_metal;

    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const bool is_sfpu_op = operation_attributes.is_sfpu;
    const bool is_quant_op = operation_attributes.is_quant_op;
    const bool is_where_op = operation_attributes.is_where_op;
    // TODO: For mixed dtypes we need to set this value to the appropriate dtype depending on which LLK is meant to be
    // used.
    const auto input_dtype = operation_attributes.input_dtype;
    if (is_quant_op) {
        TT_FATAL(is_sfpu_op, "Quantization op is SFPU-only");
    }

    const auto shard_volumes = get_shard_volumes(
        a.tensor_spec(), b.has_value() ? b->tensor_spec() : std::optional<tt::tt_metal::TensorSpec>{}, c.tensor_spec());
    const auto has_sharding = shard_volumes.has_value();
    const auto a_sharded = has_sharding and shard_volumes->a_shard_volume.has_value();
    const auto b_sharded = has_sharding and shard_volumes->b_shard_volume.has_value();
    const auto c_sharded = has_sharding and shard_volumes->c_shard_volume.has_value();
    const auto a_num_tiles_per_shard = has_sharding ? shard_volumes->a_shard_volume : std::nullopt;
    const auto b_num_tiles_per_shard = has_sharding ? shard_volumes->b_shard_volume : std::nullopt;
    const auto c_num_tiles_per_shard = has_sharding ? shard_volumes->c_shard_volume : std::nullopt;

    const auto a_dtype = a.dtype();
    // Always pass the more accurate fp32 when the quantization scale is passed as a scalar.
    // When b is a scalar, it is packed as bfloat16 for block-float input types (BFLOAT8_B,
    // BFLOAT4_B), so the CB format must be BFLOAT16 to match — not the block-float a_dtype.
    const auto b_dtype = b.has_value()                              ? b->dtype()
                         : is_quant_op                              ? DataType::FLOAT32
                         : (is_sfpu_op && !is_block_float(a_dtype)) ? a_dtype
                                                                    : DataType::BFLOAT16;
    const auto c_dtype = c.dtype();
    const auto a_data_format = datatype_to_dataformat_converter(a_dtype);
    const auto b_data_format = datatype_to_dataformat_converter(b_dtype);
    const auto c_data_format = datatype_to_dataformat_converter(c_dtype);

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);

    // we parallelize the computation across the output tiles
    const auto& all_device_cores = operation_attributes.worker_grid;

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = b.has_value() ? b->buffer() : nullptr;
    Buffer* c_buffer = c.buffer();

    auto op_type = operation_attributes.binary_op_type;

    // TODO: when handling mixed types, we must identify the appropriate dtype and pass it here to define the respective
    // LLK APIs
    const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                                      : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);

    auto compute_kernel_defines = op_config.as_defines(a_dtype);

    // Quant/requant rounding depends on the output dtype. For uint8, we need fp32->uint8 rounding instead
    // of the default fp32->int8. The packer narrows the int32 SFPU result to uint8.
    if (c_dtype == DataType::UINT8) {
        if (operation_attributes.binary_op_type == BinaryOpType::QUANT) {
            compute_kernel_defines["BINARY_SFPU_INIT"] =
                "quant_uint8_tile_init(get_arg_val<uint32_t>(QUANT_ZERO_POINT_RT_ARGS_IDX));";
        } else if (operation_attributes.binary_op_type == BinaryOpType::REQUANT) {
            compute_kernel_defines["BINARY_SFPU_INIT"] =
                "requant_uint8_tile_init(get_arg_val<uint32_t>(QUANT_ZERO_POINT_RT_ARGS_IDX));";
        }
    }

    // Indices 3 and 4 in the compute runtime args vector are reserved for rtol and atol bits.
    if (operation_attributes.binary_op_type == BinaryOpType::ISCLOSE) {
        compute_kernel_defines["ISCLOSE_OP"] = "1";
        compute_kernel_defines["ISCLOSE_EQUAL_NAN"] = operation_attributes.equal_nan ? "1" : "0";
        compute_kernel_defines["ISCLOSE_RTOL_RT_ARG_IDX"] = "3";
        compute_kernel_defines["ISCLOSE_ATOL_RT_ARG_IDX"] = "4";
    }

    {
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations = operation_attributes.lhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations = operation_attributes.rhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> post_activations = operation_attributes.post_activations;

        if (op_config.process_lhs.has_value()) {
            lhs_activations.push_back(*op_config.process_lhs);
        }

        if (op_config.process_rhs.has_value()) {
            rhs_activations.push_back(*op_config.process_rhs);
        }

        // LDEXP decomposes to EXP2(rhs) then MUL on the FPU path.  The RHS
        // intermediate CB is Float16_b (due to op_has_exp), but LHS has no
        // activation and stays in its original block-float format (BFLOAT8_B /
        // BFLOAT4_B).  The resulting data-format mismatch between the two FPU
        // binary operands produces incorrect results.  Adding a typecast for
        // LHS forces it through a Float16_b intermediate CB so both operands
        // use the same format.
        // Note: LOGADDEXP / LOGADDEXP2 are not affected because they process
        // both sides (EXP/EXP2), so lhs_activations is never empty for them.
        if (!is_sfpu_op && lhs_activations.empty() && !rhs_activations.empty() && op_type == BinaryOpType::LDEXP &&
            (a_dtype == DataType::BFLOAT8_B || a_dtype == DataType::BFLOAT4_B)) {
            lhs_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(DataType::BFLOAT16)},
            });
        }

        if (op_config.postprocess.has_value()) {
            post_activations.insert(post_activations.begin(), *op_config.postprocess);
        }

        bool is_integer_division =
            (operation_attributes.binary_op_type == BinaryOpType::DIV && a_dtype == DataType::INT32 &&
             b_dtype == DataType::INT32);

        if (binary::utils::is_typecast(a_dtype, c_dtype) and !is_quant_op and !is_integer_division) {
            post_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(c_dtype)},
            });
        }

        add_activation_defines(compute_kernel_defines, lhs_activations, "LHS", a_dtype);
        add_activation_defines(compute_kernel_defines, rhs_activations, "RHS", b_dtype);

        // The PACK_RELU fast path applies ZERO_RELU via the packer config set once at the top
        // of the compute kernel.  Subtile-broadcast kernels do an intermediate
        // `pack_tile(0, cb_llk_post)` followed by `pack_reconfig_data_format(cb_llk_post, cb_out)`
        // per iteration, which clears the packer's ZERO_RELU state, so the final pack to
        // `cb_out` no longer clips negatives and RELU is silently dropped.  Restrict the
        // PACK_RELU optimization to the non-broadcast case and fall through to the SFPU
        // activation path (used by every other unary post-activation) for broadcast cases.
        const bool is_subtile_broadcast = operation_attributes.subtile_broadcast_type != SubtileBroadcastType::NONE;

        if (lhs_activations.empty() and rhs_activations.empty() and post_activations.size() == 1) {
            compute_kernel_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            if (post_activations[0].type() == unary::UnaryOpType::RELU && !is_subtile_broadcast) {
                compute_kernel_defines["PACK_RELU"] = "1";
                unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_kernel_defines);
            } else if (post_activations[0].type() == unary::UnaryOpType::ZERO_POINT) {
                // Zero-point is passed as the 4th run-time kernel argument
                compute_kernel_defines["QUANT_ZERO_POINT_RT_ARGS_IDX"] = "3";
                unary::utils::update_macro_defines(unary::UnaryOpType::ZERO_POINT, compute_kernel_defines);
            } else {
                add_activation_defines(compute_kernel_defines, post_activations, "POST", input_dtype);
            }
        } else {
            add_activation_defines(compute_kernel_defines, post_activations, "POST", input_dtype);
        }
    }

    // Determine max tiles per cycle based on sharding and output data type
    // Multi-tile processing only enabled when all tensors are sharded
    uint32_t num_tiles_per_cycle = 1;  // Conservative default
    bool enable_multi_tile = false;

    // Enable for no-broadcast case when all tensors are sharded
    if (operation_attributes.subtile_broadcast_type == SubtileBroadcastType::NONE && a_sharded && b_sharded &&
        c_sharded) {
        enable_multi_tile = true;
    }
    // Enable for scalar value case (not scalar broadcast) when input and output are sharded
    if (!b.has_value() && a_sharded && c_sharded) {
        enable_multi_tile = true;
    }

    if (enable_multi_tile && !is_where_op) {
        if (!is_sfpu_op) {
            // FPU kernels: 16-bit types can handle 8 tiles,
            num_tiles_per_cycle = 8;  // Default for 16-bit types (BF16, BF8, BF4)
        } else {
            // SFPU kernel should handle 4, but for unknown reason, only 2 works
            // no document and example to show why 4 does not work, need further investigation
            num_tiles_per_cycle = 2;
        }
    }

    bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;
    const bool inputs_row_major =
        CMAKE_UNIQUE_NAMESPACE::should_use_row_major_path(operation_attributes, b, has_sharding);

    // CB: a (c_0)
    {
        uint32_t a_num_pages = a_num_tiles_per_shard.value_or(2);
        desc.cbs.push_back(CBDescriptor{
            .total_size = a_single_tile_size * a_num_pages,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
                .data_format = a_data_format,
                .page_size = a_single_tile_size,
            }}},
            .buffer = a_sharded ? a_buffer : nullptr,
        });
    }

    if (not compute_kernel_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty()) {
        auto a_intermediate_format = is_sfpu_op   ? a_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : a_data_format;
        uint32_t a_intermediate_single_tile_size = tt::tile_size(a_intermediate_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = a_intermediate_single_tile_size * num_tiles_per_cycle,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = a_intermediate_format,
                .page_size = a_intermediate_single_tile_size,
            }}},
        });
    }

    // CB: b (c_1)
    {
        uint32_t b_num_pages = b_buffer == nullptr ? 1 : b_num_tiles_per_shard.value_or(2);
        desc.cbs.push_back(CBDescriptor{
            .total_size = b_single_tile_size * b_num_pages,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                .data_format = b_data_format,
                .page_size = b_single_tile_size,
            }}},
            .buffer = b_sharded ? b_buffer : nullptr,
        });
    }

    if (not compute_kernel_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty()) {
        auto b_intermediate_format = is_sfpu_op   ? b_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : b_data_format;
        uint32_t b_intermediate_single_tile_size = tt::tile_size(b_intermediate_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = b_intermediate_single_tile_size * num_tiles_per_cycle,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = b_intermediate_format,
                .page_size = b_intermediate_single_tile_size,
            }}},
        });
    }

    if (operation_attributes.subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_A) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = a_single_tile_size * 2,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = a_data_format,
                .page_size = a_single_tile_size,
            }}},
        });
    }
    if (operation_attributes.subtile_broadcast_type == SubtileBroadcastType::ROW_B ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::COL_B ||
        operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = b_single_tile_size * 2,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                .data_format = b_data_format,
                .page_size = b_single_tile_size,
            }}},
        });
    }

    // CB: c (c_2)
    {
        uint32_t c_num_pages = c_num_tiles_per_shard.value_or(2);
        desc.cbs.push_back(CBDescriptor{
            .total_size = c_single_tile_size * c_num_pages,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = c_data_format,
                .page_size = c_single_tile_size,
            }}},
            .buffer = c_sharded ? c_buffer : nullptr,
        });
    }

    const bool outputs_row_major = inputs_row_major && operation_attributes.output_layout == Layout::ROW_MAJOR;
    if (inputs_row_major) {
        TT_FATAL(!has_sharding, "Row-major binary_ng path does not support sharded tensors yet");
        TT_FATAL(outputs_row_major, "Row-major binary_ng path requires row-major output layout");
    }

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);
    // WRITER KERNEL
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    if (b.has_value()) {
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;
    }

    auto writer_defines = make_dataflow_defines(b_dtype);
    writer_defines["SRC_SHARDED"] = b_sharded ? "1" : "0";
    writer_defines["DST_SHARDED"] = c_sharded ? "1" : "0";

    auto reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = a_sharded ? "1" : "0";
    reader_defines["SRC_SHARDED_B"] = b_sharded ? "1" : "0";
    if (inputs_row_major) {
        kernel_config.reader_kernel = CMAKE_UNIQUE_NAMESPACE::get_reader_rm_kernel_name_and_defines(
            operation_attributes.subtile_broadcast_type, b.has_value(), reader_defines);
        writer_kernel = KernelName::WriterRmNoBcastNg;
    } else if (b.has_value()) {
        kernel_config.reader_kernel = CMAKE_UNIQUE_NAMESPACE::get_reader_kernel_name_and_defines(
            operation_attributes.subtile_broadcast_type, reader_defines);
        writer_kernel = KernelName::WriterNoBcastNg;
    }

    // WRITER KERNEL DESCRIPTOR
    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> writer_common_runtime_args;
    tt::tt_metal::TensorAccessorArgs(*c_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(has_sharding));

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = get_kernel_file_path(writer_kernel, is_sfpu_op, is_where_op);
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = writer_common_runtime_args;

    // COMPUTE KERNEL
    // fp32 dest accumulation must be enabled whenever any input or output is fp32, otherwise
    // loading fp32 tiles into a DST configured for bf16 produces tile-aligned corruption
    // for broadcast multiply (issue 43196).
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32 || a_data_format == tt::DataFormat::Float32 ||
                            b_data_format == tt::DataFormat::Float32 ||
                            (a_data_format == tt::DataFormat::Int32 && b_data_format == tt::DataFormat::Int32) ||
                            (a_data_format == tt::DataFormat::UInt32 && b_data_format == tt::DataFormat::UInt32) ||
                            // Quant SFPU kernels compute on the fp32 input in DST; keep fp32 dest
                            // accumulation regardless of the (possibly narrow, e.g. uint8) output format.
                            operation_attributes.is_quant_op;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src0interim_cb_index = tt::CBIndex::c_3;
    uint32_t src1interim_cb_index = tt::CBIndex::c_4;

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);

    if (is_sfpu_op) {
        if (op_type != BinaryOpType::POWER) {
            unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src1_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src0interim_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src1interim_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[tt::CBIndex::c_5] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[tt::CBIndex::c_6] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        } else {
            unpack_to_dest_mode[src0_cb_index] =
                (a_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
            unpack_to_dest_mode[src1_cb_index] =
                (b_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
            unpack_to_dest_mode[src0interim_cb_index] =
                (a_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
            unpack_to_dest_mode[src1interim_cb_index] =
                (b_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
            unpack_to_dest_mode[tt::CBIndex::c_5] =
                (a_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
            unpack_to_dest_mode[tt::CBIndex::c_6] =
                (b_dtype == DataType::FLOAT32) ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32 : tt::tt_metal::UnpackToDestMode::Default;
        }
    }

    compute_kernel_defines["BCAST_INPUT"] = kernel_config.bcast_input_str();

    bool use_llk_bcast =
        !inputs_row_major && CMAKE_UNIQUE_NAMESPACE::is_llk_bcast(
                                 operation_attributes.subtile_broadcast_type, a_dtype, b_dtype, fp32_dest_acc_en);

    // The B2D broadcast path for BFP formats introduces rounding that EXP/EXP2
    // amplifies beyond acceptable tolerance.
    if (use_llk_bcast && op_has_exp) {
        use_llk_bcast = false;
    }

    // Where op does not support LLK bcast for scalar and col broadcast
    if (use_llk_bcast && is_where_op &&
        (operation_attributes.subtile_broadcast_type == SubtileBroadcastType::COL_A ||
         operation_attributes.subtile_broadcast_type == SubtileBroadcastType::COL_B ||
         operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
         operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_B)) {
        use_llk_bcast = false;
    }

    // Integer relational/equality ops on UInt16 use direct SFPU comparison
    // (LT/GT/LE/GE/EQ/NE), with DEST configured for Fp16_b accumulation
    // (fp32_dest_acc_en is false for UInt16).  Under SCALAR broadcast the B2D
    // datacopy unpacker writes a single u16 lane into all DEST positions; the
    // resulting Fp16_b-tagged DEST is then read back by the SFPU comparison
    // kernel, which interprets the integer bit pattern through the
    // format-conversion path and corrupts the comparison result (#36217).
    // Fall back to software broadcast for this combination - non-broadcast u16
    // relational ops and broadcasted arithmetic u16 ops (no postprocess) are
    // unaffected.
    if (use_llk_bcast && a_data_format == tt::DataFormat::UInt16 && b_data_format == tt::DataFormat::UInt16 &&
        (op_config.postprocess.has_value() ||
         (std::holds_alternative<OpConfig::SfpuBinaryOp>(op_config.binary_op) &&
          (std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::LT ||
           std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::GT ||
           std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::LE ||
           std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::GE ||
           std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::EQ ||
           std::get<OpConfig::SfpuBinaryOp>(op_config.binary_op) == OpConfig::SfpuBinaryOp::NE))) &&
        (operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
         operation_attributes.subtile_broadcast_type == SubtileBroadcastType::SCALAR_B)) {
        use_llk_bcast = false;
    }

    // On Blackhole, the B2D datacopy path (MOVB2D) has a known issue in FP32 dest
    // accumulation mode with non-32-bit source formats (BH Issue #449).  SCALAR and
    // ROW broadcasts use MOVB2D, which produces corrupted results when
    // fp32_dest_acc_en is true.  COL broadcast is unaffected because it uses ELWADD
    // instead of MOVB2D.  Fall back to the software-broadcast path for the affected
    // broadcast types on Blackhole when FP32 dest accumulation is active.
    if (use_llk_bcast && fp32_dest_acc_en) {
        tt::ARCH arch = tt::tt_metal::hal::get_arch();
        if (arch == tt::ARCH::BLACKHOLE) {
            auto sbt = operation_attributes.subtile_broadcast_type;
            bool uses_movb2d =
                (sbt == SubtileBroadcastType::SCALAR_A || sbt == SubtileBroadcastType::SCALAR_B ||
                 sbt == SubtileBroadcastType::ROW_A || sbt == SubtileBroadcastType::ROW_B ||
                 sbt == SubtileBroadcastType::ROW_A_COL_B || sbt == SubtileBroadcastType::ROW_B_COL_A);
            if (uses_movb2d) {
                use_llk_bcast = false;
            }
        }
    }

    if (use_llk_bcast) {
        CMAKE_UNIQUE_NAMESPACE::overwrite_compute_kernel_name_and_defines(
            compute_kernel, operation_attributes.subtile_broadcast_type, compute_kernel_defines, is_where_op);
        reader_defines["BCAST_LLK"] = "1";
    } else {
        reader_defines["BCAST_LLK"] = "0";
    }

    if (op_type == BinaryOpType::WHERE_TTS || op_type == BinaryOpType::WHERE_TST) {
        // Add common fill defines
        compute_kernel_defines["FILL_LLK"] = "fill_tile";
        if (b_dtype == DataType::INT32) {
            compute_kernel_defines["FILL_LLK"] = "fill_tile_int<DataFormat::Int32>";
            compute_kernel_defines["FILL_WITH_VALUE_INT"] = "1";
        } else if (b_dtype == DataType::UINT32) {
            compute_kernel_defines["FILL_LLK"] = "fill_tile_int<DataFormat::UInt32>";
            compute_kernel_defines["FILL_WITH_VALUE_INT"] = "1";
        } else {
            compute_kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
        }
    }
    compute_kernel_defines["WHERE_TTS"] = (op_type == BinaryOpType::WHERE_TTS) ? "1" : "0";
    compute_kernel_defines["WHERE_TST"] = (op_type == BinaryOpType::WHERE_TST) ? "1" : "0";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = get_kernel_file_path(compute_kernel, is_sfpu_op, is_where_op);
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {compute_kernel_defines.begin(), compute_kernel_defines.end()};
    compute_desc.compile_time_args = {num_tiles_per_cycle};
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };

    // READER KERNEL DESCRIPTOR
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    tt::tt_metal::TensorAccessorArgs(*a_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);
    tt::tt_metal::TensorAccessorArgs(
        b_buffer != nullptr ? *b_buffer : *a_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);
    reader_compile_time_args.push_back(static_cast<uint32_t>(has_sharding));

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = get_kernel_file_path(kernel_config.reader_kernel, is_sfpu_op, is_where_op);
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = reader_common_runtime_args;

    // === Per-core runtime arguments ===
    // Built via the shared single-source-of-truth builder so create_descriptor() (cache miss, here)
    // and BinaryNgDeviceOperation::override_runtime_arguments() (cache hit) stay byte-identical.
    {
        auto per_core = build_per_core_runtime_args(operation_attributes, a, b, c);
        for (size_t i = 0; i < per_core.cores.size(); ++i) {
            reader_desc.emplace_runtime_args(per_core.cores[i], per_core.reader[i]);
            writer_desc.emplace_runtime_args(per_core.cores[i], per_core.writer[i]);
            compute_desc.emplace_runtime_args(per_core.cores[i], per_core.compute[i]);
        }
    }

    // Push kernel descriptors: reader (index 0), writer (index 1), compute (index 2)
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void BinaryNgDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& c,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-apply ALL per-dispatch state to the cached program on a program-cache hit (the descriptor-era
    // override_runtime_arguments()).  compute_program_hash EXCLUDES the tensor volume, so one cached
    // program is reused across differently-shaped and differently-allocated (incl. in-place, out=x)
    // calls; every shape-/work-split-dependent per-core arg AND every tensor-backed CB base address
    // would otherwise stay frozen at the first miss.  We re-derive them for the CURRENT tensors from
    // the SAME shared builder create_descriptor() uses, so the two stay byte-identical by construction.
    //
    // This is correct where address inference (resolve_bindings' std::find) was not: nothing is guessed
    // from Buffer* identity, so an in-place alias (input_a == output) or a mixed in-place/out-of-place
    // reuse of one cache entry writes each slot from the tensor it actually belongs to.
    //
    // Kernel push order in create_descriptor(): reader(0), writer(1), compute(2).  The work-core
    // partition shifts with the volume (a core can flip between work and noop), so the builder emits
    // args for ALL cores; we re-apply every one, buffer-address slots included (via the current Buffer
    // address), so a core promoted to a work core on this hit is never left with a stale base address.
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    auto per_core = build_per_core_runtime_args(operation_attributes, a, b, c);

    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kComputeKernelIdx = 2;

    auto apply = [&](uint32_t kernel_idx,
                     const CoreCoord& core,
                     const std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>& args) {
        auto& data = tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core);
        for (uint32_t arg_idx = 0; arg_idx < static_cast<uint32_t>(args.size()); ++arg_idx) {
            const auto& slot = args[arg_idx];
            data[arg_idx] = std::holds_alternative<tt::tt_metal::Buffer*>(slot)
                                ? static_cast<uint32_t>(std::get<tt::tt_metal::Buffer*>(slot)->address())
                                : std::get<uint32_t>(slot);
        }
    };

    for (size_t i = 0; i < per_core.cores.size(); ++i) {
        const auto& core = per_core.cores[i];
        apply(kReaderKernelIdx, core, per_core.reader[i]);
        apply(kWriterKernelIdx, core, per_core.writer[i]);
        apply(kComputeKernelIdx, core, per_core.compute[i]);
    }

    // Re-point tensor-backed (globally-allocated) circular buffers at the CURRENT buffers, by CBIndex.
    // binary_ng convention: c_0 = input_a, c_1 = input_b, c_2 = output.  Addressing by CBIndex (not by
    // enumeration order) is what makes the in-place alias correct: the output CB always tracks the
    // output buffer even when it shares a Buffer* with an input.
    tt::tt_metal::Buffer* a_buffer = a.buffer();
    tt::tt_metal::Buffer* b_buffer = b.has_value() ? b->buffer() : nullptr;
    tt::tt_metal::Buffer* c_buffer = c.buffer();
    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& indices = cb->buffer_indices();
        if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_0)) && a_buffer != nullptr) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *a_buffer);
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_1)) && b_buffer != nullptr) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *b_buffer);
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_2)) && c_buffer != nullptr) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *c_buffer);
        }
    }
}

}  // namespace ttnn::operations::binary_ng
