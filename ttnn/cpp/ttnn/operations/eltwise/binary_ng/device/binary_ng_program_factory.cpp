// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;

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
        shape[-2] / tile.get_height(),
        shape[-1] / tile.get_width()};
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

struct AllShardSpecs {
    ShardSpec a_shard_spec;
    ShardSpec b_shard_spec;
    ShardSpec c_shard_spec;
};

ShardSpec adjust_to_shape(const ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    auto ret = shard_spec;

    ret.shape[0] = (ret.shape[0] * to_shape[-2]) / from_shape[-2];
    ret.shape[1] = (ret.shape[1] * to_shape[-1]) / from_shape[-1];

    return ret;
}

TensorMemoryLayout get_memory_layout(const Tensor& a, const std::optional<Tensor>& b, const Tensor& c) {
    if (a.memory_config().is_sharded()) {
        return a.memory_config().memory_layout();
    }
    if (b.has_value() && b->memory_config().is_sharded()) {
        return b->memory_config().memory_layout();
    }
    if (c.memory_config().is_sharded()) {
        return c.memory_config().memory_layout();
    }
    return TensorMemoryLayout::INTERLEAVED;
}

std::optional<AllShardSpecs> get_shard_specs(const Tensor& a, const std::optional<Tensor>& b, const Tensor& c) {
    bool a_sharded = a.memory_config().is_sharded();
    bool b_sharded = b.has_value() && b->memory_config().is_sharded();
    bool c_sharded = c.memory_config().is_sharded();

    if (!a_sharded && !b_sharded && !c_sharded) {
        return std::nullopt;
    }

    const auto& a_shape = a.padded_shape();
    auto b_shape = b.has_value() ? b->padded_shape() : ttnn::Shape{1, 1};
    const auto& c_shape = c.padded_shape();

    ShardSpec c_shard_spec = c_sharded   ? *c.shard_spec()
                             : a_sharded ? adjust_to_shape(*a.shard_spec(), a_shape, c_shape)
                                         : adjust_to_shape(*b->shard_spec(), b_shape, c_shape);

    return AllShardSpecs{
        a_sharded ? *a.shard_spec() : adjust_to_shape(c_shard_spec, c_shape, a_shape),
        b_sharded ? *b->shard_spec() : adjust_to_shape(c_shard_spec, c_shape, b_shape),
        c_shard_spec};
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
    bool row_major;
    TensorMemoryLayout memory_layout;
    std::array<uint32_t, 2> shard_shape;
    std::array<uint32_t, 2> last_shard_shape;

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
        // for uneven shard, HEIGHT, WIDTH, and BLOCK handling order should be all different in kernel
        // only HEIGHT sharding works naturally
        if (core == end_core) {
            if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                current_shape[majorDim] = last_shard_shape[majorDim];
                current_shape[minorDim] = last_shard_shape[minorDim];
            } else {
                TT_FATAL(
                    current_shape[majorDim] == last_shard_shape[majorDim] and
                        current_shape[minorDim] == last_shard_shape[minorDim],
                    "no un-even shard size support memory layout {}",
                    memory_layout);
            }
        }
        return current_shape;
    }
};

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    tt::tt_metal::CBHandle cb_src_a,
    tt::tt_metal::CBHandle cb_src_b,
    tt::tt_metal::CBHandle cb_src_c,
    const BinaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    BinaryNgDeviceOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const auto out_rank = c.logical_shape().rank();
    auto aND = extract_nD_dims(a, out_rank);
    auto bND = b.has_value() ? extract_nD_dims(*b, out_rank) : 1;
    auto cND = extract_nD_dims(c, out_rank);

    const auto [aD, aN, aC, aHt, aWt] = get_shape_dims(a);
    const auto [bD, bN, bC, bHt, bWt] = b.has_value() ? get_shape_dims(*b) : std::tuple{1u, 1u, 1u, 1u, 1u};
    const auto [cD, cN, cC, cHt, cWt] = get_shape_dims(c);

    const auto shard_specs = get_shard_specs(a, b, c);
    const bool has_sharding = shard_specs.has_value();
    auto grid = has_sharding ? shard_specs->a_shard_spec.grid : CoreRangeSet{};

    const auto row_major = has_sharding ? shard_specs->a_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    // This will run the original work/core distribution algorithms that are specifically for this setup, as these
    // are faster than the generic work/core distribution algorithms that work on arbitrary CoreRangeSets
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    const auto& all_device_cores = operation_attributes.worker_grid;
    if (grid.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (has_sharding) {
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

    const uint32_t tile_height = c.tensor_spec().tile().get_height();
    const uint32_t tile_width = c.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;
    const uint32_t c_num_tiles = c.physical_volume() / tile_hw;
    uint32_t c_shard_height{}, c_shard_width{}, num_shards_per_width{};

    ShardShapeGenerator a_shard_shape_generator;
    ShardShapeGenerator b_shard_shape_generator;
    ShardShapeGenerator c_shard_shape_generator;

    if (has_sharding) {
        core_group_1 = grid;
        a_shard_shape_generator = ShardShapeGenerator(shard_specs->a_shard_spec, a);
        if (b.has_value()) {
            b_shard_shape_generator = ShardShapeGenerator(shard_specs->b_shard_spec, *b);
        }
        c_shard_shape_generator = ShardShapeGenerator(shard_specs->c_shard_spec, c);
        c_shard_height = shard_specs->c_shard_spec.shape[0] / tile_height;
        c_shard_width = shard_specs->c_shard_spec.shape[1] / tile_width;
        num_shards_per_width = get_shards_per_width(shard_specs->c_shard_spec, get_memory_layout(a, b, c));

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
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid, c_num_tiles, row_major);
        cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, c_num_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t a_num_tiles = 0;
        uint32_t b_num_tiles = 0;
        uint32_t c_num_tiles = 0;
        if (core_group_1.contains(core)) {
            c_num_tiles = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            c_num_tiles = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 21>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 16>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 4>{0});
            continue;
        }

        uint32_t c_start_id = 0;
        uint32_t c_current_shard_width = 0;
        if (has_sharding) {
            auto c_shard_shape = c_shard_shape_generator(core);
            c_num_tiles = c_shard_shape[0] * c_shard_shape[1];
            c_current_shard_width = c_shard_shape[1];
            auto a_shard_shape = a_shard_shape_generator(core);
            a_num_tiles = a_shard_shape[0] * a_shard_shape[1];
            c_start_id =
                (i / num_shards_per_width) * (c_shard_height * cWt) + (i % num_shards_per_width) * c_shard_width;
        } else {
            c_start_id = start_tile_id;
        }

        const bool is_quant_op = operation_attributes.is_quant_op;
        TT_FATAL(
            is_quant_op == ((operation_attributes.post_activations.size() == 1) &&
                            (operation_attributes.post_activations[0].op_type ==
                             ttnn::operations::unary::UnaryOpType::ZERO_POINT)),
            "Quantization op needs to exactly one zero-point value as a post activation");
        const uint32_t quantization_zero_point =
            is_quant_op ? std::bit_cast<uint32_t>(operation_attributes.post_activations[0].params[0]) : 0u;

        if (b.has_value()) {
            if (has_sharding) {
                auto b_shard_shape = b_shard_shape_generator(core);
                b_num_tiles = b_shard_shape[0] * b_shard_shape[1];
            }
            // TODO: after transition, remove b from writer completely
            std::array writer_runtime_args = {
                b->buffer()->address(),
                c.buffer()->address(),
                c_start_id,
                b_num_tiles,
                c_num_tiles,
                c_current_shard_width,
                bHt * bWt * bC * bN * bD * (bND > 1),
                bHt * bWt * bC * bN * (bD > 1),
                bHt * bWt * bC * (bN > 1),
                bHt * bWt * (bC > 1),
                cD,
                cN,
                cC,
                cHt,
                cWt,
                cND};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            auto [freq, counter] =
                calculate_compute_kernel_args(operation_attributes.subtile_broadcast_type, c_start_id, cHt, cWt);
            std::array compute_runtime_args = {c_num_tiles, freq, counter, quantization_zero_point};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            const auto scalar = *operation_attributes.scalar;
            // TODO: technically we should use the b_dtype deduced by ProgramFactory::create here, but currently only
            // quant ops have different dtypes for a & b and we want to force f32 for better accuracy when scale is
            // passed as a scalar, so we'll leave this here
            const auto packed_scalar = pack_scalar_runtime_arg(scalar, a.dtype(), is_quant_op);
            std::array writer_runtime_args = {
                packed_scalar,
                c.buffer()->address(),
                c_start_id,
                c_num_tiles,
                c_current_shard_width,
                cD,
                cN,
                cC,
                cHt,
                cWt,
                cND,
                0u,
                0u,
                0u,
                0u,
                0u};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {c_num_tiles, 0u, 0u, quantization_zero_point};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        // for the specific case of subtile no_bcast type, reader also needs b's information
        // number of parameters are still small so negligible dispatch cost
        std::array reader_runtime_args = {
            a.buffer()->address(),
            c_start_id,
            a_num_tiles,
            c_num_tiles,
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
            b.has_value() ? b->buffer()->address() : 0u,
            bHt * bWt * bC * bN * bD * (bND > 1),
            bHt * bWt * bC * bN * (bD > 1),
            bHt * bWt * bC * (bN > 1),
            bHt * bWt * (bC > 1),
            b_num_tiles,
        };
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        start_tile_id += c_num_tiles;
    }
    if (has_sharding) {
        if (a.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src_a, *a.buffer());
        }
        if (b.has_value() and b->is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src_b, *b->buffer());
        }
        if (c.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src_c, *c.buffer());
        }
    }
}

KernelName get_reader_kernel_name_and_defines(
    const SubtileBroadcastType subtile_broadcast_type, std::map<std::string, std::string>& reader_defines) {
    if (subtile_broadcast_type == SubtileBroadcastType::NONE) {
        return KernelName::ReaderNoBcastNg;
    } else if (
        subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::ROW_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B ? "1" : "0";
        return KernelName::ReaderRowBcastNg;
    } else if (
        subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::COL_B ? "1" : "0";
        return KernelName::ReaderColBcastNg;
    } else if (
        subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B) {
        reader_defines["SRC_BCAST_COL"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_ROW_B"] = subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        return KernelName::ReaderRowBColABcastNg;
    } else if (
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        reader_defines["SRC_BCAST"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ? "1" : "0";
        reader_defines["SRC_BCAST_B"] = subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ? "1" : "0";
        return KernelName::ReaderScalarBcastNg;
    } else {
        TT_FATAL(false, "Unsupported subtile broadcast type {}", static_cast<int>(subtile_broadcast_type));
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::binary_ng {

// Implements c = a op b
BinaryNgDeviceOperation::ProgramFactory::cached_program_t BinaryNgDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const bool is_sfpu_op = operation_attributes.is_sfpu;
    const bool is_quant_op = operation_attributes.is_quant_op;
    if (is_quant_op) {
        TT_FATAL(is_sfpu_op, "Quantization op is SFPU-only");
    }

    auto program = CreateProgram();

    const auto shard_specs = CMAKE_UNIQUE_NAMESPACE::get_shard_specs(a, b, c);
    const bool has_sharding = shard_specs.has_value();

    auto tile_hw = c.tensor_spec().tile().get_tile_hw();
    uint32_t a_num_tiles_per_shard = has_sharding ? shard_specs->a_shard_spec.numel() / tile_hw : 0;
    uint32_t b_num_tiles_per_shard = has_sharding ? shard_specs->b_shard_spec.numel() / tile_hw : 0;
    uint32_t c_num_tiles_per_shard = has_sharding ? shard_specs->c_shard_spec.numel() / tile_hw : 0;

    const auto a_dtype = a.dtype();
    // Always pass the more accurate fp32 when the quantization scale is passed as a scalar
    const auto b_dtype = b.has_value() ? b->dtype()
                         : is_quant_op ? DataType::FLOAT32
                         : is_sfpu_op  ? a_dtype
                                       : DataType::BFLOAT16;
    const auto c_dtype = c.dtype();
    const auto a_data_format = datatype_to_dataformat_converter(a_dtype);
    const auto b_data_format = datatype_to_dataformat_converter(b_dtype);
    const auto c_data_format = datatype_to_dataformat_converter(c_dtype);

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    // we parallelize the computation across the output tiles
    const auto& all_device_cores = operation_attributes.worker_grid;

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = b.has_value() ? b->buffer() : nullptr;
    Buffer* c_buffer = c.buffer();

    auto op_type = operation_attributes.binary_op_type;

    const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>)
                                      : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>);

    auto compute_kernel_defines = op_config.as_defines(a_dtype);

    {
        ttnn::SmallVector<unary::UnaryWithParam> lhs_activations = operation_attributes.lhs_activations;
        ttnn::SmallVector<unary::UnaryWithParam> rhs_activations = operation_attributes.rhs_activations;
        ttnn::SmallVector<unary::UnaryWithParam> post_activations = operation_attributes.post_activations;

        if (op_config.process_lhs.has_value()) {
            lhs_activations.push_back(*op_config.process_lhs);
        }

        if (op_config.process_rhs.has_value()) {
            rhs_activations.push_back(*op_config.process_rhs);
        }

        if (op_config.postprocess.has_value()) {
            post_activations.insert(post_activations.begin(), *op_config.postprocess);
        }

        if (binary::utils::is_typecast(a_dtype, c_dtype) and !is_quant_op) {
            post_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(c_dtype)},
            });
        }

        add_activation_defines(compute_kernel_defines, lhs_activations, "LHS", a_dtype);
        add_activation_defines(compute_kernel_defines, rhs_activations, "RHS", b_dtype);

        if (lhs_activations.empty() and rhs_activations.empty() and post_activations.size() == 1) {
            compute_kernel_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            if (post_activations[0].op_type == unary::UnaryOpType::RELU) {
                compute_kernel_defines["PACK_RELU"] = "1";
                unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_kernel_defines);
            } else if (post_activations[0].op_type == unary::UnaryOpType::ZERO_POINT) {
                // Zero-point is passed as the 4th run-time kernel argument
                compute_kernel_defines["QUANT_ZERO_POINT_RT_ARGS_IDX"] = "3";
                unary::utils::update_macro_defines(unary::UnaryOpType::ZERO_POINT, compute_kernel_defines);
            } else {
                add_activation_defines(compute_kernel_defines, post_activations, "POST", c.dtype());
            }
        } else {
            add_activation_defines(compute_kernel_defines, post_activations, "POST", c.dtype());
        }
    }

    bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

    bool a_sharded = a.memory_config().is_sharded();
    bool b_sharded = b.has_value() && b->memory_config().is_sharded();
    bool c_sharded = c.memory_config().is_sharded();

    // How many tiles to store per input CB (double buffer)
    auto [a_cb, a_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        a_single_tile_size,
        a_sharded ? a_num_tiles_per_shard : 2,
        a_data_format,
        a_sharded ? a_buffer : nullptr);

    if (not compute_kernel_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty()) {
        auto a_intermediate_format = is_sfpu_op   ? a_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : a_data_format;
        uint32_t a_intermediate_single_tile_size = tt_metal::detail::TileSize(a_intermediate_format);
        create_cb(
            tt::CBIndex::c_3, program, all_device_cores, a_intermediate_single_tile_size, 1, a_intermediate_format);
    }

    // If b is a scalar, we only need one tile in the CB
    auto [b_cb, b_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        b_single_tile_size,
        b_buffer == nullptr ? 1 : (b_sharded ? b_num_tiles_per_shard : 2),
        b_data_format,
        b_sharded ? b_buffer : nullptr);

    if (not compute_kernel_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty()) {
        auto b_intermediate_format = is_sfpu_op   ? b_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : b_data_format;
        uint32_t b_intermediate_single_tile_size = tt_metal::detail::TileSize(b_intermediate_format);
        create_cb(
            tt::CBIndex::c_4, program, all_device_cores, b_intermediate_single_tile_size, 1, b_intermediate_format);
    }

    auto [c_cb, c_cb_handle] = create_cb(
        tt::CBIndex::c_2,
        program,
        all_device_cores,
        c_single_tile_size,
        c_sharded ? c_num_tiles_per_shard : 2,
        c_data_format,
        c_sharded ? c_buffer : nullptr);

    uint32_t a_is_dram = a_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    uint32_t b_is_dram = false;
    uint32_t c_is_dram = c_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);
    // WRITER KERNEL
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    if (b.has_value()) {
        b_is_dram = b_buffer->buffer_type() == tt_metal::BufferType::DRAM;
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;
    }

    // to maintain backward compatibility, old writer kernel only needs b_dtype
    auto writer_defines = make_dataflow_defines(b_dtype, a_dtype);
    writer_defines["SRC_SHARDED"] = b_sharded ? "1" : "0";
    writer_defines["DST_SHARDED"] = c_sharded ? "1" : "0";

    auto reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = a_sharded ? "1" : "0";
    reader_defines["SRC_SHARDED_B"] = b_sharded ? "1" : "0";

    // overwrite reader and write kernel names so that reader reads both and b and
    // writer does not read b. For the transition, it can choose the original kernels
    // or overwrite with new kernel here. If going back to old kernels, we can just
    // skip the if clause.
    if (b.has_value()) {
        kernel_config.reader_kernel = CMAKE_UNIQUE_NAMESPACE::get_reader_kernel_name_and_defines(
            operation_attributes.subtile_broadcast_type, reader_defines);
        writer_kernel = KernelName::WriterNoBcastNg;
    }
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(writer_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram, has_sharding}, std::move(writer_defines)));

    // READER KERNEL

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram, has_sharding, b_is_dram}, std::move(reader_defines)));

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src0interim_cb_index = tt::CBIndex::c_3;
    uint32_t src1interim_cb_index = tt::CBIndex::c_4;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    if (is_sfpu_op) {
        if (op_type != BinaryOpType::POWER) {
            unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src0interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[src1interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        } else {
            unpack_to_dest_mode[src0_cb_index] =
                (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
            unpack_to_dest_mode[src1_cb_index] =
                (b_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
            unpack_to_dest_mode[src0interim_cb_index] =
                (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
            unpack_to_dest_mode[src1interim_cb_index] =
                (b_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
        }
    }

    compute_kernel_defines["BCAST_INPUT"] = kernel_config.bcast_input_str();

    const uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(compute_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .compile_args = {num_tiles_per_cycle},
            .defines = std::move(compute_kernel_defines)});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        a_cb_handle,
        b_cb_handle,
        c_cb_handle,
        operation_attributes,
        tensor_args,
        c,
        set_runtime_args);

    return {
        std::move(program),
        {reader_kernel_id, writer_kernel_id, compute_kernel_id, a_cb_handle, b_cb_handle, c_cb_handle}};
}

void BinaryNgDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& c) {
    auto update_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        auto& all_args = GetRuntimeArgs(program, kernel_id);
        auto& core_args = all_args.at(core.x).at(core.y);
        std::copy(args.begin(), args.end(), core_args.data());
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.cb_src_a,
        cached_program.shared_variables.cb_src_b,
        cached_program.shared_variables.cb_src_c,
        operation_attributes,
        tensor_args,
        c,
        update_args);
}

}  // namespace ttnn::operations::binary_ng
