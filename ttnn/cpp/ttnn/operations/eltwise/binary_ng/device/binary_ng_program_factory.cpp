// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::binary_ng;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
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

std::tuple<std::optional<ShardSpec>, TensorMemoryLayout> get_shard_spec(
    const Tensor& a, const std::optional<Tensor>& b, const Tensor& c) {
    if (a.memory_config().is_sharded()) {
        return {a.shard_spec().value(), a.memory_config().memory_layout};
    } else if (b.has_value() && b->memory_config().is_sharded()) {
        return {b->shard_spec().value(), b->memory_config().memory_layout};
    } else if (c.memory_config().is_sharded()) {
        return {c.shard_spec().value(), c.memory_config().memory_layout};
    }

    return {std::nullopt, TensorMemoryLayout::INTERLEAVED};
}

uint32_t get_shards_per_width(
    const CoreRangeSet& all_cores, TensorMemoryLayout memory_layout, ShardOrientation orientation) {
    auto num_cores = all_cores.num_cores();
    if (memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }

    if (memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }

    const auto& bbox = all_cores.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    const BinaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    BinaryNgDeviceOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    const auto ashape = a.padded_shape();
    const auto bshape = b.has_value() ? b->padded_shape() : SimpleShape{1, 1};
    const auto cshape = c.padded_shape();

    const auto [aN, aC, aHt, aWt] = get_shape_dims(a);
    const auto [bN, bC, bHt, bWt] = b.has_value() ? get_shape_dims(*b) : std::tuple{1u, 1u, 1u, 1u};
    const auto [cN, cC, cHt, cWt] = get_shape_dims(c);
    const uint32_t cHt_unrolled = cN * cC * cHt;

    bool row_major = true;
    const auto [shard_spec, memory_layout] = get_shard_spec(a, b, c);
    const bool has_sharding = shard_spec.has_value();

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    // This will run the original work/core distribution algorithms that are specifically for this setup, as these
    // are faster than the generic work/core distribution algorithms that work on arbitrary CoreRangeSets
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    const auto& all_device_cores = operation_attributes.worker_grid;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (has_sharding) {
                const auto& shard_start_coord = shard_spec->grid.ranges()[0].start_coord;
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
    CoreCoord end_core;
    std::vector<CoreCoord> cores;

    const uint32_t tile_height = c.tensor_spec().tile().get_height();
    const uint32_t tile_width = c.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;
    const uint32_t num_output_tiles = c.volume() / tile_hw;

    uint32_t shard_height = cHt_unrolled, shard_width = cWt;
    uint32_t last_shard_height = shard_height, last_shard_width = shard_width;

    if (has_sharding) {
        core_group_1 = shard_spec->grid;
        num_tiles_per_core_group_1 = shard_spec->numel() / tile_hw;
        row_major = shard_spec->orientation == ShardOrientation::ROW_MAJOR;
        shard_height = shard_spec->shape[0] / tile_height;
        shard_width = shard_spec->shape[1] / tile_width;
        end_core = (*shard_spec->grid.ranges().begin()).end_coord;
        last_shard_height = shard_height - (tt::round_up(cHt_unrolled, shard_height) - cHt_unrolled);
        last_shard_width = shard_width - (tt::round_up(cWt, shard_width) - cWt);

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
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid, num_output_tiles, row_major);
        cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, num_output_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }

    auto num_shards_per_width =
        has_sharding ? get_shards_per_width(shard_spec->grid, memory_layout, shard_spec->orientation) : 0u;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 10>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 11>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
            continue;
        }

        uint32_t start_id = 0;
        uint32_t current_shard_height = 0;
        uint32_t current_shard_width = 0;
        if (has_sharding) {
            current_shard_height = shard_height;
            current_shard_width = shard_width;
            if (row_major) {
                if (core.x == end_core.x) {
                    current_shard_width = last_shard_width;
                }
                if (core.y == end_core.y) {
                    current_shard_height = last_shard_height;
                }
            } else {
                if (core.y == end_core.y) {
                    current_shard_width = last_shard_width;
                }
                if (core.x == end_core.x) {
                    current_shard_height = last_shard_height;
                }
            }
            start_id = (i / num_shards_per_width) * (shard_height * cWt) + (i % num_shards_per_width) * shard_width;
            num_tiles_per_core = current_shard_height * current_shard_width;
        } else {
            start_id = start_tile_id;
        }

        std::array reader_runtime_args = {
            a.buffer()->address(),
            start_id,
            num_tiles_per_core,
            current_shard_width,
            aHt * aWt * aC * (aN > 1),
            aHt * aWt * (aC > 1),
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        if (b.has_value()) {
            std::array writer_runtime_args = {
                b->buffer()->address(),
                c.buffer()->address(),
                start_id,
                num_tiles_per_core,
                current_shard_width,
                bHt * bWt * bC * (bN > 1),
                bHt * bWt * (bC > 1),
                cN,
                cC,
                cHt,
                cWt};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            auto [freq, counter] =
                calculate_compute_kernel_args(operation_attributes.subtile_broadcast_type, start_id, cHt, cWt);
            std::array compute_runtime_args = {num_tiles_per_core, freq, counter};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            const auto scalar = *operation_attributes.scalar;
            const auto packed_scalar = a.get_dtype() == DataType::FLOAT32 ? std::bit_cast<uint32_t>(scalar)
                                       : a.get_dtype() == DataType::INT32
                                           ? std::bit_cast<uint32_t>(static_cast<int32_t>(scalar))
                                           : pack_two_bfloat16_into_uint32({scalar, scalar});
            std::array writer_runtime_args = {
                packed_scalar,
                c.buffer()->address(),
                start_id,
                num_tiles_per_core,
                current_shard_width,
                cN,
                cC,
                cHt,
                cWt,
                0u,
                0u};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {num_tiles_per_core, 0u, 0u};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        start_tile_id += num_tiles_per_core;
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
    auto is_sfpu_op = operation_attributes.is_sfpu;

    auto program = CreateProgram();
    auto* device = a.device();

    auto [shard_spec, memory_layout] = CMAKE_UNIQUE_NAMESPACE::get_shard_spec(a, b, c);
    const bool has_sharding = shard_spec.has_value();
    uint32_t num_tiles_per_shard = has_sharding ? shard_spec->numel() / a.tensor_spec().tile().get_tile_hw() : 0;

    auto a_data_format = datatype_to_dataformat_converter(a.get_dtype());
    auto b_data_format = b.has_value() ? datatype_to_dataformat_converter(b->get_dtype())
                         : is_sfpu_op  ? datatype_to_dataformat_converter(a.get_dtype())
                                       : DataFormat::Float16_b;
    auto c_data_format = datatype_to_dataformat_converter(c.get_dtype());

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    const auto& all_device_cores = operation_attributes.worker_grid;

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = b.has_value() ? b->buffer() : nullptr;
    Buffer* c_buffer = c.buffer();

    auto op_type = operation_attributes.binary_op_type;

    const auto op_config = is_sfpu_op ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>)
                                      : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>);

    auto compute_kernel_defines = op_config.as_defines(a.get_dtype());

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

        add_activation_defines(compute_kernel_defines, lhs_activations, "LHS");
        add_activation_defines(compute_kernel_defines, rhs_activations, "RHS");

        if (lhs_activations.empty() and rhs_activations.empty() and post_activations.size() == 1 and
            post_activations[0].op_type == unary::UnaryOpType::RELU) {
            compute_kernel_defines["PACK_RELU"] = "1";
            compute_kernel_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_kernel_defines);
        } else {
            add_activation_defines(compute_kernel_defines, post_activations, "POST");
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
        a_sharded ? num_tiles_per_shard : 2,
        a_data_format,
        a_sharded ? a_buffer : nullptr);

    if (not compute_kernel_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty()) {
        auto a_intermediate_format = is_sfpu_op   ? a_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : a_data_format;
        uint32_t a_intermediate_single_tile_size = tt_metal::detail::TileSize(a_intermediate_format);
        auto [a_cb_interim, a_cb_interim_handle] = create_cb(
            tt::CBIndex::c_3, program, all_device_cores, a_intermediate_single_tile_size, 1, a_intermediate_format);
    }

    // If b is a scalar, we only need one tile in the CB
    auto [b_cb, b_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        b_single_tile_size,
        b_buffer == nullptr ? 1 : (b_sharded ? num_tiles_per_shard : 2),
        b_data_format,
        b_sharded ? b_buffer : nullptr);

    if (not compute_kernel_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty()) {
        auto b_intermediate_format = is_sfpu_op   ? b_data_format
                                     : op_has_exp ? tt::DataFormat::Float16_b
                                                  : b_data_format;
        uint32_t b_intermediate_single_tile_size = tt_metal::detail::TileSize(b_intermediate_format);
        auto [b_cb_interim, b_cb_interim_handle] = create_cb(
            tt::CBIndex::c_4, program, all_device_cores, b_intermediate_single_tile_size, 1, b_intermediate_format);
    }

    auto [c_cb, c_cb_handle] = create_cb(
        tt::CBIndex::c_2,
        program,
        all_device_cores,
        c_single_tile_size,
        c_sharded ? num_tiles_per_shard : 2,
        c_data_format,
        c_sharded ? c_buffer : nullptr);

    uint32_t a_is_dram = a_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    uint32_t b_is_dram = false;
    uint32_t c_is_dram = c_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);

    std::map<std::string, std::string> dataflow_defines;
    if (is_sfpu_op && a.get_dtype() == DataType::FLOAT32) {
        dataflow_defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        dataflow_defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<float>";
        dataflow_defines["FILL_WITH_VALUE_FLOAT"] = "fill_with_val<1024, float>";
    } else if (is_sfpu_op && a.get_dtype() == DataType::INT32) {
        dataflow_defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        dataflow_defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<int32_t>";
        dataflow_defines["FILL_WITH_VALUE"] = "fill_with_val<1024, int32_t>";
    } else {
        dataflow_defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column_bfloat16";
        dataflow_defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row_bfloat16";
        dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element_bfloat16";
        dataflow_defines["FILL_WITH_VALUE"] = "fill_with_val_bfloat16";
    }
    auto reader_defines = dataflow_defines;
    reader_defines["SRC_SHARDED"] = a_sharded ? "1" : "0";

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram, has_sharding}, std::move(reader_defines)));

    // WRITER KERNEL
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    if (b.has_value()) {
        b_is_dram = b_buffer->buffer_type() == tt_metal::BufferType::DRAM;
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;
    }
    auto writer_defines = dataflow_defines;
    writer_defines["SRC_SHARDED"] = b_sharded ? "1" : "0";
    writer_defines["DST_SHARDED"] = c_sharded ? "1" : "0";

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(writer_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram, has_sharding}, std::move(writer_defines)));

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src0interim_cb_index = tt::CBIndex::c_3;
    uint32_t src1interim_cb_index = tt::CBIndex::c_4;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (is_sfpu_op) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src0interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src1interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    compute_kernel_defines["BCAST_INPUT"] = kernel_config.bcast_input_str();

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(compute_kernel, is_sfpu_op),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .defines = std::move(compute_kernel_defines)});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        operation_attributes,
        tensor_args,
        c,
        set_runtime_args);

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id}};
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
        operation_attributes,
        tensor_args,
        c,
        update_args);
}

}  // namespace ttnn::operations::binary_ng
