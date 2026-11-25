// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_layout_change_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

// Named rather than anonymous: unity builds merge this file with the other typecast factories.
namespace layout_change {

// Bounds L1 use: the IN and OUT DFBs each hold two blocks.
constexpr uint32_t kMaxBlockWidthTiles = 8;

constexpr const char* kComputeSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/typecast_layout_change.cpp";
constexpr const char* kRowMajorReaderSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_blocks.cpp";
constexpr const char* kRowMajorWriterSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_blocks.cpp";
constexpr const char* kTileReaderSource =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp";
constexpr const char* kTileWriterSource =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp";

uint32_t element_size(DataType dtype) { return tt::datum_size(datatype_to_dataformat_converter(dtype)); }

uint32_t width_tiles(const ttnn::Shape& shape) { return tt::div_up(shape[-1], tt::constants::TILE_WIDTH); }

// Each row of a block is read from or written to a stick at offset block_index * row bytes, which
// must keep the stick's buffer alignment. Returns 0 when no block width qualifies.
uint32_t select_block_width_tiles(uint32_t num_width_tiles, uint32_t row_major_element_size, uint32_t alignment) {
    for (auto width = std::min(num_width_tiles, kMaxBlockWidthTiles); width > 0; --width) {
        const auto block_row_bytes = width * tt::constants::TILE_WIDTH * row_major_element_size;
        if (num_width_tiles % width == 0 && block_row_bytes % alignment == 0) {
            return width;
        }
    }
    return 0;
}

uint32_t buffer_alignment(const MemoryConfig& memory_config) {
    return memory_config.buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
}

// Tile pages are read as dense 32x32 tiles, numbered over the tile-padded logical shape.
bool has_default_tile_padding(const TensorSpec& spec) {
    const auto& tile = spec.tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
        return false;
    }
    const auto& logical = spec.logical_shape();
    const auto& padded = spec.padded_shape();
    if (padded.rank() != logical.rank()) {
        return false;
    }
    const auto rank = logical.rank();
    for (size_t i = 0; i < rank; ++i) {
        const bool tiled_dim = i + 2 >= rank;
        const uint32_t expected = tiled_dim ? tt::round_up(logical[i], tt::constants::TILE_WIDTH) : logical[i];
        if (padded[i] != expected) {
            return false;
        }
    }
    return true;
}

struct BlockGeometry {
    uint32_t block_width_tiles;
    uint32_t height;
    uint32_t width;
    uint32_t tile_rows_per_batch;
    uint32_t blocks_per_tile_row;
    uint32_t num_blocks;
};

BlockGeometry block_geometry(const ttnn::Shape& shape, DataType row_major_dtype, const MemoryConfig& row_major_memory) {
    const auto rank = shape.rank();
    const uint32_t width = shape[-1];
    const uint32_t height = rank >= 2 ? shape[-2] : 1;
    const auto num_width_tiles = width_tiles(shape);
    const auto block_width_tiles =
        select_block_width_tiles(num_width_tiles, element_size(row_major_dtype), buffer_alignment(row_major_memory));
    const auto tile_rows_per_batch = tt::div_up(height, tt::constants::TILE_HEIGHT);
    const auto num_batches = static_cast<uint32_t>(shape.volume() / (static_cast<uint64_t>(height) * width));
    const auto blocks_per_tile_row = block_width_tiles == 0 ? 0 : num_width_tiles / block_width_tiles;
    return BlockGeometry{
        .block_width_tiles = block_width_tiles,
        .height = height,
        .width = width,
        .tile_rows_per_batch = tile_rows_per_batch,
        .blocks_per_tile_row = blocks_per_tile_row,
        .num_blocks = num_batches * tile_rows_per_batch * blocks_per_tile_row,
    };
}

}  // namespace layout_change

bool can_fuse_typecast_layout_change(
    const Tensor& input, const TensorSpec& output_spec, const std::optional<CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value() || input.layout() == output_spec.layout()) {
        return false;
    }
    const auto& shape = input.logical_shape();
    if (shape.rank() == 0 || shape.volume() == 0) {
        return false;
    }
    const bool tilize = input.layout() == Layout::ROW_MAJOR;
    const auto& row_major_spec = tilize ? input.tensor_spec() : output_spec;
    if (!layout_change::has_default_tile_padding(tilize ? output_spec : input.tensor_spec())) {
        return false;
    }
    // Sticks are addressed as one page per logical row.
    if (row_major_spec.padded_shape() != row_major_spec.logical_shape() ||
        row_major_spec.compute_page_size_bytes() !=
            shape[-1] * layout_change::element_size(row_major_spec.data_type())) {
        return false;
    }
    const auto geometry =
        layout_change::block_geometry(shape, row_major_spec.data_type(), row_major_spec.memory_config());
    return geometry.block_width_tiles != 0;
}

ttnn::device_operation::ProgramArtifacts TypecastLayoutChangeProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using ttnn::operations::data_movement::requires_32bit_dest;
    using ttnn::operations::data_movement::requires_unpack_to_dest;

    const auto& input = tensor_args.input;
    const bool tilize = input.layout() == Layout::ROW_MAJOR;
    const bool same_dtype = input.dtype() == output.dtype();
    const auto& row_major = tilize ? input : output;
    const auto geometry =
        layout_change::block_geometry(input.logical_shape(), row_major.dtype(), row_major.memory_config());
    TT_FATAL(geometry.block_width_tiles != 0, "Typecast layout change has no aligned block width for {}", input);
    const auto block_width_tiles = geometry.block_width_tiles;

    const auto input_format = cb_dataformat_for(input.dtype());
    const auto output_format = cb_dataformat_for(output.dtype());
    const bool enable_32_bit_dest =
        args.fp32_dest_acc_en || requires_32bit_dest(input.dtype()) || requires_32bit_dest(output.dtype());
    // Typecast runs on the tile side by default. With 32-bit Dest, a UInt16 row-major side is cast on
    // row-major chunks instead so tilize / untilize only see 32-bit or 8-bit formats. The tile side is
    // then never block-float: UInt16 <-> block-float does not enable 32-bit Dest.
    const bool cast_on_row_major = enable_32_bit_dest && row_major.dtype() == DataType::UINT16;
    const bool cast_first = tilize == cast_on_row_major;
    const auto mid_format = cast_first ? output_format : input_format;

    const auto* device = input.device();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_group_1, num_blocks_group_2] =
        split_work_to_cores(device->compute_with_storage_grid_size(), geometry.num_blocks);
    (void)num_cores;

    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName MID_DFB{"mid"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_GROUP_1{"compute_group_1"};
    const KernelSpecName COMPUTE_GROUP_2{"compute_group_2"};

    // Double-buffered so data movement overlaps compute.
    constexpr uint32_t num_buffered_blocks = 2;
    std::vector<DataflowBufferSpec> dataflow_buffers{
        DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = tt::tile_size(input_format),
            .num_entries = num_buffered_blocks * block_width_tiles,
            .data_format_metadata = input_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = tt::tile_size(output_format),
            .num_entries = num_buffered_blocks * block_width_tiles,
            .data_format_metadata = output_format,
        },
    };
    if (!same_dtype) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = MID_DFB,
            .entry_size = tt::tile_size(mid_format),
            .num_entries = block_width_tiles,
            .data_format_metadata = mid_format,
        });
    }

    const KernelSpec::CompileTimeArgs block_args{
        {"block_width_tiles", block_width_tiles},
        {"element_size", layout_change::element_size(row_major.dtype())},
        {"width", geometry.width},
        {"height", geometry.height},
        {"tile_rows_per_batch", geometry.tile_rows_per_batch},
        {"blocks_per_tile_row", geometry.blocks_per_tile_row},
    };
    const KernelSpec reader = tilize ? KernelSpec{
        .unique_id = READER,
        .source = layout_change::kRowMajorReaderSource,
        .dfb_bindings = {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args = block_args,
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "num_blocks"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    } : KernelSpec{
        .unique_id = READER,
        .source = layout_change::kTileReaderSource,
        .dfb_bindings = {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    const KernelSpec writer = tilize ? KernelSpec{
        .unique_id = WRITER,
        .source = layout_change::kTileWriterSource,
        .dfb_bindings = {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    } : KernelSpec{
        .unique_id = WRITER,
        .source = layout_change::kRowMajorWriterSource,
        .dfb_bindings = {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = block_args,
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "num_blocks"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Typecast keeps its own unpack rules on the DFB it consumes; tilize / untilize take 32-bit
    // formats straight to Dest so integers and fp32 pass through unchanged.
    const auto layout_dfb = (same_dtype || !cast_first) ? IN_DFB : MID_DFB;
    const auto layout_format = same_dtype ? input_format : mid_format;
    ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    if (requires_unpack_to_dest(layout_format)) {
        unpack_modes.emplace(layout_dfb, UnpackMode::UnpackToDest);
    }
    if (!same_dtype) {
        const auto cast_dfb = cast_first ? IN_DFB : MID_DFB;
        if (args.preserve_fp32_precision) {
            unpack_modes.emplace(cast_dfb, UnpackMode::UnpackToDest);
        } else if (input_format == tt::DataFormat::Float32) {
            unpack_modes.emplace(cast_dfb, UnpackMode::UnpackToSrc);
        }
    }
    const ComputeHardwareConfig compute_config{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = enable_32_bit_dest,
        .unpack_modes = std::move(unpack_modes),
        .config_1xx =
            ComputeHardwareConfig::Compute1XXConfig{
                .bfp_pack_precision_mode = args.bfp8_pack_precise ? Precision::Precise : Precision::Approximate,
            },
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (tilize) {
        compute_defines["TILIZE_INPUT"] = "1";
    }
    if (same_dtype) {
        compute_defines["SAME_DTYPE"] = "1";
    } else if (cast_first) {
        compute_defines["CAST_FIRST"] = "1";
    }
    const auto make_compute = [&](const KernelSpecName& id, uint32_t num_blocks) {
        std::vector<DFBBinding> bindings{
            DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
        };
        if (!same_dtype) {
            bindings.push_back(DFBBinding{
                .dfb_spec_name = MID_DFB, .accessor_name = "mid", .endpoint_type = DFBEndpointType::PRODUCER});
            bindings.push_back(DFBBinding{
                .dfb_spec_name = MID_DFB, .accessor_name = "mid", .endpoint_type = DFBEndpointType::CONSUMER});
        }
        return KernelSpec{
            .unique_id = id,
            .source = layout_change::kComputeSource,
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(bindings),
            .compile_time_args =
                {{"num_blocks", num_blocks},
                 {"block_width_tiles", block_width_tiles},
                 {"in_data_format", static_cast<uint32_t>(datatype_to_dataformat_converter(input.dtype()))},
                 {"out_data_format", static_cast<uint32_t>(datatype_to_dataformat_converter(output.dtype()))}},
            .hw_config = compute_config,
        };
    };

    Group<KernelSpec> kernels = {reader, writer, make_compute(COMPUTE_GROUP_1, num_blocks_group_1)};
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "typecast_layout_change_group_1",
        .kernels = {READER, WRITER, COMPUTE_GROUP_1},
        .target_nodes = core_group_1}};
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_GROUP_2, num_blocks_group_2));
        work_units.push_back(WorkUnitSpec{
            .name = "typecast_layout_change_group_2",
            .kernels = {READER, WRITER, COMPUTE_GROUP_2},
            .target_nodes = core_group_2});
    }

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t start_block = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        const auto num_blocks = core_group_1.contains(core) ? num_blocks_group_1 : num_blocks_group_2;
        const auto num_tiles = num_blocks * block_width_tiles;
        const auto start_tile = start_block * block_width_tiles;
        auto& row_major_run_args = tilize ? reader_run_args : writer_run_args;
        auto& tile_run_args = tilize ? writer_run_args : reader_run_args;
        AddRuntimeArgsForNode(
            row_major_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"num_blocks", num_blocks}});
        AddRuntimeArgsForNode(
            tile_run_args.runtime_arg_values, core, {{"num_pages", num_tiles}, {"start_id", start_tile}});
        start_block += num_blocks;
    }

    ProgramSpec spec{
        .name = "typecast_layout_change",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
