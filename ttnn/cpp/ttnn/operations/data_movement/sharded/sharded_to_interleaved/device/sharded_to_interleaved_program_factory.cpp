// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {
// Op-private kernel paths (unique file-scope constants to avoid unity-build collisions).
constexpr const char* S2I_READER_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/kernels/dataflow/"
    "reader_unary_sharded.cpp";
constexpr const char* S2I_WRITER_TILE_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/kernels/dataflow/"
    "writer_unary_sharded_blocks_interleaved_start_id.cpp";
constexpr const char* S2I_WRITER_RM_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/kernels/dataflow/"
    "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp";
constexpr const char* S2I_COMPUTE_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/kernels/compute/"
    "eltwise_copy.cpp";
}  // namespace

ttnn::device_operation::ProgramArtifacts ShardedToInterleavedProgramFactory::create_program_artifacts(
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_height = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_cores_unpadded = num_cores;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    CoreCoord end_core = cores[num_cores - 1];
    const bool is_tile = output.layout() == Layout::TILE;
    if (is_tile) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    // re-calculate end_core in the case shard grid is larger than used grid
    if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_unpadded = div_up(num_units_height, num_units_per_shard_height);
    } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
        if (is_tile) {
            num_cores_unpadded = div_up(num_units_per_row, num_units_per_shard_width);
        } else {
            num_cores_unpadded = div_up(num_units_per_row, output_unit_size);
        }
    }
    end_core = cores[num_cores_unpadded - 1];

    // Create CoreRangeSet for only the cores that will be used (fixes NOC error when grid > data)
    CoreRangeSet used_cores = num_cores_unpadded < num_cores
                                  ? select_from_corerangeset(all_cores, 0, num_cores_unpadded - 1, rm_orientation)
                                  : all_cores;

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);
    uint32_t num_input_units = num_units_per_shard;

    // -------------------------------------------------------------------------
    // DataflowBuffers
    // -------------------------------------------------------------------------
    // Sharded input DFB (formerly CB c_0): borrowed from the input tensor's L1 shard buffer.
    m2::DataflowBufferSpec in_dfb{
        .unique_id = m2::DFBSpecName{"in0"},
        .entry_size = input_page_size,
        .num_entries = num_input_units,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = m2::TensorParamName{"input"},
    };

    // Output DFB (formerly CB c_16): only allocated when converting data formats. It is a local,
    // Program-lifetime L1 buffer that the compute kernel narrows tiles into and the writer drains.
    std::optional<m2::DataflowBufferSpec> out_dfb;
    if (convert_df) {
        uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
        out_dfb = m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = output_cb_data_format,
        };
    }
    // The writer always references its output CB via the accessor token "out". When converting, that is
    // the dedicated output DFB; otherwise it is the (single) borrowed input DFB.
    const m2::DFBSpecName writer_dfb_name = convert_df ? m2::DFBSpecName{"out"} : m2::DFBSpecName{"in0"};

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    // Reader: sole PRODUCER of the borrowed input DFB. Binds "input" via a body-unused TensorAccessor to
    // satisfy the ProgramSpec referential-integrity check for the borrowed_from input DFB.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{S2I_READER_PATH},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"in0"}, "in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units_per_core"}},
        // Reader on NCRISC so the two data-movement kernels don't collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    // Writer: CONSUMER of the writer DFB; writes the interleaved output via the real ta::output accessor.
    m2::KernelSpec::RuntimeArgSchema writer_schema;
    if (is_tile) {
        writer_schema.runtime_arg_names = {
            "block_height_tiles",
            "block_width_tiles",
            "unpadded_block_height_tiles",
            "unpadded_block_width_tiles",
            "output_width_tiles",
            "block_num_tiles",
            "start_id_offset",
            "start_id_base"};
    } else {
        writer_schema.runtime_arg_names = {
            "block_height", "block_width_bytes", "padded_block_width_bytes", "input_width_offset_bytes", "start_id"};
    }
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{is_tile ? S2I_WRITER_TILE_PATH : S2I_WRITER_RM_PATH},
        .dfb_bindings = {m2::ConsumerOf(writer_dfb_name, "out")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"}},
        .runtime_arg_schema = writer_schema,
        // Writer on BRISC (default data-movement config).
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    // Compute (convert_df only): CONSUMER of the input DFB, PRODUCER of the output DFB.
    std::optional<m2::KernelSpec> compute;
    if (convert_df) {
        compute = m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"compute"},
            .source = std::filesystem::path{S2I_COMPUTE_PATH},
            .dfb_bindings =
                {m2::ConsumerOf(m2::DFBSpecName{"in0"}, "in0"), m2::ProducerOf(m2::DFBSpecName{"out"}, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_tile_cnt"}},
            .hw_config = m2::ComputeHardwareConfig{},
        };
    }

    // -------------------------------------------------------------------------
    // ProgramSpec assembly
    // -------------------------------------------------------------------------
    m2::ProgramSpec spec;
    spec.name = "sharded_to_interleaved";
    spec.kernels = {reader, writer};
    if (convert_df) {
        spec.kernels.push_back(*compute);
    }
    spec.dataflow_buffers = {in_dfb};
    if (convert_df) {
        spec.dataflow_buffers.push_back(*out_dfb);
    }
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};
    {
        m2::WorkUnitSpec wu{.name = "sharded_to_interleaved", .target_nodes = used_cores};
        wu.kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}};
        if (convert_df) {
            wu.kernels.push_back(m2::KernelSpecName{"compute"});
        }
        spec.work_units = {std::move(wu)};
    }

    // -------------------------------------------------------------------------
    // Run-args (degenerate concept: the complete set)
    // -------------------------------------------------------------------------
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};

    // Reader runtime args: identical on every used core (publish the resident shard).
    for (const auto& core_range : used_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_args.runtime_arg_values.push_back({core, {{"num_units_per_core", num_units_per_shard}}});
            if (convert_df) {
                compute_args.runtime_arg_values.push_back({core, {{"per_core_tile_cnt", num_units_per_shard}}});
            }
        }
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = cores[core_idx];
        uint32_t shard_height = num_units_per_shard_height;
        uint32_t shard_width = is_tile ? num_units_per_shard_width : output_unit_size;
        if (is_tile) {
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            writer_args.runtime_arg_values.push_back(
                {core,
                 {{"block_height_tiles", num_units_per_shard_height},
                  {"block_width_tiles", num_units_per_shard_width},
                  {"unpadded_block_height_tiles", shard_height},
                  {"unpadded_block_width_tiles", shard_width},
                  {"output_width_tiles", num_units_offset},
                  {"block_num_tiles", num_units_per_shard},
                  {"start_id_offset", curr_idx_h + curr_idx_w},
                  {"start_id_base", starting_idx_h}}});

            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            uint32_t l1_alignment = hal::get_l1_alignment();
            uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
            if (is_blackhole or is_l1_aligned) {
                if (!dst_is_dram or is_l1_aligned) {
                    padded_shard_width = tt::align(output_unit_size, l1_alignment);
                }
            }
            writer_args.runtime_arg_values.push_back(
                {core,
                 {{"block_height", shard_height},
                  {"block_width_bytes", shard_width},
                  {"padded_block_width_bytes", padded_shard_width},
                  {"input_width_offset_bytes", curr_idx_w},
                  {"start_id", curr_idx_h}}});

            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_args));
    run_params.kernel_run_args.push_back(std::move(writer_args));
    if (convert_df) {
        run_params.kernel_run_args.push_back(std::move(compute_args));
    }
    run_params.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    run_params.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::prim
