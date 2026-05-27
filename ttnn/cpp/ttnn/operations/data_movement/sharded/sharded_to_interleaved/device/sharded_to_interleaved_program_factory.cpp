// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md / METAL2_PORT_REPORT.md alongside this
// directory for the design decisions and friction notes. This factory satisfies
// ttnn::device_operation::ProgramSpecFactoryConcept (create_program_spec returning
// ttnn::device_operation::ProgramArtifacts).

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <functional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace {

// Stable names (constexpr const char*) per the Metal 2.0 conventions in
// kernel_spec.hpp / dataflow_buffer_spec.hpp / tensor_parameter.hpp.
constexpr const char* PROGRAM_ID = "sharded_to_interleaved";
constexpr const char* READER     = "reader";
constexpr const char* WRITER     = "writer";
constexpr const char* COMPUTE    = "compute";
constexpr const char* SRC_DFB    = "src_dfb";
constexpr const char* OUT_DFB    = "out_dfb";
constexpr const char* INPUT      = "input";
constexpr const char* OUTPUT     = "output";
constexpr const char* MAIN_WU    = "main";

// Helper: convert a CoreRangeSet from the tt::tt_metal coordinate vocabulary to the
// Metal 2.0 NodeRangeSet vocabulary. Gen1 (WH/BH) keeps the two coordinate spaces in
// 1:1 correspondence; this helper centralizes the translation.
NodeRangeSet to_node_range_set(const CoreRangeSet& crs) {
    std::vector<NodeRange> ranges;
    ranges.reserve(crs.ranges().size());
    for (const auto& cr : crs.ranges()) {
        ranges.emplace_back(NodeCoord{cr.start_coord.x, cr.start_coord.y},
                            NodeCoord{cr.end_coord.x,   cr.end_coord.y});
    }
    return NodeRangeSet(std::move(ranges));
}

NodeCoord to_node_coord(const CoreCoord& c) { return NodeCoord{c.x, c.y}; }

}  // namespace

ttnn::device_operation::ProgramArtifacts ShardedToInterleavedProgramFactory::create_program_spec(
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

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_cores_unpadded = num_cores;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    CoreCoord end_core = cores[num_cores - 1];
    if (output.layout() == Layout::TILE) {
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
        if (output.layout() == Layout::TILE) {
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
    NodeRangeSet used_nodes = to_node_range_set(used_cores);

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    // -----------------------------------------------------------
    // Build the spec resources
    // -----------------------------------------------------------

    // SRC DFB: borrowed-memory DFB on top of the input shard buffer.
    DataflowBufferSpec src_dfb{
        .unique_id            = SRC_DFB,
        .entry_size           = input_page_size,
        .num_entries          = num_units_per_shard,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from        = INPUT,
    };

    // OUT DFB: allocated only when data-format conversion is needed.
    std::optional<DataflowBufferSpec> out_dfb_opt;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
    if (convert_df) {
        out_dfb_opt = DataflowBufferSpec{
            .unique_id            = OUT_DFB,
            .entry_size           = output_page_size,
            .num_entries          = num_units_per_shard,
            .data_format_metadata = output_cb_data_format,
        };
    }

    // Reader: PRODUCER of SRC_DFB. The kernel does no real I/O — it just signals
    // the consumer that the borrowed-memory shard contents are available.
    KernelSpec reader{
        .unique_id  = READER,
        .source     = KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name      = SRC_DFB,
            .local_accessor_name = "shard",
            .endpoint_type      = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles_per_core"}},
        .config_spec = DataMovementConfiguration{
            .gen1_data_movement_config = DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default,
            },
        },
    };

    // Writer: CONSUMER of whichever DFB feeds the interleaved write — SRC_DFB when
    // no data-format conversion, OUT_DFB when convert_df. TensorBinding to OUTPUT.
    const std::string writer_dfb_name = convert_df ? OUT_DFB : SRC_DFB;
    const std::string writer_source = (input.layout() == Layout::TILE)
        ? std::string{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                      "writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp"}
        : std::string{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                      "writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp"};
    // RTA schema differs between TILE and ROW_MAJOR variants — see the two forked kernels.
    KernelSpec::RuntimeArgSchema writer_rtas;
    if (input.layout() == Layout::TILE) {
        writer_rtas.named_runtime_args = {
            "block_height_tiles",
            "block_width_tiles",
            "unpadded_block_height_tiles",
            "unpadded_block_width_tiles",
            "output_width_tiles",
            "block_num_tiles",
            "start_id_offset",
            "start_id_base",
        };
    } else {
        writer_rtas.named_runtime_args = {
            "block_height",
            "block_width_bytes",
            "padded_block_width_bytes",
            "input_width_offset_bytes",
            "start_id",
        };
    }
    KernelSpec writer{
        .unique_id = WRITER,
        .source    = KernelSpec::SourceFilePath{writer_source},
        .dfb_bindings = {{
            .dfb_spec_name      = writer_dfb_name,
            .local_accessor_name = "out",
            .endpoint_type      = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name         = "out",
        }},
        .runtime_arguments_schema = std::move(writer_rtas),
        .config_spec = DataMovementConfiguration{
            .gen1_data_movement_config = DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
            },
        },
    };

    // Compute (only when convert_df): copy tile-by-tile from SRC_DFB → OUT_DFB,
    // converting data formats along the way (via the tensix Dest register).
    std::optional<KernelSpec> compute_opt;
    if (convert_df) {
        compute_opt = KernelSpec{
            .unique_id = COMPUTE,
            .source    = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp"},
            .dfb_bindings = {
                {.dfb_spec_name = SRC_DFB, .local_accessor_name = "src",
                 .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = OUT_DFB, .local_accessor_name = "dst",
                 .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER},
            },
            .compile_time_arg_bindings = {{"per_core_tile_cnt", num_units_per_shard}},
            .config_spec               = ComputeConfiguration{},
        };
    }

    // Single WorkUnitSpec — every kernel runs on every used core.
    std::vector<KernelSpecName> wu_kernels = {READER, WRITER};
    if (convert_df) {
        wu_kernels.push_back(COMPUTE);
    }
    WorkUnitSpec main_wu{
        .unique_id    = MAIN_WU,
        .kernels      = std::move(wu_kernels),
        .target_nodes = used_nodes,
    };

    ProgramSpec spec{
        .program_id        = PROGRAM_ID,
        .kernels           = {std::move(reader), std::move(writer)},
        .dataflow_buffers  = {std::move(src_dfb)},
        .tensor_parameters = {
            {.unique_id = INPUT,  .spec = input.tensor_spec()},
            {.unique_id = OUTPUT, .spec = output.tensor_spec()},
        },
        .work_units        = {std::move(main_wu)},
    };
    if (convert_df) {
        spec.kernels.push_back(std::move(*compute_opt));
        spec.dataflow_buffers.push_back(std::move(*out_dfb_opt));
    }

    // -----------------------------------------------------------
    // Build the run params
    // -----------------------------------------------------------

    using KernelRunParams = ProgramRunParams::KernelRunParams;

    // Reader: identical RTAs on every used core.
    KernelRunParams reader_rp{.kernel_spec_name = READER};
    for (const auto& core_range : used_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"num_tiles_per_core", num_units_per_shard}},
            });
        }
    }

    // Writer: per-core RTAs encoding the slice of the output tensor this core owns.
    KernelRunParams writer_rp{.kernel_spec_name = WRITER};

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = cores[core_idx];
        uint32_t shard_height = num_units_per_shard_height;
        uint32_t shard_width = input.layout() == Layout::TILE ? num_units_per_shard_width : output_unit_size;
        if (input.layout() == Layout::TILE) {
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
            writer_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {
                    {"block_height_tiles",         num_units_per_shard_height},
                    {"block_width_tiles",          num_units_per_shard_width},
                    {"unpadded_block_height_tiles", shard_height},
                    {"unpadded_block_width_tiles",  shard_width},
                    {"output_width_tiles",         num_units_offset},
                    {"block_num_tiles",            num_units_per_shard},
                    {"start_id_offset",            curr_idx_h + curr_idx_w},
                    {"start_id_base",              starting_idx_h},
                },
            });

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
            // Legacy ROW_MAJOR factory pushed 7 RTAs in this order:
            //   [0] dst_buffer*           — BufferBinding (gone — TensorBinding takes over)
            //   [1] num_units_per_row     — unread by the kernel (legacy dead slot, dropped)
            //   [2] shard_height          — kernel RTA 2 → block_height
            //   [3] shard_width           — kernel RTA 3 → block_width_bytes
            //   [4] padded_shard_width    — kernel RTA 4 → padded_block_width_bytes
            //   [5] curr_idx_w            — kernel RTA 5 → input_width_offset_bytes
            //   [6] curr_idx_h            — kernel RTA 6 → start_id
            writer_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {
                    {"block_height",             shard_height},
                    {"block_width_bytes",        shard_width},
                    {"padded_block_width_bytes", padded_shard_width},
                    {"input_width_offset_bytes", curr_idx_w},
                    {"start_id",                 curr_idx_h},
                },
            });

            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }

    ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_rp));
    run_params.kernel_run_params.push_back(std::move(writer_rp));
    if (convert_df) {
        // Compute kernel has no RTAs — but every kernel in the spec must have a
        // KernelRunParams entry (even when empty).
        run_params.kernel_run_params.push_back(KernelRunParams{.kernel_spec_name = COMPUTE});
    }
    // The framework adapter (mesh_device_operation_adapter.hpp::collect_mesh_tensors)
    // matches each TensorArg's tensor against the MeshTensor returned by
    // ttnn::Tensor::mesh_tensor() — by pointer identity, not by value. So bind to
    // the underlying MeshTensor, not the wrapping Tensor.
    run_params.tensor_args = {
        {.tensor_parameter_name = INPUT,  .tensor = std::cref(input.mesh_tensor())},
        {.tensor_parameter_name = OUTPUT, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec       = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
