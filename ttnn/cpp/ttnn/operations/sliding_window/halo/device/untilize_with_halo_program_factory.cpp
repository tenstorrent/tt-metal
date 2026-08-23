// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

constexpr std::uint32_t untilize_block_size = 32;
constexpr bool enable_untilize_double_buffering = true;

namespace {

const tt::tt_metal::experimental::TensorParamName input_param{"input"};
const tt::tt_metal::experimental::TensorParamName output_param{"output"};
const tt::tt_metal::experimental::TensorParamName pad_config0_param{"pad_config0"};
const tt::tt_metal::experimental::TensorParamName pad_config1_param{"pad_config1"};
const tt::tt_metal::experimental::TensorParamName gather_config0_param{"gather_config0"};
const tt::tt_metal::experimental::TensorParamName gather_config1_param{"gather_config1"};

const tt::tt_metal::experimental::DFBSpecName input_dfb_name{"src"};
const tt::tt_metal::experimental::DFBSpecName output_dfb_name{"out"};
const tt::tt_metal::experimental::DFBSpecName untilize_out0_dfb_name{"untilize_out0"};
const tt::tt_metal::experimental::DFBSpecName untilize_out1_dfb_name{"untilize_out1"};

const tt::tt_metal::experimental::ScratchpadSpecName pad_scratch0_name{"pad0"};
const tt::tt_metal::experimental::ScratchpadSpecName pad_scratch1_name{"pad1"};
const tt::tt_metal::experimental::ScratchpadSpecName gather_scratch0_name{"gather_scratch0"};
const tt::tt_metal::experimental::ScratchpadSpecName gather_scratch1_name{"gather_scratch1"};
const tt::tt_metal::experimental::ScratchpadSpecName pad_config_scratch0_name{"pad_config_scratch0"};
const tt::tt_metal::experimental::ScratchpadSpecName pad_config_scratch1_name{"pad_config_scratch1"};

const tt::tt_metal::experimental::KernelSpecName reader0_kernel{"reader0"};
const tt::tt_metal::experimental::KernelSpecName reader1_kernel{"reader1"};
const tt::tt_metal::experimental::KernelSpecName compute_kernel{"compute"};

constexpr const char* reader_kernel_path =
    "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp";
constexpr const char* reader_dram_kernel_path =
    "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather_dram.cpp";
constexpr const char* compute_kernel_path =
    "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp";

tt::tt_metal::MeshTensor build_config_mesh_tensor(
    const std::vector<std::vector<std::uint16_t>>& config,
    const ttnn::operations::sliding_window::ParallelConfig& parallel_config,
    bool is_block_sharded,
    tt::tt_metal::distributed::MeshDevice* device,
    bool config_tensors_in_dram) {
    const auto host_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        config, parallel_config, config_tensors_in_dram);
    Tensor device_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        host_tensor, parallel_config, is_block_sharded, device, config_tensors_in_dram);
    TT_ASSERT(device_tensor.dtype() == DataType::UINT16);
    return device_tensor.device_storage().release_mesh_tensor();
}

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeWithHaloProgramFactory::create_program_artifacts(
    const HaloParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    auto* device = input_tensor.device();
    const std::uint32_t pad_val = operation_attributes.pad_val;
    const std::uint32_t ncores_nhw = operation_attributes.config.num_cores_nhw;
    const std::uint32_t max_out_nsticks_per_core = operation_attributes.max_out_nsticks_per_core;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const bool transpose_mcast = operation_attributes.transpose_mcast;
    const bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;
    const bool is_in_tiled = input_tensor.layout() == Layout::TILE;
    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_height_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    TT_ASSERT(output_tensor.buffer() != nullptr, "Output buffer should be allocated on device");

    const auto& input_shape = input_tensor.padded_shape();
    const auto in_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const std::uint32_t out_nbytes = tt::datum_size(out_df);
    const tt::tt_metal::CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;
    const auto shard_orientation = output_tensor.shard_spec().value().orientation;
    const auto input_shard_shape = input_tensor.shard_spec().value().shape;
    const auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1], "Expected input and output shard widths to match");

    const std::uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const std::uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    const std::uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], tt::constants::TILE_WIDTH);
    const std::uint32_t input_nblocks_per_core =
        tt::div_up(remapped_input_shard_shape_for_output_grid, tt::constants::TILE_HEIGHT);
    std::uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;
    std::uint32_t in_page_size = tt::tile_size(in_df);

    const std::uint32_t stick_nbytes = output_shard_shape[1] * out_nbytes;
    const std::uint32_t aligned_stick_nbytes = stick_nbytes % input_tensor.buffer()->alignment() == 0
                                                   ? stick_nbytes
                                                   : tt::round_up(stick_nbytes, input_tensor.buffer()->alignment());
    if (skip_untilize) {
        in_page_size = aligned_stick_nbytes;
        input_npages = input_shard_shape[0];
    }

    static_assert(untilize_block_size % tt::constants::TILE_HEIGHT == 0, "Untilize block size must be tile aligned");
    const std::uint32_t clamped_block_size_height =
        std::min(untilize_block_size, input_nblocks_per_core * tt::constants::TILE_HEIGHT);

    const bool is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);

    const auto pad_metadata = ttnn::operations::sliding_window::generate_pad_metadata(operation_attributes.config);
    const auto shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(operation_attributes.config);
    const std::uint32_t input_shard_height = input_tensor.memory_config().shard_spec()->shape[0];
    const auto tensor_metadata = ttnn::operations::sliding_window::generate_tensor_metadata(
        pad_metadata, operation_attributes.config, input_shard_height);
    const std::uint32_t num_cores_x = input_tensor.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
    auto kernel_config = ttnn::operations::sliding_window::generate_halo_kernel_config_tensors(
        tensor_metadata,
        shard_boundaries,
        is_block_sharded,
        transpose_mcast,
        /*remote_read=*/false,
        device,
        num_cores_x,
        is_in_tiled,
        untilize_block_size);

    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(4);
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.pad_config0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.pad_config1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.gather_config0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.gather_config1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    const tt::tt_metal::MeshTensor& pad_config0 = op_owned_tensors[0];
    const tt::tt_metal::MeshTensor& pad_config1 = op_owned_tensors[1];
    const tt::tt_metal::MeshTensor& gather_config0 = op_owned_tensors[2];
    const tt::tt_metal::MeshTensor& gather_config1 = op_owned_tensors[3];

    const auto number_of_blocks_per_core = ttnn::operations::sliding_window::remap_nhw_scalar_argument_across_full_grid(
        kernel_config.number_of_blocks_per_core, operation_attributes.parallel_config);

    constexpr std::uint32_t empty_padding_config_buffer_size = 4;
    const bool enable_padding =
        config_tensors_in_dram ||
        pad_config0.tensor_spec().compute_page_size_bytes() != empty_padding_config_buffer_size ||
        pad_config1.tensor_spec().compute_page_size_bytes() != empty_padding_config_buffer_size;
    const bool use_pad_scratch = enable_padding && pad_val != 0;

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorParameter> tensor_parameters = {
        tt::tt_metal::experimental::TensorParameter{.unique_id = input_param, .spec = input_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = output_param, .spec = output_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = gather_config0_param, .spec = gather_config0.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = gather_config1_param, .spec = gather_config1.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = pad_config0_param, .spec = pad_config0.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = pad_config1_param, .spec = pad_config1.tensor_spec()},
    };

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dataflow_buffers = {
        tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = input_dfb_name,
            .entry_size = in_page_size,
            .num_entries = input_npages,
            .data_format_metadata = in_df,
            .borrowed_from = input_param,
        },
        tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = output_dfb_name,
            .entry_size = aligned_stick_nbytes,
            .num_entries = max_out_nsticks_per_core,
            .data_format_metadata = out_df,
            .borrowed_from = output_param,
        },
    };
    if (!skip_untilize) {
        const std::uint32_t output_ntiles = (clamped_block_size_height / tt::constants::TILE_HEIGHT) * ntiles_per_block;
        const std::uint32_t untilize_pages = enable_untilize_double_buffering ? 2 * output_ntiles : output_ntiles;
        dataflow_buffers.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = untilize_out0_dfb_name,
            .entry_size = tt::tile_size(out_df),
            .num_entries = untilize_pages,
            .data_format_metadata = out_df,
        });
        dataflow_buffers.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = untilize_out1_dfb_name,
            .entry_size = tt::tile_size(out_df),
            .num_entries = untilize_pages,
            .data_format_metadata = out_df,
        });
    }

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::ScratchpadSpec> scratchpads;
    if (use_pad_scratch) {
        scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
            .unique_id = pad_scratch0_name, .size_per_node = aligned_stick_nbytes});
        scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
            .unique_id = pad_scratch1_name, .size_per_node = aligned_stick_nbytes});
    }
    if (config_tensors_in_dram) {
        scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
            .unique_id = gather_scratch0_name,
            .size_per_node = gather_config0.tensor_spec().compute_page_size_bytes()});
        scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
            .unique_id = gather_scratch1_name,
            .size_per_node = gather_config1.tensor_spec().compute_page_size_bytes()});
        if (enable_padding) {
            scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
                .unique_id = pad_config_scratch0_name,
                .size_per_node = pad_config0.tensor_spec().compute_page_size_bytes()});
            scratchpads.push_back(tt::tt_metal::experimental::ScratchpadSpec{
                .unique_id = pad_config_scratch1_name,
                .size_per_node = pad_config1.tensor_spec().compute_page_size_bytes()});
        }
    }

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::KernelSpec> kernels;
    if (!skip_untilize) {
        auto compute_hw_config =
            ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
        // Legacy left unpack_to_dest_mode at its default, so FP32 tiles are unpacked into SrcA/B.
        // Metal 2.0 requires that default to be stated explicitly when Dest is also 32-bit.
        if (tt::tt_metal::experimental::enable_32_bit_dest(compute_hw_config) && in_df == tt::DataFormat::Float32) {
            tt::tt_metal::experimental::unpack_modes(compute_hw_config)
                .emplace(input_dfb_name, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        kernels.push_back(tt::tt_metal::experimental::KernelSpec{
            .unique_id = compute_kernel,
            .source = std::filesystem::path(compute_kernel_path),
            // Preserve the legacy ComputeConfig default; tt::tt_metal::experimental::KernelSpec otherwise defaults to
            // O2.
            .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings =
                {tt::tt_metal::experimental::ConsumerOf(input_dfb_name, "src"),
                 tt::tt_metal::experimental::ProducerOf(untilize_out0_dfb_name, "untilize_out0"),
                 tt::tt_metal::experimental::ProducerOf(untilize_out1_dfb_name, "untilize_out1")},
            .compile_time_args =
                {{"tiles_per_row", ntiles_per_block},
                 {"block_size", clamped_block_size_height / tt::constants::TILE_HEIGHT}},
            .runtime_arg_schema = {.runtime_arg_names = {"total_blocks"}},
            .hw_config = std::move(compute_hw_config),
        });
    }

    constexpr std::uint32_t block_stride = 2;
    const auto make_reader = [&](const tt::tt_metal::experimental::KernelSpecName& name,
                                 const tt::tt_metal::experimental::DFBSpecName& untilize_dfb,
                                 const tt::tt_metal::experimental::ScratchpadSpecName& pad_scratch,
                                 const tt::tt_metal::experimental::ScratchpadSpecName& gather_scratch,
                                 const tt::tt_metal::experimental::ScratchpadSpecName& pad_config_scratch,
                                 const tt::tt_metal::experimental::TensorParamName& gather_config_name,
                                 const tt::tt_metal::experimental::TensorParamName& pad_config_name,
                                 std::uint32_t block_start_offset) {
        // Preserve legacy placement exactly. Reader 0 is the larger producer build and must
        // remain on BRISC: Wormhole NCRISC has a hard 16 KiB instruction-memory limit.
        tt::tt_metal::experimental::DataMovementHardwareConfig reader_hw_config =
            tt::tt_metal::experimental::DataMovementGen1Config{
                .processor = block_start_offset == 0 ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                                     : tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc =
                    block_start_offset == 0 ? tt::tt_metal::NOC::RISCV_0_default : tt::tt_metal::NOC::RISCV_1_default,
            };
        tt::tt_metal::experimental::KernelSpec reader{
            .unique_id = name,
            .source = std::filesystem::path(config_tensors_in_dram ? reader_dram_kernel_path : reader_kernel_path),
            .compile_time_args =
                {{"pad_val", pad_val},
                 {"input_npages", input_npages},
                 {"skip_untilize", static_cast<std::uint32_t>(skip_untilize)},
                 {"aligned_stick_nbytes", aligned_stick_nbytes},
                 {"is_block_sharded", static_cast<std::uint32_t>(is_block_sharded)},
                 {"is_col_major", static_cast<std::uint32_t>(transpose_mcast)},
                 {"is_width_sharded", static_cast<std::uint32_t>(is_width_sharded)},
                 {"block_size_height", clamped_block_size_height},
                 {"block_size_width_tiles", ntiles_per_block},
                 {"block_start_offset", block_start_offset},
                 {"block_stride", block_stride},
                 {"config_tensor_in_dram", static_cast<std::uint32_t>(config_tensors_in_dram)},
                 {"enable_padding", static_cast<std::uint32_t>(enable_padding)},
                 {"use_pad_scratch", static_cast<std::uint32_t>(use_pad_scratch)}},
            .hw_config = std::move(reader_hw_config),
        };
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = gather_config_name, .accessor_name = "gather_config"});
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = pad_config_name, .accessor_name = "padding_config"});
        reader.dfb_bindings.push_back(
            block_start_offset == 0 ? tt::tt_metal::experimental::ProducerOf(output_dfb_name, "out")
                                    : tt::tt_metal::experimental::ConsumerOf(output_dfb_name, "out"));
        if (block_start_offset == 0) {
            // Gen1 plain-CB compatibility: reader0 owns reserve/push/wait/pop bookkeeping for the
            // resident input shard, so it is the declared producer even though gather also reads it.
            auto binding = tt::tt_metal::experimental::ProducerOf(input_dfb_name, "src");
            if (skip_untilize) {
                binding.accessor_aliases.push_back("untilize_out");
            }
            reader.dfb_bindings.push_back(std::move(binding));
        } else if (skip_untilize) {
            // Reader1 only reads the resident pointer and deliberately performs no pop; its consumer
            // role completes the legacy shared-CB topology without adding storage or data movement.
            auto binding = tt::tt_metal::experimental::ConsumerOf(input_dfb_name, "src");
            binding.accessor_aliases.push_back("untilize_out");
            reader.dfb_bindings.push_back(std::move(binding));
        }
        if (!skip_untilize) {
            auto binding = tt::tt_metal::experimental::ConsumerOf(untilize_dfb, "untilize_out");
            if (block_start_offset != 0) {
                binding.accessor_aliases.push_back("src");
            }
            reader.dfb_bindings.push_back(std::move(binding));
        }
        reader.scratchpad_bindings.push_back(tt::tt_metal::experimental::ScratchpadBinding{
            .scratchpad_spec_name = pad_scratch,
            .accessor_name = "pad",
            .allow_unbound_for_constexpr_discard = !use_pad_scratch});
        if (config_tensors_in_dram) {
            reader.runtime_arg_schema = {.runtime_arg_names = {"config_read_index"}};
        }
        reader.scratchpad_bindings.push_back(tt::tt_metal::experimental::ScratchpadBinding{
            .scratchpad_spec_name = gather_scratch,
            .accessor_name = "gather_config",
            .allow_unbound_for_constexpr_discard = !config_tensors_in_dram});
        reader.scratchpad_bindings.push_back(tt::tt_metal::experimental::ScratchpadBinding{
            .scratchpad_spec_name = pad_config_scratch,
            .accessor_name = "padding_config",
            .allow_unbound_for_constexpr_discard = !(config_tensors_in_dram && enable_padding)});
        return reader;
    };

    kernels.push_back(make_reader(
        reader0_kernel,
        untilize_out0_dfb_name,
        pad_scratch0_name,
        gather_scratch0_name,
        pad_config_scratch0_name,
        gather_config0_param,
        pad_config0_param,
        0));
    kernels.push_back(make_reader(
        reader1_kernel,
        untilize_out1_dfb_name,
        pad_scratch1_name,
        gather_scratch1_name,
        pad_config_scratch1_name,
        gather_config1_param,
        pad_config1_param,
        1));

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::KernelSpecName> work_unit_kernels;
    if (!skip_untilize) {
        work_unit_kernels.push_back(compute_kernel);
    }
    work_unit_kernels.push_back(reader0_kernel);
    work_unit_kernels.push_back(reader1_kernel);

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "untilize_with_halo",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {tt::tt_metal::experimental::WorkUnitSpec{
            .name = "untilize_with_halo", .kernels = std::move(work_unit_kernels), .target_nodes = all_cores}},
    };

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    if (!skip_untilize) {
        tt::tt_metal::experimental::KernelRunArgs compute_args{.kernel = compute_kernel};
        for (std::size_t core_id = 0; core_id < cores.size(); ++core_id) {
            compute_args.runtime_arg_values["total_blocks"][tt::tt_metal::experimental::NodeCoord{
                cores[core_id].x, cores[core_id].y}] = static_cast<std::uint32_t>(number_of_blocks_per_core[core_id]);
        }
        run_args.kernel_run_args.push_back(std::move(compute_args));
    }

    tt::tt_metal::experimental::KernelRunArgs reader0_args{.kernel = reader0_kernel};
    tt::tt_metal::experimental::KernelRunArgs reader1_args{.kernel = reader1_kernel};
    if (config_tensors_in_dram) {
        for (std::uint32_t core_index = 0; core_index < cores.size(); ++core_index) {
            const auto& core = cores[core_index];
            std::uint32_t read_index = 0;
            if (is_height_sharded) {
                read_index = core_index;
            } else if (is_block_sharded) {
                read_index = is_rm_orientation ? core.y : core.x;
            }
            const tt::tt_metal::experimental::NodeCoord node{core.x, core.y};
            reader0_args.runtime_arg_values["config_read_index"][node] = read_index;
            reader1_args.runtime_arg_values["config_read_index"][node] = read_index;
        }
    }
    run_args.kernel_run_args.push_back(std::move(reader0_args));
    run_args.kernel_run_args.push_back(std::move(reader1_args));
    run_args.tensor_args = {
        {input_param, tt::tt_metal::experimental::TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {output_param, tt::tt_metal::experimental::TensorArgument{std::cref(output_tensor.mesh_tensor())}},
        {gather_config0_param, tt::tt_metal::experimental::TensorArgument{std::cref(gather_config0)}},
        {gather_config1_param, tt::tt_metal::experimental::TensorArgument{std::cref(gather_config1)}},
        {pad_config0_param, tt::tt_metal::experimental::TensorArgument{std::cref(pad_config0)}},
        {pad_config1_param, tt::tt_metal::experimental::TensorArgument{std::cref(pad_config1)}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim
