// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_multi_core_program_factory.hpp"
#include <bit>
#include <filesystem>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Legacy always ran with the ComputeConfigDescriptor defaults for math_approx_mode (false)
// and dst_full_sync_en (false): the factory resolved those knobs but never copied them onto
// the descriptor. Reproduce those defaults explicitly instead of honoring the caller's
// resolved values, so behavior is unchanged.
ComputeHardwareConfig make_prefill_compute_hw_config(tt::ARCH arch, const ttnn::DeviceComputeKernelConfig& config) {
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(arch, config);
    sfpu_precision_mode(compute_hw) = Precision::Precise;  // legacy default: math_approx_mode = false
    double_buffer_dest(compute_hw) = true;                 // legacy default: dst_full_sync_en = false (inverted)
    return compute_hw;
}

ttnn::device_operation::ProgramArtifacts create_single_tile_prefill_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const auto& input_mt = input.mesh_tensor();
    const auto& cos_mt = cos.mesh_tensor();
    const auto& sin_mt = sin.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    tt::DataFormat cos_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    tt::DataFormat sin_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    tt::DataFormat trans_mat_data_format = input_data_format == tt::DataFormat::Bfp8_b    ? tt::DataFormat::Bfp8_b
                                           : input_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32
                                                                                          : tt::DataFormat::Float16_b;
    uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_data_format);

    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    constexpr uint32_t Wt = 1;
    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_rows_per_core_group_1, num_rows_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_input_tiles, num_output_tiles;

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_rows_per_core_group_2 = 0;
        num_input_tiles = in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        num_output_tiles = out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major);
        num_input_tiles = 2 * Wt;
        num_output_tiles = num_input_tiles;
    }

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    // DFB accessor names follow the shared Metal 2.0 compute kernel
    // (../rotary_embedding/device/kernels/compute/rotary_embedding_single_tile_metal2.cpp),
    // which owns the dfb::/args:: vocabulary on this path.
    const DFBSpecName IN{"in"};                                // legacy CB c_0 (input; borrowed when in_sharded)
    const DFBSpecName TRANS_MAT{"trans_mat"};                  // legacy CB c_1 (rotate-half transform matrix)
    const DFBSpecName COS_DFB{"cos"};                          // legacy CB c_2 (cos)
    const DFBSpecName SIN_DFB{"sin"};                          // legacy CB c_3 (sin)
    const DFBSpecName ROTATED_IN_INTERM{"rotated_in_interm"};  // legacy CB c_24
    const DFBSpecName COS_INTERM{"cos_interm"};                // legacy CB c_25
    const DFBSpecName SIN_INTERM{"sin_interm"};                // legacy CB c_26
    const DFBSpecName OUT{"out"};                              // legacy CB c_16 (output; borrowed when out_sharded)
    const TensorParamName INPUT{"input"};
    const TensorParamName COS_CACHE{"cos_cache"};
    const TensorParamName SIN_CACHE{"sin_cache"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    // ---- DataflowBuffers ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
    };
    if (in_sharded) {
        in_dfb.borrowed_from = INPUT;
    }
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
    };
    if (out_sharded) {
        out_dfb.borrowed_from = OUTPUT;
    }
    Group<DataflowBufferSpec> dataflow_buffers = {
        std::move(in_dfb),
        DataflowBufferSpec{
            .unique_id = TRANS_MAT,
            .entry_size = trans_mat_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = trans_mat_data_format,
        },
        DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = cos_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = sin_data_format,
        },
        DataflowBufferSpec{
            .unique_id = ROTATED_IN_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = COS_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SIN_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = input_data_format,
        },
        std::move(out_dfb),
    };

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw =
        make_prefill_compute_hw_config(device->arch(), operation_attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        // With a 32-bit Dest, Metal 2.0 requires an explicit UnpackMode for every Float32 DFB the
        // compute kernel consumes. Legacy set no unpack_to_dest_mode (Default == UnpackToSrc), so
        // the explicit entries carry that same value.
        auto& modes = unpack_modes(compute_hw);
        auto add_entry_if_fp32 = [&](const DFBSpecName& name, tt::DataFormat format) {
            if (format == tt::DataFormat::Float32) {
                modes.emplace(name, UnpackMode::UnpackToSrc);
            }
        };
        add_entry_if_fp32(IN, input_data_format);
        add_entry_if_fp32(COS_DFB, cos_data_format);
        add_entry_if_fp32(SIN_DFB, sin_data_format);
        add_entry_if_fp32(TRANS_MAT, trans_mat_data_format);
        add_entry_if_fp32(ROTATED_IN_INTERM, input_data_format);
        add_entry_if_fp32(COS_INTERM, input_data_format);
        add_entry_if_fp32(SIN_INTERM, input_data_format);
    }

    // ---- Kernels ----
    // Reader source is selected on in_sharded: the sharded variant takes no src accessor
    // (the input shard is resident; the reader just cursor-advances the borrowed DFB).
    Group<TensorBinding> reader_tensor_bindings;
    if (in_sharded) {
        reader_tensor_bindings = {
            TensorBinding{.tensor_parameter_name = COS_CACHE, .accessor_name = "cos"},
            TensorBinding{.tensor_parameter_name = SIN_CACHE, .accessor_name = "sin"},
        };
    } else {
        reader_tensor_bindings = {
            TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
            TensorBinding{.tensor_parameter_name = COS_CACHE, .accessor_name = "cos"},
            TensorBinding{.tensor_parameter_name = SIN_CACHE, .accessor_name = "sin"},
        };
    }
    KernelSpec::RuntimeArgSchema reader_arg_schema;
    if (in_sharded) {
        reader_arg_schema = {.runtime_arg_names = {"num_rows", "start_row_id", "cos_sin_start_id"}};
    } else {
        reader_arg_schema = {.runtime_arg_names = {"num_rows", "start_id", "start_row_id", "cos_sin_start_id"}};
    }
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            in_sharded ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/"
                         "dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id_sharded.cpp"
                       : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/"
                         "dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SIN_DFB,
                    .accessor_name = "sin",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TRANS_MAT,
                    .accessor_name = "trans_mat",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args =
            {
                {"Ht", Ht},
                {"HtWt", HtWt},
            },
        .runtime_arg_schema = std::move(reader_arg_schema),
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec::CompilerOptions writer_compiler_options;
    if (out_sharded) {
        writer_compiler_options.defines.emplace("OUT_SHARDED", "1");
    }
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
            "writer_rotary_embedding_hf_interleaved.cpp"),
        .compiler_options = std::move(writer_compiler_options),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // The same compute source runs on both work-split groups, differing only in the per-group
    // num_rows compile-time arg — preserved as two KernelSpecs, never demoted to a runtime arg.
    // The kernel source is the sibling rotary_embedding op's shared Metal 2.0 fork; this op
    // supplies no DECODE_MODE define, so only the prefill path of that kernel is compiled.
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_rows_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
                "rotary_embedding_single_tile_metal2.cpp"),
            // Legacy compute default opt_level is O3; Metal 2.0 defaults to O2 — set explicitly to preserve.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN,
                        .accessor_name = "in",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_DFB,
                        .accessor_name = "cos",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_DFB,
                        .accessor_name = "sin",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = TRANS_MAT,
                        .accessor_name = "trans_mat",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ROTATED_IN_INTERM,
                        .accessor_name = "rotated_in_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ROTATED_IN_INTERM,
                        .accessor_name = "rotated_in_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_INTERM,
                        .accessor_name = "cos_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_INTERM,
                        .accessor_name = "cos_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_INTERM,
                        .accessor_name = "sin_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_INTERM,
                        .accessor_name = "sin_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"num_rows", num_rows_per_core},
                },
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute_spec(COMPUTE_G1, num_rows_per_core_group_1));

    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1},
    };
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute_spec(COMPUTE_G2, num_rows_per_core_group_2));
        work_units.push_back(
            WorkUnitSpec{.name = "group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args ----
    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t cos_sin_start_id = num_tiles_written % HtWt;

        if (in_sharded) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_rows", num_rows_per_core},
                 {"start_row_id", num_tiles_written / Wt % Ht},
                 {"cos_sin_start_id", cos_sin_start_id}});
        } else {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_rows", num_rows_per_core},
                 {"start_id", num_tiles_written},
                 {"start_row_id", num_tiles_written / Wt % Ht},
                 {"cos_sin_start_id", cos_sin_start_id}});
        }

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_rows_per_core * Wt}, {"start_id", num_tiles_written}});
        num_tiles_written += num_rows_per_core * Wt;
    }

    // ---- Assemble spec + run-args ----
    ProgramSpec spec{
        .name = "rotary_embedding_hf_single_tile_prefill",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = COS_CACHE, .spec = cos.tensor_spec()},
                TensorParameter{.unique_id = SIN_CACHE, .spec = sin.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mt},
        {COS_CACHE, cos_mt},
        {SIN_CACHE, sin_mt},
        {OUTPUT, output_mt},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts create_multi_tile_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const auto& input_mt = input.mesh_tensor();
    const auto& cos_mt = cos.mesh_tensor();
    const auto& sin_mt = sin.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    tt::DataFormat cos_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    tt::DataFormat sin_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    tt::DataFormat scalar_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_data_format);

    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_rows_per_core_group_1, num_rows_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_input_tiles, num_output_tiles;

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_rows_per_core_group_2 = 0;
        num_input_tiles = in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        num_output_tiles = out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major);
        num_input_tiles = 2 * Wt;
        num_output_tiles = num_input_tiles;
    }

    uint32_t num_rotated_input_tiles = 2 * Wt;
    uint32_t num_cos_sin_tiles = 2 * Wt;
    uint32_t num_scalar_tiles = 1;
    uint32_t num_interm_tiles = 1;

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN{"in"};                                // legacy CB c_0 (input; borrowed when in_sharded)
    const DFBSpecName ROTATED_IN{"rotated_in"};                // legacy CB c_1 (rotated-half input read stream)
    const DFBSpecName COS_DFB{"cos"};                          // legacy CB c_2 (cos)
    const DFBSpecName SIN_DFB{"sin"};                          // legacy CB c_3 (sin)
    const DFBSpecName SCALAR{"scalar"};                        // legacy CB c_4 (-1.0 rotate-half scalar)
    const DFBSpecName ROTATED_IN_INTERM{"rotated_in_interm"};  // legacy CB c_24
    const DFBSpecName COS_INTERM{"cos_interm"};                // legacy CB c_25
    const DFBSpecName SIN_INTERM{"sin_interm"};                // legacy CB c_26
    const DFBSpecName OUT{"out"};                              // legacy CB c_16 (output; borrowed when out_sharded)
    const TensorParamName INPUT{"input"};
    const TensorParamName COS_CACHE{"cos_cache"};
    const TensorParamName SIN_CACHE{"sin_cache"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    // ---- DataflowBuffers ----
    // When in_sharded, the input DFB borrows the resident input shard; the reader still
    // NoC-reads the src tiles into it through the accessor (intentional legacy behavior —
    // a self-aliasing copy of resident tiles into their own storage). Preserved byte-for-byte.
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
    };
    if (in_sharded) {
        in_dfb.borrowed_from = INPUT;
    }
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
    };
    if (out_sharded) {
        out_dfb.borrowed_from = OUTPUT;
    }
    Group<DataflowBufferSpec> dataflow_buffers = {
        std::move(in_dfb),
        DataflowBufferSpec{
            .unique_id = ROTATED_IN,
            .entry_size = input_single_tile_size,
            .num_entries = num_rotated_input_tiles,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = cos_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = sin_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SCALAR,
            .entry_size = scalar_single_tile_size,
            .num_entries = num_scalar_tiles,
            .data_format_metadata = scalar_data_format,
        },
        DataflowBufferSpec{
            .unique_id = ROTATED_IN_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = COS_INTERM,
            .entry_size = cos_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = cos_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SIN_INTERM,
            .entry_size = sin_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = sin_data_format,
        },
        std::move(out_dfb),
    };

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw =
        make_prefill_compute_hw_config(device->arch(), operation_attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        // With a 32-bit Dest, Metal 2.0 requires an explicit UnpackMode for every Float32 DFB the
        // compute kernel consumes. Legacy set no unpack_to_dest_mode (Default == UnpackToSrc), so
        // the explicit entries carry that same value. (The scalar DFB is always Float16_b.)
        auto& modes = unpack_modes(compute_hw);
        auto add_entry_if_fp32 = [&](const DFBSpecName& name, tt::DataFormat format) {
            if (format == tt::DataFormat::Float32) {
                modes.emplace(name, UnpackMode::UnpackToSrc);
            }
        };
        add_entry_if_fp32(IN, input_data_format);
        add_entry_if_fp32(ROTATED_IN, input_data_format);
        add_entry_if_fp32(COS_DFB, cos_data_format);
        add_entry_if_fp32(SIN_DFB, sin_data_format);
        add_entry_if_fp32(ROTATED_IN_INTERM, input_data_format);
        add_entry_if_fp32(COS_INTERM, cos_data_format);
        add_entry_if_fp32(SIN_INTERM, sin_data_format);
    }

    // ---- Kernels ----
    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
            "reader_rotary_embedding_hf_interleaved.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ROTATED_IN,
                    .accessor_name = "rotated_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SIN_DFB,
                    .accessor_name = "sin",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALAR,
                    .accessor_name = "scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
                TensorBinding{.tensor_parameter_name = COS_CACHE, .accessor_name = "cos"},
                TensorBinding{.tensor_parameter_name = SIN_CACHE, .accessor_name = "sin"},
            },
        .compile_time_args =
            {
                {"scalar_value", bfloat16_scalar},
                {"Ht", Ht},
                {"Wt", Wt},
                {"HtWt", HtWt},
                {"half_Wt", half_Wt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_id", "start_row_id", "cos_sin_start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec::CompilerOptions writer_compiler_options;
    if (out_sharded) {
        writer_compiler_options.defines.emplace("OUT_SHARDED", "1");
    }
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
            "writer_rotary_embedding_hf_interleaved.cpp"),
        .compiler_options = std::move(writer_compiler_options),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // The same compute source runs on both work-split groups, differing only in the per-group
    // num_rows compile-time arg — preserved as two KernelSpecs, never demoted to a runtime arg.
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_rows_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
                "rotary_embedding_hf.cpp"),
            // Legacy compute default opt_level is O3; Metal 2.0 defaults to O2 — set explicitly to preserve.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN,
                        .accessor_name = "in",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ROTATED_IN,
                        .accessor_name = "rotated_in",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_DFB,
                        .accessor_name = "cos",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_DFB,
                        .accessor_name = "sin",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SCALAR,
                        .accessor_name = "scalar",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ROTATED_IN_INTERM,
                        .accessor_name = "rotated_in_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ROTATED_IN_INTERM,
                        .accessor_name = "rotated_in_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_INTERM,
                        .accessor_name = "cos_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = COS_INTERM,
                        .accessor_name = "cos_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_INTERM,
                        .accessor_name = "sin_interm",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SIN_INTERM,
                        .accessor_name = "sin_interm",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"num_rows", num_rows_per_core},
                    {"Wt", Wt},
                    {"half_Wt", half_Wt},
                },
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute_spec(COMPUTE_G1, num_rows_per_core_group_1));

    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1},
    };
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute_spec(COMPUTE_G2, num_rows_per_core_group_2));
        work_units.push_back(
            WorkUnitSpec{.name = "group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args ----
    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t cos_sin_start_id = num_tiles_written % HtWt;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_rows", num_rows_per_core},
             {"start_id", num_tiles_written},
             {"start_row_id", num_tiles_written / Wt % Ht},
             {"cos_sin_start_id", cos_sin_start_id}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_rows_per_core * Wt}, {"start_id", num_tiles_written}});
        num_tiles_written += num_rows_per_core * Wt;
    }

    // ---- Assemble spec + run-args ----
    ProgramSpec spec{
        .name = "rotary_embedding_hf_multi_tile_prefill",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = COS_CACHE, .spec = cos.tensor_spec()},
                TensorParameter{.unique_id = SIN_CACHE, .spec = sin.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mt},
        {COS_CACHE, cos_mt},
        {SIN_CACHE, sin_mt},
        {OUTPUT, output_mt},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingHfMultiCore::create_program_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    if (input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_prefill_artifacts(operation_attributes, tensor_args, output);
    }
    return create_multi_tile_artifacts(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::experimental::prim
