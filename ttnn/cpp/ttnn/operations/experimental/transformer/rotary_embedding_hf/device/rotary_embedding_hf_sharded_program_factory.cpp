// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_sharded_program_factory.hpp"
#include <bit>
#include <filesystem>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Legacy always ran with the ComputeConfigDescriptor defaults for math_approx_mode (false)
// and dst_full_sync_en (false): the factory resolved those knobs but never copied them onto
// the descriptor. Reproduce those defaults explicitly instead of honoring the caller's
// resolved values, so behavior is unchanged.
ComputeHardwareConfig make_decode_compute_hw_config(tt::ARCH arch, const ttnn::DeviceComputeKernelConfig& config) {
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(arch, config);
    sfpu_precision_mode(compute_hw) = Precision::Precise;  // legacy default: math_approx_mode = false
    double_buffer_dest(compute_hw) = true;                 // legacy default: dst_full_sync_en = false (inverted)
    return compute_hw;
}

ttnn::device_operation::ProgramArtifacts create_single_tile_decode_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const auto& input_mt = input.mesh_tensor();
    const auto& cos_mt = cos.mesh_tensor();
    const auto& sin_mt = sin.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    const tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    const tt::DataFormat cos_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    const tt::DataFormat sin_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    const tt::DataFormat trans_mat_data_format = input_data_format == tt::DataFormat::Bfp8_b ? tt::DataFormat::Bfp8_b
                                                 : input_data_format == tt::DataFormat::Float32
                                                     ? tt::DataFormat::Float32
                                                     : tt::DataFormat::Float16_b;
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_data_format);

    const tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t n_heads_per_batch_t = input.padded_shape()[2] / constants::TILE_HEIGHT;
    constexpr uint32_t head_dim_t = 1;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t num_cos_sin_tiles = head_dim_t * batch_per_core;

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN{"in"};                                // legacy CB c_0 (input) — borrowed, compute self-loop
    const DFBSpecName COS_DFB{"cos"};                          // legacy CB c_1 (cos) — borrowed, compute self-loop
    const DFBSpecName SIN_DFB{"sin"};                          // legacy CB c_2 (sin) — borrowed, compute self-loop
    const DFBSpecName TRANS_MAT{"trans_mat"};                  // legacy CB c_3 (rotate-half transform matrix)
    const DFBSpecName ROTATED_IN_INTERM{"rotated_in_interm"};  // legacy CB c_24
    const DFBSpecName COS_INTERM{"cos_interm"};                // legacy CB c_25
    const DFBSpecName SIN_INTERM{"sin_interm"};                // legacy CB c_26
    const DFBSpecName OUT{"out"};                              // legacy CB c_16 (output) — borrowed, compute self-loop
    const TensorParamName INPUT{"input"};
    const TensorParamName COS_CACHE{"cos_cache"};
    const TensorParamName SIN_CACHE{"sin_cache"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers ----
    // input/cos/sin/output are height-sharded residents: each DFB is built on the tensor's own
    // L1 shard (borrowed_from), and the compute kernel cursor-advances it itself (self-loop).
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_data_format,
            .borrowed_from = INPUT,
        },
        DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = cos_data_format,
            .borrowed_from = COS_CACHE,
        },
        DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = sin_data_format,
            .borrowed_from = SIN_CACHE,
        },
        DataflowBufferSpec{
            .unique_id = TRANS_MAT,
            .entry_size = trans_mat_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = trans_mat_data_format,
        },
        DataflowBufferSpec{
            .unique_id = ROTATED_IN_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = COS_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SIN_INTERM,
            .entry_size = input_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format,
            .borrowed_from = OUTPUT,
        },
    };

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw =
        make_decode_compute_hw_config(device->arch(), operation_attributes.compute_kernel_config);
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
        add_entry_if_fp32(OUT, output_data_format);
    }

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
            "reader_rotary_embedding_hf_single_tile_sharded.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TRANS_MAT,
                    .accessor_name = "trans_mat",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
            "rotary_embedding_hf_single_tile_sharded.cpp"),
        // Legacy compute default opt_level is O3; Metal 2.0 defaults to O2 — set explicitly to preserve.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SIN_DFB,
                    .accessor_name = "sin",
                    .endpoint_type = DFBEndpointType::PRODUCER,
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
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"heads_per_batch_t", n_heads_per_batch_t},
                {"batch_per_core", batch_per_core},
            },
        .hw_config = compute_hw,
    };

    // ---- Assemble spec + run-args (no kernel has runtime args) ----
    ProgramSpec spec{
        .name = "rotary_embedding_hf_single_tile_decode",
        .kernels = {std::move(reader), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = COS_CACHE, .spec = cos.tensor_spec()},
                TensorParameter{.unique_id = SIN_CACHE, .spec = sin.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{.name = "main", .kernels = {READER, COMPUTE}, .target_nodes = all_cores},
            },
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, input_mt},
        {COS_CACHE, cos_mt},
        {SIN_CACHE, sin_mt},
        {OUTPUT, output_mt},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts create_multi_tile_decode_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const auto& input_mt = input.mesh_tensor();
    const auto& cos_mt = cos.mesh_tensor();
    const auto& sin_mt = sin.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    const tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    const tt::DataFormat cos_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    const tt::DataFormat sin_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    const tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    const tt::DataFormat scalar_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_data_format);

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t n_heads_per_batch_t = input.padded_shape()[2] / constants::TILE_HEIGHT;
    const uint32_t head_dim_t = shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    uint32_t num_interm_tiles = head_dim_t;
    uint32_t num_scalar_tiles = 1;

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN{"in"};                                // legacy CB c_0 (input) — borrowed, compute self-loop
    const DFBSpecName COS_DFB{"cos"};                          // legacy CB c_1 (cos) — borrowed, compute self-loop
    const DFBSpecName SIN_DFB{"sin"};                          // legacy CB c_2 (sin) — borrowed, compute self-loop
    const DFBSpecName SCALAR{"scalar"};                        // legacy CB c_3 (-1.0 rotate-half scalar)
    const DFBSpecName ROTATED_IN_INTERM{"rotated_in_interm"};  // legacy CB c_24
    const DFBSpecName COS_INTERM{"cos_interm"};                // legacy CB c_25
    const DFBSpecName SIN_INTERM{"sin_interm"};                // legacy CB c_26
    const DFBSpecName OUT{"out"};                              // legacy CB c_16 (output) — borrowed, compute self-loop
    const TensorParamName INPUT{"input"};
    const TensorParamName COS_CACHE{"cos_cache"};
    const TensorParamName SIN_CACHE{"sin_cache"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers ----
    // input/cos/sin/output are height-sharded residents: each DFB is built on the tensor's own
    // L1 shard (borrowed_from), and the compute kernel cursor-advances it itself (self-loop).
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_data_format,
            .borrowed_from = INPUT,
        },
        DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = cos_data_format,
            .borrowed_from = COS_CACHE,
        },
        DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = sin_data_format,
            .borrowed_from = SIN_CACHE,
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
        DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format,
            .borrowed_from = OUTPUT,
        },
    };

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw =
        make_decode_compute_hw_config(device->arch(), operation_attributes.compute_kernel_config);
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
        add_entry_if_fp32(COS_DFB, cos_data_format);
        add_entry_if_fp32(SIN_DFB, sin_data_format);
        add_entry_if_fp32(ROTATED_IN_INTERM, input_data_format);
        add_entry_if_fp32(COS_INTERM, cos_data_format);
        add_entry_if_fp32(SIN_INTERM, sin_data_format);
        add_entry_if_fp32(OUT, output_data_format);
    }

    // ---- Kernels ----
    const uint16_t bfloat16_neg_one = std::bit_cast<uint16_t>(bfloat16(-1.0f));
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
            "reader_rotary_embedding_hf_sharded.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SCALAR,
                    .accessor_name = "scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"scalar_value", bfloat16_neg_one},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
            "rotary_embedding_hf_sharded.cpp"),
        // Legacy compute default opt_level is O3; Metal 2.0 defaults to O2 — set explicitly to preserve.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = COS_DFB,
                    .accessor_name = "cos",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SIN_DFB,
                    .accessor_name = "sin",
                    .endpoint_type = DFBEndpointType::PRODUCER,
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
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"Wt", head_dim_t},
                {"Ht", n_heads_t},
                {"heads_per_batch_t", n_heads_per_batch_t},
                {"batch_per_core", batch_per_core},
            },
        .hw_config = compute_hw,
    };

    // ---- Assemble spec + run-args (no kernel has runtime args) ----
    ProgramSpec spec{
        .name = "rotary_embedding_hf_multi_tile_decode",
        .kernels = {std::move(reader), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = COS_CACHE, .spec = cos.tensor_spec()},
                TensorParameter{.unique_id = SIN_CACHE, .spec = sin.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{.name = "main", .kernels = {READER, COMPUTE}, .target_nodes = all_cores},
            },
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, input_mt},
        {COS_CACHE, cos_mt},
        {SIN_CACHE, sin_mt},
        {OUTPUT, output_mt},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingHfMultiCoreSharded::create_program_artifacts(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    if (input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_decode_artifacts(operation_attributes, tensor_args, output);
    }
    return create_multi_tile_decode_artifacts(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::experimental::prim
