// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_tiled_program_factory.hpp"

#include <algorithm>
#include <functional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts ConcatS2STiledProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const unsigned int groups = operation_attributes.groups;

    // Work against the Metalium device-tensor type throughout; the TTNN wrappers only carry us in.
    const auto& output = tensor_return_value.mesh_tensor();
    const uint32_t num_input_tensors = tensor_args.input_tensors.size();
    std::vector<std::reference_wrapper<const MeshTensor>> inputs;
    inputs.reserve(num_input_tensors);
    for (const auto& input_tensor : tensor_args.input_tensors) {
        inputs.emplace_back(input_tensor.mesh_tensor());
    }

    TT_FATAL(
        inputs[0].get().logical_shape()[-1] == inputs[0].get().padded_shape()[-1],
        "Cannot have padding along width dimension in input tensor 0 ({} != {})",
        inputs[0].get().logical_shape()[-1],
        inputs[0].get().padded_shape()[-1]);
    TT_FATAL(
        inputs[1].get().logical_shape()[-1] == inputs[1].get().padded_shape()[-1],
        "Cannot have padding along width dimension in input tensor 1 ({} != {})",
        inputs[1].get().logical_shape()[-1],
        inputs[1].get().padded_shape()[-1]);

    TT_FATAL(
        inputs[0].get().padded_shape()[-1] % groups == 0,
        "Input tensor 0 columns must be evenly divisible by groups (W={}, groups={})",
        inputs[0].get().padded_shape()[-1],
        groups);
    TT_FATAL(
        inputs[1].get().padded_shape()[-1] % groups == 0,
        "Input tensor 1 columns must be evenly divisible by groups (W={}, groups={})",
        inputs[1].get().padded_shape()[-1],
        groups);

    // The current implementation relies on not having break up tile faces so if we would
    // need to split tiles because dim[-1] / groups < 16, we cannot proceed
    TT_FATAL(
        inputs[0].get().padded_shape()[-1] / groups >= TILE_HEIGHT / 2,
        "Group size must be at least 16 for input0 (was {})",
        inputs[0].get().padded_shape()[-1] / groups);
    TT_FATAL(
        inputs[1].get().padded_shape()[-1] / groups >= TILE_HEIGHT / 2,
        "Group size must be at least 16 for input1 (was {})",
        inputs[1].get().padded_shape()[-1] / groups);

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName INPUT0_DFB{"s2s_tiled_input0"};
    const DFBSpecName INPUT1_DFB{"s2s_tiled_input1"};
    const DFBSpecName OUTPUT_DFB{"s2s_tiled_output"};
    const DFBSpecName INPUT0_TRANSPOSE_DFB{"s2s_tiled_input0_transpose"};
    const DFBSpecName INPUT1_TRANSPOSE_DFB{"s2s_tiled_input1_transpose"};
    const DFBSpecName CONCAT_DFB{"s2s_tiled_concat"};
    const DFBSpecName OUTPUT_TRANSPOSE_DFB{"s2s_tiled_output_transpose"};
    const TensorParamName INPUT_0{"input_0"};
    const TensorParamName INPUT_1{"input_1"};
    const TensorParamName OUTPUT{"output"};

    const CoreRangeSet all_cores = inputs[0].get().shard_spec().value().grid;  // assume all inputs have same grid

    const auto get_num_tiles_per_shard = [](const ShardSpec& shard_spec) -> std::pair<uint32_t, uint32_t> {
        const std::array<uint32_t, 2> shard_shape = shard_spec.shape;
        TT_FATAL(shard_shape[0] % TILE_HEIGHT == 0, "Shard height must be aligned to tile height");
        TT_FATAL(shard_shape[1] % TILE_WIDTH == 0, "Shard width must be aligned to tile width");
        const uint32_t num_tiles_along_height = shard_shape[0] / TILE_HEIGHT;
        const uint32_t num_tiles_along_width = shard_shape[1] / TILE_WIDTH;
        TT_FATAL(num_tiles_along_height != 0 && num_tiles_along_width != 0, "Expected tensor to have at least 1 tiles");
        return {num_tiles_along_height, num_tiles_along_width};
    };
    const auto get_total_num_tiles_per_shard = [](const std::pair<uint32_t, uint32_t>& num_tiles) -> uint32_t {
        return num_tiles.first * num_tiles.second;
    };

    std::vector<std::pair<uint32_t, uint32_t>> num_tiles_for_each_input_shard;
    num_tiles_for_each_input_shard.reserve(inputs.size());
    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(num_tiles_for_each_input_shard),
        [&get_num_tiles_per_shard](const std::reference_wrapper<const MeshTensor>& input_tensor) {
            return get_num_tiles_per_shard(input_tensor.get().shard_spec().value());
        });
    const std::pair<uint32_t, uint32_t> num_tiles_for_output_shard =
        get_num_tiles_per_shard(output.shard_spec().value());

    TT_FATAL(inputs.at(0).get().dtype() == inputs.at(1).get().dtype(), "Input tensor data types must match");
    const tt::DataFormat data_format = datatype_to_dataformat_converter(inputs.at(0).get().dtype());
    const uint32_t tile_size = tt::tile_size(data_format);

    // The three tensor-backed buffers borrow their tensors' shard memory; the four below them are
    // ordinary L1 scratch.
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.reserve(num_input_tensors + 5);

    const std::array<DFBSpecName, 2> input_dfb_names = {INPUT0_DFB, INPUT1_DFB};
    const std::array<TensorParamName, 2> input_param_names = {INPUT_0, INPUT_1};
    for (uint32_t idx = 0; idx < num_input_tensors; idx++) {
        const MeshTensor& input_tensor = inputs.at(idx).get();
        const uint32_t total_num_tiles = get_total_num_tiles_per_shard(num_tiles_for_each_input_shard[idx]);
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = input_dfb_names[idx],
            .entry_size = tt::tile_size(datatype_to_dataformat_converter(input_tensor.dtype())),
            .num_entries = total_num_tiles,
            .data_format_metadata = datatype_to_dataformat_converter(input_tensor.dtype()),
            .borrowed_from = input_param_names[idx],
        });
    }

    const uint32_t total_num_output_tiles = get_total_num_tiles_per_shard(num_tiles_for_output_shard);
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = tt::tile_size(datatype_to_dataformat_converter(output.dtype())),
        .num_entries = total_num_output_tiles,
        .data_format_metadata = datatype_to_dataformat_converter(output.dtype()),
        .borrowed_from = OUTPUT,
    });

    tt::DataFormat dfb_data_format = data_format;
    uint32_t dfb_tile_size = tile_size;
    const bool is_bf8 = inputs[0].get().dtype() == DataType::BFLOAT8_B;
    if (is_bf8) {
        dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
        dfb_tile_size = tt::tile_size(dfb_data_format);
    }

    const uint32_t in0_total_tiles_width = num_tiles_for_each_input_shard[0].second;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT0_TRANSPOSE_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in0_total_tiles_width,
        .data_format_metadata = dfb_data_format,
    });

    const uint32_t in1_total_tiles_width = num_tiles_for_each_input_shard[1].second;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT1_TRANSPOSE_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in1_total_tiles_width,
        .data_format_metadata = dfb_data_format,
    });

    const uint32_t out_total_tiles_width = in0_total_tiles_width + in1_total_tiles_width;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = CONCAT_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = out_total_tiles_width,
        .data_format_metadata = dfb_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_TRANSPOSE_DFB,
        .entry_size = tile_size,
        .num_entries = out_total_tiles_width,
        .data_format_metadata = data_format,
    });

    // TODO: Skip the tile transpose in compute kernel if the following condition is true:
    // >> (input_tensors[0].padded_shape()[-1] / groups % TILE_WIDTH == 0
    // >> && input_tensors[1].padded_shape()[-1] / groups % TILE_WIDTH == 0)
    constexpr uint32_t MAX_1_BYTE_TILES_PER_BATCH = 16;
    const uint32_t batch_size = MAX_1_BYTE_TILES_PER_BATCH / inputs[0].get().element_size();

    // Calculate stride sizes to determine if we can use single-packet NOC reads
    // For BF8, the kernel uses bf16_tile_size (2048 bytes) for stride calculation
    const uint32_t stride_tile_size = is_bf8 ? dfb_tile_size : tile_size;
    const uint32_t input0_stride = stride_tile_size * num_tiles_for_each_input_shard[0].second / groups;
    const uint32_t input1_stride = stride_tile_size * num_tiles_for_each_input_shard[1].second / groups;

    // NOC_MAX_BURST_SIZE from the architecture's noc_parameters.h, via the HAL:
    // Wormhole = 8192, Blackhole = 16384, Quasar = 65536.
    const uint32_t noc_max_burst_size = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const bool use_single_packet_read = (input0_stride <= noc_max_burst_size && input1_stride <= noc_max_burst_size);

    // Legacy handed one positional list to all three kernels; named arguments let each kernel
    // declare only what it reads. The seven buffer indices legacy carried in slots 0-6 are DFB
    // bindings now, and only the compute kernel reads the batch size.
    const KernelSpec::CompileTimeArgs common_compile_time_args = {
        {"input0_num_tiles_height", num_tiles_for_each_input_shard[0].first},
        {"input0_num_tiles_width", num_tiles_for_each_input_shard[0].second},
        {"input1_num_tiles_height", num_tiles_for_each_input_shard[1].first},
        {"input1_num_tiles_width", num_tiles_for_each_input_shard[1].second},
        {"tile_size", tile_size},
        {"groups", groups},
    };
    KernelSpec::CompileTimeArgs compute_compile_time_args = common_compile_time_args;
    compute_compile_time_args.emplace("max_batch_size", batch_size);

    KernelSpec::CompilerOptions::Defines reader_defines;
    if (is_bf8) {
        reader_defines.emplace("BF8", "1");
    }
    if (use_single_packet_read) {
        reader_defines.emplace("USE_SINGLE_PACKET_READ", "1");
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors_tiled.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT0_DFB,
                    .accessor_name = "input0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT1_DFB,
                    .accessor_name = "input1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT0_TRANSPOSE_DFB,
                    .accessor_name = "input0_transpose",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT1_TRANSPOSE_DFB,
                    .accessor_name = "input1_transpose",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = CONCAT_DFB,
                    .accessor_name = "concat",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = common_compile_time_args,
        .hw_config = ttnn::create_reader_datamovement_config(inputs[0].get().device().arch()),
    };

    // The writer is the only kernel that touches the output buffer: it packs into the resident
    // output shard and nothing downstream drains it. A single toucher cannot present a producer and
    // a consumer on distinct kernels, so bind this one kernel as both — a self-loop. On Gen1 the
    // buffer lowers to a plain circular buffer that one RISC both fills and drains, so runtime
    // behaviour is identical to the legacy circular buffer and the kernel code is untouched.
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "writer_height_sharded_width_concat_two_tensors_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TRANSPOSE_DFB,
                    .accessor_name = "output_transpose",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args = common_compile_time_args,
        .hw_config = ttnn::create_writer_datamovement_config(inputs[0].get().device().arch()),
    };

    const bool fp32_dest_acc_en = data_format == tt::DataFormat::Float32 || data_format == tt::DataFormat::Int32 ||
                                  data_format == tt::DataFormat::UInt32;

    // Metal 2.0's validator requires an explicit unpack mode for every Float32 buffer a compute
    // kernel consumes while 32-bit dest is enabled; legacy defaulted silently. The legacy descriptor
    // left unpack_to_dest_mode empty — i.e. Default on every buffer — which is UnpackToSrc. When the
    // data format is Float32 all three buffers compute consumes carry that format (an is_bf8 input
    // cannot also be Float32), so all three need the entry.
    ComputeUnpackModes unpack_modes;
    if (data_format == tt::DataFormat::Float32) {
        unpack_modes[INPUT0_DFB] = UnpackMode::UnpackToSrc;
        unpack_modes[INPUT1_DFB] = UnpackMode::UnpackToSrc;
        unpack_modes[CONCAT_DFB] = UnpackMode::UnpackToSrc;
    }

    // The legacy factory set a Metal ComputeConfigDescriptor directly rather than resolving a TTNN
    // ComputeKernelConfig, so build the Gen1 config by hand and leave every field the descriptor did
    // not set at its default. ComputeGen1Config's defaults coincide with the legacy descriptor's:
    // math_approx_mode=false -> sfpu_precision_mode=Precise, dst_full_sync_en=false ->
    // double_buffer_dest=true, bfp8_pack_precise=false -> bfp_pack_precision_mode=Approximate.
    ComputeGen1Config compute_hw_config{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .unpack_modes = std::move(unpack_modes),
    };

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/compute/"
            "height_sharded_width_concat_two_tensors.cpp",
        // Legacy compute kernels default to O3 while Metal 2.0's type-agnostic compiler options
        // default to O2, so the level has to be stated to keep the compile as it was.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT0_DFB,
                    .accessor_name = "input0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT1_DFB,
                    .accessor_name = "input1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT0_TRANSPOSE_DFB,
                    .accessor_name = "input0_transpose",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT1_TRANSPOSE_DFB,
                    .accessor_name = "input1_transpose",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CONCAT_DFB,
                    .accessor_name = "concat",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TRANSPOSE_DFB,
                    .accessor_name = "output_transpose",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = std::move(compute_compile_time_args),
        .hw_config = std::move(compute_hw_config),
    };

    Group<KernelSpec> kernels;
    kernels.reserve(3);
    kernels.push_back(std::move(reader_spec));
    kernels.push_back(std::move(writer_spec));
    kernels.push_back(std::move(compute_spec));

    // No kernel builds a TensorAccessor here, so no tensor is bound on a KernelSpec. The three
    // parameters exist because the tensor-backed buffers above borrow their memory, which is a use
    // in its own right: each buffer's L1 address resolves at run time from the matching argument.
    ProgramSpec spec{
        .name = "concat_s2s_tiled",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT_0, .spec = inputs[0].get().tensor_spec()},
                TensorParameter{.unique_id = INPUT_1, .spec = inputs[1].get().tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    // The factory sets no runtime args at all, so ProgramRunArgs carries only the tensor arguments
    // the borrowed buffers resolve their addresses from.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT_0, inputs[0].get());
    run_args.tensor_args.emplace(INPUT_1, inputs[1].get());
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
