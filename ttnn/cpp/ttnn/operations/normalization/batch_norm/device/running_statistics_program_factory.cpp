// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "running_statistics_device_operation.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <bit>
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

void populate_runtime_arguments(
    tt::tt_metal::experimental::KernelRunArgs& reader_run_args,
    tt::tt_metal::experimental::KernelRunArgs& writer_run_args,
    tt::tt_metal::experimental::KernelRunArgs& compute_run_args,
    tt::tt_metal::CoreCoord compute_with_storage_grid_size,
    bool any_float32,
    const RunningStatistics::operation_attributes_t& operation_attributes,
    const RunningStatistics::tensor_args_t& tensor_args,
    RunningStatistics::tensor_return_value_t& c) {
    using tt::tt_metal::experimental::AddRuntimeArgsForNode;

    const auto& [batch_mean_tensor, batch_var_tensor, running_mean_tensor, running_var_tensor] = tensor_args;
    const auto momentum = operation_attributes.momentum;

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(batch_mean_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_var_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto
        [_unused_num_cores,
         _unused_all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // Cores outside both work groups still run the kernels, so every named runtime argument
            // must be set on them; the compute kernel early-returns on num_tiles == 0.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"momentum", 0},
                 {"start_tile_id", 0},
                 {"num_tiles", 0},
                 {"HtWt", 0},
                 {"n_stride", 0},
                 {"c_stride", 0},
                 {"N", 0},
                 {"C", 0}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_tile_id", 0},
                 {"num_tiles", 0},
                 {"HtWt", 0},
                 {"n_stride", 0},
                 {"c_stride", 0},
                 {"N", 0},
                 {"C", 0}});
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", 0}});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = momentum;
        const auto packed_scalar_momentum =
            any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"momentum", packed_scalar_momentum},
             {"start_tile_id", start_tile_id},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", cHtWt},
             {"n_stride", aHt * aWt * aC * static_cast<uint32_t>(aN > 1)},
             {"c_stride", aHt * aWt * static_cast<uint32_t>(aC > 1)},
             {"N", cN},
             {"C", cC}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_tile_id", start_tile_id},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", cHtWt},
             {"n_stride", bHt * bWt * bC * static_cast<uint32_t>(bN > 1)},
             {"c_stride", bHt * bWt * static_cast<uint32_t>(bC > 1)},
             {"N", cN},
             {"C", cC}});

        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}});

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
ttnn::device_operation::ProgramArtifacts RunningStatistics::RunningStatisticsProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& [batch_mean_tensor, batch_var_tensor, running_mean_tensor, running_var_tensor] = tensor_args;

    auto* device = batch_mean_tensor.device();

    const bool running_mean_has_value = running_mean_tensor.has_value();
    const bool running_var_has_value = running_var_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format =
        running_mean_has_value ? datatype_to_dataformat_converter(running_mean_tensor->dtype()) : DataFormat::Float16_b;
    auto e_data_format =
        running_var_has_value ? datatype_to_dataformat_converter(running_var_tensor->dtype()) : DataFormat::Float16_b;

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32);
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    auto running_stat_data_format =
        running_mean_has_value ? d_data_format : (running_var_has_value ? e_data_format : DataFormat::Float16_b);
    const bool stat_format_needs_typecast =
        (interm_data_format == DataFormat::Float32 && running_stat_data_format != DataFormat::Float32);
    const bool needs_mean_typecast = running_mean_has_value && stat_format_needs_typecast;
    const bool needs_var_typecast = running_var_has_value && stat_format_needs_typecast;

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: this factory and batch_norm_program_factory.cpp land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName BATCH_MEAN_DFB{"batch_mean"};
    const DFBSpecName BATCH_VAR_DFB{"batch_var"};
    const DFBSpecName OUT0_DFB{"out0"};
    const DFBSpecName OLD_RUNNING_MEAN_DFB{"old_running_mean"};
    const DFBSpecName OLD_RUNNING_VAR_DFB{"old_running_var"};
    const DFBSpecName MOMENTUM_DFB{"momentum"};
    const DFBSpecName ONE_DFB{"one"};
    const DFBSpecName UPDATED_M_DFB{"updated_m"};
    const DFBSpecName UPDATED_V_DFB{"updated_v"};
    const DFBSpecName TMP1_DFB{"tmp1"};
    const DFBSpecName TMP2_DFB{"tmp2"};
    const DFBSpecName TMP3_DFB{"tmp3"};
    const DFBSpecName WRITER_UPDATED_M_DFB{"writer_updated_m"};
    const DFBSpecName WRITER_UPDATED_V_DFB{"writer_updated_v"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName BATCH_MEAN_TENSOR{"batch_mean"};
    const TensorParamName BATCH_VAR_TENSOR{"batch_var"};
    const TensorParamName RUNNING_MEAN_TENSOR{"running_mean"};
    const TensorParamName RUNNING_VAR_TENSOR{"running_var"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "running_statistics";

    // ---- Dataflow buffers ----
    // Input buffers
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = BATCH_MEAN_DFB,
        .entry_size = a_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = a_data_format,
    });  // batch_mean
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = BATCH_VAR_DFB,
        .entry_size = b_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = b_data_format,
    });  // batch_var
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT0_DFB,
        .entry_size = c_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = c_data_format,
    });  // output
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OLD_RUNNING_MEAN_DFB,
        .entry_size = d_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = d_data_format,
    });  // old running mean
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OLD_RUNNING_VAR_DFB,
        .entry_size = e_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = e_data_format,
    });  // old running var
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOMENTUM_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // momentum
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ONE_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // to store 1
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UPDATED_M_DFB,
        .entry_size = needs_mean_typecast ? interm_single_tile_size : d_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = needs_mean_typecast ? interm_data_format : d_data_format,
    });  // updated running mean (staging when typecast)
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UPDATED_V_DFB,
        .entry_size = needs_var_typecast ? interm_single_tile_size : e_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = needs_var_typecast ? interm_data_format : e_data_format,
    });  // updated running var (staging when typecast)

    // The writer drains the writer-facing DFB when a typecast is needed, and the staging DFB itself
    // otherwise: with no typecast the legacy factory pointed both CB indices at the same buffer, so
    // the writer's one accessor name resolves here rather than through a kernel-side alias.
    const DFBSpecName writer_updated_m_dfb = needs_mean_typecast ? WRITER_UPDATED_M_DFB : UPDATED_M_DFB;
    const DFBSpecName writer_updated_v_dfb = needs_var_typecast ? WRITER_UPDATED_V_DFB : UPDATED_V_DFB;
    if (needs_mean_typecast) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WRITER_UPDATED_M_DFB,
            .entry_size = d_single_tile_size,
            .num_entries = b_num_tiles_per_cb,
            .data_format_metadata = d_data_format,
        });
    }
    if (needs_var_typecast) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WRITER_UPDATED_V_DFB,
            .entry_size = e_single_tile_size,
            .num_entries = b_num_tiles_per_cb,
            .data_format_metadata = e_data_format,
        });
    }

    // Intermediate buffers required for updation of running stats
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TMP1_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TMP2_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TMP3_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });

    // ---- Tensor parameters (replace the buffer-address RTAs and the TensorAccessorArgs plumbing) ----
    // The optional running statistics are declared only when present: there is no tensor to supply as
    // a TensorArgument otherwise, so their kernel-side accessors are #ifdef-gated instead.
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = BATCH_MEAN_TENSOR, .spec = batch_mean_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = BATCH_VAR_TENSOR, .spec = batch_var_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});
    if (running_mean_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = RUNNING_MEAN_TENSOR, .spec = running_mean_tensor->tensor_spec()});
    }
    if (running_var_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = RUNNING_VAR_TENSOR, .spec = running_var_tensor->tensor_spec()});
    }

    // ---- READER KERNEL ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_running_statistics.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = BATCH_MEAN_DFB,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = MOMENTUM_DFB,
                    .accessor_name = "momentum",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Filled through fill_cb_with_value, which reserves / fills / pushes internally, so
                // the reader is a genuine producer even though it builds no DataflowBuffer for it.
                DFBBinding{
                    .dfb_spec_name = ONE_DFB,
                    .accessor_name = "one",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = BATCH_MEAN_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"fill_momentum_fp32", static_cast<uint32_t>(any_float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"momentum", "start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- WRITER KERNEL ----
    // The writer is a producer on batch_var / old_running_mean / old_running_var (it reads tensor
    // memory into them) as well as the consumer of the output and the updated statistics.
    KernelSpec::CompilerOptions::Defines writer_defines;
    Group<TensorBinding> writer_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = BATCH_VAR_TENSOR, .accessor_name = "src"},
        TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"},
    };
    if (running_mean_has_value) {
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = RUNNING_MEAN_TENSOR, .accessor_name = "old_running_mean"});
        writer_defines.emplace("RUNNING_MEAN_HAS_VALUE", "1");
    }
    if (running_var_has_value) {
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = RUNNING_VAR_TENSOR, .accessor_name = "old_running_var"});
        writer_defines.emplace("RUNNING_VAR_HAS_VALUE", "1");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_running_statistics.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = BATCH_VAR_DFB,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT0_DFB,
                    .accessor_name = "dst",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // Bound unconditionally even when the optional tensor is absent: the legacy host
                // allocated these buffers in every configuration and the kernel constructs their
                // DataflowBuffer objects outside the conditional.
                DFBBinding{
                    .dfb_spec_name = OLD_RUNNING_MEAN_DFB,
                    .accessor_name = "old_running_mean",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OLD_RUNNING_VAR_DFB,
                    .accessor_name = "old_running_var",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = writer_updated_m_dfb,
                    .accessor_name = "new_mean",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = writer_updated_v_dfb,
                    .accessor_name = "new_var",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {{"old_stat_is_fp32", static_cast<uint32_t>(running_stat_data_format == DataFormat::Float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- COMPUTE KERNEL ----
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // tmp1 / tmp2 / tmp3 are packed and immediately re-read by this same kernel, so each is bound as
    // both PRODUCER and CONSUMER (a self-loop).
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = BATCH_MEAN_DFB,
            .accessor_name = "batch_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = BATCH_VAR_DFB,
            .accessor_name = "batch_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUT0_DFB,
            .accessor_name = "out0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_MEAN_DFB,
            .accessor_name = "old_running_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_VAR_DFB,
            .accessor_name = "old_running_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOMENTUM_DFB,
            .accessor_name = "momentum",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ONE_DFB,
            .accessor_name = "one",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = UPDATED_M_DFB,
            .accessor_name = "updated_m",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = UPDATED_V_DFB,
            .accessor_name = "updated_v",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TMP1_DFB,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TMP1_DFB,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = TMP2_DFB,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TMP2_DFB,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = TMP3_DFB,
            .accessor_name = "tmp3",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TMP3_DFB,
            .accessor_name = "tmp3",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };

    // On the typecast path this kernel re-reads its own FP32 staging buffer to typecast it into the
    // writer-facing DFB, so the staging DFB becomes a compute self-loop and the writer-facing DFB
    // appears. The define lets the kernel name the writer-facing token only where it is bound.
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (needs_mean_typecast) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UPDATED_M_DFB,
            .accessor_name = "updated_m",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_UPDATED_M_DFB,
            .accessor_name = "writer_updated_m",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_defines.emplace("MEAN_NEEDS_TYPECAST", "1");
    }
    if (needs_var_typecast) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UPDATED_V_DFB,
            .accessor_name = "updated_v",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_UPDATED_V_DFB,
            .accessor_name = "writer_updated_v",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_defines.emplace("VAR_NEEDS_TYPECAST", "1");
    }

    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    if (auto* compute_gen1 = std::get_if<ComputeGen1Config>(&compute_hw); compute_gen1 && fp32_dest_acc_en) {
        // Legacy set unpack_to_dest_mode[cb] = UnpackToDestFp32 on these twelve CBs when fp32
        // accumulation is on; reindexed onto DFB names and translated to the Metal 2.0 spelling.
        // The writer-facing DFBs were never in the legacy list and are not added here.
        compute_gen1->unpack_modes = ComputeUnpackModes{
            {BATCH_MEAN_DFB, UnpackMode::UnpackToDest},
            {BATCH_VAR_DFB, UnpackMode::UnpackToDest},
            {OUT0_DFB, UnpackMode::UnpackToDest},
            {OLD_RUNNING_MEAN_DFB, UnpackMode::UnpackToDest},
            {OLD_RUNNING_VAR_DFB, UnpackMode::UnpackToDest},
            {UPDATED_M_DFB, UnpackMode::UnpackToDest},
            {UPDATED_V_DFB, UnpackMode::UnpackToDest},
            {MOMENTUM_DFB, UnpackMode::UnpackToDest},
            {ONE_DFB, UnpackMode::UnpackToDest},
            {TMP1_DFB, UnpackMode::UnpackToDest},
            {TMP2_DFB, UnpackMode::UnpackToDest},
            {TMP3_DFB, UnpackMode::UnpackToDest},
        };
    }

    auto tc_out_fmt = stat_format_needs_typecast ? static_cast<uint32_t>(running_stat_data_format)
                                                 : static_cast<uint32_t>(DataFormat::Float32);

    // Both compute sources bind this one KernelSpec, so the named compile-time argument set is the
    // superset the SFPU source reads; the plain source ignores the five it does not read.
    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_{}.cpp",
            (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel"),
        // O3 is the legacy ComputeConfig default; Metal 2.0's CompilerOptions defaults to O2, so the
        // level has to be stated explicitly to keep the compute kernel where it was.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"old_running_mean_has_value", static_cast<uint32_t>(running_mean_has_value)},
                {"old_running_var_has_value", static_cast<uint32_t>(running_var_has_value)},
                {"stat_needs_typecast", static_cast<uint32_t>(stat_format_needs_typecast)},
                {"tc_in_fmt", static_cast<uint32_t>(DataFormat::Float32)},
                {"tc_out_fmt", tc_out_fmt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = compute_hw,
    });

    // ---- Work unit (placement) ----
    // All three legacy KernelDescriptors shared one core_ranges, so one work unit reproduces
    // placement exactly and satisfies the local-DFB identical-work-unit-membership invariant.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_device_cores,
    });

    // ---- Runtime arguments per core ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    CMAKE_UNIQUE_NAMESPACE::populate_runtime_arguments(
        reader_run_args,
        writer_run_args,
        compute_run_args,
        compute_with_storage_grid_size,
        any_float32,
        operation_attributes,
        tensor_args,
        output);

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    run_args.tensor_args.emplace(BATCH_MEAN_TENSOR, TensorArgument{batch_mean_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(BATCH_VAR_TENSOR, TensorArgument{batch_var_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});
    if (running_mean_has_value) {
        run_args.tensor_args.emplace(RUNNING_MEAN_TENSOR, TensorArgument{running_mean_tensor->mesh_tensor()});
    }
    if (running_var_has_value) {
        run_args.tensor_args.emplace(RUNNING_VAR_TENSOR, TensorArgument{running_var_tensor->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::normalization
