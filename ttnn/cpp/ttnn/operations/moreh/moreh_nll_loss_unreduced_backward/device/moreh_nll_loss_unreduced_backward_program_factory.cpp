// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_unreduced_backward_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// ProgramSpec resource names, shared by all three rank configurations.
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};

const DFBSpecName TARGET_DFB{"target"};
const DFBSpecName OUTPUT_GRAD_DFB{"output_grad"};
const DFBSpecName WEIGHT_DFB{"weight"};
// Scratch staging for read_line: it NoC-reads a whole DRAM-aligned chunk here, then copies out just
// the valid elements. Needed because DRAM's minimum read size exceeds L1's on some architectures.
const DFBSpecName WEIGHT_SCRATCH_DFB{"weight_scratch"};
const DFBSpecName OUTPUT_GRAD_SCRATCH_DFB{"output_grad_scratch"};
const DFBSpecName INPUT_GRAD_DFB{"input_grad"};

const TensorParamName TARGET_TENSOR{"target"};
const TensorParamName OUTPUT_GRAD_TENSOR{"output_grad"};
const TensorParamName WEIGHT_TENSOR{"weight"};
const TensorParamName INPUT_GRAD_TENSOR{"input_grad"};

// Helper: append a DFB holding `num_tiles` tiles of `data_format` (skips creation when num_tiles == 0).
void push_dfb(ProgramSpec& spec, const DFBSpecName& unique_id, uint32_t num_tiles, tt::DataFormat data_format) {
    if (num_tiles == 0) {
        return;
    }
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = tt::tile_size(data_format),
        .num_entries = num_tiles,
        .data_format_metadata = data_format,
    });
}

// Helper: the reader both fills and drains its private DFBs — some through the FIFO, some purely as an
// address source — so it is the buffer's only toucher and takes both endpoint roles. One accessor name
// for both, so the kernel keeps a single DataflowBuffer object per DFB.
void bind_self_loop(KernelSpec& kernel, const DFBSpecName& dfb, std::string_view accessor_name) {
    kernel.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    kernel.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
}

}  // namespace

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_unreduced_backward_impl_2d(
    const MeshTensor& target,
    const std::optional<std::reference_wrapper<const MeshTensor>>& weight,
    const MeshTensor& output_grad,
    const MeshTensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    const auto& input_grad_spec = input_grad.tensor_spec();
    auto input_grad_shape = input_grad_spec.padded_shape();
    auto N = input_grad_shape[0];
    uint32_t channel_size = input_grad_shape[1];

    const bool weight_has_value = weight.has_value();

    const auto& device = target.device();
    auto grid = device.compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad_shape.volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    ProgramSpec spec{.name = "moreh_nll_loss_unreduced_backward_2d"};

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad_spec.data_type());

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    auto Nt = tt::div_up(N, tt::constants::TILE_WIDTH);

    push_dfb(spec, TARGET_DFB, 1, tt::DataFormat::Int32);                 // target
    push_dfb(spec, OUTPUT_GRAD_DFB, Nt, data_format);                     // output_grad
    push_dfb(spec, WEIGHT_DFB, weight_has_value ? Ct : 0u, data_format);  // weight
    push_dfb(spec, INPUT_GRAD_DFB, 1, data_format);                       // input_grad

    if (weight_has_value) {
        // This DFB will be used as scratch storage when reading data from DRAM into L1
        push_dfb(spec, WEIGHT_SCRATCH_DFB, 1, data_format);  // weight scratch
    }
    // Need another scratch DFB for output_grad reading data from DRAM into L1.
    push_dfb(spec, OUTPUT_GRAD_SCRATCH_DFB, 1, data_format);  // output_grad scratch

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_2d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "Nt", "Ct"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device.arch()),
    };
    bind_self_loop(reader, TARGET_DFB, "target");
    bind_self_loop(reader, OUTPUT_GRAD_DFB, "output_grad");
    bind_self_loop(reader, OUTPUT_GRAD_SCRATCH_DFB, "output_grad_scratch");
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_GRAD_DFB,
        .accessor_name = "input_grad",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = TARGET_TENSOR,
        .accessor_name = "target",
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = OUTPUT_GRAD_TENSOR,
        .accessor_name = "output_grad",
    });
    if (weight_has_value) {
        bind_self_loop(reader, WEIGHT_DFB, "weight");
        bind_self_loop(reader, WEIGHT_SCRATCH_DFB, "weight_scratch");
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = WEIGHT_TENSOR,
            .accessor_name = "weight",
        });
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_GRAD_DFB,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_GRAD_TENSOR,
                    .accessor_name = "input_grad",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles_per_core", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device.arch()),
    };

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TARGET_TENSOR, .spec = target.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = OUTPUT_GRAD_TENSOR, .spec = output_grad.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_GRAD_TENSOR, .spec = input_grad_spec});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = WEIGHT_TENSOR, .spec = weight->get().tensor_spec()});
    }

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"ignore_index", ignore_index},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"Nt", Nt},
                {"Ct", Ct},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles_per_core", units_per_core}, {"start_id", tile_offset}});

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TARGET_TENSOR, target);
    run_args.tensor_args.emplace(OUTPUT_GRAD_TENSOR, output_grad);
    run_args.tensor_args.emplace(INPUT_GRAD_TENSOR, input_grad);
    if (weight_has_value) {
        run_args.tensor_args.emplace(WEIGHT_TENSOR, weight->get());
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_unreduced_backward_impl_3d(
    const MeshTensor& target,
    const std::optional<std::reference_wrapper<const MeshTensor>>& weight,
    const MeshTensor& output_grad,
    const MeshTensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    const auto& input_grad_spec = input_grad.tensor_spec();
    auto input_grad_shape = input_grad_spec.padded_shape();
    uint32_t channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Ct = channel_size / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    const auto& device = target.device();
    auto grid = device.compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad_shape.volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    ProgramSpec spec{.name = "moreh_nll_loss_unreduced_backward_3d"};

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad_spec.data_type());

    push_dfb(spec, TARGET_DFB, 1, tt::DataFormat::Int32);                 // target
    push_dfb(spec, OUTPUT_GRAD_DFB, 1, data_format);                      // output_grad
    push_dfb(spec, WEIGHT_DFB, weight_has_value ? Ct : 0u, data_format);  // weight
    push_dfb(spec, INPUT_GRAD_DFB, 1, data_format);                       // input_grad

    if (weight_has_value) {
        // This DFB will be used as scratch storage when reading data from DRAM into L1
        push_dfb(spec, WEIGHT_SCRATCH_DFB, 1, data_format);  // weight scratch
    }

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_3d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "Ct", "Wt"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device.arch()),
    };
    bind_self_loop(reader, TARGET_DFB, "target");
    bind_self_loop(reader, OUTPUT_GRAD_DFB, "output_grad");
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_GRAD_DFB,
        .accessor_name = "input_grad",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = TARGET_TENSOR,
        .accessor_name = "target",
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = OUTPUT_GRAD_TENSOR,
        .accessor_name = "output_grad",
    });
    if (weight_has_value) {
        bind_self_loop(reader, WEIGHT_DFB, "weight");
        bind_self_loop(reader, WEIGHT_SCRATCH_DFB, "weight_scratch");
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = WEIGHT_TENSOR,
            .accessor_name = "weight",
        });
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_GRAD_DFB,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_GRAD_TENSOR,
                    .accessor_name = "input_grad",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles_per_core", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device.arch()),
    };

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TARGET_TENSOR, .spec = target.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = OUTPUT_GRAD_TENSOR, .spec = output_grad.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_GRAD_TENSOR, .spec = input_grad_spec});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = WEIGHT_TENSOR, .spec = weight->get().tensor_spec()});
    }

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"ignore_index", ignore_index},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"Ct", Ct},
                {"Wt", Wt},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles_per_core", units_per_core}, {"start_id", tile_offset}});

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TARGET_TENSOR, target);
    run_args.tensor_args.emplace(OUTPUT_GRAD_TENSOR, output_grad);
    run_args.tensor_args.emplace(INPUT_GRAD_TENSOR, input_grad);
    if (weight_has_value) {
        run_args.tensor_args.emplace(WEIGHT_TENSOR, weight->get());
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_unreduced_backward_impl_4d(
    const MeshTensor& target,
    const std::optional<std::reference_wrapper<const MeshTensor>>& weight,
    const MeshTensor& output_grad,
    const MeshTensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    const auto& input_grad_spec = input_grad.tensor_spec();
    const auto& target_spec = target.tensor_spec();
    auto input_grad_shape = input_grad_spec.padded_shape();
    auto N = input_grad_shape[0];
    uint32_t channel_size = input_grad_shape[1];

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    uint32_t num_inner_tile =
        target_spec.padded_shape().volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    const auto& device = target.device();
    auto grid = device.compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad_shape.volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    ProgramSpec spec{.name = "moreh_nll_loss_unreduced_backward_4d"};

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad_spec.data_type());

    push_dfb(spec, TARGET_DFB, 1, tt::DataFormat::Int32);                 // target
    push_dfb(spec, OUTPUT_GRAD_DFB, 1, data_format);                      // output_grad
    push_dfb(spec, WEIGHT_DFB, weight_has_value ? Ct : 0u, data_format);  // weight
    push_dfb(spec, INPUT_GRAD_DFB, 1, data_format);                       // input_grad

    if (weight_has_value) {
        // This DFB will be used as scratch storage when reading data from DRAM into L1
        push_dfb(spec, WEIGHT_SCRATCH_DFB, 1, data_format);  // weight scratch
    }

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_4d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "num_inner_tile", "C", "Ct"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device.arch()),
    };
    bind_self_loop(reader, TARGET_DFB, "target");
    bind_self_loop(reader, OUTPUT_GRAD_DFB, "output_grad");
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_GRAD_DFB,
        .accessor_name = "input_grad",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = TARGET_TENSOR,
        .accessor_name = "target",
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = OUTPUT_GRAD_TENSOR,
        .accessor_name = "output_grad",
    });
    if (weight_has_value) {
        bind_self_loop(reader, WEIGHT_DFB, "weight");
        bind_self_loop(reader, WEIGHT_SCRATCH_DFB, "weight_scratch");
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = WEIGHT_TENSOR,
            .accessor_name = "weight",
        });
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_GRAD_DFB,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_GRAD_TENSOR,
                    .accessor_name = "input_grad",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles_per_core", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device.arch()),
    };

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TARGET_TENSOR, .spec = target_spec});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = OUTPUT_GRAD_TENSOR, .spec = output_grad.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_GRAD_TENSOR, .spec = input_grad_spec});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = WEIGHT_TENSOR, .spec = weight->get().tensor_spec()});
    }

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"ignore_index", ignore_index},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"num_inner_tile", num_inner_tile},
                {"C", channel_size},
                {"Ct", Ct},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles_per_core", units_per_core}, {"start_id", tile_offset}});

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TARGET_TENSOR, target);
    run_args.tensor_args.emplace(OUTPUT_GRAD_TENSOR, output_grad);
    run_args.tensor_args.emplace(INPUT_GRAD_TENSOR, input_grad);
    if (weight_has_value) {
        run_args.tensor_args.emplace(WEIGHT_TENSOR, weight->get());
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts
MorehNllLossUnreducedBackwardDeviceOperation::Factory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const MeshTensor& target = tensor_args.target_tensor.mesh_tensor();
    const std::optional<Tensor>& weight_tensor = tensor_args.weight_tensor;
    std::optional<std::reference_wrapper<const MeshTensor>> weight;
    if (weight_tensor.has_value()) {
        weight = std::cref(weight_tensor->mesh_tensor());
    }
    const MeshTensor& output_grad = tensor_args.output_grad_tensor.mesh_tensor();

    const uint32_t ignore_index = operation_attributes.ignore_index;
    const DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    const MeshTensor& input_grad = tensor_return_value.mesh_tensor();

    // split work
    const auto& input_grad_shape = input_grad.tensor_spec().logical_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_unreduced_backward_impl_2d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_unreduced_backward_impl_3d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_unreduced_backward_impl_4d(
        target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
