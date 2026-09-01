// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_backward_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE_GROUP_1{"compute_group_1"};
const KernelSpecName COMPUTE_GROUP_2{"compute_group_2"};

const DFBSpecName DFB_OUTPUT_GRAD{"output_grad"};
const DFBSpecName DFB_TARGET{"target"};
const DFBSpecName DFB_WEIGHT{"weight"};
const DFBSpecName DFB_DIVISOR{"divisor"};
const DFBSpecName DFB_WEIGHT_SCRATCH{"weight_scratch"};
const DFBSpecName DFB_TMP_WEIGHT{"tmp_weight"};
const DFBSpecName DFB_TMP1{"tmp1"};
const DFBSpecName DFB_TMP2{"tmp2"};
const DFBSpecName DFB_INPUT_GRAD{"input_grad"};

const TensorParamName TENSOR_TARGET{"target"};
const TensorParamName TENSOR_OUTPUT_GRAD{"output_grad"};
const TensorParamName TENSOR_WEIGHT{"weight"};
const TensorParamName TENSOR_DIVISOR{"divisor"};
const TensorParamName TENSOR_INPUT_GRAD{"input_grad"};

// Helper: a dataflow buffer holding a whole number of tiles of one format.
DataflowBufferSpec make_dfb(const DFBSpecName& unique_id, uint32_t num_tiles, tt::DataFormat data_format) {
    const auto tile_sz = tt::tile_size(data_format);
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = tile_sz,
        .num_entries = num_tiles,
        .data_format_metadata = data_format,
    };
}

// A compute kernel consuming a Float32 buffer while the Dest register is 32 bits wide must state
// its unpack mode outright; every other buffer keeps the implicit unpack-to-SrcA/B default. This op
// asks for the default everywhere, so the explicit entries carry that same choice.
void require_unpack_mode(
    ComputeUnpackModes& unpack_modes, bool fp32_dest_acc_en, const DFBSpecName& dfb, tt::DataFormat data_format) {
    if (fp32_dest_acc_en && data_format == tt::DataFormat::Float32) {
        unpack_modes.emplace(dfb, UnpackMode::UnpackToSrc);
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_backward_impl_2d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool /*reduction_mean*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    auto input_grad_shape = input_grad.padded_shape();
    uint32_t channel_size = input_grad_shape[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_backward_2d";

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT_GRAD, 1, data_format));
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, weight_num_tile, data_format));
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));
    if (divisor_has_value) {
        // tmp1 and tmp2 are touched only by the compute kernel's divisor branch, so they exist
        // exactly when that branch does. Allocating them unconditionally would leave two buffers
        // with no producer and no consumer in the no-divisor build, which cannot be expressed.
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP2, 1, fp32_dest_acc_en_data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT_GRAD, 1, data_format));

    if (weight_has_value) {
        // This buffer will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile of scratch, because content is read immediately after writing.
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT_SCRATCH, 1, data_format));
    }

    // declare the tensors the kernels operate on
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& input_grad_mesh = input_grad.mesh_tensor();

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_OUTPUT_GRAD, .spec = output_grad_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight.value().mesh_tensor().tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor.value().mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_INPUT_GRAD, .spec = input_grad_mesh.tensor_spec()});

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
        compute_defines.emplace("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace("DIVISOR", "1");
        compute_defines.emplace("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_2d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    Group<DFBBinding> reader_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        // The reader is the only kernel that touches `target`: it fills the entry, waits on it,
        // reads it through a local L1 pointer and pops it. So it holds both ends of the buffer.
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (weight_has_value) {
        // `weight` is read once into L1 and held there for the whole kernel, and `weight_scratch`
        // never sees a FIFO operation at all. Both are reader-only, so the reader holds both ends.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (divisor_has_value) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
        TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT_GRAD, .accessor_name = "output_grad"},
    };
    if (weight_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_WEIGHT, .accessor_name = "weight"});
    }
    if (divisor_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_DIVISOR, .accessor_name = "divisor"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "C", "weight_num_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_INPUT_GRAD,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_INPUT_GRAD, .accessor_name = "input_grad"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_INPUT_GRAD,
            .accessor_name = "input_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (divisor_has_value) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        // The compute kernel packs each intermediate and reads it straight back within its own
        // loop, so it holds both ends of tmp1 and tmp2.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    ComputeUnpackModes compute_unpack_modes;
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_OUTPUT_GRAD, data_format);
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP_WEIGHT, fp32_dest_acc_en_data_format);
    if (divisor_has_value) {
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_DIVISOR, data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP1, fp32_dest_acc_en_data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP2, fp32_dest_acc_en_data_format);
    }

    auto compute_hw_config = ttnn::to_compute_hardware_config(compute_kernel_config);
    compute_hw_config.unpack_modes = std::move(compute_unpack_modes);

    auto make_compute = [&](KernelSpecName unique_id, uint32_t units_per_core_group) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = compute_kernel_file,
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args = {{"per_core_tile_cnt", units_per_core_group}},
            .hw_config = compute_hw_config,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute(COMPUTE_GROUP_1, units_per_core_group_1));
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_GROUP_2, units_per_core_group_2));
    }

    // The reader and the writer run on every node; the two compute specialisations split the nodes
    // between them, so each node still runs exactly one of each kind.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_GROUP_1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_GROUP_2},
            .target_nodes = core_group_2,
        });
    }

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

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
                {"C", channel_size},
                {"weight_num_tile", weight_num_tile},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
            });

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    run_args.tensor_args.emplace(TENSOR_OUTPUT_GRAD, output_grad_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, weight.value().mesh_tensor());
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, divisor.value().mesh_tensor());
    }
    run_args.tensor_args.emplace(TENSOR_INPUT_GRAD, input_grad_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_backward_impl_3d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool /*reduction_mean*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    auto input_grad_shape = input_grad.padded_shape();
    uint32_t channel_size = input_grad_shape[1];

    auto target_shape = target.padded_shape();
    uint32_t num_inner_tile = target_shape[-1] / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_backward_3d";

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT_GRAD, 1, data_format));
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, weight_num_tile, data_format));
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));
    if (divisor_has_value) {
        // tmp1 and tmp2 are touched only by the compute kernel's divisor branch, so they exist
        // exactly when that branch does. Allocating them unconditionally would leave two buffers
        // with no producer and no consumer in the no-divisor build, which cannot be expressed.
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP2, 1, fp32_dest_acc_en_data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT_GRAD, 1, data_format));

    if (weight_has_value) {
        // This buffer will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile of scratch, because content is read immediately after writing.
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT_SCRATCH, 1, data_format));
    }

    // declare the tensors the kernels operate on
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& input_grad_mesh = input_grad.mesh_tensor();

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_OUTPUT_GRAD, .spec = output_grad_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight.value().mesh_tensor().tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor.value().mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_INPUT_GRAD, .spec = input_grad_mesh.tensor_spec()});

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
        compute_defines.emplace("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace("DIVISOR", "1");
        compute_defines.emplace("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_3d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    Group<DFBBinding> reader_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        // The reader is the only kernel that touches `target`: it fills the entry, waits on it,
        // reads it through a local L1 pointer and pops it. So it holds both ends of the buffer.
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (weight_has_value) {
        // `weight` is read once into L1 and held there for the whole kernel, and `weight_scratch`
        // never sees a FIFO operation at all. Both are reader-only, so the reader holds both ends.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (divisor_has_value) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
        TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT_GRAD, .accessor_name = "output_grad"},
    };
    if (weight_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_WEIGHT, .accessor_name = "weight"});
    }
    if (divisor_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_DIVISOR, .accessor_name = "divisor"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"ignore_index", "num_tiles_per_core", "start_id", "C", "num_inner_tile", "weight_num_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_INPUT_GRAD,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_INPUT_GRAD, .accessor_name = "input_grad"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_INPUT_GRAD,
            .accessor_name = "input_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (divisor_has_value) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        // The compute kernel packs each intermediate and reads it straight back within its own
        // loop, so it holds both ends of tmp1 and tmp2.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    ComputeUnpackModes compute_unpack_modes;
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_OUTPUT_GRAD, data_format);
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP_WEIGHT, fp32_dest_acc_en_data_format);
    if (divisor_has_value) {
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_DIVISOR, data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP1, fp32_dest_acc_en_data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP2, fp32_dest_acc_en_data_format);
    }

    auto compute_hw_config = ttnn::to_compute_hardware_config(compute_kernel_config);
    compute_hw_config.unpack_modes = std::move(compute_unpack_modes);

    auto make_compute = [&](KernelSpecName unique_id, uint32_t units_per_core_group) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = compute_kernel_file,
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args = {{"per_core_tile_cnt", units_per_core_group}},
            .hw_config = compute_hw_config,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute(COMPUTE_GROUP_1, units_per_core_group_1));
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_GROUP_2, units_per_core_group_2));
    }

    // The reader and the writer run on every node; the two compute specialisations split the nodes
    // between them, so each node still runs exactly one of each kind.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_GROUP_1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_GROUP_2},
            .target_nodes = core_group_2,
        });
    }

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

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
                {"C", channel_size},
                {"num_inner_tile", num_inner_tile},
                {"weight_num_tile", weight_num_tile},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
            });

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    run_args.tensor_args.emplace(TENSOR_OUTPUT_GRAD, output_grad_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, weight.value().mesh_tensor());
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, divisor.value().mesh_tensor());
    }
    run_args.tensor_args.emplace(TENSOR_INPUT_GRAD, input_grad_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_backward_impl_4d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool /*reduction_mean*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    uint32_t channel_size = input_grad_shape[1];

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    uint32_t num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_backward_4d";

    // create dataflow buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT_GRAD, 1, data_format));
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, weight_num_tile, data_format));
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));
    if (divisor_has_value) {
        // tmp1 and tmp2 are touched only by the compute kernel's divisor branch, so they exist
        // exactly when that branch does. Allocating them unconditionally would leave two buffers
        // with no producer and no consumer in the no-divisor build, which cannot be expressed.
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));
        spec.dataflow_buffers.push_back(make_dfb(DFB_TMP2, 1, fp32_dest_acc_en_data_format));
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT_GRAD, 1, data_format));

    if (weight_has_value) {
        // This buffer will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile of scratch, because content is read immediately after writing.
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT_SCRATCH, 1, data_format));
    }

    // declare the tensors the kernels operate on
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& input_grad_mesh = input_grad.mesh_tensor();

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_OUTPUT_GRAD, .spec = output_grad_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight.value().mesh_tensor().tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor.value().mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TENSOR_INPUT_GRAD, .spec = input_grad_mesh.tensor_spec()});

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
        compute_defines.emplace("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace("DIVISOR", "1");
        compute_defines.emplace("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_4d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    Group<DFBBinding> reader_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        // The reader is the only kernel that touches `target`: it fills the entry, waits on it,
        // reads it through a local L1 pointer and pops it. So it holds both ends of the buffer.
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TARGET,
            .accessor_name = "target",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (weight_has_value) {
        // `weight` is read once into L1 and held there for the whole kernel, and `weight_scratch`
        // never sees a FIFO operation at all. Both are reader-only, so the reader holds both ends.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_WEIGHT_SCRATCH,
            .accessor_name = "weight_scratch",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (divisor_has_value) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
        TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT_GRAD, .accessor_name = "output_grad"},
    };
    if (weight_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_WEIGHT, .accessor_name = "weight"});
    }
    if (divisor_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_DIVISOR, .accessor_name = "divisor"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"ignore_index", "num_tiles_per_core", "start_id", "C", "num_inner_tile", "weight_num_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_INPUT_GRAD,
                    .accessor_name = "input_grad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_INPUT_GRAD, .accessor_name = "input_grad"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT_GRAD,
            .accessor_name = "output_grad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_INPUT_GRAD,
            .accessor_name = "input_grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (divisor_has_value) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        // The compute kernel packs each intermediate and reads it straight back within its own
        // loop, so it holds both ends of tmp1 and tmp2.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP1,
            .accessor_name = "tmp1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP2,
            .accessor_name = "tmp2",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    ComputeUnpackModes compute_unpack_modes;
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_OUTPUT_GRAD, data_format);
    require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP_WEIGHT, fp32_dest_acc_en_data_format);
    if (divisor_has_value) {
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_DIVISOR, data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP1, fp32_dest_acc_en_data_format);
        require_unpack_mode(compute_unpack_modes, fp32_dest_acc_en, DFB_TMP2, fp32_dest_acc_en_data_format);
    }

    auto compute_hw_config = ttnn::to_compute_hardware_config(compute_kernel_config);
    compute_hw_config.unpack_modes = std::move(compute_unpack_modes);

    auto make_compute = [&](KernelSpecName unique_id, uint32_t units_per_core_group) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = compute_kernel_file,
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args = {{"per_core_tile_cnt", units_per_core_group}},
            .hw_config = compute_hw_config,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute(COMPUTE_GROUP_1, units_per_core_group_1));
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_GROUP_2, units_per_core_group_2));
    }

    // The reader and the writer run on every node; the two compute specialisations split the nodes
    // between them, so each node still runs exactly one of each kind.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_GROUP_1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_GROUP_2},
            .target_nodes = core_group_2,
        });
    }

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

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
                {"C", channel_size},
                {"num_inner_tile", num_inner_tile},
                {"weight_num_tile", weight_num_tile},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
            });

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    run_args.tensor_args.emplace(TENSOR_OUTPUT_GRAD, output_grad_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, weight.value().mesh_tensor());
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, divisor.value().mesh_tensor());
    }
    run_args.tensor_args.emplace(TENSOR_INPUT_GRAD, input_grad_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts MorehNllLossBackwardDeviceOperation::Factory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const Tensor& output_grad = tensor_args.output_grad_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;

    const bool reduction_mean = operation_attributes.reduction_mean;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    const Tensor& input_grad = tensor_return_value;

    // split work
    const auto& input_grad_shape = input_grad.logical_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_backward_impl_2d(
            target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_backward_impl_3d(
            target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_backward_impl_4d(
        target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
