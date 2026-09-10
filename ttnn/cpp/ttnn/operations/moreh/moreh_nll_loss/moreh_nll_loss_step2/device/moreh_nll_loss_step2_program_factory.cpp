// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_step2_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step2 {

using namespace tt::tt_metal::experimental;

namespace {

const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE_G1{"compute_g1"};
const KernelSpecName COMPUTE_G2{"compute_g2"};

const DFBSpecName DFB_INPUT{"input"};
const DFBSpecName DFB_TARGET{"target"};
const DFBSpecName DFB_WEIGHT{"weight"};
const DFBSpecName DFB_DIVISOR{"divisor"};
const DFBSpecName DFB_WEIGHT_SCRATCH{"weight_scratch"};
const DFBSpecName DFB_TMP_WEIGHT{"tmp_weight"};
const DFBSpecName DFB_TMP_INPUT{"tmp_input"};
const DFBSpecName DFB_TMP1{"tmp1"};
// Holds 1/divisor. (The name follows the compute kernel's use of the buffer, which is what the
// kernel reads and writes; it is the accurate description of what lands here.)
const DFBSpecName DFB_DIVISOR_RECIP{"divisor_recip"};
const DFBSpecName DFB_TMP3{"tmp3"};
const DFBSpecName DFB_OUTPUT{"output"};

const TensorParamName TENSOR_INPUT{"input"};
const TensorParamName TENSOR_TARGET{"target"};
const TensorParamName TENSOR_WEIGHT{"weight"};
const TensorParamName TENSOR_DIVISOR{"divisor"};
const TensorParamName TENSOR_OUTPUT{"output"};

const auto* const COMPUTE_KERNEL_FILE =
    "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
    "moreh_nll_loss_step2_kernel.cpp";

// Helper: a dataflow buffer holding a whole number of tiles of `data_format`.
DataflowBufferSpec make_dfb(const DFBSpecName& unique_id, uint32_t num_entries, tt::DataFormat data_format) {
    const auto tile_sz = tt::tile_size(data_format);
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = tile_sz,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

// Helper: bind `dfb` to `kernel_bindings` as both producer and consumer under one accessor name.
// Used for the buffers a single kernel holds both ends of, which therefore have nobody else to
// take the opposite endpoint: the reader's staging buffers (it fills each one, waits on it, reads
// it back through a local L1 pointer and pops it) and the compute kernel's scratch tiles. A buffer
// touched with no FIFO calls at all -- the 4d reader's weight scratch, which is only an async_read
// destination -- gets the same treatment, and there the two labels are purely nominal.
void bind_self_loop(Group<DFBBinding>& bindings, const DFBSpecName& dfb, std::string_view accessor_name) {
    bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
}

// Helper: the compute kernels' hardware configuration, translated from the resolved
// ComputeKernelConfig the op was given.
//
// The four knobs the op sets -- math fidelity, the approximate/precise SFPU mode, the 32-bit Dest
// register, and Dest double-buffering -- all cross over inside the helper. packer_l1_acc is
// resolved by the op but never used by it and has no counterpart here. The block-float pack
// precision stays at its default, which is the value the op's config carried.
//
// unpack_modes is the one field that has to be set by hand. The op asked for the default unpack
// mode on every buffer, which is UnpackToSrc and would normally be left implicit. But a compute
// kernel consuming a Float32 buffer while the Dest register is 32-bit has to state the mode
// explicitly, so where the 32-bit Dest register makes the five intermediates Float32, each gets an
// explicit UnpackToSrc -- the same mode as before, now spelled out. All five are bound by the
// compute kernel in every configuration, so the table needs no further conditions.
//
// It is reached through the unpack_modes() accessor rather than std::get<ComputeGen1Config>, which
// would throw on a Gen2 (Quasar) target.
ComputeHardwareConfig make_compute_hw_config(
    tt::ARCH arch, const DeviceComputeKernelConfig& compute_kernel_config, bool fp32_dest_acc_en) {
    auto hw_config = ttnn::to_compute_hardware_config(arch, compute_kernel_config);
    if (fp32_dest_acc_en) {
        unpack_modes(hw_config) = {
            {DFB_TMP_WEIGHT, UnpackMode::UnpackToSrc},
            {DFB_TMP_INPUT, UnpackMode::UnpackToSrc},
            {DFB_TMP1, UnpackMode::UnpackToSrc},
            {DFB_DIVISOR_RECIP, UnpackMode::UnpackToSrc},
            {DFB_TMP3, UnpackMode::UnpackToSrc},
        };
    }
    return hw_config;
}

// Helper: build one instance of the compute kernel.
//
// The kernel is instantiated once per work-split group, the instances differing only in
// `per_core_tile_cnt`. That count stays a compile-time argument: the kernel's main loop is bound by
// it, so demoting it to a runtime argument would cost the compile-time unrolling of that loop.
//
// opt_level is stated explicitly. A compute kernel's optimization level used to default to O3 while
// CompilerOptions defaults to O2, so leaving it unset here would quietly drop a level on both the
// compile and the link. Building both instances through this one helper is what keeps the level --
// and the unpack-mode table -- from being set on one instance and missed on the other.
KernelSpec make_compute_spec(
    const KernelSpecName& unique_id,
    uint32_t per_core_tile_cnt,
    bool weight_has_value,
    bool divisor_has_value,
    const ComputeHardwareConfig& compute_hw_config,
    const KernelSpec::CompilerOptions::Defines& defines) {
    Group<DFBBinding> dfb_bindings{
        DFBBinding{
            .dfb_spec_name = DFB_TMP_INPUT,
            .accessor_name = "tmp_input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DFB_OUTPUT,
            .accessor_name = "output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };

    if (weight_has_value) {
        // The reader fills tmp_weight with the per-target weights; this kernel multiplies by them.
        dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        // Without a weight tensor nothing fills tmp_weight and nothing reads it -- but the kernel
        // still names the buffer unconditionally, when configuring the Tensix pipeline and when
        // constructing its accessor, so the buffer has to exist in every configuration. This
        // kernel is its only toucher, so it holds both ends.
        bind_self_loop(dfb_bindings, DFB_TMP_WEIGHT, "tmp_weight");
    }

    if (divisor_has_value) {
        dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_DIVISOR,
            .accessor_name = "divisor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // The three scratch tiles are private to this kernel: it packs a partial result into each and
    // reads it straight back. Bound unconditionally, because the kernel constructs an accessor for
    // each one whether or not the configuration reaches the arithmetic that uses it.
    bind_self_loop(dfb_bindings, DFB_TMP1, "tmp1");
    bind_self_loop(dfb_bindings, DFB_DIVISOR_RECIP, "divisor_recip");
    bind_self_loop(dfb_bindings, DFB_TMP3, "tmp3");

    return KernelSpec{
        .unique_id = unique_id,
        .source = COMPUTE_KERNEL_FILE,
        .compiler_options = {.defines = defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(dfb_bindings),
        .compile_time_args = {{"per_core_tile_cnt", per_core_tile_cnt}},
        .hw_config = compute_hw_config,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_step2_impl_2d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.padded_shape();

    auto N = input_shape[0];

    // copy 32 Bytes per core
    uint32_t units_to_divide = N / tt::constants::TILE_HEIGHT;
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_step2_2d";

    // create dataflow buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT, 1, data_format));             // input
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));  // target
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, 1, data_format));  // weight
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));  // divisor
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));  // tmp_weight to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_INPUT, 1, fp32_dest_acc_en_data_format));   // tmp_input to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));        // tmp1
    spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR_RECIP, 1, fp32_dest_acc_en_data_format));  // 1/divisor
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP3, 1, fp32_dest_acc_en_data_format));           // tmp3
    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT, 1, data_format));                          // output

    // No weight-scratch buffer on this path: the 2d reader reads each weight one value at a time
    // and never uses a scratch tile. (The 4d reader does, and allocates one there.)

    // declare the tensors the kernels operate on
    const auto& input_mesh = input.mesh_tensor();
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    const MeshTensor* const weight_mesh = weight_has_value ? &weight.value().mesh_tensor() : nullptr;
    const MeshTensor* const divisor_mesh = divisor_has_value ? &divisor.value().mesh_tensor() : nullptr;

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_INPUT, .spec = input_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight_mesh->tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor_mesh->tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_OUTPUT, .spec = output_mesh.tensor_spec()});

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

    // The reader is the only kernel that touches input, target and weight: it fills each one, waits
    // on it, reads it back through a local L1 pointer and pops it. So it holds both ends of those.
    // tmp_input and tmp_weight it fills for the compute kernel to consume, and the divisor tile it
    // fetches once for the same.
    Group<DFBBinding> reader_dfb_bindings;
    bind_self_loop(reader_dfb_bindings, DFB_INPUT, "input");
    bind_self_loop(reader_dfb_bindings, DFB_TARGET, "target");
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_TMP_INPUT,
        .accessor_name = "tmp_input",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (weight_has_value) {
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT, "weight");
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
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
        TensorBinding{.tensor_parameter_name = TENSOR_INPUT, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
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
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "reader_moreh_nll_loss_step2_2d.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema = {.runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "writer_moreh_nll_loss_step2_2d.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT, .accessor_name = "output"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    const auto compute_hw_config = make_compute_hw_config(device->arch(), compute_kernel_config, fp32_dest_acc_en);

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute_spec(
        COMPUTE_G1, units_per_core_group_1, weight_has_value, divisor_has_value, compute_hw_config, compute_defines));

    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute_spec(
            COMPUTE_G2,
            units_per_core_group_2,
            weight_has_value,
            divisor_has_value,
            compute_hw_config,
            compute_defines));
    }

    // One work unit per work-split group. The reader and writer belong to both, so they run on
    // every core; each compute instance belongs to only its own group.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_G2},
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
                {"ignore_index", static_cast<uint32_t>(ignore_index)},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"N", origin_N},
                {"C", origin_C},
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
    // The compute kernel reads no runtime arguments, so it needs no run-args entry.

    run_args.tensor_args.emplace(TENSOR_INPUT, input_mesh);
    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, *weight_mesh);
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, *divisor_mesh);
    }
    run_args.tensor_args.emplace(TENSOR_OUTPUT, output_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_step2_impl_3d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // split work
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_W = input_shape_without_padding[2];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy FACE_WIDTH per core
    uint32_t units_to_divide = origin_N * div_up(origin_W, tt::constants::FACE_WIDTH);

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_step2_3d";

    // create dataflow buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT, 1, data_format));             // input
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));  // target
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, 1, data_format));  // weight
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));  // divisor
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));  // tmp_weight to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_INPUT, 1, fp32_dest_acc_en_data_format));   // tmp_input to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));        // tmp1
    spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR_RECIP, 1, fp32_dest_acc_en_data_format));  // 1/divisor
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP3, 1, fp32_dest_acc_en_data_format));           // tmp3
    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT, 1, data_format));                          // output

    // No weight-scratch buffer on this path: the 3d reader reads each weight one value at a time
    // and never uses a scratch tile. (The 4d reader does, and allocates one there.)

    // declare the tensors the kernels operate on
    const auto& input_mesh = input.mesh_tensor();
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    const MeshTensor* const weight_mesh = weight_has_value ? &weight.value().mesh_tensor() : nullptr;
    const MeshTensor* const divisor_mesh = divisor_has_value ? &divisor.value().mesh_tensor() : nullptr;

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_INPUT, .spec = input_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight_mesh->tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor_mesh->tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_OUTPUT, .spec = output_mesh.tensor_spec()});

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

    // The reader is the only kernel that touches input, target and weight: it fills each one, waits
    // on it, reads it back through a local L1 pointer and pops it. So it holds both ends of those.
    // tmp_input and tmp_weight it fills for the compute kernel to consume, and the divisor tile it
    // fetches once for the same.
    Group<DFBBinding> reader_dfb_bindings;
    bind_self_loop(reader_dfb_bindings, DFB_INPUT, "input");
    bind_self_loop(reader_dfb_bindings, DFB_TARGET, "target");
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_TMP_INPUT,
        .accessor_name = "tmp_input",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (weight_has_value) {
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT, "weight");
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
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
        TensorBinding{.tensor_parameter_name = TENSOR_INPUT, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
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
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "reader_moreh_nll_loss_step2_3d.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names = {"ignore_index", "num_tiles_per_core", "start_id", "C", "W", "element_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "writer_moreh_nll_loss_step2_3d.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT, .accessor_name = "output"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id", "W", "element_size"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    const auto compute_hw_config = make_compute_hw_config(device->arch(), compute_kernel_config, fp32_dest_acc_en);

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute_spec(
        COMPUTE_G1, units_per_core_group_1, weight_has_value, divisor_has_value, compute_hw_config, compute_defines));

    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute_spec(
            COMPUTE_G2,
            units_per_core_group_2,
            weight_has_value,
            divisor_has_value,
            compute_hw_config,
            compute_defines));
    }

    // One work unit per work-split group. The reader and writer belong to both, so they run on
    // every core; each compute instance belongs to only its own group.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    const uint32_t input_element_size = input.element_size();
    const uint32_t output_element_size = output.element_size();

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
                {"ignore_index", static_cast<uint32_t>(ignore_index)},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"C", origin_C},
                {"W", origin_W},
                {"element_size", input_element_size},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"W", origin_W},
                {"element_size", output_element_size},
            });

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    // The compute kernel reads no runtime arguments, so it needs no run-args entry.

    run_args.tensor_args.emplace(TENSOR_INPUT, input_mesh);
    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, *weight_mesh);
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, *divisor_mesh);
    }
    run_args.tensor_args.emplace(TENSOR_OUTPUT, output_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts moreh_nll_loss_step2_impl_4d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.padded_shape();
    auto target_shape = target.padded_shape();
    auto N = input_shape[0];
    uint32_t channel_size = input_shape[1];

    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    uint32_t num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy TILE per loop
    uint32_t units_to_divide = target.physical_volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_step2_4d";

    // create dataflow buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, tt::constants::TILE_WIDTH);

    spec.dataflow_buffers.push_back(make_dfb(DFB_INPUT, 1, data_format));             // input
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, 1, tt::DataFormat::Int32));  // target
    if (weight_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, weight_num_tile, data_format));  // weight
    }
    if (divisor_has_value) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR, 1, data_format));  // divisor
    }
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_WEIGHT, 1, fp32_dest_acc_en_data_format));  // tmp_weight to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP_INPUT, 1, fp32_dest_acc_en_data_format));   // tmp_input to reduce
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP1, 1, fp32_dest_acc_en_data_format));        // tmp1
    spec.dataflow_buffers.push_back(make_dfb(DFB_DIVISOR_RECIP, 1, fp32_dest_acc_en_data_format));  // 1/divisor
    spec.dataflow_buffers.push_back(make_dfb(DFB_TMP3, 1, fp32_dest_acc_en_data_format));           // tmp3
    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT, 1, data_format));                          // output

    if (weight_has_value) {
        // This buffer will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile of scratch, because content is read immediately
        // after writing.
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT_SCRATCH, 1, data_format));
    }

    // declare the tensors the kernels operate on
    const auto& input_mesh = input.mesh_tensor();
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    const MeshTensor* const weight_mesh = weight_has_value ? &weight.value().mesh_tensor() : nullptr;
    const MeshTensor* const divisor_mesh = divisor_has_value ? &divisor.value().mesh_tensor() : nullptr;

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_INPUT, .spec = input_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight_mesh->tensor_spec()});
    }
    if (divisor_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_DIVISOR, .spec = divisor_mesh->tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_OUTPUT, .spec = output_mesh.tensor_spec()});

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

    // The reader is the only kernel that touches input, target, weight and the weight scratch: it
    // fills each one and reads it back itself, so it holds both ends of those. (The scratch makes
    // no FIFO calls at all -- it is only an async_read destination -- so its two labels are purely
    // nominal; the buffer still needs both ends declared.) tmp_input and tmp_weight it fills for
    // the compute kernel to consume, and the divisor tile it fetches once for the same.
    Group<DFBBinding> reader_dfb_bindings;
    bind_self_loop(reader_dfb_bindings, DFB_INPUT, "input");
    bind_self_loop(reader_dfb_bindings, DFB_TARGET, "target");
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_TMP_INPUT,
        .accessor_name = "tmp_input",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (weight_has_value) {
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT, "weight");
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT_SCRATCH, "weight_scratch");
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFB_TMP_WEIGHT,
            .accessor_name = "tmp_weight",
            .endpoint_type = DFBEndpointType::PRODUCER,
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
        TensorBinding{.tensor_parameter_name = TENSOR_INPUT, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
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
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "reader_moreh_nll_loss_step2_4d.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"ignore_index", "num_tiles_per_core", "start_id", "C", "num_inner_tile", "weight_num_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
            "writer_moreh_nll_loss_step2_4d.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT, .accessor_name = "output"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    const auto compute_hw_config = make_compute_hw_config(device->arch(), compute_kernel_config, fp32_dest_acc_en);

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute_spec(
        COMPUTE_G1, units_per_core_group_1, weight_has_value, divisor_has_value, compute_hw_config, compute_defines));

    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute_spec(
            COMPUTE_G2,
            units_per_core_group_2,
            weight_has_value,
            divisor_has_value,
            compute_hw_config,
            compute_defines));
    }

    // One work unit per work-split group. The reader and writer belong to both, so they run on
    // every core; each compute instance belongs to only its own group.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "group_1",
        .kernels = {READER, WRITER, COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (has_core_group_2) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {READER, WRITER, COMPUTE_G2},
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
                {"ignore_index", static_cast<uint32_t>(ignore_index)},
                {"num_tiles_per_core", units_per_core},
                {"start_id", tile_offset},
                {"C", origin_C},
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
    // The compute kernel reads no runtime arguments, so it needs no run-args entry.

    run_args.tensor_args.emplace(TENSOR_INPUT, input_mesh);
    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, *weight_mesh);
    }
    if (divisor_has_value) {
        run_args.tensor_args.emplace(TENSOR_DIVISOR, *divisor_mesh);
    }
    run_args.tensor_args.emplace(TENSOR_OUTPUT, output_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts MorehNllLossStep2DeviceOperation::Factory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;
    const Tensor& output = tensor_return_value;
    const std::string reduction = operation_attributes.reduction;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // split work
    auto input_shape = input.padded_shape();
    auto rank = input_shape.rank();

    if (rank == 2) {
        return moreh_nll_loss_step2_impl_2d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    }
    if (rank == 3) {
        return moreh_nll_loss_step2_impl_3d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_step2_impl_4d(
        input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step2
