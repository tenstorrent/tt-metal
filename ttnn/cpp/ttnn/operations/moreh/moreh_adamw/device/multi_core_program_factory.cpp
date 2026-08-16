// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <optional>
#include <utility>

#include "moreh_adamw_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/reader_moreh_adamw.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/writer_moreh_adamw.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp";

namespace {

// Work split used by create_program_artifacts to derive the core list and group membership.
// It carries no combined all-cores set: a kernel's placement now comes from its WorkUnitSpec
// membership, and the reader and writer belong to both groups' work units, which is the same
// node set the legacy descriptors named as all_cores.
struct AdamwWorkSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1 = 0;
    uint32_t num_units_per_core_group_2 = 0;
};

AdamwWorkSplit compute_adamw_work_split(const Tensor& param_in) {
    auto grid = param_in.device()->compute_with_storage_grid_size();
    uint32_t num_units = param_in.physical_volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(grid, num_units);
    return {num_cores, grid.y, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2};
}

// Spec names. The strings are the role names the kernels see (`dfb::param_in`, `tensor::grad`, ...);
// the C++ identifiers carry an ADAMW_ prefix because this file shares a unity-build translation unit
// with the other moreh factories, whose anonymous namespaces merge into one scope.
const KernelSpecName ADAMW_READER{"reader"};
const KernelSpecName ADAMW_WRITER{"writer"};
const KernelSpecName ADAMW_COMPUTE_G1{"compute_g1"};
const KernelSpecName ADAMW_COMPUTE_G2{"compute_g2"};

const DFBSpecName ADAMW_DFB_PARAM_IN{"param_in"};
const DFBSpecName ADAMW_DFB_GRAD{"grad"};
const DFBSpecName ADAMW_DFB_EXP_AVG_IN{"exp_avg_in"};
const DFBSpecName ADAMW_DFB_EXP_AVG_SQ_IN{"exp_avg_sq_in"};
const DFBSpecName ADAMW_DFB_MAX_EXP_AVG_SQ_IN{"max_exp_avg_sq_in"};
// Holds lr, beta1, beta2, eps and weight_decay, one per entry — hence 5 entries where every other
// buffer here holds 1.
const DFBSpecName ADAMW_DFB_SCALAR_ARGS{"scalar_args"};
const DFBSpecName ADAMW_DFB_ONE{"one"};
const DFBSpecName ADAMW_DFB_PARAM_OUT{"param_out"};
const DFBSpecName ADAMW_DFB_EXP_AVG_OUT{"exp_avg_out"};
const DFBSpecName ADAMW_DFB_EXP_AVG_SQ_OUT{"exp_avg_sq_out"};
const DFBSpecName ADAMW_DFB_MAX_EXP_AVG_SQ_OUT{"max_exp_avg_sq_out"};
const DFBSpecName ADAMW_DFB_TMP_PARAM{"tmp_param"};
const DFBSpecName ADAMW_DFB_TMP_EXP_AVG{"tmp_exp_avg"};
const DFBSpecName ADAMW_DFB_TMP_EXP_AVG_SQ{"tmp_exp_avg_sq"};
const DFBSpecName ADAMW_DFB_TMP_MAX_EXP_AVG_SQ{"tmp_max_exp_avg_sq"};
// pow(beta1, step) / pow(beta2, step), both precomputed on the host and filled into a buffer by the
// reader; the compute kernel derives the bias corrections from them.
const DFBSpecName ADAMW_DFB_BETA1_EXPONENT{"beta1_exponent"};
const DFBSpecName ADAMW_DFB_BETA2_EXPONENT{"beta2_exponent"};
const DFBSpecName ADAMW_DFB_TMP1{"tmp1"};
const DFBSpecName ADAMW_DFB_TMP2{"tmp2"};

const TensorParamName ADAMW_T_PARAM_IN{"param_in"};
const TensorParamName ADAMW_T_GRAD{"grad"};
const TensorParamName ADAMW_T_EXP_AVG_IN{"exp_avg_in"};
const TensorParamName ADAMW_T_EXP_AVG_SQ_IN{"exp_avg_sq_in"};
const TensorParamName ADAMW_T_MAX_EXP_AVG_SQ_IN{"max_exp_avg_sq_in"};
const TensorParamName ADAMW_T_PARAM_OUT{"param_out"};
const TensorParamName ADAMW_T_EXP_AVG_OUT{"exp_avg_out"};
const TensorParamName ADAMW_T_EXP_AVG_SQ_OUT{"exp_avg_sq_out"};
const TensorParamName ADAMW_T_MAX_EXP_AVG_SQ_OUT{"max_exp_avg_sq_out"};

DataflowBufferSpec MakeAdamwDFB(
    const DFBSpecName& unique_id, uint32_t entry_size, uint32_t num_entries, tt::DataFormat data_format) {
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts MorehAdamWDeviceOperation::MultiCoreProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& param_in = tensor_args.param_in;
    const Tensor& grad = tensor_args.grad;
    const Tensor& exp_avg_in = tensor_args.exp_avg_in;
    const Tensor& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    float lr = operation_attributes.lr;
    float beta1 = operation_attributes.beta1;
    float beta2 = operation_attributes.beta2;
    float eps = operation_attributes.eps;
    float weight_decay = operation_attributes.weight_decay;
    uint32_t step = operation_attributes.step;
    bool amsgrad = operation_attributes.amsgrad;

    const std::optional<Tensor>& max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;

    // It's guarantee that param_out, exp_avg_out, exp_avg_sq_out are created.
    const Tensor& param_out = tensor_return_value.at(0).value();
    const Tensor& exp_avg_out = tensor_return_value.at(1).value();
    const Tensor& exp_avg_sq_out = tensor_return_value.at(2).value();
    // Bound by reference rather than copied out: the framework resolves each TensorArgument to its
    // tensor by MeshTensor identity against the tensors reachable from tensor_args /
    // tensor_return_value, so a binding must name those objects and not a local copy of one.
    const Tensor* max_exp_avg_sq_out = amsgrad ? &tensor_return_value.at(3).value() : nullptr;

    DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto
        [num_cores, num_cores_y, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
            compute_adamw_work_split(param_in);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = datatype_to_dataformat_converter(param_in.dtype());
    auto intermed_dfb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t data_tile_size = tile_size(data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_dfb_format);

    Group<DataflowBufferSpec> dataflow_buffers{
        MakeAdamwDFB(ADAMW_DFB_PARAM_IN, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_GRAD, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_EXP_AVG_IN, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_EXP_AVG_SQ_IN, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_SCALAR_ARGS, intermed_tile_size, 5, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_ONE, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_PARAM_OUT, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_EXP_AVG_OUT, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_EXP_AVG_SQ_OUT, data_tile_size, 1, data_format),
        MakeAdamwDFB(ADAMW_DFB_TMP_PARAM, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_TMP_EXP_AVG, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_TMP_EXP_AVG_SQ, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_BETA1_EXPONENT, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_BETA2_EXPONENT, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_TMP1, intermed_tile_size, 1, intermed_dfb_format),
        MakeAdamwDFB(ADAMW_DFB_TMP2, intermed_tile_size, 1, intermed_dfb_format),
    };

    // The amsgrad-only buffers. The legacy factory allocated all three unconditionally, but every
    // kernel reference to them sits inside `#ifdef AMSGRAD`, so with amsgrad off no kernel is an
    // endpoint — and a DFB with neither a producer nor a consumer binding is rejected as invalid.
    // Declaring them on the amsgrad path is what makes both configurations expressible; it also
    // stops the three tiles of L1 per core that the unconditional allocation wasted when off.
    if (amsgrad) {
        dataflow_buffers.push_back(MakeAdamwDFB(ADAMW_DFB_MAX_EXP_AVG_SQ_IN, data_tile_size, 1, data_format));
        dataflow_buffers.push_back(MakeAdamwDFB(ADAMW_DFB_MAX_EXP_AVG_SQ_OUT, data_tile_size, 1, data_format));
        dataflow_buffers.push_back(
            MakeAdamwDFB(ADAMW_DFB_TMP_MAX_EXP_AVG_SQ, intermed_tile_size, 1, intermed_dfb_format));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TensorParameter Setup
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters{
        TensorParameter{.unique_id = ADAMW_T_PARAM_IN, .spec = param_in.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_GRAD, .spec = grad.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_EXP_AVG_IN, .spec = exp_avg_in.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_EXP_AVG_SQ_IN, .spec = exp_avg_sq_in.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_PARAM_OUT, .spec = param_out.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_EXP_AVG_OUT, .spec = exp_avg_out.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = ADAMW_T_EXP_AVG_SQ_OUT, .spec = exp_avg_sq_out.mesh_tensor().tensor_spec()},
    };
    if (amsgrad) {
        tensor_parameters.push_back(TensorParameter{
            .unique_id = ADAMW_T_MAX_EXP_AVG_SQ_IN, .spec = max_exp_avg_sq_in.value().mesh_tensor().tensor_spec()});
        tensor_parameters.push_back(TensorParameter{
            .unique_id = ADAMW_T_MAX_EXP_AVG_SQ_OUT, .spec = max_exp_avg_sq_out->mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compiler defines
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines data_movement_defines;
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (amsgrad) {
        data_movement_defines.emplace("AMSGRAD", "1");
        compute_defines.emplace("AMSGRAD", "1");
    }
    if (fp32_dest_acc_en) {
        data_movement_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    Group<DFBBinding> reader_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_PARAM_IN,
            .accessor_name = "param_in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_GRAD,
            .accessor_name = "grad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_IN,
            .accessor_name = "exp_avg_in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_SQ_IN,
            .accessor_name = "exp_avg_sq_in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_SCALAR_ARGS,
            .accessor_name = "scalar_args",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_ONE,
            .accessor_name = "one",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_BETA1_EXPONENT,
            .accessor_name = "beta1_exponent",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_BETA2_EXPONENT,
            .accessor_name = "beta2_exponent",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = ADAMW_T_PARAM_IN, .accessor_name = "param_in"},
        TensorBinding{.tensor_parameter_name = ADAMW_T_GRAD, .accessor_name = "grad"},
        TensorBinding{.tensor_parameter_name = ADAMW_T_EXP_AVG_IN, .accessor_name = "exp_avg_in"},
        TensorBinding{.tensor_parameter_name = ADAMW_T_EXP_AVG_SQ_IN, .accessor_name = "exp_avg_sq_in"},
    };
    if (amsgrad) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ADAMW_DFB_MAX_EXP_AVG_SQ_IN,
            .accessor_name = "max_exp_avg_sq_in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = ADAMW_T_MAX_EXP_AVG_SQ_IN, .accessor_name = "max_exp_avg_sq_in"});
    }

    Group<DFBBinding> writer_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_PARAM_OUT,
            .accessor_name = "param_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_OUT,
            .accessor_name = "exp_avg_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_SQ_OUT,
            .accessor_name = "exp_avg_sq_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    Group<TensorBinding> writer_tensor_bindings{
        TensorBinding{.tensor_parameter_name = ADAMW_T_PARAM_OUT, .accessor_name = "param_out"},
        TensorBinding{.tensor_parameter_name = ADAMW_T_EXP_AVG_OUT, .accessor_name = "exp_avg_out"},
        TensorBinding{.tensor_parameter_name = ADAMW_T_EXP_AVG_SQ_OUT, .accessor_name = "exp_avg_sq_out"},
    };
    if (amsgrad) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ADAMW_DFB_MAX_EXP_AVG_SQ_OUT,
            .accessor_name = "max_exp_avg_sq_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = ADAMW_T_MAX_EXP_AVG_SQ_OUT, .accessor_name = "max_exp_avg_sq_out"});
    }

    KernelSpec reader{
        .unique_id = ADAMW_READER,
        .source = READER_KERNEL_PATH,
        .compiler_options = {.defines = data_movement_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"lr",
                  "beta1",
                  "beta2",
                  "eps",
                  "weight_decay",
                  "beta1_exponent",
                  "beta2_exponent",
                  "step",
                  "amsgrad",
                  "num_tiles_per_core",
                  "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    KernelSpec writer{
        .unique_id = ADAMW_WRITER,
        .source = WRITER_KERNEL_PATH,
        .compiler_options = {.defines = data_movement_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Compute-kernel scratch: the compute kernel both fills and drains each of these, so it is the
    // buffer's only endpoint and takes both roles — a self-loop. The two roles share one accessor
    // name so the kernel drives the buffer through a single handle, and that name is the spec name
    // itself, which is why these are bound in a loop instead of spelled out like the bindings above.
    Group<DFBSpecName> compute_scratch{
        ADAMW_DFB_TMP_PARAM,
        ADAMW_DFB_TMP_EXP_AVG,
        ADAMW_DFB_TMP_EXP_AVG_SQ,
        ADAMW_DFB_TMP1,
        ADAMW_DFB_TMP2,
    };
    if (amsgrad) {
        compute_scratch.push_back(ADAMW_DFB_TMP_MAX_EXP_AVG_SQ);
    }

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_PARAM_IN,
            .accessor_name = "param_in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_GRAD,
            .accessor_name = "grad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_IN,
            .accessor_name = "exp_avg_in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_SQ_IN,
            .accessor_name = "exp_avg_sq_in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_SCALAR_ARGS,
            .accessor_name = "scalar_args",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_ONE,
            .accessor_name = "one",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_BETA1_EXPONENT,
            .accessor_name = "beta1_exponent",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_BETA2_EXPONENT,
            .accessor_name = "beta2_exponent",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_PARAM_OUT,
            .accessor_name = "param_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_OUT,
            .accessor_name = "exp_avg_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = ADAMW_DFB_EXP_AVG_SQ_OUT,
            .accessor_name = "exp_avg_sq_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    for (const DFBSpecName& scratch : compute_scratch) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = scratch,
            .accessor_name = scratch.get(),
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = scratch,
            .accessor_name = scratch.get(),
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (amsgrad) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ADAMW_DFB_MAX_EXP_AVG_SQ_IN,
            .accessor_name = "max_exp_avg_sq_in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ADAMW_DFB_MAX_EXP_AVG_SQ_OUT,
            .accessor_name = "max_exp_avg_sq_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    // The three fields the legacy ComputeConfigDescriptor set, and only those: it destructured
    // packer_l1_acc and dst_full_sync_en from the resolved TTNN config and then dropped both, so
    // they resolved to the descriptor's own defaults no matter what the caller asked for. Those
    // defaults coincide with the ones left unset here (double_buffer_dest == !dst_full_sync_en ==
    // true, bfp_pack_precision_mode == Approximate), which is why they are left alone rather than
    // carried across from the TTNN config.
    ComputeGen1Config compute_gen1_config{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };

    // With a 32-bit Dest register, an explicit unpack mode is required for every Float32 buffer the
    // compute kernel consumes — the Src-vs-Dest choice stops having a safe default. The legacy
    // config set no unpack_to_dest_mode at all, i.e. Default for every buffer, which is UnpackToSrc.
    // Only the intermediates can be Float32: the data buffers follow the input dtype, and the op
    // accepts BFLOAT16 / BFLOAT8_B only.
    if (fp32_dest_acc_en) {
        // Every buffer carrying intermed_dfb_format that the compute kernel consumes: the four the
        // reader fills, plus all the scratch, whose self-loop binds the consumer role too.
        Group<DFBSpecName> consumed_intermediates{
            ADAMW_DFB_SCALAR_ARGS, ADAMW_DFB_ONE, ADAMW_DFB_BETA1_EXPONENT, ADAMW_DFB_BETA2_EXPONENT};
        consumed_intermediates.insert(consumed_intermediates.end(), compute_scratch.begin(), compute_scratch.end());
        for (const DFBSpecName& consumed : consumed_intermediates) {
            compute_gen1_config.unpack_modes[consumed] = UnpackMode::UnpackToSrc;
        }
    }

    // One KernelSpec per core group, each baking in its group's tile count as a compile-time
    // constant, exactly as the legacy factory built one KernelDescriptor per group.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t per_core_tile_cnt) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = COMPUTE_KERNEL_PATH,
            // Legacy defaults opt_level per kernel type and lands on O3 for a compute kernel; Metal
            // 2.0's single CompilerOptions defaults to O2 for both kinds, so leaving it unset here
            // would quietly drop a level.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args = {{"per_core_tile_cnt", per_core_tile_cnt}},
            .runtime_arg_schema = {.runtime_arg_names = {"step"}},
            .hw_config = ComputeHardwareConfig{compute_gen1_config},
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels{reader, writer, make_compute(ADAMW_COMPUTE_G1, num_units_per_core_group_1)};
    if (has_core_group_2) {
        kernels.push_back(make_compute(ADAMW_COMPUTE_G2, num_units_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      WorkUnit SetUp
    ////////////////////////////////////////////////////////////////////////////
    // One work unit per core group, since the compute kernel is the only thing that differs between
    // them. The reader and writer belong to both, so their node set is the union — the all_cores the
    // legacy descriptors named directly.
    Group<WorkUnitSpec> work_units{
        WorkUnitSpec{
            .name = "group_1",
            .kernels = {ADAMW_READER, ADAMW_WRITER, ADAMW_COMPUTE_G1},
            .target_nodes = core_group_1,
        },
    };
    if (has_core_group_2) {
        work_units.push_back(WorkUnitSpec{
            .name = "group_2",
            .kernels = {ADAMW_READER, ADAMW_WRITER, ADAMW_COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    float beta1_exponent = std::pow(beta1, step);
    float beta2_exponent = std::pow(beta2, step);

    const uint32_t f2u_lr = std::bit_cast<uint32_t>(lr);
    const uint32_t f2u_beta1 = std::bit_cast<uint32_t>(beta1);
    const uint32_t f2u_beta2 = std::bit_cast<uint32_t>(beta2);
    const uint32_t f2u_eps = std::bit_cast<uint32_t>(eps);
    const uint32_t f2u_weight_decay = std::bit_cast<uint32_t>(weight_decay);
    const uint32_t f2u_beta1_exponent = std::bit_cast<uint32_t>(beta1_exponent);
    const uint32_t f2u_beta2_exponent = std::bit_cast<uint32_t>(beta2_exponent);

    KernelRunArgs reader_run_args{.kernel = ADAMW_READER};
    KernelRunArgs writer_run_args{.kernel = ADAMW_WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = ADAMW_COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = ADAMW_COMPUTE_G2};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"lr", f2u_lr},
             {"beta1", f2u_beta1},
             {"beta2", f2u_beta2},
             {"eps", f2u_eps},
             {"weight_decay", f2u_weight_decay},
             {"beta1_exponent", f2u_beta1_exponent},
             {"beta2_exponent", f2u_beta2_exponent},
             {"step", step},
             {"amsgrad", static_cast<uint32_t>(amsgrad)},
             {"num_tiles_per_core", num_tiles_per_core},
             {"start_id", tile_offset}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles_per_core", num_tiles_per_core}, {"start_id", tile_offset}});

        // compute — runtime args go to the KernelSpec placed on this core's group
        AddRuntimeArgsForNode(
            core_group_1.contains(core) ? compute_g1_run_args.runtime_arg_values
                                        : compute_g2_run_args.runtime_arg_values,
            core,
            {{"step", step}});

        tile_offset += num_tiles_per_core;
    }

    Group<KernelRunArgs> kernel_run_args;
    kernel_run_args.reserve(has_core_group_2 ? 4 : 3);
    kernel_run_args.push_back(std::move(reader_run_args));
    kernel_run_args.push_back(std::move(writer_run_args));
    kernel_run_args.push_back(std::move(compute_g1_run_args));
    if (has_core_group_2) {
        kernel_run_args.push_back(std::move(compute_g2_run_args));
    }

    Table<TensorParamName, TensorArgument> tensor_arg_values{
        {ADAMW_T_PARAM_IN, param_in.mesh_tensor()},
        {ADAMW_T_GRAD, grad.mesh_tensor()},
        {ADAMW_T_EXP_AVG_IN, exp_avg_in.mesh_tensor()},
        {ADAMW_T_EXP_AVG_SQ_IN, exp_avg_sq_in.mesh_tensor()},
        {ADAMW_T_PARAM_OUT, param_out.mesh_tensor()},
        {ADAMW_T_EXP_AVG_OUT, exp_avg_out.mesh_tensor()},
        {ADAMW_T_EXP_AVG_SQ_OUT, exp_avg_sq_out.mesh_tensor()},
    };
    if (amsgrad) {
        tensor_arg_values.emplace(ADAMW_T_MAX_EXP_AVG_SQ_IN, max_exp_avg_sq_in.value().mesh_tensor());
        tensor_arg_values.emplace(ADAMW_T_MAX_EXP_AVG_SQ_OUT, max_exp_avg_sq_out->mesh_tensor());
    }

    ProgramSpec spec{
        .name = "moreh_adamw",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args{
        .kernel_run_args = std::move(kernel_run_args),
        .tensor_args = std::move(tensor_arg_values),
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs
MorehAdamWDeviceOperation::MultiCoreProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // attribute_names excludes lr and step, so those two plus the beta exponents derived from step and
    // the tensor bindings are all that vary per dispatch; the work split is keyed by the shapes.
    const auto& param_in = tensor_args.param_in;
    const auto& max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;
    auto& param_out = tensor_return_value.at(0).value();
    auto& exp_avg_out = tensor_return_value.at(1).value();
    auto& exp_avg_sq_out = tensor_return_value.at(2).value();
    // create_program_artifacts declares this output's parameter only when amsgrad is on; the hit path
    // has to match or it would name a binding the spec never declared.
    const bool amsgrad = operation_attributes.amsgrad;
    const Tensor* max_exp_avg_sq_out = amsgrad ? &tensor_return_value.at(3).value() : nullptr;

    const uint32_t step = operation_attributes.step;
    const uint32_t f2u_lr = std::bit_cast<uint32_t>(operation_attributes.lr);
    const uint32_t f2u_beta1_exponent =
        std::bit_cast<uint32_t>(static_cast<float>(std::pow(operation_attributes.beta1, step)));
    const uint32_t f2u_beta2_exponent =
        std::bit_cast<uint32_t>(static_cast<float>(std::pow(operation_attributes.beta2, step)));

    ProgramRunArgs params;

    // Every tensor parameter, every dispatch. Nothing refreshes these on this factory's behalf, so a
    // binding left out here would stay frozen at the address the cache-miss dispatch wrote.
    params.tensor_args = {
        {ADAMW_T_PARAM_IN, param_in.mesh_tensor()},
        {ADAMW_T_GRAD, tensor_args.grad.mesh_tensor()},
        {ADAMW_T_EXP_AVG_IN, tensor_args.exp_avg_in.mesh_tensor()},
        {ADAMW_T_EXP_AVG_SQ_IN, tensor_args.exp_avg_sq_in.mesh_tensor()},
        {ADAMW_T_PARAM_OUT, param_out.mesh_tensor()},
        {ADAMW_T_EXP_AVG_OUT, exp_avg_out.mesh_tensor()},
        {ADAMW_T_EXP_AVG_SQ_OUT, exp_avg_sq_out.mesh_tensor()},
    };
    if (amsgrad) {
        params.tensor_args.emplace(ADAMW_T_MAX_EXP_AVG_SQ_IN, max_exp_avg_sq_in.value().mesh_tensor());
        params.tensor_args.emplace(ADAMW_T_MAX_EXP_AVG_SQ_OUT, max_exp_avg_sq_out->mesh_tensor());
    }

    // step also feeds the compute kernel(s); the second one exists only when the split has a remainder
    // group, so reuse create_program_artifacts' own work-split helper rather than rebuilding the spec.
    const auto split = compute_adamw_work_split(param_in);
    const bool has_core_group_2 = !split.core_group_2.ranges().empty();

    KernelRunArgs reader_run_args{.kernel = ADAMW_READER};
    KernelRunArgs compute_g1_run_args{.kernel = ADAMW_COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = ADAMW_COMPUTE_G2};

    // The same node walk create_program_artifacts uses, so the refreshed set is exactly the set that
    // was given values on the cache miss.
    for (uint32_t i = 0; i < split.num_cores; ++i) {
        CoreCoord core = {i / split.num_cores_y, i % split.num_cores_y};

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"lr", f2u_lr},
             {"beta1_exponent", f2u_beta1_exponent},
             {"beta2_exponent", f2u_beta2_exponent},
             {"step", step}});

        AddRuntimeArgsForNode(
            split.core_group_1.contains(core) ? compute_g1_run_args.runtime_arg_values
                                              : compute_g2_run_args.runtime_arg_values,
            core,
            {{"step", step}});
    }

    // No writer entry: the ported-from override refreshed only the writer's addresses, which are now
    // tensor bindings, and none of its runtime args.
    params.kernel_run_args.reserve(has_core_group_2 ? 3 : 2);
    params.kernel_run_args.push_back(std::move(reader_run_args));
    params.kernel_run_args.push_back(std::move(compute_g1_run_args));
    if (has_core_group_2) {
        params.kernel_run_args.push_back(std::move(compute_g2_run_args));
    }

    return params;
}

}  // namespace ttnn::operations::moreh::moreh_adamw
