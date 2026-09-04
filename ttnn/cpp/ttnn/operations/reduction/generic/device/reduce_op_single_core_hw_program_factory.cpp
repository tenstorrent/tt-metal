// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <bit>
#include <cmath>
#include <variant>

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device().arch(), operation_attributes.compute_kernel_config);

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    TT_FATAL(
        operation_attributes.dim == ReduceOpDim::HW,
        "ReduceSingleCoreHwProgramFactory supports HW dim only, got dim enum value {}",
        static_cast<int>(operation_attributes.dim));

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension). Here we compensate with
    // sqrt(scaler). However, sqrt of a negative number is NaN, so negative scalers
    // must not reach this code path. Instead negative scalers are handled via the two-step
    // W-then-H path where the scaler is applied once (see the reduce function in reduce_op.cpp).
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    float scaler = std::sqrt(operation_attributes.scaler);

    TT_FATAL(
        H % tile_height == 0 && W % tile_width == 0, "Reduce HW expects tile-aligned padded shape H={}, W={}", H, W);
    uint32_t num_tensor_tiles = NC * H * W / tile_hw;
    const uint32_t num_tensor_tiles_ht_wt = NC * Ht * Wt;
    TT_FATAL(
        num_tensor_tiles == num_tensor_tiles_ht_wt,
        "Reduce HW tile count mismatch: tile_hw path={} vs Ht*Wt path={}",
        num_tensor_tiles,
        num_tensor_tiles_ht_wt);

    NodeCoord selected_node_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_node_coord = r.start_coord;
        TT_FATAL(
            operation_attributes.sub_core_grids->contains(selected_node_coord),
            "Selected core {} must be contained in provided sub_core_grids {}",
            selected_node_coord,
            *operation_attributes.sub_core_grids);
    }

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the reduce factory .cpp files land in the same unity-build
    // translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName ACC_DFB{"acc"};
    const DFBSpecName INEG_DFB{"ineg"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "reduce_single_core_hw";

    // ---- Dataflow buffers ----
    // One core owns every tile, so a tensor smaller than a batch stays unbatched.
    const uint32_t reader_tiles_per_batch = reduce_reader_batch(num_tensor_tiles);
    const uint32_t num_input_tiles = reduce_reader_input_cb_tiles(reader_tiles_per_batch);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format,
    });
    constexpr uint32_t num_output_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dst_cb_data_format,
    });
    if (operation_attributes.negate) {
        // acc holds the running negated reduction; ineg holds the negated input tile. Both are
        // compute-private scratch: the compute kernel packs into them and unpacks back out.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    // ---- Reader kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                  "reader_unary_reduce_universal_start_id.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reduce_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALER_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args =
            {{"scaler_bits", std::bit_cast<uint32_t>(scaler)}, {"tiles_per_batch", reader_tiles_per_batch}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device().arch()),
    });

    // ---- Writer kernel ----
    // Metal 2.0 fork of the eltwise/unary writer; its binding vocabulary (dfb::out, tensor::dst,
    // RTAs num_pages / start_id) is the fork's, not this op's.
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                  "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device().arch()),
    });

    // ---- Compute kernel ----
    // Legacy resolved a TTNN ComputeKernelConfig but forwarded only math_fidelity and
    // fp32_dest_acc_en onto ComputeConfigDescriptor, leaving math_approx_mode and dst_full_sync_en
    // at the *Metal* descriptor defaults (both false). Reproduce that exactly: the TTNN helper would
    // otherwise carry the caller's math_approx_mode into sfpu_precision_mode and the caller's
    // dst_full_sync_en into double_buffer_dest, silently changing precision / Dest buffering.
    auto compute_hw = ttnn::to_compute_hardware_config(a.device().arch(), operation_attributes.compute_kernel_config);
    // std::visit rather than a Gen1-only get_if: to_compute_hardware_config yields a
    // ComputeGen2Config on Quasar, and the three fields set below exist on both generations.
    // The explicit-unpack-mode requirement in particular is enforced generation-agnostically, so a
    // Gen1-only branch would leave FP32 + 32-bit-Dest programs failing ProgramSpec validation there.
    std::visit(
        [&](auto& compute_cfg) {
            compute_cfg.sfpu_precision_mode = Precision::Precise;  // legacy math_approx_mode = false
            compute_cfg.double_buffer_dest = true;                 // legacy dst_full_sync_en = false
            // Legacy left unpack_to_dest_mode unset (all Default = UnpackToSrc). Metal 2.0 nonetheless
            // requires an explicit mode for every Float32 buffer this kernel consumes under a 32-bit
            // Dest register, so state the legacy value for those.
            auto require_explicit_unpack_mode = [&](const DFBSpecName& name, tt::DataFormat format) {
                if (fp32_dest_acc_en && format == tt::DataFormat::Float32) {
                    compute_cfg.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
                }
            };
            require_explicit_unpack_mode(IN_DFB, src0_cb_data_format);
            require_explicit_unpack_mode(SCALER_DFB, scaler_cb_data_format);
            if (operation_attributes.negate) {
                require_explicit_unpack_mode(ACC_DFB, dst_cb_data_format);
                require_explicit_unpack_mode(INEG_DFB, dst_cb_data_format);
            }
        },
        compute_hw);

    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN_DFB,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = SCALER_DFB,
            .accessor_name = "scaler",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    if (operation_attributes.negate) {
        // Self-loops: the compute kernel is the only toucher of acc / ineg.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ACC_DFB,
            .accessor_name = "acc",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = ACC_DFB,
            .accessor_name = "acc",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INEG_DFB,
            .accessor_name = "ineg",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INEG_DFB,
            .accessor_name = "ineg",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // MIN on Int32 uses -MAX(-x) in reduce_hw_neg.
    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
        (operation_attributes.negate ? "_hw_neg" : "") + ".cpp";

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = compute_kernel,
        // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so the
        // level has to be stated explicitly to keep the compute kernel where it was.
        .compiler_options =
            {.defines = KernelSpec::CompilerOptions::Defines(reduce_defines),
             .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                {"NC", NC},
                // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
                {"post_mul_scaler_bits", post_mul_scaler_bits},
                // enable_fp32_sfpu: always 0 (accurate fp32 HW is forced to the two-step W-then-H path)
                {"enable_fp32_sfpu", 0u},
            },
        .hw_config = compute_hw,
    });

    // ---- Work unit (placement) ----
    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = selected_node_coord});

    // ---- Runtime args ----
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    uint32_t out_dim_divider = Ht * Wt;
    TT_FATAL(
        num_tensor_tiles % out_dim_divider == 0,
        "Reduce HW per-core input tiles {} must be divisible by Ht*Wt={}",
        num_tensor_tiles,
        out_dim_divider);

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(KernelRunArgs{
        .kernel = READER,
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(selected_node_coord, {{"num_tiles", num_tensor_tiles}, {"start_id", 0u}}),
    });
    run_args.kernel_run_args.push_back(KernelRunArgs{
        .kernel = WRITER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            selected_node_coord, {{"num_pages", num_tensor_tiles / out_dim_divider}, {"start_id", 0u}}),
    });

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{a});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
