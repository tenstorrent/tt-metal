// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_weighted_reduce_nc/device/attn_res_weighted_reduce_nc_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim::qsr {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_weighted_reduce_nc/device/kernels/";

}  // namespace

ttnn::device_operation::ProgramArtifacts AttnResWeightedReduceNCProgramFactory::create_program_artifacts(
    const AttnResWeightedReduceNCParams& operation_attributes,
    const AttnResWeightedReduceNCInputs& tensor_args,
    Tensor& tensor_return_value) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN0{"in0"};
    const DFBSpecName IN1{"in1"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName WEIGHT{"weight"};
    const TensorParamName OUTPUT{"output"};

    auto* device = tensor_args.input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(tensor_args.input.dtype());
    const auto input_tile_size = tt::tile_size(input_data_format);
    const auto weight_data_format = datatype_to_dataformat_converter(tensor_args.weight.dtype());
    const auto weight_tile_size = tt::tile_size(weight_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    // Validation pins rank to 4 and dim to 1, so there are no dims between the
    // reduced axis and the trailing two: the inner block is exactly Ht*Wt tiles.
    const auto& input_shape = tensor_args.input.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_candidates = input_shape[operation_attributes.dim];
    const uint32_t inner_tile_size = Ht * Wt;

    // The weight is [R, C, H, 1] in TILE layout, i.e. one tile column wide, so
    // its inner block is Ht tiles against the input's Ht*Wt. Every tile in an
    // input row of Wt shares one weight tile — that sharing is what the work
    // split below is arranged to exploit.
    const uint32_t weight_inner_tile_size = Ht;
    const uint32_t weight_reduce_tile_size = num_candidates * Ht;
    const uint32_t num_sites = tensor_args.weight.padded_shape()[0];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // Sites are reduced in groups, one DEST tile per site in the group, so a
    // group's worth of accumulators has to fit in DEST — which the sync mode and
    // fp32 accumulation each halve. Exceeding it does not fault, it corrupts the
    // tiles past the end, so the group size is derived rather than chosen.
    const uint32_t dest_capacity = get_dest_reg_count(operation_attributes.compute_kernel_config);
    const uint32_t sites_per_group = std::min(num_sites, dest_capacity);
    const uint32_t num_groups = (num_sites + sites_per_group - 1) / sites_per_group;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // Split the input's tile positions, not the output's tiles. Every site reads
    // the same input, so a core that owns a position produces that position for
    // all R sites and reads the candidates for it once per group. Splitting the
    // output instead would scatter one position across cores and each of them
    // would re-read the whole reduction.
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    auto [num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_tiles_group_1, num_tiles_group_2] =
        tt::tt_metal::split_work_to_cores(grid, inner_tile_size, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // No intermediate DFB: the running sums live in DEST for the whole reduction,
    // which is what lets one pass over the input do the multiply and the add.
    // A whole reduction's candidates are one DFB unit, so the group loop can walk
    // them repeatedly without the reader re-fetching between sites.
    DataflowBufferSpec in0_dfb{
        .unique_id = IN0,
        .entry_size = input_tile_size,
        .num_entries = num_candidates * 2,
        .data_format_metadata = input_data_format,
    };

    // One group's weight sets, single-buffered. The set turns over once every Wt
    // output tiles, so a prefetch buffer would idle through nearly the whole row
    // while doubling what is already the largest DFB here: sites_per_group * C
    // tiles, over a hundred kilobytes at the shape AttnRes asks for.
    DataflowBufferSpec in1_dfb{
        .unique_id = IN1,
        .entry_size = weight_tile_size,
        .num_entries = sites_per_group * num_candidates,
        .data_format_metadata = weight_data_format,
    };

    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_tile_size,
        .num_entries = sites_per_group * 2,
        .data_format_metadata = output_data_format,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Every stride is a compile-time arg. The reader divides by inner_tile_size
    // and by Wt once per output tile, and RISC-V has no divide instruction —
    // constants let the compiler emit multiply-shift instead of a libcall.
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(std::string(kKernelDir) + "reader_weighted_reduce_nc.cpp"),
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = WEIGHT, .accessor_name = "weight"},
            },
        .compile_time_args =
            {
                {"num_candidates", num_candidates},
                {"inner_tile_size", inner_tile_size},
                {"Wt", Wt},
                {"weight_inner_tile_size", weight_inner_tile_size},
                {"weight_reduce_tile_size", weight_reduce_tile_size},
                {"num_sites", num_sites},
                {"sites_per_group", sites_per_group},
                {"num_groups", num_groups},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_positions", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(std::string(kKernelDir) + "writer_weighted_reduce_nc.cpp"),
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"inner_tile_size", inner_tile_size},
                {"num_sites", num_sites},
                {"sites_per_group", sites_per_group},
                {"num_groups", num_groups},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_positions", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // One kernel over all cores, with the per-core tile count as a runtime
    // arg. fast_reduce_nc splits into two compute kernels to keep that count
    // compile-time; here the loop that matters — the MAC over candidates — is
    // already compile-time bounded, so the second kernel would buy nothing.
    ComputeHardwareConfig compute_hw_config{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = !dst_full_sync_en,
    };
    // A Float32 weight unpacked under fp32 accumulation needs an explicit mode; the legacy default was UnpackToSrc.
    if (fp32_dest_acc_en && weight_data_format == tt::DataFormat::Float32) {
        compute_hw_config.unpack_modes.emplace(IN1, UnpackMode::UnpackToSrc);
    }
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(std::string(kKernelDir) + "weighted_reduce_nc.cpp"),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {
                {"num_candidates", num_candidates},
                {"Wt", Wt},
                {"num_sites", num_sites},
                {"sites_per_group", sites_per_group},
                {"num_groups", num_groups},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_positions", "start_id"}},
        .hw_config = std::move(compute_hw_config),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Contiguous, not round-robin. fast_reduce_nc hands core i tiles
    // i, i+num_cores, i+2*num_cores, ... which is free when every output tile
    // reads a disjoint input column. It is not free here: consecutive strides of
    // num_cores land on a different token row almost every time, so the weight
    // set would be refetched per position — G times a pass over the weight.
    // Contiguous ranges give each core ~num_tiles/Wt distinct rows, so the
    // weights are read about once per group.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0, start_id = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core{i % num_cores_x, i / num_cores_x};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_positions", num_tiles_per_core}, {"start_id", start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_positions", num_tiles_per_core}, {"start_id", start_id}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"num_positions", num_tiles_per_core}, {"start_id", start_id}});

        start_id += num_tiles_per_core;
    }

    ProgramSpec spec{
        .name = "attn_res_weighted_reduce_nc",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = {std::move(in0_dfb), std::move(in1_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = tensor_args.input.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = WEIGHT, .spec = tensor_args.weight.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = tensor_return_value.mesh_tensor().tensor_spec()},
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

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT, tensor_args.input.mesh_tensor());
    run_args.tensor_args.emplace(WEIGHT, tensor_args.weight.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, tensor_return_value.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim::qsr
