// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <map>
#include <string>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    // Metal 2.0 named resource handles for the interleaved multi-core matmul ProgramSpec.
    // (Declared as locals — not in a file-scope anon namespace — to avoid unity-build collisions.)
    const DFBSpecName IN0_DFB{"in0"};  // legacy CB c_0
    const DFBSpecName IN1_DFB{"in1"};  // legacy CB c_1
    const DFBSpecName OUT_DFB{"out"};  // legacy CB c_16
    const TensorParamName IN0_TENSOR{"in0"};
    const TensorParamName IN1_TENSOR{"in1"};
    const TensorParamName OUT_TENSOR{"out"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
    const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};
    // Forked Metal 2.0 kernel sources (the legacy copies remain for the generic-op gtest, which
    // still constructs a ProgramDescriptor that binds the positional-arg originals).
    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_8bank_output_tiles_partitioned_metal2.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp";
    constexpr const char* COMPUTE_PATH = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_metal2.cpp";

    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0).mesh_tensor();
    const auto& b = tensor_args.input_tensors.at(1).mesh_tensor();
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt::tt_metal::IDevice* device = &a.mutable_device();
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());
    (void)packer_l1_acc;

    const auto& cshape = output.padded_shape();  // C=A*B, N1MK*11KN->N1MN

    TT_FATAL(
        operation_attributes.program_config.has_value(),
        "program_config must be provided for MatmulMultiCoreProgramFactory");
    auto pc = std::get<operations::matmul::MatmulMultiCoreProgramConfig>(operation_attributes.program_config.value());
    if (!pc.allowed_worker_cores.has_value()) {
        log_warning(
            tt::LogOp,
            "MatmulMultiCoreProgramFactory: program_config.allowed_worker_cores not populated; auto-populating "
            "from device compute_with_storage_grid_size. Callers that bypass ttnn::prim::matmul() should invoke "
            "ttnn::operations::matmul::normalize_program_config() on the program config first. This will become "
            "a hard error in a future release.");
        auto device_grid = device->compute_with_storage_grid_size();
        pc.allowed_worker_cores =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(device_grid.x - 1, device_grid.y - 1)));
    }
    auto compute_with_storage_grid_size = pc.allowed_worker_cores.value().bounding_box().grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers (legacy CBs)
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;
    DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = in0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = in0_data_format,
    };
    DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = in1_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = in1_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_params = {
        TensorParameter{.unique_id = IN0_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = IN1_TENSOR, .spec = b.tensor_spec()},
        TensorParameter{.unique_id = OUT_TENSOR, .spec = output.tensor_spec()}};

    ////////////////////////////////////////////////////////////////////////////
    //                      Reader kernel
    ////////////////////////////////////////////////////////////////////////////
    // last_ktile_w handles a logical K that is not a multiple of TILE_WIDTH (the reader pads the
    // last K tile). last_ktile_h is always 0 here (no transposed-last-tile padding).
    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    uint32_t last_ktile_h = 0;

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = IN0_DFB, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = IN0_TENSOR, .accessor_name = "src0"},
             TensorBinding{.tensor_parameter_name = IN1_TENSOR, .accessor_name = "src1"}},
        .compile_time_args = {{"in0_last_ktile_w", last_ktile_w}, {"in0_last_ktile_h", last_ktile_h}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"Mt",
                  "Kt",
                  "Nt",
                  "MtKt",
                  "KtNt",
                  "batch",
                  "bcast_B",
                  "output_tile_start_id",
                  "num_output_tiles",
                  "MtNt"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Writer kernel
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    Group<KernelSpec> kernels = {reader_spec, writer_spec};

    ////////////////////////////////////////////////////////////////////////////
    //                      Compute kernel(s)
    ////////////////////////////////////////////////////////////////////////////
    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // One compute KernelSpec per work-split core group, differing only in the per-group output-tile
    // count (carried as a compile-time arg — multiplicity preserved, not demoted to an RTA).
    // bmm compute kernel: B, Mt, Nt are just 3 for loops that act as 1 large loop, so only set Nt.
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_output_tiles_per_core_group) {
        ComputeHardwareConfig compute_hw{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode};
        // For a Float32 input DFB consumed under fp32_dest_acc_en, the validator requires an explicit
        // unpack-to-dest mode entry; legacy ComputeConfig defaulted the whole vector to Default.
        //
        // We MUST use Default (unpack via SrcA/B), NOT UnpackToDestFp32. matmul_tiles is an FPU
        // binary op: it reads its two operands from the SrcA/SrcB register files. UnpackToDestFp32
        // routes the unpacker straight to the Dest register and DISABLES SrcA/B access for that DFB,
        // so the FPU would multiply garbage -> inf/wrong results. fp32_dest_acc_en already gives the
        // 32-bit-wide Dest accumulator the matmul needs; the inputs still go through SrcA/B (~19-bit),
        // exactly as the legacy factory (all-Default vector) did.
        if (fp32_dest_acc_en) {
            if (in0_data_format == tt::DataFormat::Float32) {
                compute_hw.unpack_to_dest_mode.insert({IN0_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
            if (in1_data_format == tt::DataFormat::Float32) {
                compute_hw.unpack_to_dest_mode.insert({IN1_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path{COMPUTE_PATH},
            .compiler_options = {.defines = Table<std::string, std::string>(mm_kernel_defines)},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN0_DFB, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"batch", 1}, {"Mt", 1}, {"Kt", Kt}, {"Nt", num_output_tiles_per_core_group}},
            .hw_config = compute_hw,
        };
    };

    const bool group_2_present = !core_group_2.ranges().empty();
    kernels.push_back(make_compute_spec(COMPUTE_KERNEL_G1, num_output_tiles_per_core_group_1));
    if (group_2_present) {
        kernels.push_back(make_compute_spec(COMPUTE_KERNEL_G2, num_output_tiles_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Per-core runtime args (reader + writer)
    ////////////////////////////////////////////////////////////////////////////
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"Mt", Mt},
              {"Kt", Kt},
              {"Nt", Nt},
              {"MtKt", MtKt},
              {"KtNt", KtNt},
              {"batch", B},
              {"bcast_B", uint32_t(bcast_batch)},
              {"output_tile_start_id", num_tiles_written},
              {"num_output_tiles", num_output_tiles_per_core},
              {"MtNt", MtNt}}});
        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", num_output_tiles_per_core}, {"start_id", num_tiles_written}}});
        num_tiles_written += num_output_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    // Reader/writer span all_cores; compute is split per work-split group, so each group gets its
    // own WorkUnitSpec (reader + writer + per-group compute). A KernelSpec may appear in multiple
    // WorkUnitSpecs; its effective node set is the union.
    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "matmul_multi_core_g1",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1});
    if (group_2_present) {
        work_units.push_back(WorkUnitSpec{
            .name = "matmul_multi_core_g2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec + run args
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec{
        .name = "matmul_multi_core",
        .kernels = kernels,
        .dataflow_buffers = {in0_dfb_spec, in1_dfb_spec, out_dfb_spec},
        .tensor_parameters = tensor_params,
        .work_units = work_units,
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {IN0_TENSOR, TensorArgument{a}}, {IN1_TENSOR, TensorArgument{b}}, {OUT_TENSOR, TensorArgument{output}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
