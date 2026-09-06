// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

using namespace tt;
using namespace tt::constants;
using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
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
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();

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

    // Spec-scope resource names. Declared function-local rather than at file scope: the six matmul
    // factory .cpp files share one unity-build target, so file-scope constants with these names
    // would collide as sibling factories are ported.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    // Dataflow buffers
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN0_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in0_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in1_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format,
        },
    };

    // Reader kernel
    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    uint32_t last_ktile_h = 0;

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_8bank_output_tiles_partitioned_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN0,
                    .accessor_name = "in0",
                },
                TensorBinding{
                    .tensor_parameter_name = IN1,
                    .accessor_name = "in1",
                },
            },
        .compile_time_args =
            {
                {"in0_last_ktile_w", last_ktile_w},
                {"in0_last_ktile_h", last_ktile_h},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"Mt",
                     "Kt",
                     "Nt",
                     "MtKt",
                     "KtNt",
                     "batch",
                     "bcast_B",
                     "output_tile_start_id",
                     "num_output_tiles",
                     "MtNt"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer kernel
    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Per-node runtime args for reader and writer
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
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
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"Mt", Mt},
             {"Kt", Kt},
             {"Nt", Nt},
             {"MtKt", MtKt},
             {"KtNt", KtNt},
             {"batch", B},
             {"bcast_B", uint32_t(bcast_batch)},
             {"output_tile_start_id", num_tiles_written},
             {"num_output_tiles", num_output_tiles_per_core},
             {"MtNt", MtNt}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_output_tiles_per_core}, {"start_id", num_tiles_written}});
        num_tiles_written += num_output_tiles_per_core;
    }

    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // Compute kernel(s) — one per core group with different tile counts
    // bmm compute kernel: B, Mt, Nt are just 3 for loops that act as 1 large loop,
    // so only set Nt for simplicity
    auto make_compute = [&](KernelSpecName unique_id, uint32_t num_output_tiles_per_core_group) {
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        // The legacy ComputeConfigDescriptor set no unpack_to_dest_mode vector, so every buffer took
        // UnpackToDestMode::Default -- the SrcA/B path, which is UnpackMode::UnpackToSrc. Stated
        // explicitly rather than left implicit because a compute kernel that consumes an FP32
        // dataflow buffer with a 32-bit Dest register must make the choice explicit, and that is
        // exactly this kernel whenever the inputs are FP32 and fp32_dest_acc_en is set. The value is
        // the legacy one, so the lowered per-buffer vector is unchanged.
        unpack_modes(compute_hw) = {
            {IN0_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
            {IN1_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
        };
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_metal2.cpp",
            // Legacy ComputeConfigDescriptor defaults opt_level to O3; Metal 2.0's
            // type-agnostic CompilerOptions defaults to O2, so a compute kernel must say O3
            // explicitly or it silently drops a level.
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"batch", 1u},
                    {"Mt", 1u},
                    {"Kt", Kt},
                    {"Nt", num_output_tiles_per_core_group},
                },
            .hw_config = std::move(compute_hw),
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels;
    kernels.reserve(has_core_group_2 ? 4 : 3);
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(COMPUTE_G1, num_output_tiles_per_core_group_1));

    Group<WorkUnitSpec> work_units;
    work_units.reserve(has_core_group_2 ? 2 : 1);
    work_units.push_back(WorkUnitSpec{
        .name = "core_group_1",
        .kernels = {READER, WRITER, COMPUTE_G1},
        .target_nodes = core_group_1,
    });

    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_output_tiles_per_core_group_2));
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = {READER, WRITER, COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "matmul_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = IN0, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = IN1, .spec = b.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {IN0, a},
        {IN1, b},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
