// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the tilize device/ sibling .cpp files to avoid
// unity-build collisions.
constexpr const char* SC_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
    "reader_unary_stick_layout_split_rows_singlecore_m2.cpp";
constexpr const char* SC_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_m2.cpp";
constexpr const char* SC_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts TilizeSingleCoreProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t num_tiles_per_block = 1;

    if (!operation_attributes.use_low_perf) {
        // Ensure we don't intrude into storage space
        uint32_t max_l1_size =
            (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
        // Currently need the number of tiles in a row to be divisible by tiles in a block
        if (num_tiles_in_row <= max_tiles) {
            num_tiles_per_block = num_tiles_in_row;
        } else {
            for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
                if (num_tiles_in_row % n_t == 0) {
                    num_tiles_per_block = n_t;
                    break;
                }
            }
        }
    }

    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;

    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    const m2::DFBSpecName src0_dfb{"cb_id_in0"};
    const m2::DFBSpecName out0_dfb{"cb_id_out0"};

    m2::ProgramSpec spec;
    spec.name = "tilize_single_core";

    // DFBs: src0 (formerly CB c_0) is L1-allocated, produced by the reader and consumed by the compute
    // kernel; out0 (formerly CB c_16) is L1-allocated, produced by the compute kernel and consumed by the
    // writer.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = src0_dfb,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = out0_dfb,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_cb_data_format,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two DM kernels don't collide on the same
    // DM processor. The reader produces src0 from the (interleaved) input via its TensorAccessor.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{SC_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(src0_dfb, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_stick_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{SC_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(out0_dfb, "output_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    m2::ComputeHardwareConfig compute_hw{
        .fp32_dest_acc_en = fp32_llk_acc,
    };
    // Legacy set UnpackToDestFp32 on c_0 whenever fp32_llk_acc was true. Only meaningful when the input
    // DFB carries Float32; ValidateProgramSpec rejects it otherwise.
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        compute_hw.unpack_to_dest_mode.emplace(src0_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{SC_COMPUTE_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(src0_dfb, "cb_id_in0"), m2::ProducerOf(out0_dfb, "cb_id_out0")},
        // per_core_block_cnt = num_tiles / num_tiles_per_block; per_core_block_tile_cnt = num_tiles_per_block.
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles / num_tiles_per_block}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = compute_hw,
    };

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    // Local DFBs require their producer AND consumer to share a WorkUnitSpec on every node where they live.
    // reader→src0→compute→out0→writer form a single chain, so all three kernels co-locate in one work unit.
    spec.work_units = {m2::WorkUnitSpec{
        .name = "tilize_single_core",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = core_ranges}};

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    reader_args.runtime_arg_values.push_back(
        {core.start_coord,
         {{"num_sticks", num_sticks},
          {"num_tiles_per_block", num_tiles_per_block},
          {"block_width_size", block_width_size},
          {"num_full_blocks_in_row", num_full_blocks_in_row},
          {"start_stick_id", uint32_t{0}}}});
    writer_args.runtime_arg_values.push_back({core.start_coord, {{"num_pages", num_tiles}, {"start_id", uint32_t{0}}}});

    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
