// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_single_core_program_factory.hpp"

#include <cmath>
#include <filesystem>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the untilize_with_unpadding device/ sibling .cpp files
// (the _UWU_SC suffix) to avoid unity-build collisions. These are op-private Metal 2.0 copies of the
// shared kernels; the legacy kernels are still consumed positionally by the un-migrated variants.
constexpr const char* UWU_SC_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
    "reader_unary_interleaved_start_id_m2.cpp";
constexpr const char* UWU_SC_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
    "writer_unary_unpad_dims_split_rows_m2.cpp";
constexpr const char* UWU_SC_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreCoord core_0 = corerange_to_cores(core).at(0);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    log_debug(tt::LogOp, "untilize_with_unpadding_single_core");
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "output_cb_data_format: {}", output_cb_data_format);

    int32_t num_tiles = a.physical_volume() / TILE_HW;

    auto input_w = input_shape.rank() >= 4 ? input_shape[-4] : 1;
    auto input_z = input_shape.rank() >= 3 ? input_shape[-3] : 1;
    auto input_y = input_shape.rank() >= 2 ? input_shape[-2] : 1;
    auto input_x = input_shape[-1];

    auto output_w = output_shape.rank() >= 4 ? output_shape[-4] : 1;
    auto output_z = output_shape.rank() >= 3 ? output_shape[-3] : 1;
    auto output_y = output_shape.rank() >= 2 ? output_shape[-2] : 1;
    auto output_x = output_shape[-1];

    uint32_t padded_stick_size = input_x * output.element_size();  // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_x * output.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = input_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (output.element_size() * TILE_HEIGHT * 2 + output.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
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
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * output.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - (num_blocks_w_output * block_row_size);

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (input_y - output_y) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (input_z - output_z) * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (input_w - output_w) * input_z * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_y - (output_y / TILE_HEIGHT * TILE_HEIGHT);

    bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 or
                         input_cb_data_format == tt::DataFormat::UInt32 or
                         input_cb_data_format == tt::DataFormat::Int32;

    const m2::DFBSpecName input_dfb{"src0"};
    const m2::DFBSpecName output_dfb{"out0"};

    m2::ProgramSpec spec;
    spec.name = "untilize_with_unpadding_single_core";

    // Input CB (formerly c_0) and Output CB (formerly c_16): both L1-allocated, one block of tiles each.
    uint32_t num_input_tiles = num_tiles_per_block;
    uint32_t num_output_tiles = num_tiles_per_block;
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = output_dfb,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_cb_data_format,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Tilized reader on NCRISC (RISCV_1 / NOC1): reads interleaved input tiles into src0.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{UWU_SC_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(input_dfb, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    // Untilized writer on BRISC (default Gen1 config): consumes out0, writes unpadded sticks to output.
    // The (formerly positional) FLOAT32_DTYPE flag becomes a named compile-time arg; the rest become named
    // runtime args. (Legacy CTA `unpadded_stick_size` was unreferenced in this writer body and is dropped.)
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{UWU_SC_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(output_dfb, "cb_id_out0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args = {{"float32_dtype", static_cast<uint32_t>(float32_dtype)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "padded_W_diff_blocks",
                  "num_unpadded_Z",
                  "padded_Z_diff_blocks",
                  "num_unpadded_Y",
                  "padded_Y_diff_blocks",
                  "num_leftover_Y",
                  "num_unpadded_X",
                  "padded_X_size",
                  "num_blocks_w_input",
                  "num_blocks_w_output",
                  "num_blocks_w_diff",
                  "block_row_size",
                  "block_row_leftover_size"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    // Compute kernel: consumes src0, produces out0. DST_ACCUM_MODE mirrors the legacy define for 32-bit
    // formats; UnpackToDestFp32 on the input DFB when fp32_dest_acc_en.
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    m2::ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw.unpack_to_dest_mode.emplace(input_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
    }
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{UWU_SC_COMPUTE_KERNEL_PATH},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = {m2::ConsumerOf(input_dfb, "src_cb_id"), m2::ProducerOf(output_dfb, "out_cb_id")},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(num_tiles / num_tiles_per_block)},
             {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = std::move(compute_hw),
    };

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    // Local DFBs (src0, out0) require their producer AND consumer kernels co-located in the same WorkUnitSpec:
    // reader produces src0 (consumed by compute), compute produces out0 (consumed by writer). All three run
    // on the single core.
    spec.work_units = {m2::WorkUnitSpec{
        .name = "untilize_with_unpadding_single_core",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = core}};

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Reader: read all tiles starting at page 0. Writer takes the full unpad descriptor. The compute kernel
    // binds no tensor and takes only compile-time args, so it gets no KernelRunArgs entry.
    reader_args.runtime_arg_values.push_back(
        {core_0, {{"num_tiles", static_cast<uint32_t>(num_tiles)}, {"start_page_id", uint32_t{0}}}});
    writer_args.runtime_arg_values.push_back(
        {core_0,
         {{"num_unpadded_W", static_cast<uint32_t>(output_w)},
          {"padded_W_diff_blocks", padded_W_diff_blocks},
          {"num_unpadded_Z", static_cast<uint32_t>(output_z)},
          {"padded_Z_diff_blocks", padded_Z_diff_blocks},
          {"num_unpadded_Y", static_cast<uint32_t>(output_y)},
          {"padded_Y_diff_blocks", padded_Y_diff_blocks},
          {"num_leftover_Y", num_leftover_Y},
          {"num_unpadded_X", static_cast<uint32_t>(output_x)},
          {"padded_X_size", padded_stick_size},
          {"num_blocks_w_input", num_blocks_w_input},
          {"num_blocks_w_output", num_blocks_w_output},
          {"num_blocks_w_diff", num_blocks_w_diff},
          {"block_row_size", block_row_size},
          {"block_row_leftover_size", block_row_leftover_size}}});
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
