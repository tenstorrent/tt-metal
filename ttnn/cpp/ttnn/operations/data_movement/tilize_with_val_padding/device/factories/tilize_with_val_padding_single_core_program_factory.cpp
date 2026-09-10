// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_single_core_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingSingleCoreFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    tt::DataFormat input_dfb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);

    tt::DataFormat output_dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    int32_t num_tiles = output.physical_volume() / TILE_HW;

    auto true_input_shape = a.padded_shape();
    auto true_output_shape = output.padded_shape();

    auto input_w = true_input_shape.rank() >= 4 ? true_input_shape[-4] : 1;
    auto input_z = true_input_shape.rank() >= 3 ? true_input_shape[-3] : 1;
    auto input_y = true_input_shape.rank() >= 2 ? true_input_shape[-2] : 1;
    auto input_x = true_input_shape[-1];

    auto output_w = true_output_shape.rank() >= 4 ? true_output_shape[-4] : 1;
    auto output_z = true_output_shape.rank() >= 3 ? true_output_shape[-3] : 1;
    auto output_y = true_output_shape.rank() >= 2 ? true_output_shape[-2] : 1;
    auto output_x = true_output_shape[-1];

    uint32_t unpadded_row_size_bytes = input_x * a.element_size();  // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output_x * a.element_size();   // Assuming bfloat16 dataformat

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = output_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    // Memory usage is 2 DFBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
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
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - (num_blocks_w_input * block_row_size);

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (output_y - input_y) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (output_z - input_z) * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks =
        (output_w - input_w) * output_z * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = input_y - (input_y / TILE_HEIGHT * TILE_HEIGHT);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
    const uint32_t num_output_tiles = num_tiles_per_block;

    // ---------------------------------------------------------------------
    // Program-scope resource names (typed handles → generated dfb:: / tensor:: tokens)
    // ---------------------------------------------------------------------
    const DFBSpecName IN{"in"};    // legacy src0 buffer c_0: row-major staging for tilize
    const DFBSpecName OUT{"out"};  // legacy output buffer c_16: tilized output
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = "tilize_with_val_padding_single_core";

    // ---------------------------------------------------------------------
    // DataflowBufferSpecs (replaces the legacy c_0 / c_16 buffer descriptors)
    // ---------------------------------------------------------------------
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_dfb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_dfb_data_format,
        },
    };

    // ---------------------------------------------------------------------
    // Tensor parameters (typed bindings replace the Buffer* RTA slots and the
    // TensorAccessorArgs(...).append_to(...) CTA plumbing)
    // ---------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    uint32_t tile_row_size_bytes = a.element_size() * TILE_HEIGHT;

    // ---------------------------------------------------------------------
    // Tilized reader
    // ---------------------------------------------------------------------
    // NOTE: `unpadded_row_size_bytes` is carried over from the legacy CTA list even though the reader
    // kernel does not read it (it held positional slot 1 there, ahead of the accessor args). Dropping
    // it is a cleanup for the op owner, not port work.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                  "reader_unary_pad_dims_split_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"bytes_per_tile_row", tile_row_size_bytes},
             {"unpadded_row_size_bytes", unpadded_row_size_bytes},
             {"elem_size", static_cast<uint32_t>(a.element_size())}},
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
                  "pad_value",
                  "num_blocks_w_input",
                  "num_blocks_w_output",
                  "num_blocks_w_diff",
                  "block_row_size",
                  "block_row_leftover_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    });

    // ---------------------------------------------------------------------
    // Tilized writer
    // ---------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                  "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    });

    // ---------------------------------------------------------------------
    // Compute
    // ---------------------------------------------------------------------
    // Legacy ComputeConfigDescriptor set only fp32_dest_acc_en and unpack_to_dest_mode; every other
    // field stayed at its default, which ComputeGen1Config reproduces exactly. The legacy
    // unpack_to_dest_mode vector was Default everywhere except v[c_0] = UnpackToDestFp32 when
    // fp32_llk_acc, i.e. UnpackToDest on the tilize input DFB (Default == UnpackToSrc is expressed by
    // omitting the entry).
    ComputeGen1Config compute_gen1{.enable_32_bit_dest = fp32_llk_acc};
    if (fp32_llk_acc) {
        compute_gen1.unpack_modes = ComputeUnpackModes{{IN, UnpackMode::UnpackToDest}};
    }
    ComputeHardwareConfig compute_hw{std::move(compute_gen1)};

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp",
        // O3 explicitly: the legacy ComputeConfigDescriptor set no opt_level and so resolved to O3,
        // but Metal 2.0's type-agnostic CompilerOptions defaults to O2. Leaving it unset would drop
        // a level on this kernel's compile and link.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = IN,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUT,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(num_tiles / num_tiles_per_block)},
             {"per_core_block_tile_cnt", static_cast<uint32_t>(num_tiles_per_block)}},
        .hw_config = compute_hw,
    });

    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_ranges});

    // ---------------------------------------------------------------------
    // Runtime args. The compute kernel carries only CTAs, so it needs no entry.
    // ---------------------------------------------------------------------
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(KernelRunArgs{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core.start_coord,
            {{"num_unpadded_W", input_w},
             {"padded_W_diff_blocks", padded_W_diff_blocks},
             {"num_unpadded_Z", input_z},
             {"padded_Z_diff_blocks", padded_Z_diff_blocks},
             {"num_unpadded_Y", input_y},
             {"padded_Y_diff_blocks", padded_Y_diff_blocks},
             {"num_leftover_Y", num_leftover_Y},
             {"num_unpadded_X", input_x},
             {"padded_X_size", padded_row_size_bytes},
             {"pad_value", packed_pad_value},
             {"num_blocks_w_input", num_blocks_w_input},
             {"num_blocks_w_output", num_blocks_w_output},
             {"num_blocks_w_diff", num_blocks_w_diff},
             {"block_row_size", block_row_size},
             {"block_row_leftover_size", block_row_leftover_size}}),
    });
    run_args.kernel_run_args.push_back(KernelRunArgs{
        .kernel = WRITER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core.start_coord, {{"num_pages", static_cast<uint32_t>(num_tiles)}, {"start_id", 0u}}),
    });

    run_args.tensor_args.emplace(INPUT, TensorArgument{a.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
