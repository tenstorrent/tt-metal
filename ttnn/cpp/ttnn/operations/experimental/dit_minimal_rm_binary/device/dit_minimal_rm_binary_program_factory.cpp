// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_rm_binary_program_factory.hpp"

#include <map>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dit_minimal_rm_binary/device/kernels/dataflow/reader.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dit_minimal_rm_binary/device/kernels/dataflow/writer.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dit_minimal_rm_binary/device/kernels/compute/compute.cpp";

// CB indices — must match the kernel sources.
constexpr uint32_t CB_A_RM = 0;       // c_0  row-major input A (pre-tilize)
constexpr uint32_t CB_B_RM = 1;       // c_1  row-major input B
constexpr uint32_t CB_A_TILED = 2;    // c_2  properly-tiled A
constexpr uint32_t CB_B_TILED = 3;    // c_3  properly-tiled B
constexpr uint32_t CB_OUT_TILED = 4;  // c_4  tiled binary-op output
constexpr uint32_t CB_OUT_RM = 16;    // c_16 row-major output (post-untilize)

}  // namespace

DitMinimalRmBinaryProgramFactory::cached_program_t DitMinimalRmBinaryProgramFactory::create(
    const DitMinimalRmBinaryParams& operation_attributes,
    const DitMinimalRmBinaryInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;
    auto& output = tensor_return_value;

    auto* device = input_a.device();
    auto* src_a_buffer = input_a.buffer();
    auto* src_b_buffer = input_b.buffer();
    auto* dst_buffer = output.buffer();

    const bool is_fp32 = (input_a.dtype() == DataType::FLOAT32);
    const bool is_mul = (operation_attributes.op_type == BinaryOpType::MUL);
    const bool is_wormhole_b0 = (device->arch() == tt::ARCH::WORMHOLE_B0);

    const uint32_t dtype_bytes = input_a.element_size();
    const tt::DataFormat cb_fmt = datatype_to_dataformat_converter(input_a.dtype());
    const uint32_t tile_size = tt::tile_size(cb_fmt);  // bytes per hardware tile

    // Row / stick geometry -------------------------------------------------
    const auto padded_shape = input_a.padded_shape();
    const uint32_t last_dim = padded_shape[-1];
    const uint32_t row_size_bytes = last_dim * dtype_bytes;      // stick size
    const uint32_t ntiles_per_row = last_dim / TILE_WIDTH;       // tile-columns per row
    const uint32_t tile_width_bytes = TILE_WIDTH * dtype_bytes;  // bytes per tile-column per row

    const uint64_t num_elements_total = input_a.physical_volume();
    const uint32_t total_rows = static_cast<uint32_t>(num_elements_total / last_dim);
    // Ceiling division: a partial last block of < TILE_HEIGHT rows is still one block.
    // const uint32_t total_nblocks = (total_rows + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // Work split on row-blocks ---------------------------------------------
    auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_cg1, rows_per_cg2] =
        split_work_to_cores(grid, total_rows);

    Program program{};

    // ----------------------------------------------------------------------
    // Circular buffers
    //
    // CB_A_RM / CB_B_RM: row-major inputs for the tilize reader.
    //   page = tile_size; double-buffered (2 × ntiles_per_row pages) so the
    //   reader can fill the next row-block while compute processes the current.
    //
    // CB_A_TILED / CB_B_TILED / CB_OUT_TILED / CB_OUT_RM: internal pipeline
    //   buffers; one row-block at a time.  page = tile_size; ntiles_per_row pages
    //   per block.
    //
    // CB_OUT_RM layout (produced by untilize_block):
    //   ntiles_per_row tile-sized pages per block, row-major within the block.
    //   Row k of the block starts at byte offset k * row_size_bytes from the
    //   block base.  The writer pops the whole ntiles_per_row-page block at once
    //   and reads individual rows via base + k * row_size_bytes.
    // ----------------------------------------------------------------------
    auto make_cb = [&](uint32_t cb_index, uint32_t num_pages) {
        const uint32_t total_bytes = num_pages * tile_size;
        CircularBufferConfig cfg =
            CircularBufferConfig(total_bytes, {{cb_index, cb_fmt}}).set_page_size(cb_index, tile_size);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    make_cb(CB_A_RM, 4 * ntiles_per_row);  // double-buffered
    make_cb(CB_B_RM, 4 * ntiles_per_row);  // double-buffered
    make_cb(CB_A_TILED, ntiles_per_row);
    make_cb(CB_B_TILED, ntiles_per_row);
    make_cb(CB_OUT_TILED, ntiles_per_row);
    make_cb(CB_OUT_RM, 4 * ntiles_per_row);

    // ----------------------------------------------------------------------
    // Reader compile-time args
    //   [stick_size, TensorAccessorArgs_A..., TensorAccessorArgs_B...,
    //    ntiles_per_row, tile_width_bytes]
    // ----------------------------------------------------------------------
    std::vector<uint32_t> reader_ct_args = {row_size_bytes};
    TensorAccessorArgs(*src_a_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*src_b_buffer).append_to(reader_ct_args);
    reader_ct_args.push_back(ntiles_per_row);
    reader_ct_args.push_back(tile_width_bytes);

    // ----------------------------------------------------------------------
    // Writer compile-time args
    //   [cb_out, stick_size, TensorAccessorArgs_out...,
    //    ntiles_per_row, tile_width_bytes]
    // ----------------------------------------------------------------------
    std::vector<uint32_t> writer_ct_args = {CB_OUT_RM, row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(ntiles_per_row);
    writer_ct_args.push_back(tile_width_bytes);

    // ----------------------------------------------------------------------
    // Compute defines (unchanged from original)
    // ----------------------------------------------------------------------
    std::map<std::string, std::string> defines;
    std::string binary_op_init;
    std::string binary_op;

    if (is_mul) {
        defines["RM_BINARY_OP_MUL"] = "1";
        binary_op_init = is_fp32 ? "mul_binary_tile_init" : "mul_tiles_init";
        binary_op = is_fp32 ? "mul_binary_tile" : "mul_tiles";
    } else {
        defines["RM_BINARY_OP_ADD"] = "1";
        binary_op_init = is_fp32 ? "add_binary_tile_init" : "add_tiles_init";
        binary_op = is_fp32 ? "add_binary_tile" : "add_tiles";
    }
    if (is_fp32) {
        defines["IS_FP32"] = "1";
    }
    defines["BINARY_OP_INIT"] = binary_op_init;
    defines["BINARY_OP"] = binary_op;

    MathFidelity math_fidelity;
    bool fp32_dest_acc_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;

    if (is_fp32) {
        math_fidelity = MathFidelity::HiFi4;
        fp32_dest_acc_en = true;
        unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        unpack_to_dest_mode[CB_A_RM] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[CB_B_RM] = UnpackToDestMode::UnpackToDestFp32;
    } else {
        math_fidelity = (is_mul && is_wormhole_b0) ? MathFidelity::HiFi3
                        : (is_mul)                 ? MathFidelity::HiFi4
                                                   : MathFidelity::LoFi;
    }

    // ----------------------------------------------------------------------
    // Create kernels
    // ----------------------------------------------------------------------
    KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_ct_args));

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = {},
            .defines = defines,
        });

    // ----------------------------------------------------------------------
    // Per-core runtime args
    //
    // Reader  RT: [src_a_addr, src_b_addr, num_sticks, start_stick_id]
    // Writer  RT: [dst_addr,   num_sticks, start_stick_id]
    // Compute RT: [num_blocks, ntiles_per_row]
    // ----------------------------------------------------------------------
    uint32_t start_block = 0;

    auto assign_rt_args = [&](const CoreRangeSet& group, uint32_t rows_for_core) {
        if (rows_for_core == 0) {
            return;
        }
        for (const auto& range : group.ranges()) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    CoreCoord core{x, y};

                    const uint32_t start_row = start_block;
                    // Clamp to the actual number of rows for the last (possibly partial) block.
                    const uint32_t num_sticks = rows_for_core;
                    // std::cout << "{" << x << ", " << y << "} start row = " << start_block << ", num sticks = " <<
                    // rows_for_core << std::endl;

                    SetRuntimeArgs(
                        program,
                        reader_kernel_id,
                        core,
                        {src_a_buffer->address(), src_b_buffer->address(), num_sticks, start_row});

                    SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_sticks, start_row});

                    SetRuntimeArgs(program, compute_kernel_id, core, {num_sticks, ntiles_per_row});

                    start_block += rows_for_core;
                }
            }
        }
    };

    std::cout << "Assigning to core group 1" << std::endl;
    assign_rt_args(core_group_1, rows_per_cg1);

    std::cout << "Assigning to core group 2" << std::endl;
    assign_rt_args(core_group_2, rows_per_cg2);

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, all_cores}};
}

void DitMinimalRmBinaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DitMinimalRmBinaryParams& /*operation_attributes*/,
    const DitMinimalRmBinaryInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    auto* src_a_buffer = tensor_args.input_a.buffer();
    auto* src_b_buffer = tensor_args.input_b.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    for (const auto& core : corerange_to_cores(shared.all_cores, std::nullopt, true)) {
        auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id)[core.x][core.y];
        reader_rt[0] = src_a_buffer->address();
        reader_rt[1] = src_b_buffer->address();

        auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id)[core.x][core.y];
        writer_rt[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::experimental::prim
