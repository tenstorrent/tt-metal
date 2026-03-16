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

constexpr uint32_t STICK_SIZE = 1024;
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_OUT = 2;
}  // namespace

DitMinimalRmBinaryProgramFactory::cached_program_t DitMinimalRmBinaryProgramFactory::create(
    const DitMinimalRmBinaryParams& args, const DitMinimalRmBinaryInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;
    auto& output = tensor_return_value;

    auto* device = input_a.device();
    auto* src_a_buffer = input_a.buffer();
    auto* src_b_buffer = input_b.buffer();
    auto* dst_buffer = output.buffer();

    const bool is_fp32 = (input_a.dtype() == DataType::FLOAT32);
    const bool is_mul = (args.op_type == BinaryOpType::MUL);
    const bool is_wormhole_b0 = (device->arch() == tt::ARCH::WORMHOLE_B0);

    const uint32_t dtype_bytes = input_a.element_size();
    const uint32_t stick_size_bytes = STICK_SIZE * dtype_bytes;

    const auto padded_shape = input_a.padded_shape();
    const uint32_t last_dim = padded_shape[-1];
    const uint32_t row_size_bytes = last_dim * dtype_bytes;

    const uint64_t num_elements_total = input_a.physical_volume();
    const uint32_t num_full_sticks = static_cast<uint32_t>(num_elements_total / STICK_SIZE);
    const uint32_t last_chunk_bytes = static_cast<uint32_t>(num_elements_total % STICK_SIZE) * dtype_bytes;
    const uint32_t total_tiles = num_full_sticks + (last_chunk_bytes > 0 ? 1 : 0);

    auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_cg1, tiles_per_cg2] =
        split_work_to_cores(grid, total_tiles);

    // --- Circular buffers (double-buffered: 2 pages each) ---
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_a.dtype());
    const uint32_t cb_total_size = 2 * stick_size_bytes;

    Program program{};

    auto make_cb = [&](uint32_t cb_index) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_total_size, {{cb_index, cb_data_format}}).set_page_size(cb_index, stick_size_bytes);
        CreateCircularBuffer(program, all_cores, cb_config);
    };
    make_cb(CB_A);
    make_cb(CB_B);
    make_cb(CB_OUT);

    // --- Compile-time args ---
    std::vector<uint32_t> reader_ct_args = {stick_size_bytes};
    tt::tt_metal::TensorAccessorArgs(src_a_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(src_b_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {stick_size_bytes};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);

    // --- Compute defines ---
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
        unpack_to_dest_mode[CB_A] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[CB_B] = UnpackToDestMode::UnpackToDestFp32;
    } else {
        math_fidelity = (is_mul && is_wormhole_b0) ? MathFidelity::HiFi3
                        : (is_mul)                 ? MathFidelity::HiFi4
                                                   : MathFidelity::LoFi;
    }

    // --- Create kernels ---
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

    // --- Per-core runtime args ---
    uint32_t start_tile = 0;

    std::cout << "output page size = " << output.buffer()->page_size() << std::endl;

    auto assign_rt_args = [&](const CoreRangeSet& group, uint32_t tile_count) {
        if (tile_count == 0) {
            return;
        }
        for (const auto& core_range : group.ranges()) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    CoreCoord core{x, y};

                    const uint32_t start_element = start_tile * STICK_SIZE;
                    const uint32_t core_start_row = start_element / last_dim;
                    const uint32_t core_start_off = (start_element % last_dim) * dtype_bytes;
                    const uint32_t end_tile = start_tile + tile_count;
                    uint32_t core_num_full, core_last_chunk;
                    if (last_chunk_bytes > 0 && end_tile == total_tiles) {
                        core_num_full = tile_count - 1;
                        core_last_chunk = last_chunk_bytes;
                    } else {
                        core_num_full = tile_count;
                        core_last_chunk = 0;
                    }

                    SetRuntimeArgs(
                        program,
                        reader_kernel_id,
                        core,
                        {src_a_buffer->address(),
                         src_b_buffer->address(),
                         core_num_full,
                         core_start_row,
                         core_start_off,
                         core_last_chunk,
                         row_size_bytes});

                    SetRuntimeArgs(
                        program,
                        writer_kernel_id,
                        core,
                        {dst_buffer->address(),
                         core_num_full,
                         core_start_row,
                         core_start_off,
                         core_last_chunk,
                         row_size_bytes});

                    SetRuntimeArgs(program, compute_kernel_id, core, {tile_count});

                    start_tile += tile_count;
                }
            }
        }
    };

    assign_rt_args(core_group_1, tiles_per_cg1);
    assign_rt_args(core_group_2, tiles_per_cg2);

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
        auto& reader_rt_args = GetRuntimeArgs(program, shared.reader_kernel_id)[core.x][core.y];
        reader_rt_args[0] = src_a_buffer->address();
        reader_rt_args[1] = src_b_buffer->address();

        auto& writer_rt_args = GetRuntimeArgs(program, shared.writer_kernel_id)[core.x][core.y];
        writer_rt_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::experimental::prim
