// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_program_factory.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::prim {

static MatmulMultiCoreReuseProgramFactory::cached_program_t create_program(
    tt_metal::IDevice* device,
    tt::DataFormat in0_cb_data_format,
    tt::DataFormat in1_cb_data_format,
    tt::DataFormat out_cb_data_format,
    MathFidelity math_fidelity,
    uint32_t num_cores_x,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* out_buffer) {
    tt_metal::Program program{};

    uint32_t in0_single_tile_size = tt::tile_size(in0_cb_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_cb_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * out_single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (K / in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B                        // batch
    };

    uint32_t num_blocks_read = 0;
    uint32_t num_blocks_y = M / per_core_M;
    uint32_t num_blocks_x = N / per_core_N;

    CoreRangeSet all_cores(tt::tt_metal::num_cores_to_corerangeset(
        num_blocks_x * num_blocks_y, device->compute_with_storage_grid_size(), true));

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_cb_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_cb_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, out_cb_data_format}, {interm0_cb_index, out_cb_data_format}};
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, out_single_tile_size)
            .set_page_size(interm0_cb_index, out_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);

    // Create reader and writer kernels per core
    auto mm_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            // Write runtime args to device
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)in0_buffer->address(),          // in0_tensor_addr
                (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                (std::uint32_t)1,                              // in0_tensor_stride_w
                (std::uint32_t)K,                              // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,                    // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)in1_buffer->address(),      // in1_tensor_addr
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                (std::uint32_t)1,                          // in1_tensor_stride_w
                (std::uint32_t)N,                          // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * N,            // in1_tensor_next_block_stride

                (std::uint32_t)per_core_N,                // in1_block_w
                (std::uint32_t)in0_block_w,               // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)K / in0_block_w,  // num_blocks

                (std::uint32_t)M * K,       // MtKt
                (std::uint32_t)K * N,       // KtNt
                (std::uint32_t)B,           // batch
                (std::uint32_t)bcast_batch  // bcast_B
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t)out_buffer->address(),  // out_tensor_addr
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * N),  // out_tensor_start_tile_id
                (std::uint32_t)1,                     // out_tensor_stride_w
                (std::uint32_t)N,                     // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,        // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * N,    // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

                (std::uint32_t)M * N,  // MtNt
                (std::uint32_t)B       // batch
            };

            tt_metal::SetRuntimeArgs(program, mm_reader_kernel_id, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_args);

            num_blocks_read++;
        }
    }

    return {std::move(program), {mm_reader_kernel_id, unary_writer_kernel_id, num_cores_x, num_blocks_y, num_blocks_x}};
}

void MatmulMultiCoreReuseProgramFactory::override_runtime_arguments(
    MatmulMultiCoreReuseProgramFactory::cached_program_t& cached_program,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    constexpr uint32_t per_core_M = 16;
    constexpr uint32_t per_core_N = 16;

    const auto& input_tensors = tensor_args.input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& ashape = input_tensors.at(0).padded_shape();
    const auto& bshape = input_tensors.at(1).padded_shape();
    uint32_t M = ashape[-2] / TILE_HEIGHT;
    uint32_t N = bshape[-1] / TILE_WIDTH;

    uint32_t num_cores_x = cached_program.shared_variables.num_cores_x;

    uint32_t num_blocks_y = M / per_core_M;
    uint32_t num_blocks_x = N / per_core_N;

    auto* src_dram_buffer_a = input_tensors.at(0).buffer();
    auto* src_dram_buffer_b = input_tensors.at(1).buffer();

    auto* dst_dram_buffer = output_tensors.at(0).buffer();

    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    uint32_t num_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer_a->address();
                runtime_args[8] = src_dram_buffer_b->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
            }
            num_blocks_read++;
        }
    }
}

MatmulMultiCoreReuseProgramFactory::cached_program_t MatmulMultiCoreReuseProgramFactory::create(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& output = tensor_return_value.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat out_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t in0_block_w = 2;
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;
    uint32_t per_core_M = 16;
    uint32_t per_core_N = 16;

    TT_FATAL(Mt % per_core_M == 0, "Mt ({}) must be divisible by per_core_M ({})", Mt, per_core_M);
    TT_FATAL(Nt % per_core_N == 0, "Nt ({}) must be divisible by per_core_N ({})", Nt, per_core_N);
    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    // This should allocate a DRAM buffer on the device
    tt_metal::IDevice* device = a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_blocks_total = (Mt / per_core_M) * (Nt / per_core_N);
    TT_FATAL(
        num_blocks_total <= num_cores_x * num_cores_y,
        "Total number of blocks ({}) must not exceed available cores ({})",
        num_blocks_total,
        num_cores_x * num_cores_y);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    return create_program(
        device,
        in0_cb_data_format,
        in1_cb_data_format,
        out_cb_data_format,
        math_fidelity,
        num_cores_x,
        B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        in0_buffer,
        in1_buffer,
        out_buffer);
}

////////////////////////////////////////////////////////////////////////////
//                      Mesh Workload Setup
////////////////////////////////////////////////////////////////////////////

MatmulMeshWorkloadMultiCoreReuseFactory::cached_mesh_workload_t
MatmulMeshWorkloadMultiCoreReuseFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto single_device_program = MatmulMultiCoreReuseProgramFactory::create(attributes, tensor_args, output);
            shared_variables[mesh_coord_range] = single_device_program.shared_variables;
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreReuseFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = MatmulMultiCoreReuseProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        MatmulMultiCoreReuseProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::prim
