// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/add/device/program_factory/elt_nd_sharded_add_program.hpp"
#include "ttnn/operations/experimental/add/device/program_factory/elemwise_factory_common.hpp"

#include "ttnn/operations/experimental/add/device/kernels/dataflow/elt_nd_sharded_add_reader_args.hpp"
#include "ttnn/operations/experimental/add/device/kernels/dataflow/elemwise_writer_kernel_args.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

// # Wormhole: 8192 bytes, Blackhole: 16384 bytes (vs 2048 byte tile)
// page_size = (noc_max_page_size // tile_size) * tile_size

EltNDShardedAddProgram::cached_program_t EltNDShardedAddProgram::create(
    const AddParams& operation_attributes, const AddInputs& args, Tensor& output) {
    using namespace ttnn::kernel::eltwise;
    // using namespace ttnn::kernel::eltwise::add_args;
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};
    // Use worker cores (optimal for DRAM); runtime args use core index = shard index.
    auto full_dram_cores_vec =
        args.a_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    auto distribution_spec = args.a_tensor.tensor_spec().compute_buffer_sharding_args().buffer_distribution_spec();
    TT_FATAL(distribution_spec.has_value(), "Sharded add requires buffer distribution spec");
    const size_t num_cores_needed = distribution_spec->num_cores();
    TT_FATAL(
        full_dram_cores_vec.size() >= num_cores_needed,
        "Need {} worker cores for shards but get_optimal_dram_bank_to_logical_worker_assignment returned {}",
        num_cores_needed,
        full_dram_cores_vec.size());
    std::vector<CoreCoord> dram_cores_vec(full_dram_cores_vec.begin(), full_dram_cores_vec.begin() + num_cores_needed);
    CoreRangeSet shard_core_set;
    for (const auto& c : dram_cores_vec) {
        shard_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }
    const auto& all_device_cores = shard_core_set;
    (void)operation_attributes;
    auto dtype = tt_metal::datatype_to_dataformat_converter(args.a_tensor.dtype());

    /***************   CIRCULAR BUFFERS ***************/

    auto createCircularBuffer = [&program, &all_device_cores, dtype = dtype](
                                    tt::CBIndex cb_idx, uint32_t tile_size, uint32_t num_input_tiles = 1) {
        auto cb_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * tile_size, {{cb_idx, dtype}})
                             .set_page_size(cb_idx, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_config);
    };

    // auto shard_args = args.a_tensor.tensor_spec().compute_buffer_sharding_args();

    // const auto num_tiles_per_shard = shard_args.buffer_distribution_spec()->num_tiles_per_shard();

    const auto page_size = tt::tile_size(dtype);  // page is a tile
    const auto num_tiles_per_shard_width = args.a_tensor.shard_spec()->shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t num_tiles_per_cycle = num_tiles_per_shard_width;

    TT_FATAL(
        args.a_tensor.shard_spec()->shape[1] % tt::constants::TILE_WIDTH == 0,
        "Num tiles per page should be multiple of tile width");
    TT_FATAL(
        args.a_tensor.shard_spec()->shape[0] % tt::constants::TILE_HEIGHT == 0,
        "Num tiles per shard should be multiple of tile height {} {}",
        args.a_tensor.shard_spec()->shape[0],
        tt::constants::TILE_HEIGHT);

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    auto a_tensor_cb = tt::CBIndex::c_0;
    auto b_tensor_cb = tt::CBIndex::c_1;
    auto output_cb_index = tt::CBIndex::c_2;

    CBHandle a_tensor_cb_handle = createCircularBuffer(a_tensor_cb, page_size, num_tiles_per_shard_width);
    CBHandle b_tensor_cb_handle = createCircularBuffer(b_tensor_cb, page_size, num_tiles_per_shard_width);

    CBHandle cb_output = createCircularBuffer(output_cb_index, page_size, num_tiles_per_shard_width);

    add_nd_sharded_args::CompileTimeReaderKernelArgs reader_compile_time_args = {
        .a_tensor_cb = a_tensor_cb, .b_tensor_cb = b_tensor_cb, .num_tiles_per_cycle = num_tiles_per_cycle};

    /***************   READER KERNEL ***************/
    /* Specify data movement kernels for reading/writing data to/from DRAM */
    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_vec = ttnn::kernel_utils::to_vector(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(args.a_tensor.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.b_tensor.buffer()).append_to(reader_compile_time_vec);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/add/device/kernels/dataflow/elt_nd_sharded_add_reader.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_vec, reader_defines));

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    add_args::CompileTimeWriterKernelArgs writer_compile_time_args = {
        .cb_dst = output_cb_index, .num_tiles_per_cycle = num_tiles_per_cycle};

    /***************   WRITER KERNEL ***************/
    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> writer_compile_time_vec = ttnn::kernel_utils::to_vector(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_vec);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/add/device/kernels/dataflow/elt_nd_sharded_writer_kernel.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_vec, writer_defines));

    /***************   COMPUTE KERNEL ***************/
    /* Use the add_tiles operation in the compute kernel */
    add_args::CompileTimeComputeKernelArgs compute_compile_time_args = {
        .a_tensor_cb = a_tensor_cb,
        .b_tensor_cb = b_tensor_cb,
        .output_cb = output_cb_index,
        .num_tiles_per_cycle = num_tiles_per_cycle};
    std::vector<uint32_t> compute_compile_time_vec = ttnn::kernel_utils::to_vector(compute_compile_time_args);
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/add/device/kernels/compute/elemwise_add_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_vec});

    set_elt_nd_sharded_add_runtime_args<true>(
        program,
        args.a_tensor,
        args.b_tensor,
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        all_device_cores,
        &dram_cores_vec);
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         a_tensor_cb_handle,
         b_tensor_cb_handle,
         cb_output,
         all_device_cores,
         dram_cores_vec,
         page_size,
         page_size,
         page_size}};
}

void EltNDShardedAddProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const AddParams& /*operation_attributes*/,
    const AddInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& sh_var = cached_program.shared_variables;
    set_elt_nd_sharded_add_runtime_args<false>(
        cached_program.program,
        tensor_args.a_tensor,
        tensor_args.b_tensor,
        tensor_return_value,
        sh_var.reader_kernel_id,
        sh_var.writer_kernel_id,
        sh_var.eltwise_kernel_id,
        sh_var.all_device_cores,
        &sh_var.ordered_cores);
}
}  // namespace ttnn::experimental::prim
