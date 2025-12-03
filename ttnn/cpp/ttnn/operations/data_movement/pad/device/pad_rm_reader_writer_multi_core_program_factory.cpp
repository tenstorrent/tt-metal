
#include "pad_rm_reader_writer_multi_core_v2_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement::pad::program {
PadRmReaderWriterMultiCoreProgramFactory::cached_program_t PadRmReaderWriterMultiCoreProgramFactory::create(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    const float pad_value) {
    Program program{};

    auto output_shape = output_padded_shape;

    uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    distributed::MeshDevice* device = a.device();

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    // NOTE: The const buffer is always in L1
    // TODO: make a local buffer for each core?
    const Tensor pad_value_const_tensor =
        Tensor(
            std::move(pad_value_const_buffer),
            ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
            .to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    // uint32_t ntiles_h = output_tensor_shape[0] * output_tensor_shape[1] * output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_h = output_padded_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_w = output_padded_shape[3] / TILE_WIDTH;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t nbatch = output_padded_shape[0];
    uint32_t nchannel = output_padded_shape[1];
    // first the batch dim is distributed along H, and within each batch then the tiles are distributed.
    auto
        [ncores,
         ncores_h,
         ncores_w,
         all_cores,
         core_range,
         ntiles_per_core_h,
         ntiles_per_core_w,
         nbatch_per_core_h,
         ncores_per_batch_h] = split_across_cores(grid_size, nbatch, nchannel, ntiles_h, ntiles_w);

    [[maybe_unused]] int32_t src_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * a.element_size();
    int32_t dst_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * output.element_size();

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t cb_id = tt::CBIndex::c_0;
    uint32_t cb_npages = 16;  // multibuffering for perf
    // uint32_t cb_npages = 1; // multibuffering for perf
    uint32_t cb_page_alignment = std::max(tt::constants::TILE_WIDTH, src0_buffer->alignment());
    uint32_t cb_pagesize =
        static_cast<uint32_t>(std::ceil((float)dst_nbytes_per_core_w / cb_page_alignment)) * cb_page_alignment;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}}).set_page_size(cb_id, cb_pagesize);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    std::vector<uint32_t> reader_ct_args = {unpadded_row_size_nbytes, padded_row_size_nbytes};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*dst_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*pad_value_const_tensor.buffer()).append_to(reader_ct_args);
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    // int32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;
    log_rt_args(CoreCoord{0, 0}, reader_ct_args);

#if 1
    {
        log_debug(tt::LogOp, "ncores: {}", ncores);
        log_debug(tt::LogOp, "ncores_h: {}", ncores_h);
        log_debug(tt::LogOp, "ncores_w: {}", ncores_w);
        log_debug(tt::LogOp, "ntiles_per_core_h: {}", ntiles_per_core_h);
        log_debug(tt::LogOp, "ntiles_per_core_w: {}", ntiles_per_core_w);
        log_debug(tt::LogOp, "src0_buffer_addr: {}", src0_buffer->address());
        log_debug(tt::LogOp, "dst_buffer_addr: {}", dst_buffer->address());
        log_debug(tt::LogOp, "a.shape[0]: {}", a.padded_shape()[0]);
        log_debug(tt::LogOp, "out.shape[0]: {}", output_shape[0]);
        log_debug(tt::LogOp, "a.shape[1]: {}", a.padded_shape()[1]);
        log_debug(tt::LogOp, "out.shape[1]: {}", output_shape[1]);
        log_debug(tt::LogOp, "a.shape[2]: {}", a.padded_shape()[2]);
        log_debug(tt::LogOp, "out.shape[2]: {}", output_shape[2]);
        log_debug(tt::LogOp, "s.shape[3]: {}", a.padded_shape()[3]);
        log_debug(tt::LogOp, "out.shape[3]: {}", output_shape[3]);
        log_debug(tt::LogOp, "unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_size_nbytes: {}", padded_row_size_nbytes);
        // log_debug(tt::LogOp, "padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug(tt::LogOp, "pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug(tt::LogOp, "pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug(tt::LogOp, "packed_pad_value: {}", packed_pad_value);
        log_debug(tt::LogOp, "src_nbytes_per_core_w: {}", src_nbytes_per_core_w);
        log_debug(tt::LogOp, "dst_nbytes_per_core_w: {}", dst_nbytes_per_core_w);
        log_debug(tt::LogOp, "nbatch_per_core_h: {}", nbatch_per_core_h);
        log_debug(tt::LogOp, "ncores_per_batch_h: {}", ncores_per_batch_h);
    }
#endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    uint32_t start_src_stick_wi = 0;  // start of stick segment for 2d decomp
    uint32_t start_dst_stick_wi = 0;
    int32_t local_nsticks = ntiles_per_core_h * TILE_HEIGHT;
    for (int32_t b = 0; b < nbatch; ++b) {
        int32_t rem_src_nsticks = a.padded_shape()[2];
        for (uint32_t j = 0; j < ncores_per_batch_h; ++j) {
            uint32_t num_local_unpadded_nsticks = local_nsticks;
            if (rem_src_nsticks - local_nsticks >= 0) {
                // not reached padding sticks yet
                rem_src_nsticks -= local_nsticks;
            } else {
                num_local_unpadded_nsticks = rem_src_nsticks;
                rem_src_nsticks = 0;
            }
            start_src_stick_wi = 0;
            start_dst_stick_wi = 0;
            int32_t rem_src_stick_size_nbytes = unpadded_row_size_nbytes;
            for (uint32_t i = 0; i < ncores_w; ++i) {
                CoreCoord core = {i, (b * ncores_per_batch_h) + j};
                uint32_t curr_stick_size_nbytes = 0;
                int32_t curr_stick_diff_nbytes = 0;
                if (rem_src_stick_size_nbytes - dst_nbytes_per_core_w >= 0) {
                    // no padding on this core
                    curr_stick_size_nbytes = dst_nbytes_per_core_w;
                    rem_src_stick_size_nbytes -= dst_nbytes_per_core_w;
                } else {
                    // this core has padding
                    curr_stick_size_nbytes = rem_src_stick_size_nbytes;
                    curr_stick_diff_nbytes = dst_nbytes_per_core_w - curr_stick_size_nbytes;
                    rem_src_stick_size_nbytes = 0;
                }
                const std::array reader_rt_args = {
                    src0_buffer->address(),
                    dst_buffer->address(),
                    a.padded_shape()[0],
                    output_shape[0],
                    a.padded_shape()[1],
                    output_shape[1],
                    a.padded_shape()[2],
                    output_shape[2],
                    a.padded_shape()[3],
                    output_shape[3],
                    curr_stick_size_nbytes,
                    (uint32_t)dst_nbytes_per_core_w,
                    (uint32_t)curr_stick_diff_nbytes,
                    pad_value_const_tensor_addr,
                    pad_value_const_buffer_nbytes,
                    packed_pad_value,
                    start_src_stick_id,
                    start_dst_stick_id,
                    start_src_stick_wi,
                    start_dst_stick_wi,
                    start_src_stick_wi * a.element_size(),
                    (uint32_t)local_nsticks,
                    num_local_unpadded_nsticks,
                    unpadded_row_size_nbytes,
                    padded_row_size_nbytes,
                    start_dst_stick_wi * output.element_size(),
                    nbatch_per_core_h};
                // if (core.x == 0) log_rt_args(core, reader_rt_args);
                // if (core.x == 0) {
                //     log_debug(tt::LogOp, "{} :: start_src_stick_id: {}", core.y, start_src_stick_id);
                //     log_debug(tt::LogOp, "{} :: start_dst_stick_id: {}", core.y, start_dst_stick_id);
                //     log_debug(tt::LogOp, "{} :: local_nsticks: {}", core.y, local_nsticks);
                //     log_debug(tt::LogOp, "{} :: num_local_unpadded_nsticks: {}", core.y, num_local_unpadded_nsticks);
                //     log_debug(tt::LogOp, "{} :: nbatch_per_core_h: {}", core.y, nbatch_per_core_h);
                //     log_debug(tt::LogOp, "{} :: ncores_per_batch_h: {}", core.y, ncores_per_batch_h);
                // }
                const auto& writer_rt_args = reader_rt_args;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
                start_src_stick_wi += ntiles_per_core_w * TILE_WIDTH;
                start_dst_stick_wi += ntiles_per_core_w * TILE_WIDTH;
            }  // for ncores_w
            start_src_stick_id += num_local_unpadded_nsticks;
            start_dst_stick_id += local_nsticks;
        }  // for ncores_h
    }

    auto override_runtime_args_callback = [reader_kernel_id = reader_kernel_id,
                                           writer_kernel_id = writer_kernel_id,
                                           ncores_h = ncores_h,
                                           ncores_w = ncores_w](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto* src_buffer = input_tensors.at(0).buffer();
        auto* dst_buffer = output_tensors.at(0).buffer();

        for (uint32_t j = 0; j < ncores_h; ++j) {
            for (uint32_t i = 0; i < ncores_w; ++i) {
                CoreCoord core = {i, j};
                {
                    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
                {
                    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::pad::program
