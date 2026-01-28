// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
inline void log_rt_args(const CoreCoord& core, std::vector<uint32_t>& args) {
    for (auto v : args) {
        log_debug(tt::LogOp, "{},{} :: {}", core.x, core.y, v);
    }
}

// This is currently mostly hardcoded for resnet shapes
inline std::tuple<uint32_t, uint32_t, uint32_t, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
split_across_cores(CoreCoord grid_size, uint32_t nbatch, uint32_t ntiles_h, uint32_t ntiles_w) {
    uint32_t ncores, ncores_h, ncores_w, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h;

    // each batch needs to be padded independently
    switch (nbatch) {
        case 1:
            ncores_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_h = 2;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_h = 4;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_h = 8;
                    ntiles_per_core_h = 1;
                    break;
                case 64:
                    ncores_h = 8;
                    ntiles_per_core_h = 8;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            ncores_per_batch_h = ncores_h;
            break;

        case 2:
            ncores_h = 1;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_per_batch_h = 2;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 2;
                    break;
                case 64:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 16;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            break;

        case 8:
            ncores_h = 8;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = ntiles_h;
            break;

        default:
            TT_THROW("Unsupported nbatch value {} for pad operation. Supported values are 1, 2, and 8.", nbatch);

            // generic case -- TODO

            // one of the following will be 0 when grid_size.y != nbatch
            nbatch_per_core_h = nbatch / grid_size.y;   // floor
            ncores_per_batch_h = grid_size.y / nbatch;  // floor
            if (nbatch == grid_size.y) {
                nbatch_per_core_h = 1;
                ncores_per_batch_h = 1;
            }

            // currently uses hardcoded values for resnet50
            // TT_ASSERT(ntiles_h == 1 || ntiles_h == 2 || ntiles_h == 4 || ntiles_h == 16, "Only Resnet50 shapes are
            // supported in multicore version for now."); TT_ASSERT(ntiles_w == 64, "Only Resnet50 shapes are supported
            // in multicore version for now.");

            TT_ASSERT(nbatch <= grid_size.y, "Unsupported case with nbatch > grid_size.y!");

            if (nbatch_per_core_h == 0) {
                // there are multiple cores along h per batch
                nbatch_per_core_h = 1;
            } else if (ncores_per_batch_h == 0) {
                // unsupported case. TODO.
                TT_THROW(
                    "Unsupported configuration: multiple batches per core along height dimension "
                    "(nbatch={}, grid_size.y={})",
                    nbatch,
                    grid_size.y);
                // there are multiple batch per core along h
                // ncores_per_batch_h = 1;
            } else {
                TT_THROW("Something went terribly wrong in splitting across cores");
            }
            break;
    }

    switch (ntiles_w) {
        case 2: ncores_w = 2; break;
        case 4: ncores_w = 4; break;
        case 8:
        case 64: ncores_w = 8; break;
        default: TT_THROW("Unsupported ntiles_w value {}", ntiles_w);
    }
    ncores = ncores_h * ncores_w;
    ntiles_per_core_w = ntiles_w / ncores_w;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range;

    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));

    return std::make_tuple(
        ncores,
        ncores_h,
        ncores_w,
        all_cores,
        core_range,
        ntiles_per_core_h,
        ntiles_per_core_w,
        nbatch_per_core_h,
        ncores_per_batch_h);
}
}  // namespace

PadRmReaderWriterMultiCoreProgramFactory::cached_program_t PadRmReaderWriterMultiCoreProgramFactory::create(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
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
         ncores_per_batch_h] = split_across_cores(grid_size, nbatch, ntiles_h, ntiles_w);

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

    return cached_program_t{std::move(program), {ncores_h, ncores_w, reader_kernel_id, writer_kernel_id}};
}

void PadRmReaderWriterMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PadParams& /*operation_attributes*/,
    const PadInputs& tensor_args,
    Tensor& output) {
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    for (uint32_t j = 0; j < cached_program.shared_variables.ncores_h; ++j) {
        for (uint32_t i = 0; i < cached_program.shared_variables.ncores_w; ++i) {
            CoreCoord core = {i, j};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
                    cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
                    cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
    }
}

}  // namespace ttnn::prim
