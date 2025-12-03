
#include "pad_rm_reader_writer_multi_core_v2_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement::pad::program {
PadRmReaderWriterMultiCoreV2ProgramFactory::cached_program_t PadRmReaderWriterMultiCoreV2ProgramFactory::create(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    const float pad_value) {
    Program program{};

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];
    uint32_t NCH_padded = H_padded * C_padded * N_padded;

    const auto& front_pad = input_tensor_start;

    auto stick_size = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_padded_front = front_pad[-1] * a.element_size();
    auto stick_size_padded_end = stick_size_padded - stick_size - stick_size_padded_front;
    uint32_t stick_size_padded_aligned = tt::align(stick_size_padded, hal::get_l1_alignment());
    uint32_t stick_size_padded_DRAM_aligned = tt::align(stick_size_padded, hal::get_dram_alignment());
    uint32_t row_major_min_bytes = 16;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_sticks_padded_per_core_group_1,
         num_sticks_padded_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);

    uint32_t src0_cb_index = tt::CBIndex::c_0;

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(stick_size_padded_DRAM_aligned, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, stick_size_padded_DRAM_aligned);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);

    bool unaligned = stick_size_padded_aligned % hal::get_dram_alignment() != 0;
    if (stick_size_padded_front != 0 || unaligned) {
        uint32_t src2_cb_index = tt::CBIndex::c_2;
        tt::tt_metal::CircularBufferConfig cb_src2_config =
            tt::tt_metal::CircularBufferConfig(stick_size_padded_DRAM_aligned, {{src2_cb_index, cb_data_format}})
                .set_page_size(src2_cb_index, stick_size_padded_DRAM_aligned);
        tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src2_config);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)N + front_pad[-4],
        (std::uint32_t)H + front_pad[-2],
        (std::uint32_t)C + front_pad[-3],
        (std::uint32_t)stick_size,
        (std::uint32_t)N_padded,
        (std::uint32_t)H_padded,
        (std::uint32_t)C_padded,
        (std::uint32_t)stick_size_padded,
        (std::uint32_t)stick_size_padded_front,
        (std::uint32_t)stick_size_padded_end,
        (std::uint32_t)tt::div_up(stick_size_padded, 512),  // max zero size is 512B
        (std::uint32_t)(stick_size_padded % 512 == 0 ? 512 : stick_size_padded % 512),
        (std::uint32_t)not_pad_by_zero,
        (std::uint32_t)packed_pad_value,
        (std::uint32_t)row_major_min_bytes,
        (std::uint32_t)(stick_size_padded_front / row_major_min_bytes),
        (std::uint32_t)(stick_size_padded_end / row_major_min_bytes),
        (std::uint32_t)(stick_size_padded / row_major_min_bytes),
        (std::uint32_t)stick_size_padded_aligned,
        (std::uint32_t)unaligned};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)src0_cb_index, (std::uint32_t)stick_size_padded, (std::uint32_t)stick_size_padded_aligned};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved_v2.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved_v2.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_runtime_args_rm(
        a,
        output,
        input_tensor_start,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        num_sticks_padded_per_core_group_1,
        core_group_2,
        num_sticks_padded_per_core_group_2);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);

        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, all_runtime_args[i].second

        );
    }
    uint32_t cb_npages = get_num_stick_per_barrier(a);
    const uint32_t buffer_reader_writer_async_factor = 16;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_reader_writer_async_factor * cb_npages * stick_size_padded_aligned,
            {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_padded_aligned);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size, input_tensor_start](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& src_tensor = input_tensors.at(0);

            auto dst_tensor = output_tensors.at(0);

            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;

            uint32_t num_cores_total = num_cores_x * num_cores_y;

            auto output_tensor_shape = dst_tensor.logical_shape();
            uint32_t H_padded = output_tensor_shape[2], C_padded = output_tensor_shape[1],
                     N_padded = output_tensor_shape[0];
            uint32_t NCH_padded = H_padded * C_padded * N_padded;

            auto
                [num_cores,
                 all_cores,
                 core_group_1,
                 core_group_2,
                 num_sticks_padded_per_core_group_1,
                 num_sticks_padded_per_core_group_2] =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);
            auto all_runtime_args = get_runtime_args_rm(
                src_tensor,
                dst_tensor,
                input_tensor_start,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                num_sticks_padded_per_core_group_1,
                core_group_2,
                num_sticks_padded_per_core_group_2);

            for (uint32_t i = 0; i < num_cores_total; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};

                {
                    SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
                }

                {
                    SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::pad::program
