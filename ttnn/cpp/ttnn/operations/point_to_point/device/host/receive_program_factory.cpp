
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

namespace ttnn::operations::point_to_point::detail {

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensors) {
    auto mesh_device = operation_attributes.mesh_device();

    const auto& intermediate_tensor = output_tensors.at(0);
    const auto& output_tensor = output_tensors.at(1);

    // basic accounting
    const uint32_t output_num_pages = data_movement::get_num_pages(output_tensor);
    const uint32_t output_page_size_bytes = output_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // figure out packets
    // !TODO see what happens if page size is larger than packet size.
    const auto [packet_size_bytes, num_pages_per_packet, total_packets] =
        compute_aligned_packet_dims(output_tensor.get_dtype(), output_page_size_bytes, output_num_pages, l1_alignment);

    // distribute work
    const auto use_cores = mesh_device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    // program!
    tt::tt_metal::Program program{};

    tt::DataFormat inter_dataformat = tt::tt_metal::datatype_to_dataformat_converter(intermediate_tensor.get_dtype());

    // Scratch CB for loading up pages that are collected into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, inter_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    tt::tt_metal::CBHandle cb_cb_handle = CreateCircularBuffer(program, all_cores, cb_packet_config);

    // CB for sender reader->writer kernels
    constexpr auto receiver_cb_id = tt::CBIndex::c_1;
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;

    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(intermediate_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_receiver_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * output_page_size_bytes, {{receiver_cb_id, inter_dataformat}})
            .set_page_size(receiver_cb_id, output_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_handle = CreateCircularBuffer(program, all_cores, cb_receiver_config);

    const bool intermediate_is_dram = output_tensors.at(0).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<uint32_t> reader_ct_args = {intermediate_is_dram, packet_cb_id, receiver_cb_id, l1_alignment};
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    const bool output_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // And the writer
    std::map<std::string, std::string> writer_defines;
    if (output_tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) {
        writer_defines["ROWMAJOR"] = "1";
    }
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({output_is_dram, receiver_cb_id}, writer_defines));

    uint32_t page_idx_start = 0, page_idx_end = 0;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_packets_per_core_group_1 * num_pages_per_packet;
        } else if (core_group_2.contains(c)) {
            increment = num_packets_per_core_group_2 * num_pages_per_packet;
        } else {
            continue;
        }
        increment = std::min(increment, output_num_pages - page_idx_start);
        page_idx_end += increment;

        const std::vector<uint32_t> reader_runtime_args = {
            page_idx_start,
            page_idx_end,
            num_pages_per_packet,
            intermediate_tensor.buffer()->address(),
            packet_size_bytes,
            output_page_size_bytes,
            operation_attributes.receiver_semaphore.address()};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(), increment, page_idx_start, packet_size_bytes};

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
    }

    // !TODO
    return {std::move(program), PointToPointOp::SendReceive::shared_variables_t{}};
}
}  // namespace ttnn::operations::point_to_point::detail
