#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

namespace ttnn::operations::point_to_point::detail {

std::tuple<uint32_t, bool, IDevice*> calculate_fabric_connection(
    const MeshDevice* mesh_device,
    const MeshCoordinate& src_coord,
    const MeshCoordinate& dest_coord,
    const ccl::Topology& topology) {
    auto devices = mesh_device->get_devices();
    auto begin_it = devices.cbegin();
    int src_idx =
        std::find_if(begin_it, devices.cend(), [&](auto& d) { return d == mesh_device->get_device(src_coord); }) -
        begin_it;
    int dest_idx =
        std::find_if(begin_it, devices.cend(), [&](auto& d) { return d == mesh_device->get_device(dest_coord); }) -
        begin_it;

    // sign indicates direction
    int line_hops = dest_idx - src_idx;

    TT_ASSERT(line_hops != 0);

    if (topology == ccl::Topology::Ring) {
        int ring_hops = line_hops + (line_hops < 0 ? -1 : 1) * mesh_device->get_devices().size();

        if (std::abs(ring_hops) < std::abs(line_hops)) {
            bool dst_is_forward = (ring_hops > 0);
            auto next_device = devices.at(src_idx + (dst_is_forward ? 1 : -1));
            TT_ASSERT(next_device != nullptr);
            return std::make_tuple(std::abs(ring_hops), dst_is_forward, next_device);
        }
    }

    bool dst_is_forward = (line_hops > 0);
    auto next_device = devices.at(src_idx + (dst_is_forward ? 1 : -1));
    TT_ASSERT(next_device != nullptr);
    return std::make_tuple(std::abs(line_hops), dst_is_forward, next_device);
}

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    PointToPointOp::tensor_return_value_t& output_tensors) {
    auto mesh_device = operation_attributes.mesh_device();
    const auto& receive_coord = operation_attributes.receive_coord;
    const auto& topology = operation_attributes.topology;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& receiver_semaphore = operation_attributes.receiver_semaphore;

    // basic accounting
    const uint32_t input_num_pages = data_movement::get_num_pages(input_tensor);
    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();

    // figure out packets
    // !TODO see what happens if page size is larger than packet size.
    const auto [packet_size_bytes, num_pages_per_packet, total_packets] =
        compute_packet_dims(input_tensor.get_dtype(), input_page_size_bytes, input_num_pages);

    // distribute work
    // !TODO debug
    // auto use_cores = mesh_device->compute_with_storage_grid_size()
    CoreCoord use_cores = {1, 1};
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    // program!
    tt::tt_metal::Program program{};

    // CB for sender reader->writer kernels
    // Note this ID is hardcoded in the reader kernel
    constexpr auto sender_cb_id = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 1;

    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * input_page_size_bytes, {{sender_cb_id, input_dataformat}})
            .set_page_size(sender_cb_id, input_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_handle = CreateCircularBuffer(program, all_cores, cb_sender_config);

    // allocate space for packet headers for payload and maybe sempahore ?
    constexpr auto packet_header_cb_id = tt::CBIndex::c_6;
    constexpr auto buffering_factor = 2;             // ?
    constexpr auto num_packet_headers_storable = 8;  // ?
    constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);
    auto cb_header_handle = CreateCircularBuffer(program, all_cores, cb_header_config);

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    tt::tt_metal::CBHandle cb_cb_handle = CreateCircularBuffer(program, all_cores, cb_packet_config);

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // basic reader kernel set up
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({input_is_dram}));

    auto this_device = mesh_device->get_device(send_coord);
    const auto [num_hops, dst_is_forward, next_device] =
        calculate_fabric_connection(mesh_device, send_coord, receive_coord, topology);
    const bool output_is_dram = output_tensors.at(0).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const std::vector<uint32_t> writer_ct_args = {
        sender_cb_id, packet_cb_id, packet_header_cb_id, output_is_dram, l1_alignment};
    std::cout << "------SEND PF------" << std::endl;
    std::cout << "CT ARGS: ";
    for (auto& a : writer_ct_args) {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    constexpr auto link_idx = 0;  // equivalent to num_links = 0

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
        increment = std::min(increment, input_num_pages - page_idx_start);
        page_idx_end += increment;

        const std::vector<uint32_t> reader_runtime_args = {input_tensor.buffer()->address(), increment, page_idx_start};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        std::cout << "READER RT ARGS: ";
        for (auto& a : reader_runtime_args) {
            std::cout << a << " ";
        }
        std::cout << std::endl;

        std::vector<uint32_t> writer_runtime_args = {
            output_tensors.at(0).buffer()->address(),
            page_idx_start,
            page_idx_end,
            num_hops,
            input_page_size_bytes,
            packet_size_bytes,
            num_pages_per_packet,
            receiver_semaphore.address(),
            c.x,
            c.y,
            dst_is_forward,
        };

        if (dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_device->id(), next_device->id(), link_idx, program, c, writer_runtime_args);
        }
        writer_runtime_args.emplace_back(!dst_is_forward);
        if (!dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_device->id(), next_device->id(), link_idx, program, c, writer_runtime_args);
        }

        std::cout << "WRITER RT ARGS: ";
        for (auto& a : writer_runtime_args) {
            std::cout << a << " ";
        }
        std::cout << std::endl;

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
    }

    std::cout << "------END SEND PF------" << std::endl;

    // !TODO
    return {std::move(program), PointToPointOp::SendReceive::shared_variables_t{}};
}
}  // namespace ttnn::operations::point_to_point::detail
