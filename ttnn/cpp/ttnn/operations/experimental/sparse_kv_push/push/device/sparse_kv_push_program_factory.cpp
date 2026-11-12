// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_program_factory.hpp"
#include "sparse_kv_push_device_operation.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Kernel handles: create_descriptor pushes the kernels in this order.
constexpr tt::tt_metal::KernelHandle kReaderKernel = 0;
constexpr tt::tt_metal::KernelHandle kWriterKernel = 1;

constexpr uint8_t kMuxBuffersPerChannel = 2;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kUserIdCbBytes = 64;

// The reader takes the part as the kernels' own PushPart, value for value.
static_assert(
    static_cast<uint32_t>(SparseKvPushPart::kAll) == static_cast<uint32_t>(::sparse_push::PushPart::kAll) &&
    static_cast<uint32_t>(SparseKvPushPart::kPrefix) == static_cast<uint32_t>(::sparse_push::PushPart::kPrefix) &&
    static_cast<uint32_t>(SparseKvPushPart::kRest) == static_cast<uint32_t>(::sparse_push::PushPart::kRest));

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/push/device/kernels/";

CoreRangeSet to_range_set(const std::vector<CoreCoord>& cores) {
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    return CoreRangeSet(ranges);
}

// Sender w serves channel w / kSendersPerChannel; channel c = link * directions.size() + d sends towards directions[d].
// The senders come first in the grid, so sender 0 (the sync core the fences address on every chip) is the same core on
// every chip, however many directions a chip has.
struct CoreLayout {
    uint32_t links = 0;
    std::vector<FabricDirection> directions;
    std::vector<CoreCoord> senders;
    std::vector<CoreCoord> muxes;
    CoreRangeSet sender_range;
    CoreRangeSet mux_range;

    uint32_t num_directions() const { return static_cast<uint32_t>(directions.size()); }
    uint32_t channels() const { return links * num_directions(); }
    uint32_t num_senders() const { return static_cast<uint32_t>(senders.size()); }
    const FabricDirection& direction_of(uint32_t channel) const { return directions[channel % num_directions()]; }
};

CoreLayout core_layout(const CoreRangeSet& grid, uint32_t links, std::vector<FabricDirection> directions) {
    TT_FATAL(links >= 1 && links <= 2, "the sparse KV push supports 1 or 2 links, got {}", links);
    TT_FATAL(!directions.empty(), "the sparse KV push found no fabric neighbour");
    const auto cores = corerange_to_cores(grid, std::nullopt, true);
    const auto dirs = static_cast<uint32_t>(directions.size());
    const uint32_t senders = links * dirs * kSendersPerChannel;
    TT_FATAL(
        cores.size() >= cores_needed(links, dirs),
        "the sparse KV push needs {} sender + {} mux cores, the grid has {}",
        senders,
        links * dirs,
        cores.size());
    CoreLayout layout{.links = links, .directions = std::move(directions)};
    layout.senders.assign(cores.begin(), cores.begin() + senders);
    layout.muxes.assign(cores.begin() + senders, cores.begin() + cores_needed(links, dirs));
    layout.sender_range = to_range_set(layout.senders);
    layout.mux_range = to_range_set(layout.muxes);
    return layout;
}

// The mux's L1 map lives in program-local L1 on the mux cores, allocated with the program, so it cannot overlap live L1
// buffers there. Its addresses are offsets into the map until the writers read its base from the mux.
tt::tt_fabric::FabricMuxConfig mux_config(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreLayout& layout) {
    tt::tt_fabric::FabricMuxConfig mux(
        *mesh_device,
        /*num_full_size_channels=*/kSendersPerChannel,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/kMuxBuffersPerChannel,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device),
        /*base_l1_address=*/0);
    mux.place_in_program_local_l1(desc, layout.mux_range);
    return mux;
}

// Both kernels look up every destination's first hop, indexed by row-major rank.
tt::tt_metal::KernelDescriptor::Defines kernel_defines(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto mesh_shape = mesh_device->get_view().shape();
    std::vector<uint32_t> chip_ids, mesh_ids;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto node = mesh_device->get_fabric_node_id(coord);
        chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
        mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
    }
    return {
        {"DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(chip_ids)},
        {"DEST_MESH_ID", ttnn::operations::ccl::common::stringify(mesh_ids)},
        {"FABRIC_2D", "1"},
    };
}

// The chip's fabric directions. The kernels send each requester's rows, and their completion counts, on the channel of
// the fabric's first hop towards it, so every first hop must be one of them, or that requester would wait forever. The
// directions must also differ: two neighbours the fabric reaches the same way mean the topology was misread (e.g. a
// line taken for a ring).
std::vector<FabricDirection> checked_directions(
    const Tensor& tensor, const tt::tt_metal::distributed::MeshCoordinate& coord) {
    auto* mesh_device = tensor.device();
    auto directions = fabric_directions(tensor, coord);
    for (size_t i = 0; i < directions.size(); ++i) {
        for (size_t j = i + 1; j < directions.size(); ++j) {
            TT_FATAL(
                directions[i].eth_direction != directions[j].eth_direction,
                "the sparse KV push from chip {} reaches neighbours {} and {} in the same fabric direction",
                coord,
                directions[i].neighbor,
                directions[j].neighbor);
        }
    }
    const auto my_node = mesh_device->get_fabric_node_id(coord);
    for (const auto& other : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (other == coord) {
            continue;
        }
        const auto first_hop =
            tt::tt_fabric::get_eth_forwarding_direction(my_node, mesh_device->get_fabric_node_id(other));
        TT_FATAL(
            first_hop.has_value() &&
                std::any_of(
                    directions.begin(),
                    directions.end(),
                    [&](const FabricDirection& d) { return d.eth_direction == static_cast<uint32_t>(*first_hop); }),
            "the sparse KV push from chip {} has no channel on the fabric's route to chip {}",
            coord,
            other);
    }
    return directions;
}

std::vector<uint32_t> reader_compile_args(
    const SparseKvPushParams& params,
    const SparseKvPushInputs& tensor_args,
    uint32_t num_devices,
    const PacketShape& shape) {
    const auto layout = PlanTableLayout::make(params.max_rows, params.stripe_chunk);
    std::vector<uint32_t> ct = {
        tensor_args.input_tensor.buffer()->aligned_page_size(),
        shape.pages_per_packet,
        shape.banks,
        tensor_args.input_tensor.buffer()->num_pages() /
            tensor_args.input_tensor.padded_shape()[0],  // pages per cache slot
        tensor_args.batch_index_tensor.has_value() ? 1u : 0u,
        num_devices,
        params.stripe_chunk,
        layout.num_blocks,
        layout.pages,
        tensor_args.output_tensor.buffer()->num_pages(),
        static_cast<uint32_t>(params.part),
        tensor_args.input_tensor.padded_shape()[0],  // cache slots
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(tensor_args.table.buffer()).append_to(ct);
    if (tensor_args.batch_index_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.batch_index_tensor->buffer()).append_to(ct);
    }
    return ct;
}

std::vector<uint32_t> writer_compile_args(
    const SparseKvPushInputs& tensor_args,
    uint32_t num_devices,
    const PacketShape& shape,
    const CoreLayout& layout,
    const tt::tt_fabric::FabricMuxConfig& mux) {
    const uint32_t page_bytes = tensor_args.output_tensor.buffer()->aligned_page_size();
    std::vector<uint32_t> ct = {
        page_bytes, num_devices, layout.num_senders(), shape.pages_per_packet * page_bytes, layout.links};
    tt::tt_metal::TensorAccessorArgs(tensor_args.output_tensor.buffer()).append_to(ct);
    ttnn::ccl::fabric_mux_connection_ct_args(
        kSendersPerChannel, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux, ct);
    return ct;
}

std::vector<uint32_t> reader_runtime_args(const CoreLayout& layout, uint32_t sender, uint32_t my_rank) {
    const uint32_t channel = sender / kSendersPerChannel;
    return {
        channel / layout.num_directions(),  // link
        layout.direction_of(channel).eth_direction,
        my_rank,
        sender % kSendersPerChannel,
        kSendersPerChannel,
        layout.links,
        layout.directions.front().eth_direction};  // the channels of the first direction also carry this chip's rows
}

std::vector<uint32_t> writer_runtime_args(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreLayout& layout,
    uint32_t sender,
    uint32_t my_rank,
    const tt::tt_fabric::FabricMuxConfig& mux,
    tt::tt_metal::ProgramDescriptor& desc) {
    const uint32_t channel = sender / kSendersPerChannel, sub = sender % kSendersPerChannel;
    const auto sync_core = mesh_device->worker_core_from_logical_core(layout.senders[0]);
    std::vector<uint32_t> rt = {
        sender,
        layout.direction_of(channel).eth_direction,
        channel / layout.num_directions(),
        sub,
        my_rank,
        static_cast<uint32_t>(sync_core.x),
        static_cast<uint32_t>(sync_core.y)};
    for (const auto& core : layout.senders) {
        const auto phys = mesh_device->worker_core_from_logical_core(core);
        rt.push_back(static_cast<uint32_t>(phys.x));
        rt.push_back(static_cast<uint32_t>(phys.y));
    }
    ttnn::ccl::fabric_mux_connection_rt_args(
        /*mux_connection_valid=*/true,
        /*is_termination_master=*/sub == 0,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mesh_device->worker_core_from_logical_core(layout.muxes[channel]),
        /*worker_id=*/sub,
        layout.senders[sender],
        mux,
        desc,
        mesh_device->worker_core_from_logical_core(layout.senders[channel * kSendersPerChannel]),
        rt);
    return rt;
}

tt::tt_metal::KernelDescriptor data_movement_kernel(
    const std::string& source,
    const CoreRangeSet& cores,
    std::vector<uint32_t> compile_args,
    tt::tt_metal::KernelDescriptor::Defines defines,
    tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = source;
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = std::move(compile_args);
    kernel.defines = std::move(defines);
    kernel.config = std::move(config);
    return kernel;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

tt::tt_metal::ProgramDescriptor SparseKvPushProgramFactory::create_descriptor(
    const SparseKvPushParams& params,
    const SparseKvPushInputs& tensor_args,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "sparse_kv_push programs are per device");
    const auto& coord = *mesh_dispatch_coordinate;
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = tensor_args.input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.shape().mesh_size();
    const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
    const auto layout =
        core_layout(params.core_grid, params.num_links, checked_directions(tensor_args.input_tensor, coord));
    const auto mux = mux_config(mesh_device, desc, layout);
    const auto defines = kernel_defines(mesh_device);
    const uint32_t page_bytes = tensor_args.input_tensor.buffer()->aligned_page_size();
    const auto shape = packet_shape(mesh_device, page_bytes);
    const uint32_t packet_bytes = shape.pages_per_packet * page_bytes;

    using namespace ::sparse_push;
    const struct {
        uint32_t id, page_bytes, pages;
        tt::DataFormat format;
    } cbs[] = {
        {kData, packet_bytes, kPacketSlots, tt::DataFormat::RawUInt8},
        {kMeta, kMetaEntryBytes, kPacketSlots, tt::DataFormat::RawUInt32},
        {kTable, kTableCbBytes, 1, tt::DataFormat::RawUInt32},
        {kUserId, kUserIdCbBytes, 1, tt::DataFormat::RawUInt32},
        {kHeaders,
         static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()),
         kHeaderSlots,
         tt::DataFormat::RawUInt32},
    };
    for (const auto& cb : cbs) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = cb.page_bytes * cb.pages,
            .core_ranges = layout.sender_range,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb.id), .data_format = cb.format, .page_size = cb.page_bytes}}},
        });
    }

    const std::string kdir = kKernelDir;
    auto reader = data_movement_kernel(
        kdir + "sparse_push_table_reader.cpp",
        layout.sender_range,
        reader_compile_args(params, tensor_args, num_devices, shape),
        defines,
        tt::tt_metal::ReaderConfigDescriptor{});
    tt::tt_metal::KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(tensor_args.input_tensor.buffer());
    reader_common.push_back(tensor_args.table.buffer());
    if (tensor_args.batch_index_tensor.has_value()) {
        reader_common.push_back(tensor_args.batch_index_tensor->buffer());
    } else {
        reader_common.push_back(0u);
    }
    reader_common.push_back(params.batch_slot_num_layers);
    reader_common.push_back(params.batch_slot_layer_idx);
    reader_common.push_back(params.batch_index);
    reader.emplace_common_runtime_args(reader_common);

    auto writer = data_movement_kernel(
        kdir + "sparse_push_writer.cpp",
        layout.sender_range,
        writer_compile_args(tensor_args, num_devices, shape, layout, mux),
        defines,
        tt::tt_metal::WriterConfigDescriptor{});
    tt::tt_metal::KernelDescriptor::RTArgList writer_common;
    writer_common.push_back(tensor_args.output_tensor.buffer());
    writer_common.push_back(static_cast<uint32_t>(params.ready_semaphore->address()));  // smuggled-rta-ok: semaphore
    writer_common.push_back(static_cast<uint32_t>(params.done_semaphore->address()));   // smuggled-rta-ok: same
    writer.emplace_common_runtime_args(writer_common);

    auto mux_kernel = data_movement_kernel(
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        layout.mux_range,
        mux.get_fabric_mux_compile_time_args(),
        {},
        tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    mux_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;

    for (uint32_t c = 0; c < layout.channels(); ++c) {
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(mux.get_fabric_mux_run_time_args(
            *mesh_device,
            mesh_device->get_fabric_node_id(coord),
            mesh_device->get_fabric_node_id(layout.direction_of(c).neighbor),
            layout.direction_of(c).links[c / layout.num_directions()],
            desc,
            layout.muxes[c]));
        mux_kernel.emplace_runtime_args(layout.muxes[c], rt);
    }
    for (uint32_t w = 0; w < layout.num_senders(); ++w) {
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
        reader_rt.append(reader_runtime_args(layout, w, my_rank));
        reader.emplace_runtime_args(layout.senders[w], reader_rt);
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.append(writer_runtime_args(mesh_device, layout, w, my_rank, mux, desc));
        writer.emplace_runtime_args(layout.senders[w], writer_rt);
    }

    // In the order of kReaderKernel and kWriterKernel, which override_runtime_arguments patches.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(mux_kernel));
    return desc;
}

void SparseKvPushProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseKvPushParams& params,
    const SparseKvPushInputs& tensor_args,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, kReaderKernel);
    reader[0] = tensor_args.input_tensor.buffer()->address();
    reader[1] = tensor_args.table.buffer()->address();
    reader[2] = tensor_args.batch_index_tensor.has_value() ? tensor_args.batch_index_tensor->buffer()->address() : 0u;
    reader[3] = params.batch_slot_num_layers;
    reader[4] = params.batch_slot_layer_idx;
    reader[5] = params.batch_index;
    auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, kWriterKernel);
    writer[0] = tensor_args.output_tensor.buffer()->address();
    // The semaphores are not in the program hash: check new ones before the program uses them.
    if (writer[1] != params.ready_semaphore->address() || writer[2] != params.done_semaphore->address()) {
        validate_semaphore_placement(params, tensor_args.input_tensor.device());
    }
    writer[1] = params.ready_semaphore->address();
    writer[2] = params.done_semaphore->address();
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
