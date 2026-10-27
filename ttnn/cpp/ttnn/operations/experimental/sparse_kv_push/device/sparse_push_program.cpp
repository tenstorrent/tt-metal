// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_program.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "sparse_push_plan_device_operation.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::sparse_push {

namespace {

constexpr uint8_t kMuxBuffersPerChannel = 2;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kScalarCbBytes = 64;
constexpr uint32_t kRowDim = 2;  // the cache is sharded over its rows across every chip

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/kernels/";

CoreRangeSet to_range_set(const std::vector<CoreCoord>& cores) {
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    return CoreRangeSet(ranges);
}

// Sender w serves channel w / kSendersPerChannel; channel c = link * kDirections + dir.
struct CoreLayout {
    uint32_t links = 0;
    std::vector<CoreCoord> senders;
    std::vector<CoreCoord> muxes;
    CoreRangeSet sender_range;
    CoreRangeSet mux_range;

    uint32_t channels() const { return links * kDirections; }
    uint32_t num_senders() const { return static_cast<uint32_t>(senders.size()); }
};

CoreLayout core_layout(const CoreRangeSet& grid, uint32_t links) {
    TT_FATAL(links >= 1 && links <= 2, "the sparse KV push supports 1 or 2 links, got {}", links);
    const auto cores = corerange_to_cores(grid, std::nullopt, true);
    const uint32_t senders = links * kDirections * kSendersPerChannel;
    TT_FATAL(
        cores.size() >= cores_needed(links),
        "the sparse KV push needs {} sender + {} mux cores, the grid has {}",
        senders,
        links * kDirections,
        cores.size());
    CoreLayout layout{.links = links};
    layout.senders.assign(cores.begin(), cores.begin() + senders);
    layout.muxes.assign(cores.begin() + senders, cores.begin() + cores_needed(links));
    layout.sender_range = to_range_set(layout.senders);
    layout.mux_range = to_range_set(layout.muxes);
    return layout;
}

tt::tt_fabric::FabricMuxConfig mux_config(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    return tt::tt_fabric::FabricMuxConfig(
        /*num_full_size_channels=*/kSendersPerChannel,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/kMuxBuffersPerChannel,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device),
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
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
        {"MESH_ROWS", std::to_string(mesh_shape[0])},
        {"MESH_COLS", std::to_string(mesh_shape[1])},
    };
}

// The chip's four torus neighbours, in channel direction order.
std::vector<tt::tt_metal::distributed::MeshCoordinate> torus_neighbors(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_view,
    const tt::tt_metal::distributed::MeshCoordinate& coord) {
    const auto [neighbors, directions] =
        ttnn::operations::ccl::common::get_neighbors(mesh_view, coord, tt::tt_fabric::Topology::Torus, std::nullopt);
    TT_FATAL(
        neighbors.size() == kDirections && std::all_of(directions.begin(), directions.end(), [](bool d) { return d; }),
        "the sparse KV push needs all four torus neighbours");
    return neighbors;
}

std::vector<uint32_t> reader_compile_args(const PushConfig& cfg, uint32_t num_devices, const PacketShape& shape) {
    const auto layout = SparsePushPlanLayout::make(cfg.max_rows, cfg.stripe_chunk);
    std::vector<uint32_t> ct = {
        cfg.input.buffer()->aligned_page_size(),
        shape.pages_per_packet,
        shape.banks,
        cfg.input.buffer()->num_pages() / cfg.input.padded_shape()[0],  // pages per cache slot
        cfg.batch_index_tensor.has_value() ? 1u : 0u,
        num_devices,
        cfg.stripe_chunk,
        layout.num_blocks,
        layout.pages,
        cfg.output.buffer()->num_pages(),
        cfg.part,
    };
    tt::tt_metal::TensorAccessorArgs(cfg.input.buffer()).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(cfg.table.buffer()).append_to(ct);
    if (cfg.batch_index_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(cfg.batch_index_tensor->buffer()).append_to(ct);
    }
    return ct;
}

std::vector<uint32_t> writer_compile_args(
    const PushConfig& cfg,
    uint32_t num_devices,
    const PacketShape& shape,
    const CoreLayout& layout,
    const tt::tt_fabric::FabricMuxConfig& mux) {
    const uint32_t page_bytes = cfg.output.buffer()->aligned_page_size();
    std::vector<uint32_t> ct = {
        page_bytes, num_devices, layout.num_senders(), shape.pages_per_packet * page_bytes, layout.links};
    tt::tt_metal::TensorAccessorArgs(cfg.output.buffer()).append_to(ct);
    ttnn::ccl::fabric_mux_connection_ct_args(
        kSendersPerChannel, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux, ct);
    return ct;
}

std::vector<uint32_t> reader_runtime_args(const CoreLayout& layout, uint32_t sender, uint32_t my_rank) {
    const uint32_t channel = sender / kSendersPerChannel;
    return {
        channel / kDirections,  // link
        channel % kDirections,  // direction
        my_rank,
        sender % kSendersPerChannel,
        kSendersPerChannel,
        layout.links};
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
        channel % kDirections,
        channel / kDirections,
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

void validate_semaphores(const PushConfig& cfg) {
    TT_FATAL(
        cfg.ready_semaphore.has_value() && cfg.data_valid_semaphore.has_value(),
        "the sparse KV push needs caller-owned ready and data_valid semaphores");
    const auto& ready = *cfg.ready_semaphore;
    const auto& data_valid = *cfg.data_valid_semaphore;
    TT_FATAL(ready.address() != data_valid.address(), "the sparse KV push's two semaphores must differ");
    auto* mesh_device = cfg.input.device();
    TT_FATAL(
        ready.device() == mesh_device && data_valid.device() == mesh_device,
        "the sparse KV push's semaphores must be created on the input tensor's mesh device");
    const auto ready_buffer_type = std::get<1>(ready.attribute_values());
    const auto data_valid_buffer_type = std::get<1>(data_valid.attribute_values());
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(BufferType::L1_SMALL) > 0;
    const auto expected_buffer_type = has_l1_small ? BufferType::L1_SMALL : BufferType::L1;
    TT_FATAL(
        ready_buffer_type == expected_buffer_type && data_valid_buffer_type == expected_buffer_type,
        "the sparse KV push's semaphores must use buffer type {}, got ready={} and data_valid={}",
        expected_buffer_type,
        ready_buffer_type,
        data_valid_buffer_type);
}

void validate_batch_slot(const PushConfig& cfg) {
    const uint32_t cache_batch = cfg.input.padded_shape()[0];
    if (cfg.batch_index_tensor.has_value()) {
        TT_FATAL(
            cfg.batch_slot_layer_idx < cfg.batch_slot_num_layers,
            "sparse KV push batch_slot_layer_idx {} must be < batch_slot_num_layers {}",
            cfg.batch_slot_layer_idx,
            cfg.batch_slot_num_layers);
        TT_FATAL(
            cfg.batch_slot_num_layers <= cache_batch && cache_batch % cfg.batch_slot_num_layers == 0,
            "sparse KV push batch_slot_num_layers {} must divide the cache batch {}",
            cfg.batch_slot_num_layers,
            cache_batch);
    } else {
        TT_FATAL(
            cfg.batch_index < cache_batch,
            "sparse KV push batch_index {} must be < cache batch {}",
            cfg.batch_index,
            cache_batch);
    }
}

void validate_batch_index_tensor(const Tensor& batch_index_tensor) {
    TT_FATAL(
        batch_index_tensor.storage_type() == StorageType::DEVICE && batch_index_tensor.buffer() != nullptr,
        "sparse KV push batch_index_tensor must be allocated on device");
    TT_FATAL(
        batch_index_tensor.dtype() == DataType::UINT32 && batch_index_tensor.layout() == Layout::ROW_MAJOR &&
            batch_index_tensor.logical_volume() == 1 &&
            batch_index_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            batch_index_tensor.memory_config().buffer_type() == BufferType::DRAM,
        "sparse KV push batch_index_tensor must be a 1-element UINT32 ROW_MAJOR interleaved DRAM tensor");
}

// Row-major DRAM, one row per page. Each packet is one contiguous write into one bank of an interleaved output; the
// input is read through a tensor accessor, so it may also be sharded (the model's cache is ND-sharded).
void validate_rows(const Tensor& tensor, const char* name, bool interleaved) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "sparse KV push {} must be allocated on device",
        name);
    const auto& shape = tensor.padded_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[1] == 1, "sparse KV push {} must be [batch, 1, rows, width], got {}", name, shape);
    const auto& memory_config = tensor.memory_config();
    TT_FATAL(
        tensor.layout() == Layout::ROW_MAJOR && memory_config.buffer_type() == BufferType::DRAM,
        "sparse KV push {} must be a ROW_MAJOR DRAM tensor",
        name);
    TT_FATAL(
        !interleaved || memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sparse KV push {} must be interleaved, got {}",
        name,
        memory_config);
}

}  // namespace

void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows) {
    TT_FATAL(
        stripe_chunk > 0 && stripe_chunk % 32 == 0 && stripe_chunk <= ::sparse_push::kMaxStripeChunk,
        "sparse KV push stripe_chunk {} must be a positive multiple of 32, at most {}",
        stripe_chunk,
        ::sparse_push::kMaxStripeChunk);
    TT_FATAL(
        max_rows > 0 && max_rows % stripe_chunk == 0,
        "sparse KV push max_rows {} must be a positive multiple of stripe_chunk {}",
        max_rows,
        stripe_chunk);
}

void validate_config(const PushConfig& cfg) {
    const auto& input = cfg.input;
    const auto& output = cfg.output;
    const auto& table = cfg.table;
    validate_rows(input, "input", /*interleaved=*/false);
    validate_rows(output, "output", /*interleaved=*/true);
    auto* mesh_device = input.device();
    TT_FATAL(
        output.device() == mesh_device && table.device() == mesh_device,
        "sparse KV push tensors must be on one mesh device");
    const auto mesh_shape = mesh_device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    TT_FATAL(mesh_shape.dims() == 2, "the sparse KV push needs a 2D device mesh, got {}", mesh_shape);
    TT_FATAL(
        num_devices <= ::sparse_push::plan_table::kMaxDevices,
        "the plan table holds at most {} owners, the mesh has {}",
        ::sparse_push::plan_table::kMaxDevices,
        num_devices);
    TT_FATAL(
        cfg.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY,
        "the sparse KV push needs a 2D torus fabric (FABRIC_2D_TORUS_XY), got {}",
        cfg.fabric_config);
    TT_FATAL(cfg.part <= 2, "sparse KV push part must be 0 (all), 1 (prefix) or 2 (remainder), got {}", cfg.part);
    TT_FATAL(
        cfg.num_links >= 1 && cfg.num_links <= 2, "the sparse KV push supports 1 or 2 links, got {}", cfg.num_links);
    TT_FATAL(
        cfg.core_grid.num_cores() >= cores_needed(cfg.num_links),
        "the sparse KV push needs {} cores on {} links, core_grid has {}",
        cores_needed(cfg.num_links),
        cfg.num_links,
        cfg.core_grid.num_cores());
    validate_stripe_chunk(cfg.stripe_chunk, cfg.max_rows);

    TT_FATAL(
        ttnn::operations::ccl::common::has_row_major_mesh_coordinates(input),
        "the sparse KV push requires row-major tensor mesh coordinates");
    TT_FATAL(
        ttnn::operations::ccl::common::tensor_dim_shard_factor(input, kRowDim) == num_devices,
        "the sparse KV push requires the input to be sharded over dim {} across all {} devices",
        kRowDim,
        num_devices);
    if (cfg.batch_index_tensor.has_value()) {
        validate_batch_index_tensor(*cfg.batch_index_tensor);
    }

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    TT_FATAL(
        output.dtype() == input.dtype() && output.tensor_spec().page_config() == input.tensor_spec().page_config(),
        "sparse KV push output dtype and page config must match the input");
    TT_FATAL(
        output_shape[0] == 1 && output_shape[3] == input_shape[3],
        "sparse KV push output must be [1, 1, rows, {}], got {}",
        input_shape[3],
        output_shape);
    TT_FATAL(
        output.buffer()->num_pages() == output_shape[2], "sparse KV push output needs one page per compact-buffer row");
    const uint32_t page_bytes = input.buffer()->aligned_page_size();
    TT_FATAL(
        output.buffer()->aligned_page_size() == page_bytes,
        "sparse KV push input and output row pages differ ({} vs {})",
        page_bytes,
        output.buffer()->aligned_page_size());
    // The exact need is sum_o round_up(count_o, unit) over the requester's per-owner counts, known only on device.
    const uint32_t unit = packet_shape(mesh_device, page_bytes).unit();
    TT_FATAL(
        output_shape[2] >= num_devices * unit,
        "sparse KV push output needs at least {} rows (a {}-row bank-lane unit per owner), got {}",
        num_devices * unit,
        unit,
        output_shape[2]);

    const auto& table_shape = table.padded_shape();
    const auto layout = SparsePushPlanLayout::make(cfg.max_rows, cfg.stripe_chunk);
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.buffer() != nullptr,
        "sparse KV push table must be allocated on device");
    TT_FATAL(
        table.dtype() == DataType::UINT32 && table.layout() == Layout::ROW_MAJOR &&
            table.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            table.memory_config().buffer_type() == BufferType::DRAM,
        "sparse KV push table must be a UINT32 ROW_MAJOR interleaved DRAM tensor");
    TT_FATAL(
        table_shape.rank() == 4 && table_shape[0] == 1 && table_shape[1] == 1 &&
            table_shape[2] == layout.pages * num_devices && table_shape[3] == SparsePushPlanLayout::kPageWords,
        "sparse KV push table must be [1, 1, {}, {}] ({} pages per device), got {}",
        layout.pages * num_devices,
        SparsePushPlanLayout::kPageWords,
        layout.pages,
        table_shape);

    validate_runtime(cfg);
}

void validate_runtime(const PushConfig& cfg) {
    TT_FATAL(
        ttnn::operations::ccl::common::has_row_major_mesh_coordinates(cfg.input),
        "the sparse KV push requires row-major tensor mesh coordinates");
    TT_FATAL(cfg.output.buffer() != nullptr, "sparse KV push output must be allocated on device");
    validate_semaphores(cfg);
    validate_batch_slot(cfg);
}

ttsl::hash::hash_t program_hash(const PushConfig& cfg) {
    // Routes are resolved from the fabric node ids, so they are part of the key.
    auto* mesh_device = cfg.input.device();
    std::vector<tt::tt_fabric::FabricNodeId> fabric_nodes;
    fabric_nodes.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        fabric_nodes.push_back(mesh_device->get_fabric_node_id(coord));
    }
    const auto batch_index_spec = cfg.batch_index_tensor.has_value()
                                      ? std::optional<tt::tt_metal::TensorSpec>(cfg.batch_index_tensor->tensor_spec())
                                      : std::nullopt;
    // The output's row count is a compile-time arg (the reader's write bound), so its whole spec is keyed.
    return ttsl::hash::hash_objects_with_default_seed(
        cfg.stripe_chunk,
        cfg.max_rows,
        cfg.part,
        cfg.num_links,
        cfg.core_grid,
        cfg.fabric_config,
        mesh_device->shape(),
        fabric_nodes,
        cfg.input.tensor_spec(),
        cfg.output.tensor_spec(),
        cfg.table.tensor_spec(),
        batch_index_spec);
}

PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes) {
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    TT_FATAL(
        page_bytes > 0 && page_bytes <= max_payload,
        "the sparse KV push needs a row page of 1 to {} bytes (the fabric's max payload), got {}",
        max_payload,
        page_bytes);
    return PacketShape{
        .pages_per_packet = max_payload / page_bytes,
        .banks = static_cast<uint32_t>(mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM))};
}

Kernels append(tt::tt_metal::ProgramDescriptor& desc, const PushConfig& cfg, const MeshCoordinate& coord) {
    auto* mesh_device = cfg.input.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.shape().mesh_size();
    const uint32_t my_rank = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
    const auto neighbors = torus_neighbors(mesh_view, coord);
    const auto layout = core_layout(cfg.core_grid, cfg.num_links);
    const auto mux = mux_config(mesh_device);
    const auto defines = kernel_defines(mesh_device);
    const uint32_t page_bytes = cfg.input.buffer()->aligned_page_size();
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
        {kScalar, kScalarCbBytes, 1, tt::DataFormat::RawUInt32},
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
        reader_compile_args(cfg, num_devices, shape),
        defines,
        tt::tt_metal::ReaderConfigDescriptor{});
    tt::tt_metal::KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(cfg.input.buffer());
    reader_common.push_back(cfg.table.buffer());
    if (cfg.batch_index_tensor.has_value()) {
        reader_common.push_back(cfg.batch_index_tensor->buffer());
    } else {
        reader_common.push_back(0u);
    }
    reader_common.push_back(cfg.batch_slot_num_layers);
    reader_common.push_back(cfg.batch_slot_layer_idx);
    reader_common.push_back(cfg.batch_index);
    reader.emplace_common_runtime_args(reader_common);

    auto writer = data_movement_kernel(
        kdir + "sparse_push_writer.cpp",
        layout.sender_range,
        writer_compile_args(cfg, num_devices, shape, layout, mux),
        defines,
        tt::tt_metal::WriterConfigDescriptor{});
    tt::tt_metal::KernelDescriptor::RTArgList writer_common;
    writer_common.push_back(cfg.output.buffer());
    writer_common.push_back(static_cast<uint32_t>(cfg.ready_semaphore->address()));       // smuggled-rta-ok: semaphore
    writer_common.push_back(static_cast<uint32_t>(cfg.data_valid_semaphore->address()));  // smuggled-rta-ok: same
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
            mesh_device->get_fabric_node_id(coord),
            mesh_device->get_fabric_node_id(neighbors[c % kDirections]),
            c / kDirections,
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

    const Kernels kernels{
        .reader = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()),
        .writer = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size() + 1)};
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(mux_kernel));
    return kernels;
}

void override_runtime_arguments(tt::tt_metal::Program& program, const Kernels& kernels, const PushConfig& cfg) {
    auto& reader = tt::tt_metal::GetCommonRuntimeArgs(program, kernels.reader);
    reader[0] = cfg.input.buffer()->address();
    reader[1] = cfg.table.buffer()->address();
    reader[2] = cfg.batch_index_tensor.has_value() ? cfg.batch_index_tensor->buffer()->address() : 0u;
    reader[3] = cfg.batch_slot_num_layers;
    reader[4] = cfg.batch_slot_layer_idx;
    reader[5] = cfg.batch_index;
    auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, kernels.writer);
    writer[0] = cfg.output.buffer()->address();
    writer[1] = cfg.ready_semaphore->address();
    writer[2] = cfg.data_valid_semaphore->address();
}

}  // namespace ttnn::operations::experimental::sparse_kv_push::sparse_push
