// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "persistent_moe_request_device_operation.hpp"

#include <variant>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "d2d_routes.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

namespace {

using tt::tt_metal::distributed::MeshSocket;
using tt::tt_metal::distributed::SocketEndpoint;

constexpr uint32_t kRequestHeaderBytes = 64;
constexpr uint32_t kRequestStopLayer = 0xFFFFFFFFu;
constexpr uint32_t kRequestNoPartner = 0xFFFFFFFFu;  // moe_request.cpp's kNoPartner
constexpr uint32_t kMinPartnerPackets = 4;

// This chip's side of the request socket: the core that sends the request, and the request's receivers.
struct RequestEndpoints {
    tt::tt_metal::CoreCoord core;
    std::vector<tt::tt_fabric::FabricNodeId> receivers;
};

std::optional<RequestEndpoints> request_endpoints(
    const PersistentMoeRequestParams& attributes, const ttnn::MeshCoordinate& coord) {
    std::optional<tt::tt_metal::CoreCoord> core;
    std::optional<tt::tt_metal::CoreCoord> receiver_core;
    std::vector<tt::tt_fabric::FabricNodeId> receivers;
    for (const auto& c : attributes.request_socket.get_config().socket_connection_config) {
        if (c.sender_core.device_coord != coord) {
            continue;
        }
        TT_FATAL(
            (!core.has_value() || *core == c.sender_core.core_coord) &&
                (!receiver_core.has_value() || *receiver_core == c.receiver_core.core_coord),
            "persistent_moe_request: {}'s request connections must share one sender core and one receiver core, "
            "so that the page is one multicast",
            coord);
        core = c.sender_core.core_coord;
        receiver_core = c.receiver_core.core_coord;
        receivers.push_back(
            attributes.request_socket.get_fabric_node_id(SocketEndpoint::RECEIVER, c.receiver_core.device_coord));
    }
    if (!core.has_value()) {
        return std::nullopt;
    }
    return RequestEndpoints{.core = *core, .receivers = std::move(receivers)};
}

// The request page on the socket: [64-byte header | the op's row]. The socket's page (its FIFO) is the op's: a
// several-row block's when the op serves blocks. A one-row request to such an op (the input one row, the page a
// block's) puts its row in row 0 of the block's (rows x 32) tiles -- 32 B, one face row, every 32 * block_rows B --
// and says so in the header (persistent.h kFeHdrRows): the op's other rows are never used. Its partial comes back
// compacted (the op's senders take row 0) into the front of the partials, its flags where a block's go.
// With persistent_moe_native_one_row the op runs such a request in one-row tiles instead, so the page carries the
// header and the plain row only (payload_bytes; the socket still counts its whole page).
struct RequestPage {
    uint32_t page_bytes;      // the socket page
    uint32_t page_row_bytes;  // the op's row: page_bytes less the header
    uint32_t block_rows;      // the op's rows per input row: 1, or the block's for a one-row request
    bool native;              // a one-row request to a block op, sent as its plain row
    uint32_t payload_bytes;   // the bytes sent: the page, or a native request's header and row
};

RequestPage request_page(const PersistentMoeRequestParams& attributes, const Tensor& input) {
    const uint32_t row_bytes = input.buffer()->page_size();
    const uint32_t page_bytes = attributes.request_socket.get_config().socket_mem_config.fifo_size;
    TT_FATAL(
        page_bytes > kRequestHeaderBytes && (page_bytes - kRequestHeaderBytes) % row_bytes == 0,
        "persistent_moe_request: the socket's {} B page is not the header and a whole number of {} B input rows",
        page_bytes,
        row_bytes);
    const uint32_t page_row_bytes = page_bytes - kRequestHeaderBytes;
    const uint32_t block_rows = page_row_bytes / row_bytes;
    const bool native = block_rows > 1u && persistent_moe_native_one_row();
    return RequestPage{
        .page_bytes = page_bytes,
        .page_row_bytes = page_row_bytes,
        .block_rows = block_rows,
        .native = native,
        .payload_bytes = native ? kRequestHeaderBytes + row_bytes : page_bytes};
}

// The request's second sender, on fabric link 1: a core next to the socket's sender core, when the page spans
// kMinPartnerPackets packets or more and the route's first hop has a second link. It sends the page's back part.
// A link carries about a packet per 0.84 us; the partner starts about 0.7 us behind the main core (it waits for
// the page's place), so a several-row block's 41 KB page gains ~3.5 us, while a one-row 10 KB page (3 packets) loses.
struct RequestPartner {
    tt::tt_metal::CoreCoord core;
    uint32_t link0;
    uint32_t link1;
    uint32_t split;  // the page bytes the main core sends; the partner sends the rest
};

std::optional<RequestPartner> request_partner(
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& next,
    const tt::tt_metal::CoreCoord& core,
    uint32_t page_bytes) {
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    const uint32_t packets = (page_bytes + max_payload - 1) / max_payload;
    const auto links = tt::tt_fabric::get_forwarding_link_indices(mesh, src, next);
    const auto grid = mesh.compute_with_storage_grid_size();
    if (packets < kMinPartnerPackets || links.size() < 2 || grid.x < 2) {
        return std::nullopt;
    }
    const tt::tt_metal::CoreCoord partner =
        core.x + 1 < grid.x ? tt::tt_metal::CoreCoord(core.x + 1, core.y) : tt::tt_metal::CoreCoord(core.x - 1, core.y);
    return RequestPartner{
        .core = partner, .link0 = links[0], .link1 = links[1], .split = ((packets + 1) / 2) * max_payload};
}

// The partials buffer's layout (see PersistentMoeRequestParams): its sum cores, one per part of the row, the
// part's bytes, and the slots (the persistent op's chips) per shard.
struct PartialsLayout {
    std::vector<tt::tt_metal::CoreCoord> sum_cores;
    uint32_t part_bytes;
    uint32_t num_slots;
    // How far past this request's slots the flags sit (a one-row request to a block op: the block's slots' end).
    uint32_t flags_shift = 0;
};

// The sum views a part as tiles of this shape; the sum is elementwise, so any whole-tile view does. 4 rows: a
// stream-collapse core's 640 columns are 5 of them.
constexpr uint32_t kSumTileH = 4;
constexpr uint32_t kSumTileW = 32;
constexpr uint32_t kSumTileBytes = kSumTileH * kSumTileW * 2;
// The HC post kernel views each part (a stream-collapse core's columns of one row) as ONE 32x32 bf16 tile through
// CBs whose pages are part_bytes apart (unpack and pack address tile i at i * page size): every step is elementwise,
// so the tile's elements past the part are don't-cares, and the FPU runs on whole tiles, as tt-blaze's BcastMulAdd
// does with its [T, 32] views. post and comb are scalars at element 0 of pages kHcScalarPage apart.
constexpr uint32_t kHcTileBytes = 32 * 32 * 2;
constexpr uint32_t kHcScalarPage = 32;
// Stream counts the HC post kernel reads post and comb for: their (h, g) elements must lie in face 0's first 4 rows.
constexpr uint32_t kMaxHcStreams = 4;
// A several-row block's HC post step: its per-row post and comb values are column 0 of 32x32 tiles (faces 0 and 2)
// kHcColumnPage apart, an odd multiple of 16 B past 1 KB, so one tile's face 2 column and the next one's face 0 column
// interleave (see dataflow/moe_partials_hcpost.cpp).
constexpr uint32_t kHcColumnPage = 1024 + 16;
// The tile rows (block rows) a request carries, and so the HC post step's streams' tile height.
constexpr uint32_t kMaxBlockRows = 4;

// The HC post step's views of a several-row part (see compute/moe_partials_hcpost.cpp): 32x32 views page_bytes apart,
// part_pages pages per part.
struct BlockViews {
    uint32_t views;
    uint32_t page_bytes;
    uint32_t part_pages;
    uint32_t tile_pages;
};

BlockViews block_views(uint32_t part_bytes) {
    uint32_t page = kHcTileBytes;
    while (part_bytes % page != 0) {
        page /= 2;
    }
    return BlockViews{
        .views = (part_bytes + kHcTileBytes - 1) / kHcTileBytes,
        .page_bytes = page,
        .part_pages = part_bytes / page,
        .tile_pages = kHcTileBytes / page};
}

bool hc_post(const PersistentMoeRequestInputs& tensors) { return tensors.streams.has_value(); }

// The block rows of an HC post request: the input row's elements over the streams' width.
uint32_t hc_rows(const PersistentMoeRequestInputs& tensors) {
    const auto width = static_cast<uint32_t>(tensors.streams->logical_shape()[-1]);
    const uint32_t row_elems = tensors.input.buffer()->page_size() / 2u;
    return width > 0 && row_elems % width == 0 ? row_elems / width : 0u;
}

// The HC post request's stream count.
uint32_t hc_streams(const PersistentMoeRequestInputs& tensors) {
    const uint32_t rows = hc_rows(tensors);
    return rows == 0 ? 0u : static_cast<uint32_t>(tensors.streams->logical_shape()[2]) / rows;
}

PartialsLayout partials_layout(
    const PersistentMoeRequestParams& attributes, uint32_t row_bytes, uint32_t block_rows = 1) {
    const auto& partials = attributes.partials;
    const auto& shard = partials.shard_spec().value();
    PartialsLayout layout{
        .sum_cores = corerange_to_cores(shard.grid, std::nullopt, /*row_wise=*/true), .part_bytes = 0, .num_slots = 0};
    const auto parts = static_cast<uint32_t>(layout.sum_cores.size());
    TT_FATAL(
        parts > 0 && row_bytes % parts == 0 && (row_bytes / parts) % kSumTileBytes == 0,
        "persistent_moe_request: a {} B row does not split into {} parts of whole {} B tiles",
        row_bytes,
        parts,
        kSumTileBytes);
    layout.part_bytes = row_bytes / parts;
    // num_slots slots of the op's part (a block's: block_rows of this request's), then a flag word per slot; a one-row
    // request's slots are compacted at the front, its flags where the block's go.
    layout.num_slots =
        static_cast<uint32_t>(partials.buffer()->aligned_size_per_bank()) / (layout.part_bytes * block_rows + 4u);
    layout.flags_shift = layout.num_slots * layout.part_bytes * (block_rows - 1u);
    TT_FATAL(
        layout.num_slots > 0,
        "persistent_moe_request: the partials shards hold no slot of a {} B part",
        layout.part_bytes);
    return layout;
}

}  // namespace

namespace {

// The HC post inputs (see PersistentMoeRequestInputs).
void validate_hc_post(const PersistentMoeRequestParams& attributes, const PersistentMoeRequestInputs& tensors) {
    TT_FATAL(
        tensors.post.has_value() && tensors.comb.has_value(),
        "persistent_moe_request: streams, post and comb come together");
    const auto& streams = *tensors.streams;
    const auto& post = *tensors.post;
    const auto& comb = *tensors.comb;
    auto* mesh = tensors.input.device();
    const uint32_t row_elems = tensors.input.buffer()->page_size() / 2u;
    const auto& shape = streams.logical_shape();
    const uint32_t rows = hc_rows(tensors);
    TT_FATAL(
        rows >= 1 && rows <= kMaxBlockRows && (rows & (rows - 1)) == 0,
        "persistent_moe_request: the {}-element input row is not 1, 2 or 4 rows of the streams' width {}",
        row_elems,
        shape[-1]);
    const uint32_t width = row_elems / rows;
    const uint32_t hc = hc_streams(tensors);
    // One row: [1, 1, hc, D] ROW_MAJOR. A block of rows: [1, 1, hc * rows, D] in (rows x 32) tiles, row h * rows + j
    // stream h's block row j, so each shard holds a stream's part after another, in the partials' byte order.
    const bool layout_ok = rows == 1 ? streams.layout() == Layout::ROW_MAJOR
                                     : streams.layout() == Layout::TILE &&
                                           streams.tensor_spec().tile().get_height() == rows &&
                                           streams.tensor_spec().tile().get_width() == 32;
    TT_FATAL(
        streams.storage_type() == StorageType::DEVICE && streams.device() == mesh &&
            streams.dtype() == DataType::BFLOAT16 && layout_ok && shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 &&
            hc >= 1 && hc <= kMaxHcStreams && hc * rows == static_cast<uint32_t>(shape[2]) &&
            static_cast<uint32_t>(shape[3]) == width,
        "persistent_moe_request: streams must be [1, 1, hc <= {}, {}] bfloat16 ROW_MAJOR (one row) or [1, 1, hc * {}, "
        "{}] bfloat16 TILE with ({}, 32) tiles (a block of {} rows) on the input's mesh, got {} {} {}",
        kMaxHcStreams,
        width,
        rows,
        width,
        rows,
        rows,
        shape,
        streams.layout(),
        streams.dtype());
    TT_FATAL(
        streams.memory_config().buffer_type() == BufferType::L1 &&
            streams.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "persistent_moe_request: streams must be L1 WIDTH_SHARDED");
    const auto& shard = streams.shard_spec().value();
    const auto stream_cores =
        corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
    const auto partials_cores =
        corerange_to_cores(attributes.partials.shard_spec().value().grid, std::nullopt, /*row_wise=*/true);
    TT_FATAL(
        stream_cores == partials_cores && shard.shape[0] == hc * rows && shard.shape[1] * stream_cores.size() == width,
        "persistent_moe_request: the streams' [{}, {}] shards must sit on the partials' {} cores, in the same order",
        shard.shape[0],
        shard.shape[1],
        partials_cores.size());
    for (const auto* t : {&post, &comb}) {
        TT_FATAL(
            t->storage_type() == StorageType::DEVICE && t->device() == mesh && t->dtype() == DataType::BFLOAT16 &&
                t->layout() == Layout::TILE && t->tensor_spec().tile().get_height() == 32 &&
                t->tensor_spec().tile().get_width() == 32 &&
                t->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "persistent_moe_request: post and comb must be bfloat16 32x32 TILE interleaved tensors on the input's "
            "mesh");
    }
    TT_FATAL(
        post.logical_shape() == ttnn::Shape({1, rows, hc, 1}) && comb.logical_shape() == ttnn::Shape({1, rows, hc, hc}),
        "persistent_moe_request: post must be [1, {}, {}, 1] and comb [1, {}, {}, {}] (a tile per row), got {} and {}",
        rows,
        hc,
        rows,
        hc,
        hc,
        post.logical_shape(),
        comb.logical_shape());
}

}  // namespace

void PersistentMoeRequestDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    const auto& input = tensors.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "persistent_moe_request: the input must be on device");
    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR && input.dtype() == DataType::BFLOAT16 &&
            input.logical_shape().volume() == input.logical_shape()[-1] && input.buffer()->num_pages() == 1,
        "persistent_moe_request: the input must be one ROW_MAJOR bfloat16 row, got {} {} {}",
        input.logical_shape(),
        input.layout(),
        input.dtype());
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "persistent_moe_request: the input must be interleaved");
    auto* mesh = input.device();
    TT_FATAL(
        attributes.request_socket.get_socket_endpoint_type() == SocketEndpoint::SENDER &&
            attributes.request_socket.get_mesh_device() == mesh,
        "persistent_moe_request: it sends on request_socket, whose sending end must be on the input's mesh");
    const auto& partials = attributes.partials;
    TT_FATAL(
        partials.storage_type() == StorageType::DEVICE && partials.device() == mesh &&
            partials.memory_config().buffer_type() == BufferType::L1 &&
            partials.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "persistent_moe_request: partials must be an L1 HEIGHT_SHARDED tensor on the input's mesh");
    partials_layout(attributes, input.buffer()->page_size(), request_page(attributes, input).block_rows);
    if (hc_post(tensors)) {
        validate_hc_post(attributes, tensors);
    } else {
        TT_FATAL(
            !tensors.post.has_value() && !tensors.comb.has_value(),
            "persistent_moe_request: streams, post and comb come together");
    }
}

PersistentMoeRequestDeviceOperation::spec_return_value_t PersistentMoeRequestDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensors) {
    return hc_post(tensors) ? tensors.streams->tensor_spec() : tensors.input.tensor_spec();
}

PersistentMoeRequestDeviceOperation::tensor_return_value_t PersistentMoeRequestDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(attributes, tensors), tensors.input.device());
}

ttsl::hash::hash_t PersistentMoeRequestDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    return tt::tt_metal::operation::hash_operation<PersistentMoeRequestDeviceOperation>(
        attributes.request_socket,
        attributes.request_socket.get_config_buffer_address(),
        attributes.partials.buffer()->address(),
        attributes.partials.memory_config(),
        tensors.input,
        tensors.streams,
        tensors.post,
        tensors.comb);
}

tt::tt_metal::ProgramDescriptor PersistentMoeRequestDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensors,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;
    const auto& input = tensors.input;
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "persistent_moe_request: built per mesh coordinate");
    const auto& coord = *mesh_dispatch_coordinate;
    // Every chip sums the partials it receives; only a chip that sends on the request socket sends the page.
    const auto endpoints = request_endpoints(attributes, coord);
    const uint32_t row_bytes = input.buffer()->page_size();
    const bool stop = attributes.layer == kRequestStopLayer;
    ProgramDescriptor desc;
    if (endpoints.has_value()) {
        auto* mesh = input.device();
        const auto src = mesh->get_fabric_node_id(coord);
        const auto routes = d2d_routes(src, endpoints->receivers);
        const auto& core = endpoints->core;
        const auto page = request_page(attributes, input);
        const uint32_t page_bytes = page.page_bytes;
        // A padded one-row request to a block op scatters its row over the page, so the main core sends all of it.
        const auto partner = page.block_rows == 1u || page.native
                                 ? request_partner(*mesh, src, routes.front().next, core, page.payload_bytes)
                                 : std::nullopt;
        CoreRangeSet senders(CoreRange(core, core));
        if (partner.has_value()) {
            senders = senders.merge(CoreRangeSet(CoreRange(partner->core, partner->core)));
            // The partner's page place, the receivers' bytes_sent address and their core, from the main core.
            for (uint32_t id = 0; id < 3; ++id) {
                desc.semaphores.push_back(SemaphoreDescriptor{.id = id, .core_ranges = senders, .initial_value = 0});
            }
        }

        constexpr uint32_t cb_page = tt::CBIndex::c_0;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_bytes,
            .core_ranges = senders,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_page,
                .data_format = tt::DataFormat::UInt32,
                .page_size = page_bytes,
            }}},
        });

        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(attributes.request_socket.get_config_buffer_address()),
            row_bytes,
            static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()),
            cb_page,
            page.page_row_bytes,
            page.block_rows,
            page.native ? 1u : 0u,
        };
        TensorAccessorArgs(*input.buffer()).append_to(ct_args);
        KernelDescriptor kernel;
        kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/moe_request.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = senders;
        kernel.compile_time_args = std::move(ct_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        tt::tt_metal::KernelHandle kernel_id = desc.kernels.size();
        desc.kernels.push_back(std::move(kernel));

        TT_FATAL(
            routes.size() <= 16,
            "persistent_moe_request: {} sends to {} chips, more than the kernel's 16",
            coord,
            routes.size());
        // One sender's args: its part of the page, the partner's NoC coordinates (main core), the routes, and the
        // fabric link it leaves by.
        auto sender_args = [&](const CoreCoord& at, uint32_t start, uint32_t bytes, uint32_t link, bool main) {
            std::vector<uint32_t> args = {
                attributes.layer, static_cast<uint32_t>(input.buffer()->address()), start, bytes};
            if (main && partner.has_value()) {
                const auto noc = mesh->worker_core_from_logical_core(partner->core);
                args.push_back(noc.x);
                args.push_back(noc.y);
            } else {
                args.push_back(kRequestNoPartner);
                args.push_back(kRequestNoPartner);
            }
            args.push_back(static_cast<uint32_t>(routes.size()));
            for (const auto& route : routes) {
                append_d2d_route_args(route, args);
            }
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *mesh,
                src,
                {routes.front().next},
                {link},
                desc,
                kernel_id,
                at,
                args,
                tt::tt_fabric::FabricApiType::Mesh);
            desc.kernels[kernel_id].runtime_args.emplace_back(
                at, KernelDescriptor::CoreRuntimeArgs(args.begin(), args.end()));
        };
        if (partner.has_value()) {
            sender_args(core, 0, partner->split, partner->link0, true);
            sender_args(partner->core, partner->split, page.payload_bytes - partner->split, partner->link1, false);
        } else {
            sender_args(core, 0, page.payload_bytes, 0u, true);
        }
    }

    // The sum of each part of the row, on its sum core: the slots are a CB over the partials shard, which the
    // persistent op's chips fill.
    const auto layout = partials_layout(attributes, row_bytes, request_page(attributes, input).block_rows);
    // A one-row request to a block op: the sums' flags past the block's slots (the kernels find them past their own).
    const std::variant<uint32_t, Buffer*> partials_arg =
        layout.flags_shift == 0u ? std::variant<uint32_t, Buffer*>(attributes.partials.buffer())
                                 : std::variant<uint32_t, Buffer*>(static_cast<uint32_t>(
                                       attributes.partials.buffer()->address() + layout.flags_shift));
    // The plain sum's view: the tallest tile (up to 32x32) that a part is a whole number of -- per-tile overhead is
    // what the sum costs, so a several-row block's part (20 KB) goes in 32x32 tiles rather than 80 4x32 ones.
    uint32_t sum_tile_h = kSumTileH;
    if (!hc_post(tensors)) {
        for (uint32_t h : {32u, 16u, 8u}) {
            if (layout.part_bytes % (h * kSumTileW * 2u) == 0u) {
                sum_tile_h = h;
                break;
            }
        }
    }
    const uint32_t sum_tile_bytes = sum_tile_h * kSumTileW * 2u;
    const uint32_t tiles_per_part = layout.part_bytes / sum_tile_bytes;
    CoreRangeSet sum_set;
    for (const auto& c : layout.sum_cores) {
        sum_set = sum_set.merge(CoreRangeSet(CoreRange(c, c)));
    }
    const TileDescriptor sum_tile(tt::tt_metal::Tile({sum_tile_h, kSumTileW}));
    auto add_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format, Buffer* buffer = nullptr) {
        const uint32_t page = format == tt::DataFormat::Float32 ? 2u * sum_tile_bytes : sum_tile_bytes;
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * page,
            .core_ranges = sum_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index,
                .data_format = format,
                .page_size = page,
                .tile = sum_tile,
            }}},
            .buffer = buffer,
        });
    };
    constexpr uint32_t cb_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_2;
    const TileDescriptor hc_tile(tt::tt_metal::Tile({32, 32}));
    // A CB of `pages` pages of `page` bytes holding 32x32 bf16 tiles (see kHcTileBytes).
    auto add_view_cb = [&](uint32_t index, uint32_t pages, uint32_t page, Buffer* buffer = nullptr) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = pages * page,
            .core_ranges = sum_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index,
                .data_format = tt::DataFormat::Float16_b,
                .page_size = page,
                .tile = hc_tile,
            }}},
            .buffer = buffer,
        });
    };
    const uint32_t rows = hc_post(tensors) ? hc_rows(tensors) : 1u;
    const BlockViews views = block_views(layout.part_bytes);
    if (hc_post(tensors) && rows > 1) {
        TT_FATAL(
            views.page_bytes % 32 == 0 && views.views <= 8,
            "persistent_moe_request: a {} B part does not split into the HC post step's 32x32 views",
            layout.part_bytes);
        add_view_cb(cb_in, layout.num_slots * views.part_pages, views.page_bytes, attributes.partials.buffer());
    } else if (hc_post(tensors)) {
        TT_FATAL(
            layout.part_bytes <= kHcTileBytes && layout.part_bytes % 32 == 0,
            "persistent_moe_request: a {} B part does not fit the HC post step's one-tile view",
            layout.part_bytes);
        add_view_cb(cb_in, layout.num_slots, layout.part_bytes, attributes.partials.buffer());
    } else {
        add_cb(cb_in, layout.num_slots * tiles_per_part, tt::DataFormat::Float16_b, attributes.partials.buffer());
    }

    KernelDescriptor sum_reader;
    sum_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sum_reader.core_ranges = sum_set;
    // RISCV_1: the request core may be a sum core, and the request kernel is RISCV_0.
    sum_reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    KernelDescriptor sum_compute;
    sum_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sum_compute.core_ranges = sum_set;
    const uint32_t skip = stop ? 1u : 0u;
    if (hc_post(tensors)) {
        // The new streams on each stream-collapse core (see dataflow/moe_partials_hcpost.cpp).
        const uint32_t hc = hc_streams(tensors);
        constexpr uint32_t cb_s = tt::CBIndex::c_3;
        constexpr uint32_t cb_post = tt::CBIndex::c_4;
        constexpr uint32_t cb_comb = tt::CBIndex::c_5;
        constexpr uint32_t cb_mix = tt::CBIndex::c_6;
        constexpr uint32_t cb_x = tt::CBIndex::c_7;
        constexpr uint32_t cb_tmp = tt::CBIndex::c_8;
        constexpr uint32_t cb_prod = tt::CBIndex::c_9;
        uint32_t scalar_page = kHcScalarPage;
        if (rows > 1) {
            // Views read or packed past a part's end: one spare view of slack. A coefficient tile's face 2 runs
            // one page and a half past its start.
            add_view_cb(cb_out, hc * views.part_pages + views.tile_pages, views.page_bytes);
            add_view_cb(cb_s, hc * views.part_pages + views.tile_pages, views.page_bytes);
            scalar_page = kHcColumnPage;
            add_view_cb(cb_post, hc + 1, kHcColumnPage);
            add_view_cb(cb_comb, hc * hc + 1, kHcColumnPage);
            add_view_cb(cb_mix, hc * views.views, kHcTileBytes);
            add_view_cb(cb_x, views.views, kHcTileBytes);
            add_cb(cb_tmp, rows, tt::DataFormat::Float16_b);  // each row's post and comb face-0 heads, 128 B each
        } else {
            // Tiles read or packed through a part_bytes view run past the last page: one spare tile of slack.
            const uint32_t slack_pages = (kHcTileBytes + layout.part_bytes - 1) / layout.part_bytes;
            add_view_cb(cb_out, hc + slack_pages, layout.part_bytes);
            add_view_cb(cb_s, hc + slack_pages, layout.part_bytes);
            const uint32_t scalar_slack = kHcTileBytes / kHcScalarPage;
            add_view_cb(cb_post, hc + scalar_slack, kHcScalarPage);
            add_view_cb(cb_comb, hc * hc + scalar_slack, kHcScalarPage);
            add_view_cb(cb_mix, hc, kHcTileBytes);
            add_view_cb(cb_x, 1, kHcTileBytes);
            add_view_cb(cb_prod, hc, kHcTileBytes);
            add_cb(cb_tmp, 1, tt::DataFormat::Float16_b);  // post's and comb's face-0 heads, 128 B each
        }

        std::vector<uint32_t> reader_ct_args = {
            layout.num_slots,
            1,  // tiles per part
            hc,
            layout.part_bytes,
            cb_in,
            cb_s,
            cb_post,
            cb_comb,
            cb_out,
            cb_tmp,
            scalar_page,
            scalar_page,
            rows,
            views.part_pages};
        TensorAccessorArgs(*tensors.post->buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*tensors.comb->buffer()).append_to(reader_ct_args);
        sum_reader.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/"
            "moe_partials_hcpost.cpp";
        sum_reader.compile_time_args = std::move(reader_ct_args);
        sum_compute.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/compute/"
            "moe_partials_hcpost.cpp";
        sum_compute.compile_time_args = {
            layout.num_slots,
            hc,
            cb_in,
            cb_s,
            cb_post,
            cb_comb,
            cb_mix,
            cb_x,
            cb_out,
            cb_prod,
            rows,
            rows > 1 ? views.views : 1u,
            rows > 1 ? views.part_pages : 1u,
            rows > 1 ? views.tile_pages : 1u};
        sum_compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false};
        for (const auto& c : layout.sum_cores) {
            sum_reader.emplace_runtime_args(
                c,
                {skip,
                 partials_arg,
                 tensors.post->buffer(),
                 tensors.comb->buffer(),
                 tensors.streams->buffer(),
                 output.buffer()});
            sum_compute.emplace_runtime_args(c, {skip});
        }
    } else {
        add_cb(cb_out, tiles_per_part, tt::DataFormat::Float16_b);
        std::vector<uint32_t> sum_ct_args = {layout.num_slots, layout.part_bytes, cb_in, cb_out, tiles_per_part};
        TensorAccessorArgs(*output.buffer()).append_to(sum_ct_args);
        sum_reader.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/"
            "moe_partials_sum.cpp";
        sum_reader.compile_time_args = std::move(sum_ct_args);
        sum_compute.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/compute/"
            "moe_partials_sum.cpp";
        sum_compute.compile_time_args = {layout.num_slots, tiles_per_part, cb_in, cb_out};
        sum_compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false};
        for (uint32_t part = 0; part < layout.sum_cores.size(); ++part) {
            const auto& c = layout.sum_cores[part];
            sum_reader.emplace_runtime_args(c, {skip, partials_arg, output.buffer(), part});
            sum_compute.emplace_runtime_args(c, {skip});
        }
    }
    desc.kernels.push_back(std::move(sum_reader));
    desc.kernels.push_back(std::move(sum_compute));
    return desc;
}

void PersistentMoeRequestDeviceOperation::MultiCore::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensors,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "persistent_moe_request: built per mesh coordinate");
    const auto& input = tensors.input;
    const auto endpoints = request_endpoints(attributes, *mesh_dispatch_coordinate);
    const uint32_t skip = attributes.layer == kRequestStopLayer ? 1u : 0u;
    // Kernel 0 is the request on a sending chip; the sums' data movement and compute follow, on every sum core.
    uint32_t sum_kernel = 0;
    if (endpoints.has_value()) {
        auto* mesh = input.device();
        const auto src = mesh->get_fabric_node_id(*mesh_dispatch_coordinate);
        const auto routes = d2d_routes(src, endpoints->receivers);
        const auto page = request_page(attributes, input);
        const auto partner = page.block_rows == 1u || page.native
                                 ? request_partner(*mesh, src, routes.front().next, endpoints->core, page.payload_bytes)
                                 : std::nullopt;
        std::vector<tt::tt_metal::CoreCoord> senders = {endpoints->core};
        if (partner.has_value()) {
            senders.push_back(partner->core);
        }
        for (const auto& c : senders) {
            auto& args = tt::tt_metal::GetRuntimeArgs(program, 0, c);
            args[0] = attributes.layer;
            args[1] = input.buffer()->address();
        }
        sum_kernel = 1;
    }
    const auto layout =
        partials_layout(attributes, input.buffer()->page_size(), request_page(attributes, input).block_rows);
    for (const auto& c : layout.sum_cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, sum_kernel, c);
        reader_args[0] = skip;
        if (hc_post(tensors)) {
            reader_args[2] = tensors.post->buffer()->address();
            reader_args[3] = tensors.comb->buffer()->address();
            reader_args[4] = tensors.streams->buffer()->address();
            reader_args[5] = output.buffer()->address();
        } else {
            reader_args[2] = output.buffer()->address();
        }
        tt::tt_metal::GetRuntimeArgs(program, sum_kernel + 1, c)[0] = skip;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {

Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const Tensor& partials,
    const std::optional<Tensor>& streams,
    const std::optional<Tensor>& post,
    const std::optional<Tensor>& comb) {
    using OperationType =
        ttnn::operations::experimental::deepseek::moe::fused_experts::PersistentMoeRequestDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t(layer, request_socket, partials),
        OperationType::tensor_args_t{input_tensor, streams, post, comb});
}

}  // namespace ttnn::prim
