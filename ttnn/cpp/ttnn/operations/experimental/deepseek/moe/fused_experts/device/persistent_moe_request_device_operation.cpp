// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "persistent_moe_request_device_operation.hpp"

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

// The partials buffer's layout (see PersistentMoeRequestParams): its sum cores, one per part of the row, the
// part's bytes, and the slots (the persistent op's chips) per shard.
struct PartialsLayout {
    std::vector<tt::tt_metal::CoreCoord> sum_cores;
    uint32_t part_bytes;
    uint32_t num_slots;
};

// The sum views a part as tiles of this shape; the sum is elementwise, so any whole-tile view does. 4 rows: a
// stream-collapse core's 640 columns are 5 of them.
constexpr uint32_t kSumTileH = 4;
constexpr uint32_t kSumTileW = 32;
constexpr uint32_t kSumTileBytes = kSumTileH * kSumTileW * 2;
constexpr uint32_t kSumTileBytesFp32 = kSumTileH * kSumTileW * 4;
// Stream counts the HC post kernel reads post and comb for: their (h, g) elements must lie in face 0's first 4 rows.
constexpr uint32_t kMaxHcStreams = 4;

bool hc_post(const PersistentMoeRequestInputs& tensors) { return tensors.streams.has_value(); }

PartialsLayout partials_layout(const PersistentMoeRequestParams& attributes, uint32_t row_bytes) {
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
    // num_slots slots of a part, then a flag word per slot.
    layout.num_slots = static_cast<uint32_t>(partials.buffer()->aligned_size_per_bank()) / (layout.part_bytes + 4u);
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
    TT_FATAL(
        streams.storage_type() == StorageType::DEVICE && streams.device() == mesh &&
            streams.dtype() == DataType::BFLOAT16 && streams.layout() == Layout::ROW_MAJOR && shape.rank() == 4 &&
            shape[0] == 1 && shape[1] == 1 && shape[2] >= 1 && static_cast<uint32_t>(shape[2]) <= kMaxHcStreams &&
            static_cast<uint32_t>(shape[3]) == row_elems,
        "persistent_moe_request: streams must be [1, 1, hc <= {}, {}] bfloat16 ROW_MAJOR on the input's mesh, got {} "
        "{} {}",
        kMaxHcStreams,
        row_elems,
        shape,
        streams.layout(),
        streams.dtype());
    const uint32_t hc = shape[2];
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
        stream_cores == partials_cores && shard.shape[0] == hc && shard.shape[1] * stream_cores.size() == row_elems,
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
        post.logical_shape() == ttnn::Shape({1, 1, hc, 1}) && comb.logical_shape() == ttnn::Shape({1, 1, hc, hc}),
        "persistent_moe_request: post must be [1, 1, {}, 1] and comb [1, 1, {}, {}], got {} and {}",
        hc,
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
    partials_layout(attributes, input.buffer()->page_size());
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
        const uint32_t page_bytes = kRequestHeaderBytes + row_bytes;

        constexpr uint32_t cb_page = tt::CBIndex::c_0;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_bytes,
            .core_ranges = CoreRangeSet(CoreRange(core, core)),
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
        };
        TensorAccessorArgs(*input.buffer()).append_to(ct_args);
        KernelDescriptor kernel;
        kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/moe_request.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = CoreRangeSet(CoreRange(core, core));
        kernel.compile_time_args = std::move(ct_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        tt::tt_metal::KernelHandle kernel_id = desc.kernels.size();
        desc.kernels.push_back(std::move(kernel));

        std::vector<uint32_t> args = {attributes.layer, static_cast<uint32_t>(input.buffer()->address())};
        TT_FATAL(
            routes.size() <= 16,
            "persistent_moe_request: {} sends to {} chips, more than the kernel's 16",
            coord,
            routes.size());
        args.push_back(static_cast<uint32_t>(routes.size()));
        for (const auto& route : routes) {
            append_d2d_route_args(route, args);
        }
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            *mesh, src, {routes.front().next}, {0u}, desc, kernel_id, core, args, tt::tt_fabric::FabricApiType::Mesh);
        desc.kernels[kernel_id].runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs(args.begin(), args.end()));
    }

    // The sum of each part of the row, on its sum core: the slots are a CB over the partials shard, which the
    // persistent op's chips fill.
    const auto layout = partials_layout(attributes, row_bytes);
    const uint32_t tiles_per_part = layout.part_bytes / kSumTileBytes;
    CoreRangeSet sum_set;
    for (const auto& c : layout.sum_cores) {
        sum_set = sum_set.merge(CoreRangeSet(CoreRange(c, c)));
    }
    const TileDescriptor sum_tile(tt::tt_metal::Tile({kSumTileH, kSumTileW}));
    auto add_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format, Buffer* buffer = nullptr) {
        const uint32_t page = format == tt::DataFormat::Float32 ? kSumTileBytesFp32 : kSumTileBytes;
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
    add_cb(cb_in, layout.num_slots * tiles_per_part, tt::DataFormat::Float16_b, attributes.partials.buffer());

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
        const auto hc = static_cast<uint32_t>(tensors.streams->logical_shape()[2]);
        const uint32_t row_tiles = hc * tiles_per_part;
        constexpr uint32_t cb_s = tt::CBIndex::c_3;
        constexpr uint32_t cb_post = tt::CBIndex::c_4;
        constexpr uint32_t cb_comb = tt::CBIndex::c_5;
        constexpr uint32_t cb_mix = tt::CBIndex::c_6;
        constexpr uint32_t cb_x = tt::CBIndex::c_7;
        constexpr uint32_t cb_tmp = tt::CBIndex::c_8;
        add_cb(cb_out, row_tiles, tt::DataFormat::Float16_b);
        add_cb(cb_s, row_tiles, tt::DataFormat::Float16_b);
        add_cb(cb_post, hc, tt::DataFormat::Float32);
        add_cb(cb_comb, hc * hc, tt::DataFormat::Float16_b);
        add_cb(cb_mix, row_tiles, tt::DataFormat::Float32);
        add_cb(cb_x, tiles_per_part, tt::DataFormat::Float32);
        add_cb(cb_tmp, 1, tt::DataFormat::Float16_b);  // post's and comb's face-0 heads, 128 B each

        std::vector<uint32_t> reader_ct_args = {
            layout.num_slots,
            tiles_per_part,
            hc,
            layout.part_bytes,
            cb_in,
            cb_s,
            cb_post,
            cb_comb,
            cb_out,
            cb_tmp,
            kSumTileBytes,
            kSumTileBytesFp32};
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
            layout.num_slots, tiles_per_part, hc, cb_in, cb_s, cb_post, cb_comb, cb_mix, cb_x, cb_out};
        sum_compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};
        for (const auto& c : layout.sum_cores) {
            sum_reader.emplace_runtime_args(
                c,
                {skip,
                 attributes.partials.buffer(),
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
            sum_reader.emplace_runtime_args(c, {skip, attributes.partials.buffer(), output.buffer(), part});
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
        auto& args = tt::tt_metal::GetRuntimeArgs(program, 0, endpoints->core);
        args[0] = attributes.layer;
        args[1] = input.buffer()->address();
        sum_kernel = 1;
    }
    const auto layout = partials_layout(attributes, input.buffer()->page_size());
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
