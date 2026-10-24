// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/mpi_socket.hpp"

#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ttnn::distributed {

namespace {

constexpr uint64_t kDescriptorMagic = 0x54544D504954454EULL;  // "TTMPITEN"
constexpr uint64_t kDescriptorVersion = 1;
constexpr size_t kMaxDescriptorBytes = 1024 * 1024;

constexpr auto kPayloadTag = tt::tt_metal::distributed::multihost::Tag{0};
constexpr auto kPreambleTag = tt::tt_metal::distributed::multihost::Tag{1};
constexpr auto kDescriptorTag = tt::tt_metal::distributed::multihost::Tag{2};
constexpr auto kAckTag = tt::tt_metal::distributed::multihost::Tag{3};

enum class AgreementAck : uint32_t { READY = 1, ACCEPT = 2, REJECT = 3 };

template <typename T>
ttsl::Span<std::byte> as_writable_bytes(T& value) {
    return {reinterpret_cast<std::byte*>(&value), sizeof(T)};
}

template <typename T>
ttsl::Span<std::byte> as_writable_bytes(std::vector<T>& values) {
    return {reinterpret_cast<std::byte*>(values.data()), values.size() * sizeof(T)};
}

void receive_exact(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    ttsl::Span<std::byte> buffer,
    tt::tt_metal::distributed::multihost::Rank peer,
    tt::tt_metal::distributed::multihost::Tag tag) {
    const auto status = context->irecv(buffer, peer, tag)->wait();
    TT_FATAL(
        status.count == static_cast<int>(buffer.size()),
        "MPI tensor message size mismatch for tag {}: expected {} bytes, received {}",
        *tag,
        buffer.size(),
        status.count);
}

void send_ack(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    AgreementAck ack) {
    context->send(as_writable_bytes(ack), peer, kAckTag);
}

AgreementAck receive_ack(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer) {
    AgreementAck ack{};
    receive_exact(context, as_writable_bytes(ack), peer, kAckTag);
    return ack;
}

std::vector<tt::tt_metal::HostBuffer> get_as(const ttnn::Tensor& tensor) {
    TT_FATAL(ttnn::is_cpu_tensor(tensor), "Tensor must be on host");
    const auto& storage = tensor.host_storage();
    std::vector<tt::tt_metal::HostBuffer> buffers;
    buffers.reserve(storage.buffer().shard_coords().size());
    storage.buffer().apply([&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
    return buffers;
}

std::vector<std::span<std::byte>> get_bytes_from_cpu_tensor(ttnn::Tensor& cpu_tensor) {
    auto buffers = get_as(cpu_tensor);

    std::vector<std::span<std::byte>> res;
    res.reserve(buffers.size());
    for (auto& buffer : buffers) {
        auto view = buffer.view_bytes();
        auto span = std::as_writable_bytes(std::span{view.begin(), view.end()});
        res.push_back(span);
    }
    return res;
}

detail::MPITensorDescriptor make_descriptor(
    const ttnn::Tensor& tensor, const std::vector<std::span<std::byte>>& buffers) {
    detail::MPITensorDescriptor descriptor{
        .dtype = static_cast<uint32_t>(tensor.dtype()),
        .layout = static_cast<uint32_t>(tensor.layout()),
    };
    const auto& logical_shape = tensor.logical_shape();
    descriptor.logical_shape.assign(logical_shape.begin(), logical_shape.end());
    const auto& padded_shape = tensor.padded_shape();
    descriptor.padded_shape.assign(padded_shape.begin(), padded_shape.end());
    descriptor.segment_sizes.reserve(buffers.size());
    for (const auto buffer : buffers) {
        descriptor.segment_sizes.push_back(buffer.size());
    }
    return descriptor;
}

void send_descriptor_and_wait_for_acceptance(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const detail::MPITensorDescriptor& descriptor) {
    auto bytes = detail::serialize_mpi_tensor_descriptor(descriptor);
    uint64_t descriptor_size = bytes.size();
    context->send(as_writable_bytes(descriptor_size), peer, kPreambleTag);
    TT_FATAL(receive_ack(context, peer) == AgreementAck::READY, "MPI tensor receiver rejected descriptor preamble");

    context->send(as_writable_bytes(bytes), peer, kDescriptorTag);
    TT_FATAL(receive_ack(context, peer) == AgreementAck::ACCEPT, "MPI tensor receiver rejected descriptor");
}

void receive_and_validate_descriptor(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const detail::MPITensorDescriptor& expected) {
    uint64_t descriptor_size = 0;
    receive_exact(context, as_writable_bytes(descriptor_size), peer, kPreambleTag);
    if (descriptor_size == 0 || descriptor_size > kMaxDescriptorBytes) {
        send_ack(context, peer, AgreementAck::REJECT);
        TT_FATAL(false, "Invalid MPI tensor descriptor size: {}", descriptor_size);
    }
    send_ack(context, peer, AgreementAck::READY);

    std::vector<std::byte> bytes(descriptor_size);
    receive_exact(context, as_writable_bytes(bytes), peer, kDescriptorTag);
    detail::MPITensorDescriptor received;
    std::string parse_error;
    if (!detail::deserialize_mpi_tensor_descriptor(bytes, received, parse_error)) {
        send_ack(context, peer, AgreementAck::REJECT);
        TT_FATAL(false, "Invalid MPI tensor descriptor: {}", parse_error);
    }
    const auto mismatch = detail::compare_mpi_tensor_descriptors(expected, received);
    if (!mismatch.empty()) {
        send_ack(context, peer, AgreementAck::REJECT);
        TT_FATAL(false, "MPI tensor descriptor mismatch: {}", mismatch);
    }
    send_ack(context, peer, AgreementAck::ACCEPT);
}

}  // namespace

namespace detail {

namespace {

void append_u64(std::vector<std::byte>& bytes, uint64_t value) {
    const auto* begin = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

bool read_u64(std::span<const std::byte> bytes, size_t& offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) {
        return false;
    }
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    offset += sizeof(value);
    return true;
}

void append_vector(std::vector<std::byte>& bytes, const auto& values) {
    append_u64(bytes, values.size());
    for (const auto value : values) {
        append_u64(bytes, value);
    }
}

bool read_u32_vector(
    std::span<const std::byte> bytes, size_t& offset, std::vector<uint32_t>& values, std::string& error) {
    uint64_t size = 0;
    if (!read_u64(bytes, offset, size) || size > ttnn::MAX_NUM_DIMENSIONS) {
        error = "invalid tensor rank";
        return false;
    }
    values.reserve(size);
    for (uint64_t i = 0; i < size; ++i) {
        uint64_t value = 0;
        if (!read_u64(bytes, offset, value) || value > std::numeric_limits<uint32_t>::max()) {
            error = "invalid tensor dimension";
            return false;
        }
        values.push_back(static_cast<uint32_t>(value));
    }
    return true;
}

}  // namespace

std::vector<std::byte> serialize_mpi_tensor_descriptor(const MPITensorDescriptor& descriptor) {
    std::vector<std::byte> bytes;
    bytes.reserve(
        8 * (7 + descriptor.logical_shape.size() + descriptor.padded_shape.size() + descriptor.segment_sizes.size()));
    append_u64(bytes, kDescriptorMagic);
    append_u64(bytes, kDescriptorVersion);
    append_u64(bytes, descriptor.dtype);
    append_u64(bytes, descriptor.layout);
    append_vector(bytes, descriptor.logical_shape);
    append_vector(bytes, descriptor.padded_shape);
    append_vector(bytes, descriptor.segment_sizes);
    return bytes;
}

bool deserialize_mpi_tensor_descriptor(
    std::span<const std::byte> bytes, MPITensorDescriptor& descriptor, std::string& error) {
    if (bytes.size() > kMaxDescriptorBytes) {
        error = "descriptor is too large";
        return false;
    }
    size_t offset = 0;
    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t dtype = 0;
    uint64_t layout = 0;
    if (!read_u64(bytes, offset, magic) || magic != kDescriptorMagic) {
        error = "bad magic";
        return false;
    }
    if (!read_u64(bytes, offset, version) || version != kDescriptorVersion) {
        error = "unsupported version";
        return false;
    }
    if (!read_u64(bytes, offset, dtype) || dtype > std::numeric_limits<uint32_t>::max() ||
        !read_u64(bytes, offset, layout) || layout > std::numeric_limits<uint32_t>::max()) {
        error = "invalid dtype or layout";
        return false;
    }
    descriptor = MPITensorDescriptor{
        .dtype = static_cast<uint32_t>(dtype),
        .layout = static_cast<uint32_t>(layout),
    };
    if (!read_u32_vector(bytes, offset, descriptor.logical_shape, error) ||
        !read_u32_vector(bytes, offset, descriptor.padded_shape, error)) {
        return false;
    }
    uint64_t segment_count = 0;
    if (!read_u64(bytes, offset, segment_count) || segment_count > (kMaxDescriptorBytes - offset) / sizeof(uint64_t)) {
        error = "invalid segment count";
        return false;
    }
    descriptor.segment_sizes.reserve(segment_count);
    for (uint64_t i = 0; i < segment_count; ++i) {
        uint64_t size = 0;
        if (!read_u64(bytes, offset, size)) {
            error = "truncated segment sizes";
            return false;
        }
        descriptor.segment_sizes.push_back(size);
    }
    if (offset != bytes.size()) {
        error = "trailing descriptor bytes";
        return false;
    }
    return true;
}

std::string compare_mpi_tensor_descriptors(const MPITensorDescriptor& expected, const MPITensorDescriptor& received) {
    if (expected.dtype != received.dtype) {
        return "dtype differs";
    }
    if (expected.layout != received.layout) {
        return "layout differs";
    }
    if (expected.logical_shape != received.logical_shape) {
        return "logical shape differs";
    }
    if (expected.padded_shape != received.padded_shape) {
        return "padded shape differs";
    }
    if (expected.segment_sizes != received.segment_sizes) {
        return "payload segmentation differs";
    }
    return {};
}

void send_mpi_tensor_descriptor(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const MPITensorDescriptor& descriptor) {
    send_descriptor_and_wait_for_acceptance(context, peer, descriptor);
}

void receive_and_validate_mpi_tensor_descriptor(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const MPITensorDescriptor& expected) {
    receive_and_validate_descriptor(context, peer, expected);
}

}  // namespace detail

MPISocket::MPISocket(
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context,
    tt::tt_metal::distributed::multihost::Rank sender_rank,
    tt::tt_metal::distributed::multihost::Rank receiver_rank) :
    distributed_context_(std::move(distributed_context)), sender_rank_(sender_rank), receiver_rank_(receiver_rank) {}

void MPISocket::send(const ttnn::Tensor& tensor) {
    std::lock_guard lock(send_mutex_);
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    auto receiver_rank = get_rank();
    send_descriptor_and_wait_for_acceptance(distributed_context_, receiver_rank, make_descriptor(cpu_tensor, buffers));
    for (auto buffer : buffers) {
        distributed_context_->send(buffer, receiver_rank, kPayloadTag);
    }
}

void MPISocket::recv(ttnn::Tensor& tensor) {
    std::lock_guard lock(recv_mutex_);
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    auto sender_rank = get_rank();
    receive_and_validate_descriptor(distributed_context_, sender_rank, make_descriptor(cpu_tensor, buffers));
    for (auto buffer : buffers) {
        receive_exact(distributed_context_, buffer, sender_rank, kPayloadTag);
    }

    ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
}

tt::tt_metal::distributed::multihost::Rank MPISocket::get_rank() const {
    const auto local_rank = distributed_context_->rank();
    if (local_rank == sender_rank_) {
        return receiver_rank_;
    }
    TT_FATAL(local_rank == receiver_rank_, "Local rank is not an endpoint of this MPI socket");
    return sender_rank_;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> MPISocket::get_distributed_context() const {
    return distributed_context_;
}

std::unique_ptr<MPISocket> MPISocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank rank,
    tt::tt_metal::distributed::SocketConfig socket_config) {
    (void)mesh_device;
    if (socket_config.distributed_context->rank() < rank) {
        socket_config.sender_rank = socket_config.distributed_context->rank();
        socket_config.receiver_rank = rank;
    } else {
        socket_config.sender_rank = rank;
        socket_config.receiver_rank = socket_config.distributed_context->rank();
    }
    return std::make_unique<MPISocket>(
        socket_config.distributed_context, socket_config.sender_rank, socket_config.receiver_rank);
}

}  // namespace ttnn::distributed
