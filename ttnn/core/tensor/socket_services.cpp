// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/socket_services.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "tensor/tensor_ops.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace tt::tt_metal {

namespace {

// Build a single-shard host tensor with zero-initialised data of size `spec`.
// Used purely to feed the mapper at construction time so we can extract a
// TensorTopology before any user data exists. The bytes are never read.
//
// TODO: replace with a direct "topology from MeshMapperConfig + global shape"
// helper once one exists upstream, so we can skip allocating `spec`-many bytes
// just to throw them away.
Tensor make_zero_host_tensor(const TensorSpec& spec) {
    const size_t bytes = spec.compute_packed_buffer_size_bytes();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_vector<bfloat16>(std::vector<bfloat16>(bytes / sizeof(bfloat16)), spec);
        case DataType::FLOAT32: return Tensor::from_vector<float>(std::vector<float>(bytes / sizeof(float)), spec);
        case DataType::INT32: return Tensor::from_vector<int32_t>(std::vector<int32_t>(bytes / sizeof(int32_t)), spec);
        case DataType::UINT8: return Tensor::from_vector<uint8_t>(std::vector<uint8_t>(bytes / sizeof(uint8_t)), spec);
        case DataType::UINT16:
            return Tensor::from_vector<uint16_t>(std::vector<uint16_t>(bytes / sizeof(uint16_t)), spec);
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return Tensor::from_vector<uint32_t>(std::vector<uint32_t>(bytes / sizeof(uint32_t)), spec);
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Zero-copy wrap of caller-provided raw bytes into a host Tensor whose spec
// matches `spec` exactly (ROW_MAJOR + default MemoryConfig only — see caller).
//
// Why const_cast: Tensor::from_borrowed_data takes a non-const Span as an API
// artifact, but the resulting tensor is treated as read-only along the
// forward_to_tensor pipeline (mapper reads; H2DSocket::write copies out).
// Why empty MemoryPin: the borrowed tensor never escapes forward_to_tensor's
// stack frame, and H2DSocket::write is synchronous, so the caller's bytes are
// guaranteed alive for the whole transit.
Tensor make_borrowed_host_tensor(ttsl::Span<const std::byte> bytes, const TensorSpec& spec) {
    auto* raw = const_cast<std::byte*>(bytes.data());
    const auto& shape = spec.logical_shape();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_borrowed_data<bfloat16>(
                ttsl::Span<bfloat16>(reinterpret_cast<bfloat16*>(raw), bytes.size() / sizeof(bfloat16)),
                shape,
                MemoryPin{});
        case DataType::FLOAT32:
            return Tensor::from_borrowed_data<float>(
                ttsl::Span<float>(reinterpret_cast<float*>(raw), bytes.size() / sizeof(float)), shape, MemoryPin{});
        case DataType::INT32:
            return Tensor::from_borrowed_data<int32_t>(
                ttsl::Span<int32_t>(reinterpret_cast<int32_t*>(raw), bytes.size() / sizeof(int32_t)),
                shape,
                MemoryPin{});
        case DataType::UINT8:
            return Tensor::from_borrowed_data<uint8_t>(
                ttsl::Span<uint8_t>(reinterpret_cast<uint8_t*>(raw), bytes.size() / sizeof(uint8_t)),
                shape,
                MemoryPin{});
        case DataType::UINT16:
            return Tensor::from_borrowed_data<uint16_t>(
                ttsl::Span<uint16_t>(reinterpret_cast<uint16_t*>(raw), bytes.size() / sizeof(uint16_t)),
                shape,
                MemoryPin{});
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return Tensor::from_borrowed_data<uint32_t>(
                ttsl::Span<uint32_t>(reinterpret_cast<uint32_t*>(raw), bytes.size() / sizeof(uint32_t)),
                shape,
                MemoryPin{});
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Picks the largest `pages_per_chunk` (and therefore largest `socket_page_size`)
// that:
//   * fits in `scratch_cb_size_bytes` (pages_per_chunk * tensor_page_size),
//   * divides `tensor_num_pages` evenly (no ragged last chunk),
// falling back to 1 in the worst case (e.g. `tensor_num_pages` is prime).
//
// Mirrors the logic in copy_tensor_over_socket (tensor_ops.cpp). Pulled into a
// helper so the persistent service can reuse the exact same chunking strategy
// without depending on copy_tensor_over_socket's anonymous namespace.
struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
};

ChunkPlan derive_chunk_plan(uint32_t tensor_page_size, uint32_t tensor_num_pages, uint32_t scratch_cb_size_bytes) {
    TT_FATAL(tensor_page_size > 0, "device_tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "device_tensor must have at least one page");
    TT_FATAL(
        scratch_cb_size_bytes >= tensor_page_size,
        "scratch_cb_size_bytes ({} B) must be >= tensor page size ({} B); "
        "consider a layout with smaller pages or a larger CB budget",
        scratch_cb_size_bytes,
        tensor_page_size);

    const uint32_t max_pages_per_chunk_by_cb = scratch_cb_size_bytes / tensor_page_size;
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk_by_cb);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    return ChunkPlan{
        .socket_page_size = pages_per_chunk * tensor_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
    };
}

// Builds the single-core persistent H2D program for one socket / device buffer.
//
// CT-arg layout (must stay in sync with persistent_h2d_receiver.cpp):
//   [0] socket_config_addr
//   [1] termination_semaphore_addr
//   [2] socket_page_size
//   [3] num_socket_pages
//   [4] output_tensor_addr
//   [5] output_tensor_page_size
//   [6] pages_per_chunk
//   [7] scratch_buffer_cb_index
//   [8..] TensorAccessorArgs
Program build_persistent_h2d_program(
    const Buffer& device_buffer,
    const CoreCoord& recv_core,
    uint32_t socket_config_buffer_address,
    uint32_t termination_semaphore_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    DataType dtype) {
    auto program = CreateProgram();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;
    auto cb_cfg =
        CircularBufferConfig(plan.socket_page_size, {{scratch_cb_index, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(scratch_cb_index, plan.socket_page_size);
    CreateCircularBuffer(program, recv_core, cb_cfg);

    auto tensor_accessor_args = TensorAccessorArgs(device_buffer);
    auto tensor_accessor_compile_args = tensor_accessor_args.get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_buffer_address,
        termination_semaphore_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        static_cast<uint32_t>(device_buffer.address()),
        tensor_page_size,
        plan.pages_per_chunk,
        static_cast<uint32_t>(scratch_cb_index),
    };
    ct_args.insert(ct_args.end(), tensor_accessor_compile_args.begin(), tensor_accessor_compile_args.end());

    CreateKernel(
        program,
        "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_receiver.cpp",
        recv_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    return program;
}

}  // namespace

H2DStreamService::H2DStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg) :
    mesh_device_(mesh_device), cfg_(std::move(cfg)) {
    // --- B1: config validation -------------------------------------------------
    TT_FATAL(mesh_device_ != nullptr, "H2DStreamService: mesh_device must not be null");
    TT_FATAL(cfg_.fifo_size_bytes > 0, "H2DStreamService: fifo_size_bytes must be > 0");
    TT_FATAL(cfg_.scratch_cb_size_bytes > 0, "H2DStreamService: scratch_cb_size_bytes must be > 0");

    // --- B2: take ownership of the mapper (defaulting to replicate-on-all if
    //         the caller didn't supply one), then derive per-shard spec &
    //         topology in one pass --------------------------------------------
    // If no mapper is provided we default to replicating on every mesh dim.
    // For a 1x1 mesh this is the identity mapping; for an NxM mesh it puts the
    // full tensor on every device. Sharded distributions require an explicit
    // user-supplied mapper because the placement decision is non-default.
    //
    // The mapper is normally caller-built (via create_mesh_mapper) so the same
    // instance can be reused outside this service if desired. Running it once
    // on a zero-filled host tensor gives us both the per-shard TensorSpec
    // (mapper preserves layout, only resizes the shape) and the TensorTopology
    // (participating coords + placement). Doing it once and reusing both
    // outputs avoids a second mapper invocation.
    if (cfg_.mapper == nullptr) {
        ttsl::SmallVector<distributed::MeshMapperConfig::Placement> replicate_all(
            mesh_device_->shape().dims(), distributed::MeshMapperConfig::Replicate{});
        cfg_.mapper = ttnn::distributed::create_mesh_mapper(
            *mesh_device_, distributed::MeshMapperConfig{.placements = std::move(replicate_all)});
    }
    mapper_ = std::move(cfg_.mapper);

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    const auto& per_shard_spec = distributed_dummy.tensor_spec();
    const auto& topology = distributed_dummy.tensor_topology();

    // --- B3: allocate backing device tensor -----------------------------------
    device_tensor_ = create_device_tensor(per_shard_spec, mesh_device_.get(), topology);

    // --- B4: create one socket per participating mesh coord -------------------
    // Iterating topology.mesh_coords() (not the full mesh shape) keeps replication-
    // collapsed or shape-overridden mappings working correctly.
    const auto& coords = topology.mesh_coords();
    sockets_.reserve(coords.size());
    for (const auto& coord : coords) {
        sockets_.push_back(std::make_unique<distributed::H2DSocket>(
            mesh_device_,
            distributed::MeshCoreCoord(coord, cfg_.recv_core),
            cfg_.socket_buffer_type,
            cfg_.fifo_size_bytes,
            cfg_.socket_mode));
    }

    // --- B5: derive a chunk plan ----------------------------------------------
    // Every per-device buffer in the MeshBuffer shares the same spec (page size,
    // num pages), so `device_tensor_.buffer()` (the MeshBuffer's reference
    // buffer) is representative for every socket. Per-coord buffers are still
    // needed in B8 for the per-device address baked into the kernel CT args.
    const uint32_t tensor_page_size = device_tensor_.buffer()->page_size();
    const uint32_t tensor_num_pages = device_tensor_.buffer()->num_pages();
    const ChunkPlan plan = derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg_.scratch_cb_size_bytes);
    socket_page_size_ = plan.socket_page_size;
    num_socket_pages_ = plan.num_socket_pages;

    // --- B6: configure each socket's page size --------------------------------
    // The kernel calls set_receiver_socket_page_size on its side too; the host
    // side needs it for H2DSocket::write() byte arithmetic.
    for (auto& s : sockets_) {
        s->set_page_size(plan.socket_page_size);
    }

    // --- B7: allocate the termination semaphore -------------------------------
    // Single GlobalSemaphore on the CoreRangeSet covering every recv core
    // (same logical core on every participating device). Initial value 0;
    // the dtor flips it to 1.
    termination_semaphore_.emplace(ttnn::global_semaphore::create_global_semaphore(
        mesh_device_.get(),
        CoreRangeSet(CoreRange(cfg_.recv_core)),
        /*initial_value=*/0,
        BufferType::L1));
    const auto termination_addr = static_cast<uint32_t>(termination_semaphore_->address());

    // --- B8: build one persistent program per socket, bundle into a workload --
    for (auto& s : sockets_) {
        const auto core = s->get_active_cores()[0];
        const Buffer* dbuf = device_tensor_.mesh_buffer().get_device_buffer(core.device_coord);
        TT_FATAL(dbuf != nullptr, "H2DStreamService: device buffer missing for coord {}", core.device_coord);
        auto program = build_persistent_h2d_program(
            *dbuf,
            core.core_coord,
            s->get_config_buffer_address(),
            termination_addr,
            plan,
            tensor_page_size,
            device_tensor_.dtype());
        workload_.add_program(distributed::MeshCoordinateRange(core.device_coord), std::move(program));
    }

    // --- B9: launch the persistent kernels (non-blocking) ---------------------
    // The kernels now sit in their outer while-loop polling their sockets.
    // forward_to_tensor calls feed those sockets; the dtor shuts them down.
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload_, /*blocking=*/false);
}

H2DStreamService::~H2DStreamService() {
    // Best-effort clean shutdown. Wrap in try/catch so a failure in any one
    // step (e.g. mesh device already torn down by a faulty caller) doesn't
    // throw from the destructor.
    try {
        // 1. Drain any in-flight host writes so no kernel iteration is mid-
        //    transfer when we signal termination.
        barrier();

        // 2. Flip the semaphore to 1. The kernels exit on the next poll.
        signal_termination();

        // 3. Wait for the workload to actually finish before sockets / device
        //    tensor go out of scope.
        if (mesh_device_) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed with unknown exception");
    }
}

void H2DStreamService::barrier() {
    for (auto& s : sockets_) {
        s->barrier();
    }
}

void H2DStreamService::signal_termination() {
    if (termination_semaphore_.has_value()) {
        termination_semaphore_->reset_semaphore_value(1);
    }
}

std::vector<distributed::H2DSocket*> H2DStreamService::get_sockets() const {
    std::vector<distributed::H2DSocket*> out;
    out.reserve(sockets_.size());
    for (const auto& s : sockets_) {
        out.push_back(s.get());
    }
    return out;
}

void H2DStreamService::forward_to_tensor(ttsl::Span<const std::byte> bytes) {
    // --- S1: validate -----------------------------------------------------------
    // Bytes must equal the packed size of one full global tensor; partial transfers
    // aren't supported because the persistent kernel's chunk count is baked into
    // its CT args.
    const size_t expected = cfg_.global_spec.compute_packed_buffer_size_bytes();
    TT_FATAL(
        bytes.size() == expected,
        "H2DStreamService::forward_to_tensor: span size {} B does not match global_spec packed size {} B",
        bytes.size(),
        expected);

    // ROW_MAJOR + default MemoryConfig is the only combination where the
    // borrowed-data wrap and the mapper's zero-copy fast path both stay engaged.
    // Anything else would either (a) trip the spec-mismatch check in the Tensor
    // overload below because Tensor::from_borrowed_data hardcodes ROW_MAJOR /
    // default MemoryConfig in its produced spec, or (b) eat a normalization copy
    // inside the mapper's create_host_buffer_from_span. Defer TILE / sharded-
    // memory support to a future pass.
    TT_FATAL(
        cfg_.global_spec.layout() == Layout::ROW_MAJOR,
        "H2DStreamService::forward_to_tensor(span): global_spec must be ROW_MAJOR (got {}). "
        "Use the Tensor overload with a pre-distributed host tensor for other layouts.",
        cfg_.global_spec.layout());

    // --- S2: wrap bytes as a borrowed host tensor (zero-copy) -------------------
    Tensor borrowed = make_borrowed_host_tensor(bytes, cfg_.global_spec);

    // --- S3: distribute via the cached mapper -----------------------------------
    // For Replicate placements this stays zero-copy (mapper emplaces one shared
    // HostBuffer at every coord). For Shard placements the mapper materializes
    // per-shard buffers via xtensor chunking — one full input-size copy.
    Tensor distributed = (*mapper_)(borrowed);

    // --- S4: delegate to the Tensor path ----------------------------------------
    // The Tensor overload re-asserts that `distributed.tensor_spec()` matches
    // `device_tensor_.tensor_spec()`; since `distributed` came from the same
    // mapper that produced the backing tensor's per-shard spec at construction,
    // this should hold by construction.
    forward_to_tensor(distributed);
}

void H2DStreamService::forward_to_tensor(const Tensor& host_tensor) {
    // --- W1: input validation -------------------------------------------------
    TT_FATAL(
        host_tensor.storage_type() == StorageType::HOST,
        "H2DStreamService::forward_to_tensor: expected host tensor, got storage_type={}",
        host_tensor.storage_type());

    const auto& host_mesh_tensor = host_tensor.host_storage().host_tensor();
    TT_FATAL(
        host_mesh_tensor.tensor_spec() == device_tensor_.tensor_spec(),
        "H2DStreamService::forward_to_tensor: host tensor per-shard spec ({}) does not match backing "
        "per-shard spec ({}). Did you distribute the host tensor with a different mapper config?",
        host_mesh_tensor.tensor_spec(),
        device_tensor_.tensor_spec());

    const auto& dhb = host_mesh_tensor.buffer();
    const uint64_t expected_shard_bytes =
        static_cast<uint64_t>(num_socket_pages_) * static_cast<uint64_t>(socket_page_size_);

    // --- W2: per-socket shard lookup ------------------------------------------
    // No HostBuffer retention needed: H2DSocket::write synchronously copies
    // into the shared FIFO, so by the time it returns the source bytes are no
    // longer referenced. The caller's `host_tensor` is alive for the whole
    // synchronous call anyway. Sockets were built from `topology.mesh_coords()`
    // in the ctor and are de-duplicated by construction, so no runtime
    // coord-dedup check is needed.
    std::vector<std::byte*> bases;
    bases.reserve(sockets_.size());

    for (auto& s : sockets_) {
        // NOTE: H2DSocket::get_active_cores() returns a `std::vector<MeshCoreCoord>` BY VALUE.
        // Binding `const auto& coord = ...[0].device_coord;` would dangle: lifetime extension
        // does not propagate through operator[] (a function-call-returned reference), so the
        // temporary vector dies at the `;` and the MeshCoordinate's internal SmallVector
        // reads as garbage on the next access. Copy the coord by value instead.
        const auto coord = s->get_active_cores()[0].device_coord;
        TT_FATAL(
            dhb.is_local(coord),
            "H2DStreamService::forward_to_tensor: host tensor has no local shard for coord {}",
            coord);
        auto shard_opt = dhb.get_shard(coord);
        TT_FATAL(
            shard_opt.has_value(),
            "H2DStreamService::forward_to_tensor: host shard for coord {} is not populated",
            coord);

        auto shard_span = shard_opt->view_bytes();
        TT_FATAL(
            shard_span.size() == expected_shard_bytes,
            "H2DStreamService::forward_to_tensor: host shard at coord {} has {} B, expected {} "
            "({} socket pages * {} B). Layout drift between mapper output and backing tensor.",
            coord,
            shard_span.size(),
            expected_shard_bytes,
            num_socket_pages_,
            socket_page_size_);

        bases.push_back(shard_span.data());
    }

    // --- W3: page-major write loop --------------------------------------------
    // Send socket page `i` to every socket before page `i+1`, so every kernel
    // can start progressing on the first round. H2DSocket::write enforces
    // num_pages*page_size <= fifo_size per call, so we always pass num_pages=1;
    // the host blocks naturally inside reserve_bytes() when a FIFO fills up.
    //
    // No barrier here on purpose: the whole point of the persistent service is
    // that callers can pipeline writes and only sync via `barrier()` when they
    // actually need to read the backing tensor.
    for (uint32_t i = 0; i < num_socket_pages_; ++i) {
        const size_t offset = static_cast<size_t>(i) * socket_page_size_;
        for (size_t s = 0; s < sockets_.size(); ++s) {
            sockets_[s]->write(bases[s] + offset, /*num_pages=*/1);
        }
    }
}

}  // namespace tt::tt_metal
