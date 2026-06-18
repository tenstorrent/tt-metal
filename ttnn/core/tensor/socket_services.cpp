// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/socket_services.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/driver_atomics.hpp>

#include "tensor/tensor_ops.hpp"
#include "tt_metal/distributed/h2d_stream_service_descriptor.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace tt::tt_metal {

namespace {

// Zero-filled host tensor of size `spec`, used to feed the mapper at construction.
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
        case DataType::FP8_E4M3: TT_THROW("H2DStreamService: FP8_E4M3 is not supported");
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Zero-copy wrap of caller-provided raw bytes into a host Tensor matching `spec`.
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
        case DataType::FP8_E4M3: TT_THROW("H2DStreamService: FP8_E4M3 is not supported");
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Reader/writer split: the scratch CB holds `slot_count` full socket pages so the reader
// can stage pages ahead while the writer drains earlier ones. We target kMinDataSlots
// slots (double-buffering) and pick the largest socket page that still leaves room for
// them; if the budget cannot hold that many we fall back to a single slot (no device-side
// overlap, but still correct).
constexpr uint32_t kMinDataSlots = 2;

struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
    uint32_t slot_count;        // full-page data-CB slots backing the reader/writer pipeline
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

    // Largest pages_per_chunk that leaves room for kMinDataSlots full slots; fall back to
    // the largest chunk that fits a single slot when the budget is too small for two.
    uint32_t max_pages_per_chunk = scratch_cb_size_bytes / (kMinDataSlots * tensor_page_size);
    if (max_pages_per_chunk == 0) {
        max_pages_per_chunk = scratch_cb_size_bytes / tensor_page_size;
    }
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    const uint32_t socket_page_size = pages_per_chunk * tensor_page_size;
    return ChunkPlan{
        .socket_page_size = socket_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
        .slot_count = scratch_cb_size_bytes / socket_page_size,
    };
}

// Worker-sync CT-arg block. Populated when Config::worker_cores is set; all
// fields zero when disabled (the kernel's `if constexpr (worker_sync_enabled)`
// gate skips the block entirely).
struct WorkerSyncArgs {
    bool enabled = false;
    uint32_t data_ready_sem_addr = 0;     // worker-grid L1 (mesh-wide GlobalSemaphore)
    uint32_t consumed_counter_addr = 0;   // service-core L1 (per-coord, allocated via ServiceCoreManager)
    uint32_t mcast_noc_x_start = 0;       // physical NoC bbox of worker_cores on this device
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;             // mcast destination count + sync arithmetic target
};

// The writer half owns both worker-grid multicasts (data_ready and metadata) and runs on
// RISCV_1's default NOC. build_persistent_h2d_program (kernel placement) and
// set_worker_mcast_corners (coord ordering) must agree on this single source of truth, so a
// future NOC reassignment stays correct without introducing the multicast-direction bug.
constexpr NOC kWriterNoc = NOC::RISCV_1_default;

// Order a worker-grid multicast box for the NOC the writer will issue the multicast on.
//
// A NOC multicast must lead with the corner its routing direction reaches first: NOC 0
// traverses low->high coords so it leads with the low (logical-start) corner, while NOC 1
// traverses high->low so its start/end corners are swapped.
void set_worker_mcast_corners(WorkerSyncArgs& args, NOC noc, CoreCoord start_phys, CoreCoord end_phys) {
    const bool reverse = (noc == NOC::NOC_1);
    const CoreCoord lead = reverse ? end_phys : start_phys;
    const CoreCoord trail = reverse ? start_phys : end_phys;
    args.mcast_noc_x_start = static_cast<uint32_t>(lead.x);
    args.mcast_noc_y_start = static_cast<uint32_t>(lead.y);
    args.mcast_noc_x_end = static_cast<uint32_t>(trail.x);
    args.mcast_noc_y_end = static_cast<uint32_t>(trail.y);
}

// Metadata multicast CT-arg block. Populated when Config::metadata_size_bytes > 0.
struct MetadataArgs {
    bool enabled = false;
    uint32_t metadata_size_bytes = 0;     // user-specified size; <= socket_page_size
    uint32_t metadata_l1_addr = 0;        // worker-grid L1 (mesh-wide L1-sharded Buffer)
};

// Writer DRAM-completion push CT-arg block. The writer pushes a per-transfer count to its
// slot in the shared host-pinned completion region so barrier() can confirm the backing
// tensor has drained.
struct CompletionArgs {
    uint32_t pcie_xy_enc = 0;   // PCIe core encoding of the host pinned slot (TRANSLATED coords)
    uint32_t pcie_addr_lo = 0;  // low 32 bits of the 64-bit PCIe offset
    uint32_t pcie_addr_hi = 0;  // high 32 bits
    uint32_t src_l1_addr = 0;   // service-core L1 scratch the writer stages the count in
};

struct CompletionLayout {
    uint64_t shm_size = 0;
    uint32_t issued_offset = 0;
    uint32_t completed_offset = 0;
    uint32_t completed_stride = sizeof(uint32_t);
};

CompletionLayout make_completion_layout(uint32_t num_counters) {
    TT_FATAL(num_counters > 0, "H2DStreamService: completion state requires at least one counter");
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uint32_t pcie_alignment = hal::get_pcie_alignment();
    const uint64_t raw_size =
        static_cast<uint64_t>(pcie_alignment) + static_cast<uint64_t>(num_counters) * pcie_alignment;
    const uint64_t shm_size = ((raw_size + page - 1) / page) * page;
    return CompletionLayout{
        .shm_size = shm_size,
        .issued_offset = 0,
        .completed_offset = pcie_alignment,
        .completed_stride = pcie_alignment,
    };
}

volatile uint32_t* completion_word(void* base, uint32_t offset) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<std::byte*>(base) + offset);
}

// Builds the two-kernel persistent H2D program for one socket / device buffer: a reader on
// RISCV_0 (its default NOC, PCIe -> L1) and a writer on RISCV_1 (its default NOC, L1 -> DRAM),
// connected by a multi-slot data CB (and a metadata CB when metadata is enabled). Splitting
// the phases onto two RISCs lets the reader stage ahead while the writer drains.
//
// Reader CT-arg layout (must stay in sync with persistent_h2d_reader.cpp):
//   [0] socket_page_size
//   [1] num_socket_pages
//   [2] data_cb_index
//   [3] metadata_enabled                  (uint32 0/1)
//   [4] metadata_cb_index
// Reader RT-arg layout:
//   [0] socket_config_addr
//   [1] termination_semaphore_addr
//
// Writer CT-arg layout (must stay in sync with persistent_h2d_writer.cpp):
//   [0] num_socket_pages
//   [1] output_tensor_page_size
//   [2] pages_per_chunk
//   [3] data_cb_index
//   [4] metadata_enabled                  (uint32 0/1)
//   [5] metadata_cb_index
//   [6] worker_sync_enabled               (uint32 0/1)
//   [7..] TensorAccessorArgs
// Writer RT-arg layout:
//   [0]  termination_semaphore_addr
//   [1]  output_tensor_addr
//   [2]  data_ready_sem_addr              (uint32, worker-grid L1)
//   [3]  consumed_counter_addr            (uint32, local service-core L1)
//   [4]  worker_mcast_noc_x_start          (corners ordered for kWriterNoc; see
//   [5]  worker_mcast_noc_y_start           set_worker_mcast_corners -- "start" is the
//   [6]  worker_mcast_noc_x_end             corner the writer's NOC reaches first, which
//   [7]  worker_mcast_noc_y_end             is the high corner on NOC 1)
//   [8]  num_workers
//   [9]  metadata_size_bytes              (uint32, un-padded user size)
//   [10] metadata_l1_addr                 (uint32, worker-grid L1)
//   [11] completion_pcie_xy_enc           (uint32, host pinned slot PCIe core)
//   [12] completion_pcie_addr_lo          (uint32, low 32 bits of PCIe offset)
//   [13] completion_pcie_addr_hi          (uint32, high 32 bits of PCIe offset)
//   [14] completion_src_l1_addr           (uint32, local service-core L1 scratch)
Program build_persistent_h2d_program(
    const Buffer& device_buffer,
    const CoreCoord& recv_core,
    uint32_t socket_config_buffer_address,
    uint32_t termination_semaphore_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    DataType dtype,
    const WorkerSyncArgs& worker_sync,
    const MetadataArgs& metadata,
    const CompletionArgs& completion) {
    auto program = CreateProgram();

    constexpr tt::CBIndex data_cb_index = tt::CBIndex::c_0;
    constexpr tt::CBIndex metadata_cb_index = tt::CBIndex::c_1;
    const auto data_format = datatype_to_dataformat_converter(dtype);

    // Data CB: slot_count full socket pages so the reader can stage ahead of the writer.
    auto data_cb_cfg = CircularBufferConfig(plan.slot_count * plan.socket_page_size, {{data_cb_index, data_format}})
                           .set_page_size(data_cb_index, plan.socket_page_size);
    CreateCircularBuffer(program, recv_core, data_cb_cfg);

    // Metadata CB: a single socket page; only created when metadata multicast is enabled.
    if (metadata.enabled) {
        auto metadata_cb_cfg = CircularBufferConfig(plan.socket_page_size, {{metadata_cb_index, data_format}})
                                   .set_page_size(metadata_cb_index, plan.socket_page_size);
        CreateCircularBuffer(program, recv_core, metadata_cb_cfg);
    }

    std::vector<uint32_t> reader_ct_args = {
        plan.socket_page_size,
        plan.num_socket_pages,
        static_cast<uint32_t>(data_cb_index),
        static_cast<uint32_t>(metadata.enabled ? 1u : 0u),
        static_cast<uint32_t>(metadata_cb_index),
    };
    auto reader_kernel = CreateKernel(
        program,
        "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_reader.cpp",
        recv_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_ct_args,
        });
    SetRuntimeArgs(program, reader_kernel, recv_core, {socket_config_buffer_address, termination_semaphore_addr});

    auto tensor_accessor_args = TensorAccessorArgs(device_buffer);
    auto tensor_accessor_compile_args = tensor_accessor_args.get_compile_time_args();

    std::vector<uint32_t> writer_ct_args = {
        plan.num_socket_pages,
        tensor_page_size,
        plan.pages_per_chunk,
        static_cast<uint32_t>(data_cb_index),
        static_cast<uint32_t>(metadata.enabled ? 1u : 0u),
        static_cast<uint32_t>(metadata_cb_index),
        static_cast<uint32_t>(worker_sync.enabled ? 1u : 0u),
    };
    writer_ct_args.insert(
        writer_ct_args.end(), tensor_accessor_compile_args.begin(), tensor_accessor_compile_args.end());

    auto writer_kernel = CreateKernel(
        program,
        "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp",
        recv_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = kWriterNoc,
            .compile_args = writer_ct_args,
        });
    SetRuntimeArgs(
        program,
        writer_kernel,
        recv_core,
        {
            termination_semaphore_addr,
            static_cast<uint32_t>(device_buffer.address()),
            worker_sync.data_ready_sem_addr,
            worker_sync.consumed_counter_addr,
            worker_sync.mcast_noc_x_start,
            worker_sync.mcast_noc_y_start,
            worker_sync.mcast_noc_x_end,
            worker_sync.mcast_noc_y_end,
            worker_sync.num_workers,
            metadata.metadata_size_bytes,
            metadata.metadata_l1_addr,
            completion.pcie_xy_enc,
            completion.pcie_addr_lo,
            completion.pcie_addr_hi,
            completion.src_l1_addr,
        });

    return program;
}

}  // namespace

H2DStreamService::H2DStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg) :
    mesh_device_(mesh_device), cfg_(std::move(cfg)) {
    TT_FATAL(mesh_device_ != nullptr, "H2DStreamService: mesh_device must not be null");
    TT_FATAL(cfg_.fifo_size_bytes > 0, "H2DStreamService: fifo_size_bytes must be > 0");
    TT_FATAL(cfg_.scratch_cb_size_bytes > 0, "H2DStreamService: scratch_cb_size_bytes must be > 0");
    TT_FATAL(
        cfg_.metadata_size_bytes == 0 || cfg_.worker_cores.has_value(),
        "H2DStreamService: metadata_size_bytes={} requires Config::worker_cores to be set "
        "(no workers to multicast metadata to)",
        cfg_.metadata_size_bytes);

    // Default to replicate-on-every-mesh-dim when no mapper is supplied.
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

    device_tensor_ = create_device_tensor(per_shard_spec, mesh_device_.get(), topology);
    per_shard_spec_ = device_tensor_.tensor_spec();

    // Each device may resolve a different free service core; record it per coord.
    auto& svc = tt::tt_metal::internal::service_core_manager();
    const auto& coords = topology.mesh_coords();
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(
            !claimable.empty(),
            "H2DStreamService: no claimable service core on device at coord {}",
            coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores_.emplace(coord, chosen);
    }

    // Iterate participating coords (not the full mesh shape) so replication-
    // collapsed or shape-overridden mappings stay correct.
    sockets_.reserve(coords.size());
    for (const auto& coord : coords) {
        sockets_.push_back(std::make_unique<distributed::H2DSocket>(
            mesh_device_,
            distributed::MeshCoreCoord(coord, service_cores_.at(coord)),
            cfg_.socket_buffer_type,
            cfg_.fifo_size_bytes,
            cfg_.socket_mode));
    }

    // Every per-device buffer shares the same spec, so this buffer is representative.
    const uint32_t tensor_page_size = device_tensor_.buffer()->page_size();
    const uint32_t tensor_num_pages = device_tensor_.buffer()->num_pages();
    const ChunkPlan plan = derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg_.scratch_cb_size_bytes);
    socket_page_size_ = plan.socket_page_size;
    num_socket_pages_ = plan.num_socket_pages;

    // Metadata travels as exactly one trailing socket page, so it must fit.
    TT_FATAL(
        cfg_.metadata_size_bytes <= socket_page_size_,
        "H2DStreamService: metadata_size_bytes={} exceeds derived socket_page_size={} "
        "(single-metadata-page constraint). Either reduce metadata or increase "
        "scratch_cb_size_bytes / per-shard page size.",
        cfg_.metadata_size_bytes,
        socket_page_size_);

    for (auto& s : sockets_) {
        s->set_page_size(plan.socket_page_size);
    }

    // Per-device termination signal: one uint32 in L1, written directly rather
    // than via a GlobalSemaphore (which can't target per-coord service cores).
    std::vector<uint32_t> zero_word{0};
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        const DeviceAddr sem_addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
        termination_addrs_.emplace(coord, sem_addr);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(sem_addr), zero_word);
    }

    if (cfg_.worker_cores.has_value()) {
        const auto& worker_range = cfg_.worker_cores.value();
        num_workers_ = worker_range.size();
        TT_FATAL(num_workers_ > 0, "H2DStreamService: cfg.worker_cores must contain at least one core");

        // Mesh-wide, same address on every (device, worker core). Workers poll
        // their local copy; the service kernel multicasts atomic-inc into it.
        data_ready_sem_.emplace(ttnn::global_semaphore::create_global_semaphore(
            mesh_device_.get(),
            CoreRangeSet(worker_range),
            /*initial_value=*/0,
            BufferType::L1));

        // Consumed counter: per-coord L1 word on the service core.
        for (const auto& coord : coords) {
            auto* d = mesh_device_->get_device(coord);
            const CoreCoord chosen = service_cores_.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
            consumed_addrs_.emplace(coord, addr);
            tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), zero_word);
        }
    }

    // Mesh-wide L1-sharded Buffer across worker_cores: REPLICATED per device,
    // HEIGHT_SHARDED so each worker core gets one L1-aligned shard.
    if (cfg_.metadata_size_bytes > 0) {
        const uint32_t l1_align = hal::get_l1_alignment();
        const DeviceAddr aligned_shard_size =
            tt::align(static_cast<DeviceAddr>(cfg_.metadata_size_bytes), static_cast<DeviceAddr>(l1_align));

        const CoreRangeSet shard_grid(cfg_.worker_cores.value());

        // Mirrors the L1-sharded allocation pattern in h2d_socket.cpp.
        distributed::DeviceLocalBufferConfig device_local = {
            .page_size = aligned_shard_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(
                ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_workers_, 1}),
                TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        distributed::MeshBufferConfig mesh_config = distributed::ReplicatedBufferConfig{
            .size = aligned_shard_size * static_cast<DeviceAddr>(num_workers_),
        };

        metadata_buffer_ = distributed::MeshBuffer::create(mesh_config, device_local, mesh_device_.get());
        metadata_l1_addr_ = metadata_buffer_->address();

        // Zero-init so trailing padding is deterministic; only the leading
        // metadata_size_bytes are overwritten per call.
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }

    if (cfg_.preprocessor) {
        preprocess_scratch_.assign(cfg_.global_spec.compute_packed_buffer_size_bytes(), std::byte{0});
    }

    workload_ = std::make_unique<distributed::MeshWorkload>();

    // One shared completion region backs all writer counters. The first word is
    // the host-issued transfer count; each writer pushes its committed count to
    // its assigned completed[i] slot.
    const CompletionLayout completion_layout = make_completion_layout(static_cast<uint32_t>(sockets_.size()));
    completion_shm_size_ = completion_layout.shm_size;
    completion_issued_offset_ = completion_layout.issued_offset;
    completion_completed_offset_ = completion_layout.completed_offset;
    completion_completed_stride_ = completion_layout.completed_stride;
    completion_shm_ = std::make_unique<distributed::NamedShm>(
        distributed::NamedShm::create(distributed::generate_shm_name("h2d_completion"), completion_shm_size_));
    completion_host_mem_ =
        std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(completion_shm_->ptr()), [](uint32_t*) {});
    completion_issued_ = completion_word(completion_shm_->ptr(), completion_issued_offset_);
    completion_counters_.reserve(sockets_.size());
    for (uint32_t i = 0; i < sockets_.size(); ++i) {
        completion_counters_.push_back(
            completion_word(completion_shm_->ptr(), completion_completed_offset_ + i * completion_completed_stride_));
    }
    HostBuffer completion_host_view(
        ttsl::Span<uint32_t>(completion_host_mem_.get(), static_cast<size_t>(completion_shm_size_ / sizeof(uint32_t))),
        MemoryPin(completion_host_mem_));
    distributed::MeshCoordinateRangeSet completion_coord_range;
    for (const auto& coord : coords) {
        completion_coord_range.merge(distributed::MeshCoordinateRange(coord));
    }
    completion_pinned_ = experimental::PinnedMemory::Create(
        *mesh_device_, completion_coord_range, completion_host_view, /*map_to_noc=*/true);

    uint32_t completion_index = 0;
    for (auto& s : sockets_) {
        const auto core = s->get_active_cores()[0];
        auto* d = mesh_device_->get_device(core.device_coord);
        const Buffer* dbuf = device_tensor_.mesh_buffer().get_device_buffer(core.device_coord);
        TT_FATAL(dbuf != nullptr, "H2DStreamService: device buffer missing for coord {}", core.device_coord);
        const uint32_t term_addr = static_cast<uint32_t>(termination_addrs_.at(core.device_coord));

        WorkerSyncArgs worker_sync;
        if (cfg_.worker_cores.has_value()) {
            const auto& worker_range = cfg_.worker_cores.value();
            const auto start_phys = d->worker_core_from_logical_core(worker_range.start_coord);
            const auto end_phys = d->worker_core_from_logical_core(worker_range.end_coord);
            worker_sync.enabled = true;
            worker_sync.data_ready_sem_addr = static_cast<uint32_t>(data_ready_sem_->address());
            worker_sync.consumed_counter_addr = static_cast<uint32_t>(consumed_addrs_.at(core.device_coord));
            // The writer issues both multicasts on kWriterNoc, so order the box for that NOC.
            set_worker_mcast_corners(worker_sync, kWriterNoc, start_phys, end_phys);
            worker_sync.num_workers = num_workers_;
        }

        MetadataArgs metadata;
        if (cfg_.metadata_size_bytes > 0) {
            metadata.enabled = true;
            metadata.metadata_size_bytes = cfg_.metadata_size_bytes;
            metadata.metadata_l1_addr = static_cast<uint32_t>(metadata_l1_addr_);
        }

        // Per-coord writer DRAM-completion counter: a slot in the shared host-pinned
        // completion region. The writer stages the count in local L1 before pushing it.
        CompletionArgs completion;
        {
            const auto noc_addr = completion_pinned_->get_noc_addr(d->id());
            TT_FATAL(
                noc_addr.has_value(),
                "H2DStreamService: completion counter not mapped to NOC for coord {}",
                core.device_coord);
            const uint64_t completion_counter_pcie_addr =
                noc_addr->addr + completion_completed_offset_ +
                static_cast<uint64_t>(completion_index) * completion_completed_stride_;

            const DeviceAddr src_addr = svc.allocate_l1(d, core.core_coord, sizeof(uint32_t));
            tt::tt_metal::detail::WriteToDeviceL1(d, core.core_coord, static_cast<uint32_t>(src_addr), zero_word);
            completion_src_addrs_.emplace(core.device_coord, src_addr);

            completion.pcie_xy_enc = noc_addr->pcie_xy_enc;
            completion.pcie_addr_lo = static_cast<uint32_t>(completion_counter_pcie_addr & 0xFFFFFFFFull);
            completion.pcie_addr_hi = static_cast<uint32_t>(completion_counter_pcie_addr >> 32);
            completion.src_l1_addr = static_cast<uint32_t>(src_addr);
        }

        auto program = build_persistent_h2d_program(
            *dbuf,
            core.core_coord,
            s->get_config_buffer_address(),
            term_addr,
            plan,
            tensor_page_size,
            device_tensor_.dtype(),
            worker_sync,
            metadata,
            completion);
        workload_->add_program(distributed::MeshCoordinateRange(core.device_coord), std::move(program));
        ++completion_index;
    }

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload_, /*blocking=*/false);
}

H2DStreamService::H2DStreamService(
    Config cfg,
    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
    uint32_t socket_page_size,
    uint32_t num_socket_pages,
    std::string completion_shm_name,
    uint64_t completion_shm_size,
    uint32_t completion_issued_offset,
    uint32_t completion_completed_offset,
    uint32_t completion_completed_stride) :
    is_owner_(false), cfg_(std::move(cfg)) {
    TT_FATAL(!sockets.empty(), "H2DStreamService(connector): sockets vector must not be empty");
    TT_FATAL(cfg_.mapper != nullptr, "H2DStreamService(connector): mapper must be pre-built and supplied");
    TT_FATAL(socket_page_size > 0, "H2DStreamService(connector): socket_page_size must be > 0");
    TT_FATAL(num_socket_pages > 0, "H2DStreamService(connector): num_socket_pages must be > 0");
    TT_FATAL(!completion_shm_name.empty(), "H2DStreamService(connector): completion SHM name must not be empty");
    TT_FATAL(completion_shm_size > 0, "H2DStreamService(connector): completion SHM size must be > 0");
    TT_FATAL(completion_completed_stride >= sizeof(uint32_t), "H2DStreamService(connector): invalid completion stride");

    mapper_ = std::move(cfg_.mapper);

    socket_page_size_ = socket_page_size;
    num_socket_pages_ = num_socket_pages;
    sockets_ = std::move(sockets);
    for (auto& s : sockets_) {
        s->set_page_size(socket_page_size_);
    }

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    per_shard_spec_ = distributed_dummy.tensor_spec();

    completion_shm_size_ = completion_shm_size;
    completion_issued_offset_ = completion_issued_offset;
    completion_completed_offset_ = completion_completed_offset;
    completion_completed_stride_ = completion_completed_stride;
    completion_shm_ =
        std::make_unique<distributed::NamedShm>(distributed::NamedShm::open(completion_shm_name, completion_shm_size_));
    completion_host_mem_ =
        std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(completion_shm_->ptr()), [](uint32_t*) {});
    completion_issued_ = completion_word(completion_shm_->ptr(), completion_issued_offset_);
    completion_counters_.reserve(sockets_.size());
    for (uint32_t i = 0; i < sockets_.size(); ++i) {
        completion_counters_.push_back(
            completion_word(completion_shm_->ptr(), completion_completed_offset_ + i * completion_completed_stride_));
    }

    if (cfg_.metadata_size_bytes > 0) {
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }

    if (cfg_.preprocessor) {
        preprocess_scratch_.assign(cfg_.global_spec.compute_packed_buffer_size_bytes(), std::byte{0});
    }
}

H2DStreamService::~H2DStreamService() {
    // try/catch so a teardown failure (e.g. mesh device already gone) never
    // escapes the destructor.
    try {
        if (!is_owner_) {
            // Connector owns no device-side resources; sockets free their own SHM.
            sockets_.clear();
            return;
        }

        // Drain in-flight socket acks, then flip the per-device termination signals so
        // each persistent kernel exits on its next poll. We deliberately use the
        // socket-only drain (not barrier()): wait_done() below already blocks until the
        // writer has drained the data CB and committed every page to DRAM, and a full
        // barrier() could hang here if a worker-sync wait is still outstanding at teardown.
        drain_socket_acks();
        signal_termination();

        if (mesh_device_) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        }

        // Wait for each kernel to actually return (RUN_MSG_DONE), not just for
        // dispatch to drain, or a later instance finds the service core occupied.
        auto& svc = tt::tt_metal::internal::service_core_manager();
        if (mesh_device_) {
            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.wait_done(d, core);
            }
        }

        if (mesh_device_) {
            for (const auto& [coord, addr] : termination_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            termination_addrs_.clear();

            for (const auto& [coord, addr] : consumed_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            consumed_addrs_.clear();

            for (const auto& [coord, addr] : completion_src_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            completion_src_addrs_.clear();

            // Release the writer DRAM-completion pinned host buffer while the device is still
            // alive (UMD unpin happens in PinnedMemory's dtor); unpin before unmapping SHM.
            completion_counters_.clear();
            completion_issued_ = nullptr;
            completion_pinned_.reset();
            completion_host_mem_.reset();
            if (completion_shm_) {
                completion_shm_->unlink();
                completion_shm_.reset();
            }

            // Destroy sockets before releasing the service-core claims: H2DSocket
            // dtors deallocate their own L1 and TT_FATAL if the core is already
            // released.
            sockets_.clear();

            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.release(d, {core});
            }
        }

        // Unlink + untrack the exported descriptor so it doesn't linger in
        // ShmResourceTracker until process exit.
        if (!descriptor_path_.empty()) {
            if (std::remove(descriptor_path_.c_str()) == 0 || errno == ENOENT) {
                distributed::ShmResourceTracker::instance().untrack_file(descriptor_path_);
            }
            descriptor_path_.clear();
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed with unknown exception");
    }
}

void H2DStreamService::drain_socket_acks() {
    for (auto& s : sockets_) {
        s->barrier();
    }
}

void H2DStreamService::barrier() {
    drain_socket_acks();

    // The socket ack now fires when the reader stages a page into L1 (recycling the host FIFO
    // slot early), not when the writer commits it to DRAM -- so the socket barrier alone no
    // longer guarantees the backing tensor is readable. Wait for each writer's DRAM-completion
    // counter (pushed to host pinned memory over PCIe) to catch up to the transfers issued.
    // The issued/completed counters live in shared memory so owner and connector barriers use
    // the same target. uint32_t modulo equality is valid as long as fewer than 2^32 logical
    // transfers are uncompleted at once.
    TT_FATAL(completion_issued_ != nullptr, "H2DStreamService::barrier: completion issued counter unavailable");
    tt_driver_atomics::mfence();
    const uint32_t target = *completion_issued_;
    for (auto* counter : completion_counters_) {
        while (true) {
            tt_driver_atomics::mfence();
            if (static_cast<uint32_t>(target - *counter) == 0) {
                break;
            }
        }
    }
}

void H2DStreamService::signal_termination() {
    if (mesh_device_ == nullptr) {
        return;
    }
    std::vector<uint32_t> one_word{1};
    for (const auto& [coord, addr] : termination_addrs_) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), one_word);
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

// The owner-only getters below return state for wiring up a consumer workload;
// the connector has no MeshDevice, so calling them there is a programming error.
namespace {
inline void require_owner(bool is_owner, const char* api) {
    TT_FATAL(
        is_owner,
        "{}: this getter is owner-only. The connector-mode service (built via "
        "H2DStreamService::connect) has no MeshDevice and cannot dispatch consumer "
        "workloads, so the worker-sync / metadata addresses it would return are "
        "meaningless. Call this from the owner process instead.",
        api);
}
}  // namespace

CoreRange H2DStreamService::get_worker_cores() const {
    TT_FATAL(
        cfg_.worker_cores.has_value(),
        "H2DStreamService::get_worker_cores: worker-sync was not configured (Config::worker_cores unset).");
    return *cfg_.worker_cores;
}

DeviceAddr H2DStreamService::get_data_ready_sem_addr() const {
    require_owner(is_owner_, "H2DStreamService::get_data_ready_sem_addr");
    TT_FATAL(
        data_ready_sem_.has_value(),
        "H2DStreamService::get_data_ready_sem_addr: worker-sync was not configured (Config::worker_cores unset).");
    return data_ready_sem_->address();
}

DeviceAddr H2DStreamService::get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const {
    require_owner(is_owner_, "H2DStreamService::get_consumed_counter_addr");
    auto it = consumed_addrs_.find(coord);
    TT_FATAL(
        it != consumed_addrs_.end(),
        "H2DStreamService::get_consumed_counter_addr: no consumed-counter at coord {} (worker-sync was not "
        "configured or the coord does not participate in this service).",
        coord);
    return it->second;
}

CoreCoord H2DStreamService::get_service_core(const distributed::MeshCoordinate& coord) const {
    require_owner(is_owner_, "H2DStreamService::get_service_core");
    auto it = service_cores_.find(coord);
    TT_FATAL(
        it != service_cores_.end(),
        "H2DStreamService::get_service_core: no service core claimed at coord {} (does this coord participate "
        "in this service?).",
        coord);
    return it->second;
}

DeviceAddr H2DStreamService::get_metadata_addr() const {
    require_owner(is_owner_, "H2DStreamService::get_metadata_addr");
    TT_FATAL(
        cfg_.metadata_size_bytes > 0,
        "H2DStreamService::get_metadata_addr: metadata multicast was not configured "
        "(Config::metadata_size_bytes is 0).");
    return metadata_l1_addr_;
}

const TensorSpec& H2DStreamService::get_per_shard_spec() const {
    TT_FATAL(per_shard_spec_.has_value(), "H2DStreamService::get_per_shard_spec: per-shard spec not derived");
    return *per_shard_spec_;
}

const Tensor& H2DStreamService::get_backing_tensor() const {
    require_owner(is_owner_, "H2DStreamService::get_backing_tensor");
    return device_tensor_;
}

std::size_t H2DStreamService::payload_size_bytes() const {
    return cfg_.global_spec.compute_packed_buffer_size_bytes();
}

std::size_t H2DStreamService::metadata_size_bytes() const {
    return cfg_.metadata_size_bytes;
}

std::string H2DStreamService::export_descriptor(const std::string& service_id) {
    TT_FATAL(is_owner_, "H2DStreamService::export_descriptor: only owner-side services can be exported");
    TT_FATAL(mesh_device_ != nullptr, "H2DStreamService::export_descriptor: mesh device unavailable");
    TT_FATAL(mapper_ != nullptr, "H2DStreamService::export_descriptor: mapper unavailable");

    distributed::H2DStreamServiceDescriptor desc;
    desc.global_shape = cfg_.global_spec.logical_shape();
    desc.global_dtype = cfg_.global_spec.data_type();
    desc.mesh_shape = mesh_device_->shape();
    desc.mapper_config = mapper_->config();
    desc.socket_page_size = socket_page_size_;
    desc.num_socket_pages = num_socket_pages_;
    desc.metadata_size_bytes = cfg_.metadata_size_bytes;
    desc.socket_buffer_type = cfg_.socket_buffer_type;
    desc.socket_mode = cfg_.socket_mode;
    TT_FATAL(
        completion_shm_ && completion_shm_->is_open(),
        "H2DStreamService::export_descriptor: completion SHM unavailable");
    desc.completion_shm_name = completion_shm_->name();
    desc.completion_shm_size = completion_shm_size_;
    desc.completion_issued_offset = completion_issued_offset_;
    desc.completion_completed_offset = completion_completed_offset_;
    desc.completion_completed_stride = completion_completed_stride_;

    // Embed each socket's descriptor inline so the whole service is one file;
    // avoids a visibility race between service- and socket-level descriptors.
    desc.per_coord_entries.reserve(sockets_.size());
    for (auto& s : sockets_) {
        const auto coord = s->get_active_cores()[0].device_coord;
        desc.per_coord_entries.emplace_back(coord, s->populate_descriptor());
    }

    auto path = distributed::descriptor_path_for_service(service_id);
    desc.write_to_file(path);
    distributed::ShmResourceTracker::instance().track_file(path);
    descriptor_path_ = path;
    return path;
}

std::unique_ptr<H2DStreamService> H2DStreamService::connect(
    const std::string& service_id,
    std::optional<uint32_t> timeout_ms,
    std::function<void(ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata)> preprocessor) {
    auto desc = distributed::H2DStreamServiceDescriptor::wait_and_read(
        distributed::descriptor_path_for_service(service_id), timeout_ms.value_or(10000));

    const TensorLayout tensor_layout(
        desc.global_dtype,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    TensorSpec global_spec(desc.global_shape, tensor_layout);

    auto mapper = ttnn::distributed::create_mesh_mapper(desc.mesh_shape, desc.mapper_config);

    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets;
    sockets.reserve(desc.per_coord_entries.size());
    for (const auto& [coord, socket_desc] : desc.per_coord_entries) {
        // socket_desc already carries the owner-side coord; the entry key is unused here.
        (void)coord;
        sockets.push_back(distributed::H2DSocket::connect_from_descriptor(socket_desc));
    }

    Config cfg{
        .global_spec = std::move(global_spec),
        .mapper = std::move(mapper),
        .socket_buffer_type = desc.socket_buffer_type,
        .fifo_size_bytes = 0,
        .scratch_cb_size_bytes = 0,
        .socket_mode = desc.socket_mode,
        .worker_cores = std::nullopt,
        .metadata_size_bytes = desc.metadata_size_bytes,
        // Preprocessor is process-local and not carried by the descriptor.
        .preprocessor = std::move(preprocessor),
    };

    return std::unique_ptr<H2DStreamService>(new H2DStreamService(
        std::move(cfg),
        std::move(sockets),
        desc.socket_page_size,
        desc.num_socket_pages,
        desc.completion_shm_name,
        desc.completion_shm_size,
        desc.completion_issued_offset,
        desc.completion_completed_offset,
        desc.completion_completed_stride));
}

void H2DStreamService::forward_to_tensor(
    ttsl::Span<const std::byte> bytes, ttsl::Span<const std::byte> metadata) {
    // Partial transfers aren't supported: the kernel's chunk count is baked into its CT args.
    const size_t expected = cfg_.global_spec.compute_packed_buffer_size_bytes();
    TT_FATAL(
        bytes.size() == expected,
        "H2DStreamService::forward_to_tensor: span size {} B does not match global_spec packed size {} B",
        bytes.size(),
        expected);

    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "H2DStreamService::forward_to_tensor: metadata span size {} B does not match "
        "Config::metadata_size_bytes={} (must be exact match, including 0 when metadata is disabled)",
        metadata.size(),
        cfg_.metadata_size_bytes);

    // ROW_MAJOR + default MemoryConfig is what keeps the borrowed-data wrap and
    // the mapper's zero-copy fast path engaged.
    TT_FATAL(
        cfg_.global_spec.layout() == Layout::ROW_MAJOR,
        "H2DStreamService::forward_to_tensor(span): global_spec must be ROW_MAJOR (got {}). "
        "Use the Tensor overload with a pre-distributed host tensor for other layouts.",
        cfg_.global_spec.layout());

    ttsl::Span<const std::byte> mapper_input = bytes;
    if (cfg_.preprocessor) {
        std::memcpy(preprocess_scratch_.data(), bytes.data(), bytes.size());
        cfg_.preprocessor(
            ttsl::Span<std::byte>(preprocess_scratch_.data(), preprocess_scratch_.size()),
            metadata);
        mapper_input = ttsl::Span<const std::byte>(
            preprocess_scratch_.data(), preprocess_scratch_.size());
    }

    Tensor borrowed = make_borrowed_host_tensor(mapper_input, cfg_.global_spec);
    Tensor distributed = (*mapper_)(borrowed);
    forward_to_tensor(distributed, metadata);
}

void H2DStreamService::forward_to_tensor(const Tensor& host_tensor, ttsl::Span<const std::byte> metadata) {
    TT_FATAL(
        host_tensor.storage_type() == StorageType::HOST,
        "H2DStreamService::forward_to_tensor: expected host tensor, got storage_type={}",
        host_tensor.storage_type());
    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "H2DStreamService::forward_to_tensor: metadata span size {} B does not match "
        "Config::metadata_size_bytes={} (must be exact match, including 0 when metadata is disabled)",
        metadata.size(),
        cfg_.metadata_size_bytes);

    const auto& host_mesh_tensor = host_tensor.host_storage().host_tensor();
    TT_FATAL(per_shard_spec_.has_value(), "H2DStreamService::forward_to_tensor: per-shard spec not derived");
    TT_FATAL(
        host_mesh_tensor.tensor_spec() == *per_shard_spec_,
        "H2DStreamService::forward_to_tensor: host tensor per-shard spec ({}) does not match expected "
        "per-shard spec ({}). Did you distribute the host tensor with a different mapper config?",
        host_mesh_tensor.tensor_spec(),
        *per_shard_spec_);

    const auto& dhb = host_mesh_tensor.buffer();
    const uint64_t expected_shard_bytes =
        static_cast<uint64_t>(num_socket_pages_) * static_cast<uint64_t>(socket_page_size_);

    std::vector<std::byte*> bases;
    bases.reserve(sockets_.size());

    for (auto& s : sockets_) {
        // get_active_cores() returns by value; copy the coord so it outlives the temporary.
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

    // Page-major: send page `i` to every socket before `i+1` so every kernel can progress.
    for (uint32_t i = 0; i < num_socket_pages_; ++i) {
        const size_t offset = static_cast<size_t>(i) * socket_page_size_;
        for (size_t s = 0; s < sockets_.size(); ++s) {
            sockets_[s]->write(bases[s] + offset, /*num_pages=*/1);
        }
    }

    // Trailing metadata page: the kernel multicasts the leading metadata_size_bytes
    // to every worker. Same bytes to every device.
    if (cfg_.metadata_size_bytes > 0) {
        std::memcpy(metadata_scratch_.data(), metadata.data(), metadata.size());
        for (auto& s : sockets_) {
            s->write(metadata_scratch_.data(), /*num_pages=*/1);
        }
    }

    // One logical transfer is now in flight on every socket. The shared issued counter is
    // process-visible so owner and connector barriers wait on the same completion target.
    TT_FATAL(
        completion_issued_ != nullptr, "H2DStreamService::forward_to_tensor: completion issued counter unavailable");
    *completion_issued_ = *completion_issued_ + 1;
    tt_driver_atomics::sfence();
}

}  // namespace tt::tt_metal
