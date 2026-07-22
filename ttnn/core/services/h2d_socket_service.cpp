// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/h2d_socket_service.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <unistd.h>
#include <unordered_set>

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

#include "socket_service_common.hpp"
#include "tensor/tensor_ops.hpp"
#include "tt_metal/distributed/h2d_stream_service_descriptor.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace tt::tt_metal {

namespace {

// Zero-copy wrap of caller-provided raw bytes into a host ttnn::Tensor matching `spec`.
ttnn::Tensor make_borrowed_host_tensor(ttsl::Span<const std::byte> bytes, const TensorSpec& spec) {
    auto* raw = const_cast<std::byte*>(bytes.data());
    const auto& shape = spec.logical_shape();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return ttnn::Tensor::from_borrowed_data<bfloat16>(
                ttsl::Span<bfloat16>(reinterpret_cast<bfloat16*>(raw), bytes.size() / sizeof(bfloat16)),
                shape,
                MemoryPin{});
        case DataType::FLOAT32:
            return ttnn::Tensor::from_borrowed_data<float>(
                ttsl::Span<float>(reinterpret_cast<float*>(raw), bytes.size() / sizeof(float)), shape, MemoryPin{});
        case DataType::INT32:
            return ttnn::Tensor::from_borrowed_data<int32_t>(
                ttsl::Span<int32_t>(reinterpret_cast<int32_t*>(raw), bytes.size() / sizeof(int32_t)),
                shape,
                MemoryPin{});
        case DataType::UINT8:
            return ttnn::Tensor::from_borrowed_data<uint8_t>(
                ttsl::Span<uint8_t>(reinterpret_cast<uint8_t*>(raw), bytes.size() / sizeof(uint8_t)),
                shape,
                MemoryPin{});
        case DataType::UINT16:
            return ttnn::Tensor::from_borrowed_data<uint16_t>(
                ttsl::Span<uint16_t>(reinterpret_cast<uint16_t*>(raw), bytes.size() / sizeof(uint16_t)),
                shape,
                MemoryPin{});
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return ttnn::Tensor::from_borrowed_data<uint32_t>(
                ttsl::Span<uint32_t>(reinterpret_cast<uint32_t*>(raw), bytes.size() / sizeof(uint32_t)),
                shape,
                MemoryPin{});
        case DataType::FP8_E4M3: TT_THROW("H2DStreamService: FP8_E4M3 is not supported");
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Reader/writer split: the data CB holds `slot_count` full socket pages so the reader can stage
// pages ahead while the writer drains earlier ones. The socket page is sized to a few NOC bursts
// (read coalescing); slot depth then fills the remaining service-core L1 as producer/consumer
// buffering that hides the writer's per-transfer worker-sync stall from the reader. See
// derive_h2d_chunk_plan.
constexpr uint32_t kMinDataSlots = 2;  // double-buffering floor (reader/writer overlap)

// Host-side sizing heuristics. The reader chunks each socket page by the real device NOC burst
// (BH 16K / WH 8K) internally; kNocBurstBytes is only a host granularity target, NOT device truth
// (we deliberately don't pull NOC headers into host code).
constexpr uint32_t kNocBurstBytes = 16u * 1024;
constexpr uint32_t kTargetReadBursts = 4;  // default socket-page target ~= 4 bursts (64 KB) of coalescing
constexpr uint32_t kSlotCap = 64;          // upper bound on data-CB slots ("fill L1" knob; sweep validates)
// Host FIFO depth (in socket pages) when Config::fifo_size_bytes == 0 (auto). The DEVICE_PULL FIFO is
// host pinned memory, so a generous default is cheap.
constexpr uint32_t kAutoFifoSocketPages = 8;
// The data CB (program allocator, bottom-up) and the service-core scratch (ServiceCoreManager,
// top-down) share the unreserved L1 with no cross-allocator overflow check. We compute the CB budget
// from svc.bytes_available() BEFORE constructing sockets (so the socket page is known in time to
// auto-size the FIFO), at which point nothing is allocated on the service core yet -- so this reserve
// must hold back headroom for the socket config buffer AND the post-plan scratch words (termination,
// worker-sync consumed, per-coord completion-src) plus a safety pad. Generous on purpose (>> their sum).
constexpr uint64_t kServiceScratchReserveBytes = 16u * 1024;

// H2D-specific chunk plan: like the shared ChunkPlan in socket_service_common.hpp but adds the
// reader/writer-split `slot_count`. Named distinctly so it doesn't collide with the shared one
// (which d2h still uses).
struct H2DChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
    uint32_t slot_count;        // full-page data-CB slots backing the reader/writer pipeline
};

// Usable service-core L1 for the data CB, given the service core's measured free L1
// (`free_l1_bytes` == svc.bytes_available()), after reserving the metadata CB (a bottom-up program
// allocation) and kServiceScratchReserveBytes. The caller measures free_l1_bytes BEFORE constructing
// sockets, so it's the full unreserved region -- the socket config buffer and the post-plan scratch
// words aren't allocated yet, but kServiceScratchReserveBytes is sized to cover them. Keeps the CB
// from colliding with the top-down scratch; the caller also asserts the final footprint fits. NB: use
// bytes_available (spans [DEFAULT_UNRESERVED, L1_end], where CBs actually start) rather than
// hal::get_max_worker_l1_unreserved_size(), which is keyed off KERNEL_CONFIG and overshoots by the
// kernel-config ringbuffer (CBs are placed after it, so that figure overflows L1).
uint64_t service_core_cb_l1_budget(uint64_t free_l1_bytes, uint64_t metadata_cb_bytes) {
    const uint64_t reserved = metadata_cb_bytes + kServiceScratchReserveBytes;
    TT_FATAL(
        free_l1_bytes > reserved,
        "H2DStreamService: service-core free L1 ({} B) too small for reservations ({} B)",
        free_l1_bytes,
        reserved);
    return free_l1_bytes - reserved;
}

// Size the socket page to a few NOC bursts for read coalescing (capped by `page_budget_hint`, which
// is Config::max_socket_page_size_bytes -- 0 means use the burst-derived default), then fill the
// remaining L1 with full-page slots above the kMinDataSlots overlap floor. No double-buffer
// fallback: capping the page budget at usable/kMinDataSlots yields >= kMinDataSlots slots as long as
// a single tensor page is itself <= usable/kMinDataSlots. A page that fits L1 but exceeds that (a
// very large per-shard page) falls to a single slot -- correct, but no reader/writer overlap -- and
// is warned about below. Slots are pure producer/consumer depth (they decouple the reader from the
// writer's worker-sync stall) -- help-or-neutral for throughput, at a per-transfer latency cost.
H2DChunkPlan derive_h2d_chunk_plan(
    uint32_t tensor_page_size, uint32_t tensor_num_pages, uint64_t usable_cb_l1_bytes, uint64_t page_budget_hint) {
    TT_FATAL(tensor_page_size > 0, "device_tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "device_tensor must have at least one page");
    // The only un-recoverable case: a single tensor page must physically fit in the CB L1 budget.
    TT_FATAL(
        tensor_page_size <= usable_cb_l1_bytes,
        "H2DStreamService: tensor page {} B exceeds service-core CB L1 budget {} B; "
        "use a layout with smaller pages",
        tensor_page_size,
        usable_cb_l1_bytes);
    // max_socket_page_size_bytes is an upper bound, but the socket page can't be smaller than one
    // tensor page (the indivisible chunk unit). A nonzero hint below that would silently round UP to
    // one tensor page, violating the "max" contract -- reject it instead so the bound stays honest.
    TT_FATAL(
        page_budget_hint == 0 || page_budget_hint >= tensor_page_size,
        "H2DStreamService: max_socket_page_size_bytes ({} B) must be >= the tensor page size ({} B); "
        "the socket page can't be smaller than one tensor page (pass 0 to auto-size)",
        page_budget_hint,
        tensor_page_size);

    const uint64_t burst_target = static_cast<uint64_t>(kTargetReadBursts) * kNocBurstBytes;
    const uint64_t requested = page_budget_hint > 0 ? page_budget_hint : burst_target;
    // Never let one socket page exceed usable/kMinDataSlots, so kMinDataSlots full pages always fit.
    const uint64_t page_budget = std::min<uint64_t>(requested, usable_cb_l1_bytes / kMinDataSlots);

    uint32_t pages_per_chunk = std::max<uint32_t>(1, static_cast<uint32_t>(page_budget / tensor_page_size));
    pages_per_chunk = std::min(pages_per_chunk, tensor_num_pages);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    const uint32_t socket_page_size = pages_per_chunk * tensor_page_size;

    const uint32_t slots_l1_max = static_cast<uint32_t>(usable_cb_l1_bytes / socket_page_size);  // >= 1
    const uint32_t slot_count = std::min(slots_l1_max, kSlotCap);
    if (slot_count < kMinDataSlots) {
        // Reachable only when one tensor page > usable/kMinDataSlots: correct, but no reader/writer
        // overlap (the reader's early host-FIFO-recycling ack still applies).
        log_warning(
            tt::LogOp,
            "H2DStreamService: tensor page {} B leaves only {} CB slot(s) in {} B of L1; "
            "reader/writer overlap disabled (use a smaller per-shard page to double-buffer)",
            tensor_page_size,
            slot_count,
            usable_cb_l1_bytes);
    }
    return H2DChunkPlan{
        .socket_page_size = socket_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
        .slot_count = slot_count,
    };
}

// Worker-sync CT-arg block. Populated when Config::worker_cores is set; all
// fields zero when disabled (the kernel's `if constexpr (worker_sync_enabled)`
// gate skips the block entirely).
struct H2DWorkerSyncArgs {
    bool enabled = false;
    uint32_t data_ready_sem_addr = 0;    // worker-grid L1 (mesh-wide GlobalSemaphore)
    uint32_t consumed_counter_addr = 0;  // service-core L1 (per-coord, allocated via ServiceCoreManager)
    uint32_t mcast_noc_x_start = 0;      // physical NoC bbox of worker_cores on this device
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;  // mcast destination count + sync arithmetic target
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
void set_worker_mcast_corners(H2DWorkerSyncArgs& args, NOC noc, CoreCoord start_phys, CoreCoord end_phys) {
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
    uint32_t metadata_size_bytes = 0;    // user-specified size; multicast verbatim by the writer
    uint32_t metadata_l1_addr = 0;       // worker-grid L1 (mesh-wide L1-sharded Buffer)
    uint32_t metadata_cb_page_size = 0;  // staged bytes / metadata-CB page (== align(metadata_size),
                                         // << socket_page_size: reader reads only this much L1 of the
                                         // full metadata socket page, then pops the whole page)
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
    uint32_t completed_stride = 0;  // make_completion_layout always sets this to the PCIe alignment
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
//   [5] metadata_read_size                (uint32, L1 bytes staged per metadata page; 0 if disabled)
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
    const H2DChunkPlan& plan,
    uint32_t tensor_page_size,
    DataType dtype,
    const H2DWorkerSyncArgs& worker_sync,
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

    // Metadata CB: one slot sized to the (aligned) metadata payload -- not a full socket page -- so a
    // tiny metadata multicast doesn't reserve a large socket-page slot of L1. Only created when enabled.
    if (metadata.enabled) {
        auto metadata_cb_cfg = CircularBufferConfig(metadata.metadata_cb_page_size, {{metadata_cb_index, data_format}})
                                   .set_page_size(metadata_cb_index, metadata.metadata_cb_page_size);
        CreateCircularBuffer(program, recv_core, metadata_cb_cfg);
    }

    std::vector<uint32_t> reader_ct_args = {
        plan.socket_page_size,
        plan.num_socket_pages,
        static_cast<uint32_t>(data_cb_index),
        static_cast<uint32_t>(metadata.enabled ? 1u : 0u),
        static_cast<uint32_t>(metadata_cb_index),
        metadata.metadata_cb_page_size,  // [5] metadata_read_size: L1 bytes staged per metadata page
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
    // fifo_size_bytes == 0 means auto (service sizes the host FIFO to a few socket pages); a non-zero
    // value is validated against the derived socket page size once the chunk plan is known, below.
    // max_socket_page_size_bytes is an OPTIONAL upper bound on the socket page: 0 means use the
    // burst-derived default, and slot depth is auto-sized to fill service-core L1 either way.
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

    device_tensor_ = ttnn::create_device_tensor(per_shard_spec, mesh_device_.get(), topology);
    per_shard_spec_ = device_tensor_.tensor_spec();

    // Each device may resolve a different free service core; record it per coord.
    auto& svc = tt::tt_metal::internal::service_core_manager();
    const auto& coords = topology.mesh_coords();
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(!claimable.empty(), "H2DStreamService: no claimable service core on device at coord {}", coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores_.emplace(coord, chosen);
    }

    // Every per-device buffer shares the same spec, so this buffer is representative.
    const uint32_t tensor_page_size = device_tensor_.buffer()->page_size();
    const uint32_t tensor_num_pages = device_tensor_.buffer()->num_pages();

    // Metadata CB is sized to the (aligned) metadata payload, not a full socket page: keeps L1 for
    // data slots AND makes the CB budget computable before the chunk plan (no circular dependency).
    // Aligned to max(L1, PCIe) so it satisfies both the CB and the reader's PCIe read.
    const uint32_t metadata_cb_page_size =
        cfg_.metadata_size_bytes > 0
            ? tt::align(cfg_.metadata_size_bytes, std::max(hal::get_l1_alignment(), hal::get_pcie_alignment()))
            : 0u;

    // Derive the chunk plan BEFORE constructing sockets, so socket_page_size is known in time to
    // auto-size the FIFO (which the sockets need at construction). bytes_available() here is the full
    // unreserved region -- the service core is claimed but nothing is allocated on it yet -- so the
    // socket config buffer and the post-plan scratch words are not yet subtracted; the reserve inside
    // service_core_cb_l1_budget (>> their combined size) covers them. L1 layout is uniform across
    // coords, so a representative service core suffices.
    auto* rep_device = mesh_device_->get_device(coords.front());
    const uint64_t service_core_free_l1 = svc.bytes_available(rep_device, service_cores_.at(coords.front()));
    const uint64_t usable_cb_l1 = service_core_cb_l1_budget(service_core_free_l1, metadata_cb_page_size);
    const H2DChunkPlan plan =
        derive_h2d_chunk_plan(tensor_page_size, tensor_num_pages, usable_cb_l1, cfg_.max_socket_page_size_bytes);
    socket_page_size_ = plan.socket_page_size;
    num_socket_pages_ = plan.num_socket_pages;
    slot_count_ = plan.slot_count;

    // Metadata travels as exactly one trailing socket page on the wire, and its (shrunk) CB slot
    // must also fit within one socket page.
    TT_FATAL(
        cfg_.metadata_size_bytes <= socket_page_size_ && metadata_cb_page_size <= socket_page_size_,
        "H2DStreamService: metadata_size_bytes={} (aligned CB page {}) exceeds derived socket_page_size={} "
        "(single-metadata-page constraint). Either reduce metadata or increase the per-shard page size.",
        cfg_.metadata_size_bytes,
        metadata_cb_page_size,
        socket_page_size_);

    // Belt-and-suspenders L1 guard: the data CB (program allocator, bottom-up) and the service-core
    // scratch (ServiceCoreManager, top-down) share the unreserved region with no cross-allocator check.
    const uint64_t data_cb_bytes = static_cast<uint64_t>(plan.slot_count) * plan.socket_page_size;
    TT_FATAL(
        data_cb_bytes + metadata_cb_page_size + kServiceScratchReserveBytes <= service_core_free_l1,
        "H2DStreamService: data CB {} B + metadata CB {} B + scratch reserve {} B exceeds service-core "
        "free L1 {} B",
        data_cb_bytes,
        metadata_cb_page_size,
        kServiceScratchReserveBytes,
        service_core_free_l1);

    // Host FIFO size: caller's value, or auto-sized to kAutoFifoSocketPages socket pages when 0.
    // Sizing in socket-page terms keeps the host run-ahead meaningful regardless of the derived page
    // size; the DEVICE_PULL FIFO is host pinned memory, so a generous default is cheap. Either way it
    // must hold at least one socket page (else the first push can't fit and the socket FATALs).
    const uint32_t effective_fifo_size_bytes =
        cfg_.fifo_size_bytes > 0 ? cfg_.fifo_size_bytes : kAutoFifoSocketPages * socket_page_size_;
    TT_FATAL(
        effective_fifo_size_bytes >= socket_page_size_,
        "H2DStreamService: fifo_size_bytes ({} B) must be >= the derived socket_page_size ({} B); "
        "increase fifo_size_bytes (or pass 0 to auto-size) or lower max_socket_page_size_bytes",
        effective_fifo_size_bytes,
        socket_page_size_);

    log_debug(
        tt::LogOp,
        "H2DStreamService L1: socket_page={} B, pages_per_chunk={}, num_socket_pages={}, slots={}, "
        "data_cb={} B, metadata_cb={} B, usable_for_cb={} B, fifo={} B",
        plan.socket_page_size,
        plan.pages_per_chunk,
        plan.num_socket_pages,
        plan.slot_count,
        data_cb_bytes,
        metadata_cb_page_size,
        usable_cb_l1,
        effective_fifo_size_bytes);

    // Iterate participating coords (not the full mesh shape) so replication-
    // collapsed or shape-overridden mappings stay correct.
    sockets_.reserve(coords.size());
    for (const auto& coord : coords) {
        sockets_.push_back(std::make_unique<distributed::H2DSocket>(
            mesh_device_,
            distributed::MeshCoreCoord(coord, service_cores_.at(coord)),
            cfg_.socket_buffer_type,
            effective_fifo_size_bytes,
            // DEVICE_PULL only: the persistent reader pulls each socket page from host pinned memory over
            // PCIe; it has no local-L1 / HOST_PUSH read path. Not exposed as a Config knob.
            distributed::H2DMode::DEVICE_PULL));
    }

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

        // Per-coord worker-sync args. Populated only when cfg.worker_cores is
        // set; otherwise everything stays zero and the kernel skips the sync
        // block via `if constexpr (worker_sync_enabled == 0)`.
        H2DWorkerSyncArgs worker_sync;
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
            metadata.metadata_cb_page_size = metadata_cb_page_size;
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
    start_host_push_workers();
}

H2DStreamService::H2DStreamService(
    Config cfg,
    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
    uint32_t socket_page_size,
    uint32_t num_socket_pages,
    const std::string& completion_shm_name,
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

    start_host_push_workers();
}

H2DStreamService::~H2DStreamService() {
    // try/catch so a teardown failure (e.g. mesh device already gone) never
    // escapes the destructor.
    try {
        stop_host_push_workers();

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

size_t H2DStreamService::effective_host_push_worker_count() const {
    if (sockets_.size() <= 1 || !cfg_.parallel_host_push) {
        return 0;
    }
    if (cfg_.host_push_thread_count == 0) {
        return std::min<size_t>(kAutoHostPushThreadCount, sockets_.size());
    }
    if (cfg_.host_push_thread_count <= 1) {
        return 0;
    }
    return std::min<size_t>(cfg_.host_push_thread_count, sockets_.size());
}

void H2DStreamService::start_host_push_workers() {
    const size_t worker_count = effective_host_push_worker_count();
    if (worker_count == 0 || !host_push_workers_.empty()) {
        return;
    }

    host_push_worker_states_.reserve(worker_count);
    host_push_workers_.reserve(worker_count);
    for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        auto state = std::make_unique<HostPushWorkerState>();
        state->socket_begin = (worker_index * sockets_.size()) / worker_count;
        state->socket_end = ((worker_index + 1) * sockets_.size()) / worker_count;
        TT_FATAL(
            state->socket_begin < state->socket_end,
            "H2DStreamService: host push worker {} got an empty socket range [{}, {})",
            worker_index,
            state->socket_begin,
            state->socket_end);
        host_push_worker_states_.push_back(std::move(state));
        host_push_workers_.emplace_back([this, worker_index]() { host_push_worker_loop(worker_index); });
    }
}

void H2DStreamService::stop_host_push_workers() {
    for (auto& state_ptr : host_push_worker_states_) {
        auto& state = *state_ptr;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.job = HostPushJobKind::Stop;
            state.done = false;
        }
        state.cv.notify_one();
    }

    for (auto& worker : host_push_workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    host_push_workers_.clear();
    host_push_worker_states_.clear();
}

void H2DStreamService::write_payload_with_host_push_workers(const std::vector<std::byte*>& bases) {
    TT_FATAL(
        bases.size() == sockets_.size(),
        "H2DStreamService::write_payload_with_host_push_workers: bases size {} does not match sockets size {}",
        bases.size(),
        sockets_.size());

    for (size_t worker_index = 0; worker_index < host_push_worker_states_.size(); ++worker_index) {
        submit_host_push_job(worker_index, HostPushJobKind::Payload, &bases);
    }
    wait_host_push_jobs();
}

void H2DStreamService::write_metadata_with_host_push_workers() {
    TT_FATAL(
        !host_push_worker_states_.empty(),
        "H2DStreamService::write_metadata_with_host_push_workers: no host push workers");

    for (size_t worker_index = 0; worker_index < host_push_worker_states_.size(); ++worker_index) {
        submit_host_push_job(worker_index, HostPushJobKind::Metadata);
    }
    wait_host_push_jobs();
}

void H2DStreamService::submit_host_push_job(
    size_t worker_index, HostPushJobKind job, const std::vector<std::byte*>* payload_bases) {
    TT_FATAL(
        worker_index < host_push_worker_states_.size(),
        "H2DStreamService::submit_host_push_job: worker index {} out of range",
        worker_index);
    TT_FATAL(
        job == HostPushJobKind::Payload || job == HostPushJobKind::Metadata || job == HostPushJobKind::Stop,
        "H2DStreamService::submit_host_push_job: invalid job {}",
        static_cast<int>(job));
    TT_FATAL(
        job != HostPushJobKind::Payload || payload_bases != nullptr,
        "H2DStreamService::submit_host_push_job: payload job requires payload bases");

    auto& state = *host_push_worker_states_[worker_index];
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        TT_FATAL(
            state.done && state.job == HostPushJobKind::None,
            "H2DStreamService::submit_host_push_job: worker {} is still busy",
            worker_index);
        state.payload_bases = payload_bases;
        state.error = nullptr;
        state.done = false;
        state.job = job;
    }
    state.cv.notify_one();
}

void H2DStreamService::wait_host_push_jobs() {
    std::exception_ptr first_error;
    for (auto& state_ptr : host_push_worker_states_) {
        auto& state = *state_ptr;
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cv.wait(lock, [&state]() { return state.done; });
        if (state.error != nullptr && first_error == nullptr) {
            first_error = state.error;
        }
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

void H2DStreamService::host_push_worker_loop(size_t worker_index) {
    auto& state = *host_push_worker_states_[worker_index];

    while (true) {
        HostPushJobKind job = HostPushJobKind::None;
        const std::vector<std::byte*>* payload_bases = nullptr;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&state]() { return state.job != HostPushJobKind::None; });
            job = state.job;
            payload_bases = state.payload_bases;
            state.job = HostPushJobKind::None;
        }

        if (job == HostPushJobKind::Stop) {
            break;
        }

        std::exception_ptr error;
        try {
            if (job == HostPushJobKind::Metadata) {
                for (size_t socket_index = state.socket_begin; socket_index < state.socket_end; ++socket_index) {
                    sockets_[socket_index]->write(metadata_scratch_.data(), /*num_pages=*/1);
                }
            } else {
                TT_FATAL(
                    payload_bases != nullptr,
                    "H2DStreamService::host_push_worker_loop: payload job missing payload bases");
                for (uint32_t i = 0; i < num_socket_pages_; ++i) {
                    const size_t offset = static_cast<size_t>(i) * socket_page_size_;
                    for (size_t socket_index = state.socket_begin; socket_index < state.socket_end; ++socket_index) {
                        sockets_[socket_index]->write((*payload_bases)[socket_index] + offset, /*num_pages=*/1);
                    }
                }
            }
        } catch (...) {
            error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.payload_bases = nullptr;
            state.error = error;
            state.done = true;
        }
        state.cv.notify_one();
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

const ttnn::Tensor& H2DStreamService::get_backing_tensor() const {
    require_owner(is_owner_, "H2DStreamService::get_backing_tensor");
    return device_tensor_;
}

std::size_t H2DStreamService::payload_size_bytes() const { return cfg_.global_spec.compute_packed_buffer_size_bytes(); }

std::size_t H2DStreamService::metadata_size_bytes() const { return cfg_.metadata_size_bytes; }

uint32_t H2DStreamService::get_slot_count() const {
    require_owner(is_owner_, "H2DStreamService::get_slot_count");
    return slot_count_;
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
    std::function<void(ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata)> preprocessor,
    bool parallel_host_push,
    uint32_t host_push_thread_count) {
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
        .max_socket_page_size_bytes = 0,
        .worker_cores = std::nullopt,
        .metadata_size_bytes = desc.metadata_size_bytes,
        // Preprocessor and host push threading are process-local; not carried by the descriptor.
        .preprocessor = std::move(preprocessor),
        .parallel_host_push = parallel_host_push,
        .host_push_thread_count = host_push_thread_count,
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

void H2DStreamService::forward_to_tensor(ttsl::Span<const std::byte> bytes, ttsl::Span<const std::byte> metadata) {
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
        "Use the ttnn::Tensor overload with a pre-distributed host tensor for other layouts.",
        cfg_.global_spec.layout());

    ttsl::Span<const std::byte> mapper_input = bytes;
    if (cfg_.preprocessor) {
        std::memcpy(preprocess_scratch_.data(), bytes.data(), bytes.size());
        cfg_.preprocessor(ttsl::Span<std::byte>(preprocess_scratch_.data(), preprocess_scratch_.size()), metadata);
        mapper_input = ttsl::Span<const std::byte>(preprocess_scratch_.data(), preprocess_scratch_.size());
    }

    ttnn::Tensor borrowed = make_borrowed_host_tensor(mapper_input, cfg_.global_spec);
    ttnn::Tensor distributed = (*mapper_)(borrowed);
    forward_to_tensor(distributed, metadata);
}

void H2DStreamService::forward_to_tensor(const ttnn::Tensor& host_tensor, ttsl::Span<const std::byte> metadata) {
    TT_FATAL(
        host_tensor.storage_type() == ttnn::StorageType::HOST,
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

    if (!host_push_worker_states_.empty()) {
        write_payload_with_host_push_workers(bases);
    } else {
        // Page-major: send page `i` to every socket before `i+1` so every kernel can progress.
        for (uint32_t i = 0; i < num_socket_pages_; ++i) {
            const size_t offset = static_cast<size_t>(i) * socket_page_size_;
            for (size_t s = 0; s < sockets_.size(); ++s) {
                sockets_[s]->write(bases[s] + offset, /*num_pages=*/1);
            }
        }
    }

    // Trailing metadata page: the kernel multicasts the leading metadata_size_bytes
    // to every worker. Same bytes to every device.
    if (cfg_.metadata_size_bytes > 0) {
        std::memcpy(metadata_scratch_.data(), metadata.data(), metadata.size());
        if (!host_push_worker_states_.empty()) {
            write_metadata_with_host_push_workers();
        } else {
            for (auto& s : sockets_) {
                s->write(metadata_scratch_.data(), /*num_pages=*/1);
            }
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
