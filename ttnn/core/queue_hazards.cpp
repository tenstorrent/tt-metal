// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/queue_hazards.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/kernel_cores.hpp>
#include <tt-metalium/experimental/queue_fence.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/worker_cores.hpp"

namespace ttnn::queue_hazards {

namespace {

using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::SubDeviceId;
using tt::tt_metal::distributed::MeshCommandQueue;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshEvent;
using tt::tt_metal::experimental::QueueFence;

constexpr uint32_t kMaxQueues = 2;
constexpr uint32_t kMaxSubDevices = 16;
// Each queue has a lane per sub-device for its programs, and one for its host transfers.
constexpr uint32_t kTransferLane = kMaxSubDevices;
constexpr uint32_t kLanes = kMaxSubDevices + 1;
constexpr uint32_t kStreams = kMaxQueues * kLanes;
// Bounds on the history that carries knowledge through waits. Forgetting can only add waits or refusals.
constexpr std::size_t kMaxVersions = 4096;
constexpr std::size_t kMaxEvents = 4096;

struct Stream {
    uint32_t queue = 0;
    uint32_t lane = 0;

    uint32_t index() const { return (queue * kLanes) + lane; }
    bool transfers() const { return lane == kTransferLane; }
};

std::string describe(Stream stream) {
    return stream.transfers() ? fmt::format("command queue {}'s host transfers", stream.queue)
                              : fmt::format("command queue {} on sub-device {}", stream.queue, stream.lane);
}

// Experiment: TTNN_QUEUE_ORDERING_LOG=<file> writes a line per launch, per host transfer, per earlier access a launch
// or transfer is ordered after (an edge), and per wait it enqueues.
std::FILE* log_file() {
    static std::FILE* file = [] {
        const char* path = std::getenv("TTNN_QUEUE_ORDERING_LOG");
        std::FILE* opened = path == nullptr ? nullptr : std::fopen(path, "w");
        if (opened != nullptr) {
            std::setvbuf(opened, nullptr, _IOLBF, 0);
        }
        return opened;
    }();
    return file;
}

std::string short_name(Stream stream) {
    return stream.transfers() ? fmt::format("cq{}/host", stream.queue)
                              : fmt::format("cq{}/sd{}", stream.queue, stream.lane);
}

// A point in a stream: a fence's (epoch, count) for programs, a sequence number for transfers. The zero point precedes
// everything.
struct Point {
    uint64_t epoch = 0;
    uint64_t count = 0;
    auto operator<=>(const Point&) const = default;
};

// For each stream, the last point known to be done.
using Clock = std::array<Point, kStreams>;

void merge(Clock& into, const Clock& from) {
    for (uint32_t i = 0; i < kStreams; ++i) {
        into[i] = std::max(into[i], from[i]);
    }
}

struct Access {
    Stream stream;
    Point point;
    std::string_view operation;  // static storage
    std::size_t buffer = 0;      // Buffer::unique_id: another value at the same address is another tenant
    uint64_t version = 0;        // what the stream's queue had waited for when the access was enqueued
};

struct Region {
    uint64_t end = 0;  // exclusive
    std::optional<Access> last_write;
    std::vector<Access> reads;  // since the last write, the latest per stream
};

struct QueueState {
    // Everything this queue has waited for, which all it enqueues from now on follows. Earlier versions are kept so an
    // access can name what its queue had waited for when it was enqueued.
    std::deque<Clock> versions{Clock{}};
    uint64_t first_version = 0;
    uint64_t transfers = 0;  // host transfers so far

    uint64_t version() const { return first_version + versions.size() - 1; }
    const Clock& waited() const { return versions.back(); }
    const Clock* at(uint64_t version) const {
        return version >= first_version ? &versions[version - first_version] : nullptr;
    }
    void wait(const Clock& clock) {
        Clock next = versions.back();
        merge(next, clock);
        if (next != versions.back()) {
            versions.push_back(next);
            if (versions.size() > kMaxVersions) {
                versions.pop_front();
                ++first_version;
            }
        }
    }
};

// The latest full collective on a group and stream: the stream's point before it, and what its queue had waited for
// then. What the host has seen finish is left out: blocking reads and event synchronizes can differ between the hosts
// of a multi-host mesh, and every rank must decide a handshake the same way.
struct Rendezvous {
    Point start;
    Clock followed;
};

struct MeshState {
    std::array<QueueState, kMaxQueues> queues;
    std::map<std::pair<int64_t, uint32_t>, Rendezvous> rendezvous;  // by (group, stream)
    // The latest collective per group. Collectives from two streams over the same fabric links at once deadlock
    // (measured on one axis), so each waits for the previous one on its links when that ran on another stream. A
    // collective on a mesh axis uses that axis's links; one across the whole mesh (group -1) uses every axis's.
    std::map<int64_t, Access> last_collective;
    Clock host{};  // what the host has seen finish: everything enqueued from now on follows it
    // (queue, event id) -> the event and what it covers. A wait for a point on another queue uses the earliest event
    // recorded at or after it: one recorded when the wait is needed sits behind everything that queue was given since.
    struct RecordedEvent {
        Clock covered;
        MeshEvent event;
    };
    std::map<std::pair<uint32_t, uint32_t>, RecordedEvent> events;
    std::map<tt::tt_metal::BufferType, std::map<uint64_t, Region>> regions;  // by memory space, by start address

    // Whether an access by `stream`, enqueued now, follows `earlier`.
    bool follows(Stream stream, const Access& earlier, std::span<const SubDeviceId> stall_group) const {
        if (earlier.stream.index() == stream.index()) {
            return true;
        }
        if (earlier.stream.queue == stream.queue) {
            // A queue runs its programs after its earlier transfers, and its transfers after its earlier programs on
            // the stall group.
            if (earlier.stream.transfers()) {
                return true;
            }
            if (stream.transfers() &&
                std::ranges::any_of(stall_group, [&](SubDeviceId id) { return *id == earlier.stream.lane; })) {
                return true;
            }
        }
        const uint32_t i = earlier.stream.index();
        return std::max(host[i], queues[stream.queue].waited()[i]) >= earlier.point;
    }

    // What is done once `access` is: the access and everything its queue had waited for before it.
    Clock implied_by(const Access& access) const {
        Clock clock{};
        if (const Clock* waited = queues[access.stream.queue].at(access.version)) {
            clock = *waited;
        }
        auto& point = clock[access.stream.index()];
        point = std::max(point, access.point);
        return clock;
    }
};

std::mutex mutex;
std::unordered_map<int, MeshState> states;

std::atomic<Mode>& mode_flag() {
    static std::atomic<Mode> flag = [] {
        const char* value = std::getenv("TTNN_CHECK_QUEUE_HAZARDS");
        if (value == nullptr || std::string_view(value) == "0") {
            return Mode::off;
        }
        return std::string_view(value) == "order" ? Mode::order : Mode::check;
    }();
    return flag;
}

struct Range {
    tt::tt_metal::BufferType type;
    uint64_t begin;
    uint64_t end;
    std::size_t buffer;
};

std::optional<Range> range_of(const Tensor& tensor) {
    if (tensor.storage_type() != StorageType::DEVICE || !tensor.is_allocated()) {
        return std::nullopt;
    }
    const auto* buffer = tensor.buffer();
    if (buffer == nullptr || buffer->size() == 0) {
        return std::nullopt;
    }
    // Interleaved and sharded buffers take the same address range in every bank they use.
    return Range{
        buffer->buffer_type(),
        buffer->address(),
        buffer->address() + buffer->aligned_size_per_bank(),
        buffer->unique_id()};
}

template <typename Fn>
void for_each_overlap(const std::map<uint64_t, Region>& regions, const Range& range, Fn&& fn) {
    auto it = regions.upper_bound(range.begin);
    if (it != regions.begin()) {
        --it;
    }
    for (; it != regions.end() && it->first < range.end; ++it) {
        if (it->second.end > range.begin) {
            fn(it->second);
        }
    }
}

// Splits the region that straddles `at`, if any, so that a region starts there.
void split(std::map<uint64_t, Region>& regions, uint64_t at) {
    auto it = regions.upper_bound(at);
    if (it == regions.begin()) {
        return;
    }
    --it;
    if (it->first < at && it->second.end > at) {
        Region upper = it->second;
        it->second.end = at;
        regions.emplace(at, std::move(upper));
    }
}

void record_read(std::map<uint64_t, Region>& regions, const Range& range, const Access& access) {
    split(regions, range.begin);
    split(regions, range.end);
    std::vector<std::pair<uint64_t, uint64_t>> gaps;
    uint64_t next = range.begin;
    for (auto it = regions.lower_bound(range.begin); it != regions.end() && it->first < range.end; ++it) {
        if (it->first > next) {
            gaps.emplace_back(next, it->first);
        }
        auto& reads = it->second.reads;
        std::erase_if(reads, [&](const Access& read) { return read.stream.index() == access.stream.index(); });
        reads.push_back(access);
        next = it->second.end;
    }
    if (next < range.end) {
        gaps.emplace_back(next, range.end);
    }
    for (const auto& [begin, end] : gaps) {
        regions.emplace(begin, Region{.end = end, .reads = {access}});
    }
}

void record_write(std::map<uint64_t, Region>& regions, const Range& range, const Access& access) {
    split(regions, range.begin);
    split(regions, range.end);
    regions.erase(regions.lower_bound(range.begin), regions.lower_bound(range.end));
    regions.emplace(range.begin, Region{.end = range.end, .last_write = access});
}

// The sub-device a workload runs on: the one whose cores hold its kernels (metal refuses workloads that span several).
SubDeviceId sub_device_of(const MeshDevice& device, const tt::tt_metal::distributed::MeshWorkload& workload) {
    const auto& ids = device.get_sub_device_ids();
    if (ids.size() == 1) {
        return ids.front();
    }
    for (const auto& [range, program] : workload.get_programs()) {
        const auto tensix = tt::tt_metal::experimental::GetKernelCores(program, HalProgrammableCoreType::TENSIX);
        const auto ethernet = tt::tt_metal::experimental::GetKernelCores(program, HalProgrammableCoreType::ACTIVE_ETH);
        if (tensix.empty() && ethernet.empty()) {
            continue;
        }
        for (const SubDeviceId id : ids) {
            if (device.worker_cores(HalProgrammableCoreType::TENSIX, id).contains(tensix) &&
                device.worker_cores(HalProgrammableCoreType::ACTIVE_ETH, id).contains(ethernet)) {
                return id;
            }
        }
    }
    return ids.front();
}

// The sub-device of the thread's worker-cores scope, if there is one and it lies in a single sub-device.
std::optional<SubDeviceId> scoped_sub_device(const MeshDevice& device) {
    const auto cores = ttnn::get_scoped_worker_cores(device);
    if (!cores.has_value()) {
        return device.get_sub_device_ids().size() == 1 ? std::optional(device.get_sub_device_ids().front())
                                                       : std::nullopt;
    }
    for (const SubDeviceId id : device.get_sub_device_ids()) {
        if (device.worker_cores(HalProgrammableCoreType::TENSIX, id).contains(*cores)) {
            return id;
        }
    }
    return std::nullopt;
}

Point program_point(MeshCommandQueue& cq, SubDeviceId sub_device) {
    const QueueFence fence = tt::tt_metal::experimental::GetQueueFence(cq, sub_device);
    return Point{fence.epoch, fence.count};
}

// What `queue` has done once everything it enqueued so far on `sub_device_ids` (empty: the stall group), and its
// transfers, are done.
Clock queue_progress(
    MeshState& state, MeshDevice& device, uint32_t queue, std::span<const SubDeviceId> sub_device_ids) {
    auto& cq = device.mesh_command_queue(static_cast<uint8_t>(queue));
    Clock clock = state.queues[queue].waited();
    const auto& ids =
        sub_device_ids.empty() ? std::span<const SubDeviceId>(device.get_sub_device_stall_group()) : sub_device_ids;
    for (const SubDeviceId id : ids) {
        clock[Stream{queue, *id}.index()] = program_point(cq, id);
    }
    clock[Stream{queue, kTransferLane}.index()] = Point{0, state.queues[queue].transfers};
    return clock;
}

[[noreturn]] void refuse(
    std::string_view operation,
    Stream stream,
    bool writes,
    const Range& range,
    const Access& earlier,
    bool earlier_writes) {
    TT_THROW(
        "{} on {} {} {} memory [{:#x}, {:#x}) that {} on {} {}, and nothing orders the two. Call ttnn.wait_for on the "
        "tensor first, have command queue {} wait for an event recorded after {}, or synchronize.",
        operation,
        describe(stream),
        writes ? "writes" : "reads",
        enchantum::to_string(range.type),
        range.begin,
        range.end,
        earlier.operation,
        describe(earlier.stream),
        earlier_writes ? "wrote" : "may still be reading",
        stream.queue,
        earlier.operation);
}

// Makes `stream`'s queue wait for each earlier access in `earlier`, the latest per stream, with fences.
void wait_for_accesses(MeshState& state, MeshDevice& device, Stream stream, const std::vector<const Access*>& earlier) {
    std::map<uint32_t, const Access*> latest;  // per stream
    for (const Access* access : earlier) {
        auto& slot = latest[access->stream.index()];
        if (slot == nullptr || slot->point < access->point) {
            slot = access;
        }
    }
    auto& cq = device.mesh_command_queue(static_cast<uint8_t>(stream.queue));
    Clock implied{};
    for (const auto& [index, access] : latest) {
        const Stream other = access->stream;
        QueueFence fence{.cq_id = other.queue};
        if (!other.transfers()) {
            fence.sub_device_id = SubDeviceId{static_cast<uint8_t>(other.lane)};
            fence.epoch = access->point.epoch;
            fence.count = static_cast<uint32_t>(access->point.count);
        }
        const MeshState::RecordedEvent* recorded = nullptr;
        if (other.queue != stream.queue) {
            for (auto it = state.events.lower_bound({other.queue, 0});
                 it != state.events.end() && it->first.first == other.queue;
                 ++it) {
                if (it->second.covered[index] >= access->point) {
                    recorded = &it->second;
                    break;
                }
            }
        }
        if (recorded != nullptr) {
            cq.enqueue_wait_for_event(recorded->event);
            merge(implied, recorded->covered);
        } else {
            tt::tt_metal::experimental::EnqueueWaitForQueueFence(cq, fence);
        }
        if (std::FILE* log = log_file()) {
            fmt::print(
                log,
                "wait\t{}\t{}\t{}\n",
                short_name(stream),
                short_name(other),
                recorded != nullptr ? "event" : "fence");
        }
        merge(implied, state.implied_by(*access));
    }
    state.queues[stream.queue].wait(implied);
}

struct Footprint {
    std::vector<Range> reads;
    std::vector<Range> writes;
};

Footprint footprint_of(std::span<const Tensor* const> reads, std::span<const Tensor* const> writes) {
    Footprint footprint;
    for (const Tensor* tensor : writes) {
        if (auto range = range_of(*tensor)) {
            footprint.writes.push_back(*range);
        }
    }
    for (const Tensor* tensor : reads) {
        auto range = range_of(*tensor);
        // An argument the operation also returns, as with preallocated outputs, is written.
        if (range.has_value() && std::ranges::none_of(footprint.writes, [&](const Range& written) {
                return written.type == range->type && written.begin == range->begin && written.end == range->end;
            })) {
            footprint.reads.push_back(*range);
        }
    }
    return footprint;
}

// Orders `footprint`, about to be accessed by `stream`, after every earlier access on another stream it races with:
// waits where the memory changed tenants or the mode orders, and refuses otherwise.
void order(
    MeshState& state, MeshDevice& device, Stream stream, std::string_view operation, const Footprint& footprint) {
    const auto& stall_group = device.get_sub_device_stall_group();
    std::vector<const Access*> waits;
    auto resolve = [&](const Range& range, bool writes, const Access& earlier, bool earlier_writes) {
        if (state.follows(stream, earlier, stall_group)) {
            return;
        }
        // Experiment (diagnostic, unsafe): TTNN_QUEUE_ORDERING_SKIP_REUSE=1 leaves memory reuse unordered.
        if (earlier.buffer != range.buffer && std::getenv("TTNN_QUEUE_ORDERING_SKIP_REUSE") != nullptr) {
            return;
        }
        if (std::FILE* log = log_file()) {
            fmt::print(
                log,
                "edge\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                operation,
                short_name(stream),
                writes ? "writes" : "reads",
                earlier.operation,
                short_name(earlier.stream),
                earlier_writes ? "wrote" : "read",
                earlier.buffer == range.buffer ? "same" : "reuse",
                enchantum::to_string(range.type));
        }
        if (earlier.buffer == range.buffer && mode_flag().load(std::memory_order_relaxed) != Mode::order) {
            refuse(operation, stream, writes, range, earlier, earlier_writes);
        }
        waits.push_back(&earlier);
    };
    for (bool writes : {false, true}) {
        for (const Range& range : writes ? footprint.writes : footprint.reads) {
            for_each_overlap(state.regions[range.type], range, [&](const Region& region) {
                if (region.last_write.has_value()) {
                    resolve(range, writes, *region.last_write, true);
                }
                if (writes) {
                    for (const Access& read : region.reads) {
                        resolve(range, writes, read, false);
                    }
                }
            });
        }
    }
    if (!waits.empty()) {
        wait_for_accesses(state, device, stream, waits);
    }
}

void record(MeshState& state, const Footprint& footprint, Access access) {
    for (const Range& range : footprint.reads) {
        access.buffer = range.buffer;
        record_read(state.regions[range.type], range, access);
    }
    for (const Range& range : footprint.writes) {
        access.buffer = range.buffer;
        record_write(state.regions[range.type], range, access);
    }
}

bool capturing(MeshCommandQueue& cq) { return cq.trace_id().has_value(); }

int64_t group_key(std::optional<uint32_t> group) { return group.has_value() ? static_cast<int64_t>(*group) : -1; }

}  // namespace

Mode mode() { return mode_flag().load(std::memory_order_relaxed); }

void set_mode(Mode mode) {
    std::lock_guard lock(mutex);
    mode_flag().store(mode, std::memory_order_relaxed);
    states.clear();
}

void launch(
    MeshDevice& device,
    const tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation,
    std::span<const Tensor* const> reads,
    std::span<const Tensor* const> writes,
    const std::function<void()>& enqueue) {
    auto& cq = device.mesh_command_queue();
    // Programs captured into a trace run when the trace is replayed, which this does not track.
    if (capturing(cq)) {
        enqueue();
        return;
    }
    TT_FATAL(cq.id() < kMaxQueues, "Command queue {} out of range", cq.id());
    std::lock_guard lock(mutex);
    auto& state = states[device.id()];
    const SubDeviceId sub_device = sub_device_of(device, workload);
    const Stream stream{cq.id(), *sub_device};
    const Footprint footprint = footprint_of(reads, writes);
    if (std::FILE* log = log_file()) {
        fmt::print(log, "launch\t{}\t{}\n", operation, short_name(stream));
    }
    order(state, device, stream, operation, footprint);
    enqueue();
    record(
        state,
        footprint,
        Access{
            .stream = stream,
            .point = program_point(cq, sub_device),
            .operation = operation,
            .version = state.queues[stream.queue].version()});
}

void transfer(
    MeshDevice& device,
    QueueId queue,
    std::string_view what,
    const Tensor& tensor,
    bool writes,
    bool blocking,
    const std::function<void()>& run) {
    auto& cq = device.mesh_command_queue(static_cast<uint8_t>(queue.get()));
    if (capturing(cq)) {
        run();
        return;
    }
    TT_FATAL(cq.id() < kMaxQueues, "Command queue {} out of range", cq.id());
    std::lock_guard lock(mutex);
    auto& state = states[device.id()];
    const Stream stream{cq.id(), kTransferLane};
    const Tensor* const tensors[] = {&tensor};
    const Footprint footprint =
        writes ? footprint_of({}, tensors) : footprint_of(tensors, std::span<const Tensor* const>{});
    if (std::FILE* log = log_file()) {
        fmt::print(log, "transfer\t{}\t{}\n", what, short_name(stream));
    }
    order(state, device, stream, what, footprint);
    run();
    auto& queue_state = state.queues[stream.queue];
    record(
        state,
        footprint,
        Access{
            .stream = stream,
            .point = Point{0, ++queue_state.transfers},
            .operation = what,
            .version = queue_state.version()});
    if (blocking && !writes) {
        merge(state.host, queue_progress(state, device, stream.queue, {}));
    }
}

void wait_for(const Tensor& tensor, QueueId queue) {
    auto* device = tensor.device();
    TT_FATAL(device != nullptr, "ttnn.wait_for needs a device tensor");
    if (!enabled()) {
        return;
    }
    auto range = range_of(tensor);
    if (!range.has_value()) {
        return;
    }
    std::lock_guard lock(mutex);
    auto& state = states[device->id()];
    // Within the thread's sub-device scope, the queue's own earlier programs on that sub-device come first anyway.
    // Outside one, the next access may run on any sub-device, so the queue's transfer lane stands for the queue: its
    // own transfers come first, its programs do not.
    const auto sub_device = scoped_sub_device(*device);
    const Stream stream{queue.get(), sub_device.has_value() ? static_cast<uint32_t>(**sub_device) : kTransferLane};
    std::vector<const Access*> waits;
    auto consider = [&](const Access& earlier) {
        if (!state.follows(stream, earlier, {})) {
            waits.push_back(&earlier);
        }
    };
    for_each_overlap(state.regions[range->type], *range, [&](const Region& region) {
        if (region.last_write.has_value()) {
            consider(*region.last_write);
        }
        for (const Access& read : region.reads) {
            consider(read);
        }
    });
    if (!waits.empty()) {
        wait_for_accesses(state, *device, stream, waits);
    }
}

bool peers_may_write_without_handshake(
    MeshDevice& device,
    QueueId queue,
    SubDeviceId sub_device,
    std::optional<uint32_t> group,
    std::span<const Tensor* const> targets) {
    // A captured collective runs at every replay, after whatever ran in between, which this does not track.
    if (!enabled() || capturing(device.mesh_command_queue(static_cast<uint8_t>(queue.get())))) {
        return false;
    }
    std::lock_guard lock(mutex);
    auto& state = states[device.id()];
    const Stream stream{queue.get(), *sub_device};
    const auto last = state.rendezvous.find({group_key(group), stream.index()});
    if (last == state.rendezvous.end()) {
        return false;
    }
    const Rendezvous& rendezvous = last->second;
    // On every member, an access on this stream before the rendezvous ran before it; an access on another stream did if
    // this stream had followed it by then.
    auto before = [&](const Access& access) {
        return access.stream.index() == stream.index() ? access.point <= rendezvous.start
                                                       : rendezvous.followed[access.stream.index()] >= access.point;
    };
    bool proven = true;
    for (const Tensor* target : targets) {
        const auto range = range_of(*target);
        if (!range.has_value()) {
            continue;
        }
        for_each_overlap(state.regions[range->type], *range, [&](const Region& region) {
            proven = proven && (!region.last_write.has_value() || before(*region.last_write)) &&
                     std::ranges::all_of(region.reads, before);
        });
    }
    return proven;
}

void full_collective(
    MeshDevice& device,
    QueueId queue,
    SubDeviceId sub_device,
    std::optional<uint32_t> group,
    const std::function<void()>& launch) {
    auto& cq = device.mesh_command_queue(static_cast<uint8_t>(queue.get()));
    if (!enabled() || capturing(cq)) {
        launch();
        return;
    }
    const Stream stream{queue.get(), *sub_device};
    Rendezvous rendezvous;
    {
        std::lock_guard lock(mutex);
        auto& state = states[device.id()];
        std::vector<const Access*> sharing_links;
        for (const auto& [other_group, access] : state.last_collective) {
            const bool shares_links = other_group == group_key(group) || other_group == -1 || group_key(group) == -1;
            if (shares_links && !state.follows(stream, access, {})) {
                sharing_links.push_back(&access);
            }
        }
        if (!sharing_links.empty()) {
            wait_for_accesses(state, device, stream, sharing_links);
        }
        rendezvous.start = program_point(cq, sub_device);
        rendezvous.followed = state.queues[stream.queue].waited();
        // The queue runs the collective after its earlier transfers.
        auto& transfers = rendezvous.followed[Stream{stream.queue, kTransferLane}.index()];
        transfers = std::max(transfers, Point{0, state.queues[stream.queue].transfers});
    }
    launch();
    std::lock_guard lock(mutex);
    auto& state = states[device.id()];
    state.rendezvous[{group_key(group), stream.index()}] = rendezvous;
    state.last_collective[group_key(group)] = Access{
        .stream = stream,
        .point = program_point(cq, sub_device),
        .operation = "a collective",
        .version = state.queues[stream.queue].version()};
}

void on_record_event(const MeshEvent& event, std::span<const SubDeviceId> sub_device_ids) {
    auto* device = event.device();
    // An event over part of the mesh does not order the rest of it, and this tracks the mesh as a whole.
    if (event.device_range() != tt::tt_metal::distributed::MeshCoordinateRange(device->shape())) {
        return;
    }
    std::lock_guard lock(mutex);
    auto& state = states[device->id()];
    const Clock covered = queue_progress(state, *device, event.mesh_cq_id(), sub_device_ids);
    state.events.insert_or_assign({event.mesh_cq_id(), event.id()}, MeshState::RecordedEvent{covered, event});
    // A record blocks its queue until what it covers is done, so everything the queue runs later follows it.
    state.queues[event.mesh_cq_id()].wait(covered);
    while (state.events.size() > kMaxEvents) {
        state.events.erase(state.events.begin());
    }
}

void on_wait_for_event(QueueId waiting_queue, const MeshEvent& event) {
    std::lock_guard lock(mutex);
    auto& state = states[event.device()->id()];
    const auto it = state.events.find({event.mesh_cq_id(), event.id()});
    if (it != state.events.end()) {
        state.queues[waiting_queue.get()].wait(it->second.covered);
    }
}

void on_event_synchronize(const MeshEvent& event) {
    std::lock_guard lock(mutex);
    auto& state = states[event.device()->id()];
    const auto it = state.events.find({event.mesh_cq_id(), event.id()});
    if (it != state.events.end()) {
        merge(state.host, it->second.covered);
    }
}

void on_synchronize(MeshDevice& device, std::optional<QueueId> queue, std::span<const SubDeviceId> sub_device_ids) {
    std::lock_guard lock(mutex);
    auto& state = states[device.id()];
    const uint32_t first = queue.has_value() ? queue->get() : 0;
    const uint32_t last = queue.has_value() ? queue->get() : std::min<uint32_t>(device.num_hw_cqs(), kMaxQueues) - 1;
    for (uint32_t q = first; q <= last; ++q) {
        merge(state.host, queue_progress(state, device, q, sub_device_ids));
    }
}

}  // namespace ttnn::queue_hazards
