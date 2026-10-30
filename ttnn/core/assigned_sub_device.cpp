// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/assigned_sub_device.hpp"
#include "ttnn/sub_device_scope.hpp"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/kernel_cores.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::SubDeviceId;
using tt::tt_metal::SubDeviceManagerId;
using tt::tt_metal::distributed::MeshDevice;

namespace {

struct SubDeviceScope {
    SubDeviceId id;
    // Opened by the user, as opposed to the sub-device a composite operation carries to the operations it launches.
    bool strict = true;
    std::uint64_t token = 0;
    // For each device, the sub-device manager that was active when the scope first assigned an operation there.
    std::vector<std::pair<int, SubDeviceManagerId>> managers;
};

struct ThreadScopes {
    std::vector<SubDeviceScope> scopes;
    // The outermost composite operation (ttnn::CompositeSubDevice) running inside a scope or carrying its sub-device:
    // the call its caller made. Refusals and warnings name it; it assigns nothing.
    std::string_view composite;
    std::uint64_t next_token = 1;
};

ThreadScopes& thread_scopes() {
    thread_local ThreadScopes scopes;
    return scopes;
}

std::uint64_t push_scope(SubDeviceId id, bool strict) {
    auto& state = thread_scopes();
    const std::uint64_t token = state.next_token++;
    state.scopes.push_back(SubDeviceScope{.id = id, .strict = strict, .token = token});
    return token;
}

// Closes the scope `token` opened. Returns whether it was the innermost; one that was not is closed all the same.
bool pop_scope(std::uint64_t token) {
    auto& scopes = thread_scopes().scopes;
    const auto it = std::find_if(
        scopes.begin(), scopes.end(), [token](const SubDeviceScope& scope) { return scope.token == token; });
    if (it == scopes.end()) {
        return false;
    }
    const bool innermost = std::next(it) == scopes.end();
    scopes.erase(it);
    return innermost;
}

// The innermost scope, bound to the sub-device manager active on `mesh_device` when it first assigns an operation
// there.
const SubDeviceScope* innermost_scope_on(const MeshDevice& mesh_device, std::string_view operation_name) {
    auto& scopes = thread_scopes().scopes;
    if (scopes.empty()) [[likely]] {
        return nullptr;
    }
    SubDeviceScope& scope = scopes.back();
    const auto manager = mesh_device.get_active_sub_device_manager_id();
    const auto bound = std::find_if(scope.managers.begin(), scope.managers.end(), [&](const auto& entry) {
        return entry.first == mesh_device.id();
    });
    if (bound == scope.managers.end()) {
        scope.managers.emplace_back(mesh_device.id(), manager);
    } else {
        TT_FATAL(
            bound->second == manager,
            "{} runs inside a scope on sub-device {} whose earlier operations ran under another sub-device manager of "
            "device {}. Load sub-device managers outside sub-device scopes.",
            operation_name,
            *scope.id,
            mesh_device.id());
    }
    return &scope;
}

// Where `cores` lie among the loaded sub-devices: the one holding all of them, if any, and the ones they reach when
// they span several.
struct Holding {
    std::optional<SubDeviceId> holder;
    std::string spanned;  // "0 and 1"
    std::size_t num_spanned = 0;
};

Holding sub_device_holding(const MeshDevice& mesh_device, const CoreRangeSet& cores) {
    Holding holding;
    for (const SubDeviceId sub_device_id : mesh_device.get_sub_device_ids()) {
        const CoreRangeSet sub_device_cores = mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        if (sub_device_cores.contains(cores)) {
            holding.holder = sub_device_id;
            return holding;
        }
        if (sub_device_cores.intersects(cores)) {
            holding.spanned += fmt::format("{}{}", holding.num_spanned++ == 0 ? "" : " and ", *sub_device_id);
        }
    }
    return holding;
}

}  // namespace

std::uint64_t push_sub_device_for_thread(SubDeviceId sub_device_id) { return push_scope(sub_device_id, true); }

bool in_sub_device_scope() { return !thread_scopes().scopes.empty(); }

void pop_sub_device_for_thread(std::uint64_t token) {
    TT_FATAL(pop_scope(token), "Sub-device scopes on this thread were closed out of order; close the innermost first");
}

ScopeGuard with_sub_device(SubDeviceId sub_device_id) {
    const std::uint64_t token = push_sub_device_for_thread(sub_device_id);
    return make_guard([token] {
        if (!pop_scope(token)) {
            log_error(
                tt::LogOp, "Sub-device scopes on this thread were closed out of order; close the innermost first");
        }
    });
}

AssignedSubDevice assign_sub_device(
    const MeshDevice* device, const SubDeviceRequest& request, std::string_view operation_name) {
    if (device == nullptr) {
        return AssignedSubDevice{};
    }
    const MeshDevice& mesh_device = *device;
    if (mesh_device.is_remote_only()) {
        // This host drives none of the device's chips, so it loads no sub-devices and launch skips the operation.
        const auto& scopes = thread_scopes().scopes;
        return AssignedSubDevice{scopes.empty() ? request.sub_device.value_or(SubDeviceId{0}) : scopes.back().id, {}};
    }
    const auto& sub_device_ids = mesh_device.get_sub_device_ids();
    std::optional<SubDeviceId> assigned;
    if (const SubDeviceScope* scope = innermost_scope_on(mesh_device, operation_name)) {
        TT_FATAL(
            !request.sub_device.has_value() || *request.sub_device == scope->id,
            "{} was asked to run on sub-device {} inside a {} on sub-device {}",
            operation_name,
            **request.sub_device,
            scope->strict ? "sub-device scope" : "composite operation",
            *scope->id);
        assigned = scope->id;
    } else if (request.sub_device.has_value()) {
        assigned = request.sub_device;
    } else if (request.cores.has_value()) {
        const Holding holding = sub_device_holding(mesh_device, *request.cores);
        TT_FATAL(
            holding.num_spanned < 2,
            "{}: its cores {} span sub-devices {}; an operation runs on one sub-device",
            operation_name,
            *request.cores,
            holding.spanned);
        TT_FATAL(
            holding.holder.has_value(),
            "{}: its cores {} do not all lie in one sub-device",
            operation_name,
            *request.cores);
        assigned = holding.holder;
    } else if (request.data.has_value()) {
        const Holding holding = sub_device_holding(mesh_device, *request.data);
        TT_FATAL(
            holding.num_spanned < 2,
            "{}: its L1 shards {} span sub-devices {}, so they name none of them. Reshard them within one sub-device, "
            "or name one with `with ttnn.sub_device(i)` or through its arguments: the operation then reads shards "
            "beyond it over the NoC, or refuses them if it reads them in place.",
            operation_name,
            *request.data,
            holding.spanned);
        assigned = holding.holder;
    }
    if (!assigned.has_value() && sub_device_ids.size() == 1) {
        assigned = sub_device_ids.front();
    }
    TT_FATAL(
        assigned.has_value(),
        "{} does not know which of the {} loaded sub-devices to run on: nothing names one. Run it inside "
        "`with ttnn.sub_device(i)`, or name one through its arguments (a sub-device id, sub_core_grids, or L1-sharded "
        "tensors within one sub-device).",
        operation_name,
        sub_device_ids.size());
    TT_FATAL(
        **assigned < sub_device_ids.size(),
        "{} was asked to run on sub-device {}, but the loaded sub-device manager has {}",
        operation_name,
        **assigned,
        sub_device_ids.size());
    AssignedSubDevice result{*assigned, mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, *assigned)};
    TT_FATAL(result.assigned(), "{} was assigned sub-device {}, which has no Tensix cores", operation_name, **assigned);
    TT_FATAL(
        !request.cores.has_value() || result.holds(*request.cores),
        "{} was given cores {} outside sub-device {} ({})",
        operation_name,
        *request.cores,
        *result.id,
        result.worker_cores);
    return result;
}

CompositeSubDevice::CompositeSubDevice(
    const MeshDevice* mesh_device, SubDeviceRequest request, std::string_view operation_name) :
    mesh_device_(mesh_device), request_(std::move(request)), operation_name_(operation_name) {
    auto& state = thread_scopes();
    const bool local = mesh_device != nullptr && !mesh_device->is_remote_only();
    if (!state.scopes.empty()) {
        if (local) {
            // Arguments that contradict the scope are refused here, before the composite launches anything.
            assigned_ = assign_sub_device(mesh_device, request_, operation_name_);
        }
    } else if (local && mesh_device->get_sub_device_ids().size() > 1) {
        assigned_ = assign_sub_device(mesh_device, request_, operation_name_);
        token_ = push_scope(assigned_->id, /*strict=*/false);
    } else {
        return;
    }
    if (state.composite.empty()) {
        state.composite = operation_name_;
        outermost_ = true;
    }
}

CompositeSubDevice::~CompositeSubDevice() noexcept {
    if (outermost_) {
        thread_scopes().composite = {};
    }
    if (token_.has_value() && !pop_scope(*token_)) {
        log_error(tt::LogOp, "A composite operation's sub-device was closed out of order");
    }
}

const AssignedSubDevice& CompositeSubDevice::assigned() const {
    if (!assigned_.has_value()) {
        assigned_ = assign_sub_device(mesh_device_, request_, operation_name_);
    }
    return *assigned_;
}

std::optional<CoreRangeSet> l1_shard_grid(const tt::tt_metal::MemoryConfig& config) {
    if (!config.is_l1()) {
        return std::nullopt;
    }
    if (config.shard_spec().has_value() && !config.shard_spec()->grid.empty()) {
        return config.shard_spec()->grid;
    }
    if (config.nd_shard_spec().has_value() && !config.nd_shard_spec()->grid.empty()) {
        return config.nd_shard_spec()->grid;
    }
    return std::nullopt;
}

CoreRange rectangle_in(const CoreRangeSet& cores) {
    TT_FATAL(!cores.empty(), "There is no rectangle in an empty set of cores");
    const CoreRange box = cores.bounding_box();
    if (cores.num_cores() == box.size()) {
        return box;  // The cores fill their bounding box (a CoreRangeSet's ranges never overlap).
    }
    const std::size_t width = box.end_coord.x - box.start_coord.x + 1;
    const std::size_t height = box.end_coord.y - box.start_coord.y + 1;
    // held(x, y): how many cores of the box's first x columns and y rows the set holds.
    std::vector<std::uint32_t> prefix((width + 1) * (height + 1), 0);
    const auto held = [&](std::size_t x, std::size_t y) -> std::uint32_t& { return prefix[(y * (width + 1)) + x]; };
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const bool in_set = cores.contains(CoreCoord{box.start_coord.x + x, box.start_coord.y + y});
            held(x + 1, y + 1) = held(x, y + 1) + held(x + 1, y) - held(x, y) + (in_set ? 1 : 0);
        }
    }
    CoreRange best = cores.ranges().front();
    std::size_t best_area = 0;
    for (std::size_t y0 = 0; y0 < height; ++y0) {
        for (std::size_t x0 = 0; x0 < width; ++x0) {
            for (std::size_t y1 = y0; y1 < height; ++y1) {
                for (std::size_t x1 = x0; x1 < width; ++x1) {
                    const std::size_t area = (x1 - x0 + 1) * (y1 - y0 + 1);
                    const std::size_t count = held(x1 + 1, y1 + 1) - held(x0, y1 + 1) - held(x1 + 1, y0) + held(x0, y0);
                    if (count != area) {
                        break;  // Wider rectangles from this start in these rows hold a gap too.
                    }
                    if (area > best_area) {
                        best_area = area;
                        best = CoreRange(
                            CoreCoord{box.start_coord.x + x0, box.start_coord.y + y0},
                            CoreCoord{box.start_coord.x + x1, box.start_coord.y + y1});
                    }
                }
            }
        }
    }
    return best;
}

}  // namespace ttnn

namespace ttnn::device_operation::detail {

void check_sub_device_scope(const ttnn::AssignedSubDevice* recorded, std::string_view operation_name) {
    const auto& scopes = ttnn::thread_scopes().scopes;
    if (scopes.empty()) [[likely]] {
        return;
    }
    const auto& scope = scopes.back();
    if (recorded == nullptr) {
        // A composite that built a ttnn::CompositeSubDevice is named, and only the limit is stated: launched outside
        // the scope, its operation would run unordered against the rest of the call. Otherwise the caller's own
        // operation and one an unmigrated composite launched look alike here, so the advice covers both.
        const std::string_view composite = ttnn::thread_scopes().composite;
        TT_FATAL(
            !scope.strict,
            "{}{} cannot run inside a sub-device scope (sub-device {}): it does not record the sub-device it runs on "
            "(ttnn::AssignedSubDevice), so it would place its kernels wherever it chooses. {}",
            operation_name,
            composite.empty() ? std::string() : fmt::format(", launched within {},", composite),
            *scope.id,
            composite.empty() ? fmt::format(
                                    "Launch it outside the scope; if an operation you called launched it, that "
                                    "operation cannot run inside a scope until {} records its sub-device.",
                                    operation_name)
                              : fmt::format(
                                    "{} cannot run inside a sub-device scope on this path until the operations it "
                                    "launches record their sub-device.",
                                    composite));
        return;
    }
    TT_FATAL(
        recorded->id == scope.id,
        "{} records sub-device {}, but runs inside a {} on sub-device {}",
        operation_name,
        *recorded->id,
        scope.strict ? "sub-device scope" : "composite operation",
        *scope.id);
}

void check_assigned(const ttnn::AssignedSubDevice& recorded, std::string_view operation_name) {
    TT_FATAL(
        recorded.assigned(),
        "{} was launched without the sub-device it runs on; its prim assigns one with ttnn::assign_sub_device",
        operation_name);
}

L1OutputCheck l1_output_check(const MeshDevice& mesh_device) {
    if (mesh_device.get_sub_device_ids().size() < 2) [[likely]] {
        return L1OutputCheck::none;
    }
    const auto& scopes = ttnn::thread_scopes().scopes;
    return !scopes.empty() && scopes.back().strict ? L1OutputCheck::refuse : L1OutputCheck::warn;
}

void check_l1_output_in_sub_device(
    const ttnn::AssignedSubDevice& recorded,
    const Tensor& output,
    std::string_view operation_name,
    L1OutputCheck check) {
    const auto& config = output.memory_config();
    if (check == L1OutputCheck::none || !config.is_l1()) {
        return;
    }
    const auto shards = ttnn::l1_shard_grid(config);
    if (shards.has_value() && recorded.holds(*shards)) {
        return;
    }
    const std::string placement =
        shards.has_value() ? fmt::format("sharded on {}", *shards) : std::string("interleaved");
    TT_FATAL(
        check != L1OutputCheck::refuse,
        "{} allocates an output in L1 ({}) inside a scope on sub-device {}, outside its cores {}. Interleaved L1, or "
        "L1 shards beyond the sub-device, reach the cores of the other sub-devices, where programs running "
        "concurrently keep their circular buffers. Shard the output within the sub-device, or allocate it in DRAM.",
        operation_name,
        placement,
        *recorded.id,
        recorded.worker_cores);
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned;
    {
        const std::lock_guard lock(warned_mutex);
        if (!warned.emplace(operation_name).second) {
            return;
        }
    }
    log_warning(
        tt::LogOp,
        "{} allocated an output in L1 ({}) on sub-device {} of a device with several sub-devices. Interleaved L1, or "
        "L1 shards beyond the sub-device, reach the cores of the other sub-devices: if a program is running there, "
        "the two can overwrite each other without an error. Shard the output within the sub-device or allocate it in "
        "DRAM; inside `with ttnn.sub_device(i)` this is refused. (Logged once per operation.)",
        operation_name,
        placement,
        *recorded.id);
}

void check_unrecorded_workload_in_composite(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const MeshDevice& mesh_device,
    std::string_view operation_name) {
    const auto& scopes = ttnn::thread_scopes().scopes;
    if (scopes.empty() || scopes.back().strict) {
        return;
    }
    const auto& carry = scopes.back();
    if (*carry.id >= mesh_device.get_sub_device_ids().size()) {
        return;
    }
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned;
    const std::string name(operation_name);
    {
        const std::lock_guard lock(warned_mutex);
        if (warned.contains(name)) {
            return;
        }
    }
    const CoreRangeSet carried_cores = mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, carry.id);
    for (const auto& [range, program] : workload.get_programs()) {
        const auto kernel_cores = tt::tt_metal::experimental::GetKernelCores(program, HalProgrammableCoreType::TENSIX);
        if (carried_cores.contains(kernel_cores)) {
            continue;
        }
        {
            const std::lock_guard lock(warned_mutex);
            if (!warned.emplace(name).second) {
                return;
            }
        }
        // A carry is pushed only when no scope is open, so the composite carrying it is the outermost one.
        const std::string_view composite = ttnn::thread_scopes().composite;
        log_warning(
            tt::LogOp,
            "{} ran inside {} on sub-device {}, but records no sub-device: its kernels landed on {}, outside "
            "sub-device {}. Nothing orders it against the call's other programs. Make {} record its sub-device. "
            "(Logged once per operation.)",
            operation_name,
            composite.empty() ? std::string_view("a composite operation") : composite,
            *carry.id,
            kernel_cores.subtract(carried_cores),
            *carry.id,
            operation_name);
        return;
    }
}

void check_workload_on_sub_device(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const ttnn::AssignedSubDevice& sub_device,
    std::string_view operation_name) {
    for (const auto& [range, program] : workload.get_programs()) {
        const auto kernel_cores = tt::tt_metal::experimental::GetKernelCores(program, HalProgrammableCoreType::TENSIX);
        TT_FATAL(
            sub_device.holds(kernel_cores),
            "{} placed kernels on {} for {}, outside sub-device {} ({})",
            operation_name,
            kernel_cores.subtract(sub_device.worker_cores),
            range,
            *sub_device.id,
            sub_device.worker_cores);
    }
}

}  // namespace ttnn::device_operation::detail
