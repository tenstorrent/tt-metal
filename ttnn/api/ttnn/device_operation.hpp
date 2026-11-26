// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/overloaded.hpp>
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <unordered_map>

#include <tt-metalium/utils/program_cache.hpp>
#include <tracy/Tracy.hpp>
#include "tools/profiler/op_profiler_serialize.hpp"  // lightweight; heavy JSON lives in op_profiler_json.cpp
#include <tt_stl/concepts.hpp>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_serialization.hpp"  // serialize_tracked_arg<T> definitions, used via track_function_start
#include "ttnn/distributed/api.hpp"
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <internal/graph_function_abort.hpp>
#include <type_traits>
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation_detail.hpp"
#include "ttnn/compiled_program.hpp"
#include <tt-metalium/experimental/kernel_config_buffer.hpp>

namespace ttnn::device_operation {

template <typename T>
using CachedProgram = tt::tt_metal::program_cache::detail::CachedProgram<T>;

// Used for adapting Ops that work with single-device programs and shared variables.
// Prefer to natively implement `CachedMeshWorkload` that overrides `override_runtime_arguments` at the
// workload level.
template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

template <typename T>
using CachedMeshWorkload = tt::tt_metal::program_cache::detail::CachedMeshWorkload<T>;

namespace detail {

using ::tt::tt_metal::program_cache::detail::CachedProgramFactory;
using ::tt::tt_metal::program_cache::detail::ProgramCacheKey;

template <typename... Ts>
[[nodiscard]] std::variant<Ts...> map_index_to_variant(std::size_t i, std::variant<Ts...>) {
    assert(i < sizeof...(Ts));
    static constexpr std::variant<Ts...> table[] = {Ts{}...};
    return table[i];
}

inline bool graph_capture_blocks_dispatch() {
    if (auto hook = tt::tt_metal::GraphTracker::instance().get_hook()) {
        if (auto* processor_hooks = dynamic_cast<ttnn::graph::ProcessorHooks*>(hook.get())) {
            return processor_hooks->get_block();
        }
    }
    return false;
}

template <typename device_operation_t>
auto compute_program_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (DeviceOperationWithCustomProgramCacheConcept<device_operation_t>) {
        ZoneScopedN("Compute custom program hash");
        // Fold type_hash so distinct ops cannot alias on a custom-hash collision
        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<device_operation_t>,
            device_operation_t::compute_program_hash(operation_attributes, tensor_args));
    } else {
        ZoneScopedN("Compute default program hash");
        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    }
}

// Helper to create a mesh workload from a WorkloadFactory that may or may not
// provide create_mesh_workload. If missing, synthesize it from create_at.
template <typename WorkloadFactory, typename device_operation_t>
auto create_mesh_workload_from_workload_factory(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    using cached_mesh_workload_t = typename WorkloadFactory::cached_mesh_workload_t;
    if constexpr (requires(
                      const typename device_operation_t::operation_attributes_t& operation_attributes,
                      const ttnn::MeshCoordinateRangeSet& tensor_coords,
                      const typename device_operation_t::tensor_args_t& tensor_args,
                      typename device_operation_t::tensor_return_value_t& tensor_return_value) {
                      WorkloadFactory::create_mesh_workload(
                          operation_attributes,
                          tensor_coords,
                          tensor_args,
                          tensor_return_value,
                          ttnn::MergePerDevicePrograms{});
                  }) {
        // The factory runs the op itself, including its cache hits: it may merge per-device programs.
        return WorkloadFactory::create_mesh_workload(
            operation_attributes, tensor_coords, tensor_args, tensor_return_value, ttnn::MergePerDevicePrograms{});
    } else if constexpr (requires(
                             const typename device_operation_t::operation_attributes_t& operation_attributes,
                             const ttnn::MeshCoordinateRangeSet& tensor_coords,
                             const typename device_operation_t::tensor_args_t& tensor_args,
                             typename device_operation_t::tensor_return_value_t& tensor_return_value) {
                             WorkloadFactory::create_mesh_workload(
                                 operation_attributes, tensor_coords, tensor_args, tensor_return_value);
                         }) {
        return WorkloadFactory::create_mesh_workload(
            operation_attributes, tensor_coords, tensor_args, tensor_return_value);
    } else if constexpr (requires(
                             const typename device_operation_t::operation_attributes_t& operation_attributes,
                             const ttnn::MeshCoordinate& mesh_coordinate,
                             const typename device_operation_t::tensor_args_t& tensor_args,
                             typename device_operation_t::tensor_return_value_t& tensor_return_value) {
                             WorkloadFactory::create_at(
                                 operation_attributes, mesh_coordinate, tensor_args, tensor_return_value);
                         }) {
        using shared_variables_t = typename WorkloadFactory::shared_variables_t;
        tt::tt_metal::distributed::MeshWorkload mesh_workload;
        std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
        for (const auto& coord : tensor_coords.coords()) {
            auto cached_program =
                WorkloadFactory::create_at(operation_attributes, coord, tensor_args, tensor_return_value);
            mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
            shared_variables.emplace(coord, std::move(cached_program.shared_variables));
        }
        return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
    } else {
        static_assert(
            ttsl::concepts::always_false_v<WorkloadFactory>,
            "WorkloadFactory must implement create_mesh_workload(operation_attributes, tensor_coords, tensor_args, "
            "tensor_return_value) or create_at(operation_attributes, mesh_coordinate, tensor_args, "
            "tensor_return_value, tensor_coords)");
    }
}

template <typename device_operation_t>
auto get_operation_name(const typename device_operation_t::operation_attributes_t& operation_attributes) {
    if constexpr (is_mesh_device_operation_adapter_v<device_operation_t>) {
        // For MeshAdapter operations, we recurse to get the name of the underlying device operation
        return get_operation_name<typename device_operation_t::device_operation_t>(operation_attributes);
    } else {
        return ttsl::get_type_name<device_operation_t>();
    }
}

// GCC 12 has a bug that causes a segfault when using reflection to log tensors in debug mode.
// If building with clang, allow this to be compiled
#if !defined(NDEBUG) && defined(__clang__)

template <typename device_operation_t>
inline void log_operation(
    std::size_t /*device_id*/,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    ttsl::hash::hash_t program_hash,
    bool program_cache_hit) {
    log_debug(
        tt::LogOp, "Launching Device Operation: \"{}\"", get_operation_name<device_operation_t>(operation_attributes));

    log_debug(tt::LogOp, "Program Hash: {}", program_hash);
    log_debug(tt::LogOp, "Program Cache Hit: {}", program_cache_hit);

    log_debug(tt::LogOp, "Attributes:");
    for ([[maybe_unused]] const auto& [key, value] : ttsl::reflection::get_attributes(operation_attributes)) {
        log_debug(tt::LogOp, "\t{} = {}", key, value);
    }

    log_debug(tt::LogOp, "Tensors Args:");
    auto index = 0;
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&index](auto&& tensor) {
            log_debug(tt::LogOp, "\t{}: {}", index, tensor);
            index++;
        },
        tensor_args);

    log_debug(tt::LogOp, "");
}

#else

template <typename device_operation_t>
inline void log_operation(
    std::size_t /*device_id*/,
    const typename device_operation_t::operation_attributes_t& /*operation_attributes*/,
    const typename device_operation_t::tensor_args_t& /*tensor_args*/,
    ttsl::hash::hash_t /*program_hash*/,
    bool /*program_cache_hit*/) {}

#endif

// Records `workload` into the calling thread's TTNN CommandListBuilder, with the op-owned tensors and GlobalSemaphores
// of a spec-adapter workload. With binds pushed and a spec-adapter workload, first collects each program's stored
// CmdListBindings and the tensors this call binds.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
[[gnu::noinline]] void record_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    const typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    const typename mesh_device_operation_t::SpecSharedVariablesMap* spec_shared_variables) {
    const auto operation_name = get_operation_name<mesh_device_operation_t>(operation_attributes);
    std::vector<std::shared_ptr<const void>> op_owned_resources;
    if (spec_shared_variables != nullptr) {
        for (const auto& [_, sv] : *spec_shared_variables) {
            if (sv.op_owned_tensors && !sv.op_owned_tensors->empty()) {
                op_owned_resources.push_back(sv.op_owned_tensors);
            }
            if (sv.global_semaphores) {
                op_owned_resources.push_back(sv.global_semaphores);
            }
        }
    }
    if (spec_shared_variables == nullptr || !has_active_cmd_list_binds()) {
        record_to_command_list(operation_name, mesh_device, workload, {}, op_owned_resources);
        return;
    }
    const auto io_mesh_tensors =
        mesh_device_operation_t::SpecBindingSupport::collect_mesh_tensors(tensor_args, tensor_return_value);
    std::vector<CmdListProgramBindings> programs;
    for (const auto& [range, program] : workload.get_programs()) {
        const auto& sv = spec_shared_variables->at(range);
        if (!sv.cmd_list_bindings) {
            continue;
        }
        CmdListProgramBindings& entry = programs.emplace_back(
            CmdListProgramBindings{.program = &program, .bindings = sv.cmd_list_bindings.get(), .bound_tensors = {}});
        for (const auto& binding : sv.bindings) {
            entry.bound_tensors.emplace_back(
                binding.tensor_parameter_name,
                &mesh_device_operation_t::SpecBindingSupport::bound_mesh_tensor(
                    sv, io_mesh_tensors, binding.tensor_idx));
        }
    }
    record_to_command_list(operation_name, mesh_device, workload, programs, op_owned_resources);
}

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void enqueue_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    [[maybe_unused]] typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    const typename mesh_device_operation_t::SpecSharedVariablesMap* spec_shared_variables,
    [[maybe_unused]] bool program_cache_hit = false,
    const tt::tt_metal::experimental::KernelConfigBuffer* kernel_config_buffer = nullptr) {
    // Generate and set runtime_id for all programs (always, for dispatcher)
    auto runtime_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();
    for (auto& [_, program] : workload.get_programs()) {
        program.set_runtime_id(runtime_id);
    }

    const auto cq_id = ttnn::core::get_current_command_queue_id_for_thread();

    // Inspector: emit debug entry with tensor parameters
    if (tt::tt_metal::experimental::inspector::IsEnabled()) {
        auto operation_name = get_operation_name<mesh_device_operation_t>(operation_attributes);

        std::vector<tt::tt_metal::TensorSpec> spec_copies;
        if (tt::tt_metal::experimental::inspector::ShouldCaptureTensorSpecs()) {
            size_t specs_count = ttsl::reflection::count_object_of_type<Tensor>(tensor_args);
            spec_copies.reserve(specs_count);

            ttsl::reflection::visit_object_of_type<Tensor>(
                [&](const Tensor& t) { spec_copies.emplace_back(t.tensor_spec()); }, tensor_args);
        }

        // A NO_DISPATCH graph capture enqueues nothing, so there is no trace to report. Skipping the lookup also
        // keeps such a capture working when the selected queue does not exist on this device.
        std::optional<tt::tt_metal::distributed::MeshTraceId> trace_id;
        if (!graph_capture_blocks_dispatch()) {
            trace_id = tt::tt_metal::experimental::inspector::GetCurrentMeshTraceId(
                ttnn::core::current_mesh_command_queue(*mesh_device, cq_id));
        }
        tt::tt_metal::experimental::inspector::EmitMeshWorkloadDebugEntry(
            workload, runtime_id, operation_name, std::move(spec_copies), trace_id);
    }

    if (mesh_device_operation_utils::track_workload(workload, mesh_device)) {
        return;
    }

    if (is_recording_command_list()) [[unlikely]] {
        // A command list replays its workloads from the KERNEL_CONFIG window, so it cannot hold one that runs from a
        // kernel config buffer.
        TT_FATAL(
            kernel_config_buffer == nullptr,
            "{} cannot be recorded into a command list: it runs from a kernel config buffer",
            get_operation_name<mesh_device_operation_t>(operation_attributes));
        record_mesh_workload<mesh_device_operation_t>(
            operation_attributes, tensor_args, tensor_return_value, mesh_device, workload, spec_shared_variables);
    } else if (kernel_config_buffer != nullptr) {
        tt::tt_metal::experimental::EnqueueMeshWorkload(
            ttnn::core::current_mesh_command_queue(*mesh_device, cq_id), workload, *kernel_config_buffer, false);
    } else {
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            ttnn::core::current_mesh_command_queue(*mesh_device, cq_id), workload, false);
    }

    ttnn::graph::track_mesh_workload_execution(workload, mesh_device, runtime_id);

    TracyOpMeshWorkload(
        mesh_device,
        workload,
        mesh_device_operation_t{},
        operation_attributes,
        tensor_args,
        tensor_return_value,
        program_cache_hit);
}

// Dispatches `fn` to `program_factory` through either the `MeshWorkloadFactoryConcept` directly, or through the adapted
// path for `ProgramFactoryConcept` / `ProgramDescriptorFactoryConcept` / `ProgramSpecFactoryConcept` /
// `CustomProgramSpecFactoryConcept` / `MeshWorkloadSpecFactoryConcept` factories.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t, typename ProgramFactory, typename Fn>
void dispatch_to_mesh_workload_factory(const ProgramFactory& program_factory, const Fn& fn) {
    std::visit(
        ttsl::overloaded{
            [&]<ProgramFactoryConcept T>(const T&) {
                // Adapt ProgramFactory to MeshWorkloadFactory concept.
                using AdaptedMeshWorkloadFactory = mesh_device_operation_t::template MeshWorkloadFactoryAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<ProgramDescriptorFactoryConcept T>(const T&) {
                using AdaptedMeshWorkloadFactory = mesh_device_operation_t::template DescriptorMeshWorkloadAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<ProgramSpecFactoryConcept T>(const T&) {
                using AdaptedMeshWorkloadFactory =
                    mesh_device_operation_t::template ProgramSpecMeshWorkloadFactoryAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<CustomProgramSpecFactoryConcept T>(const T&) {
                using AdaptedMeshWorkloadFactory =
                    mesh_device_operation_t::template CustomProgramSpecMeshWorkloadFactoryAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<MeshWorkloadSpecFactoryConcept T>(const T&) {
                using AdaptedMeshWorkloadFactory = mesh_device_operation_t::template MeshWorkloadSpecFactoryAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<MeshWorkloadFactoryConcept WorkloadFactory>(const WorkloadFactory&) {
                fn.template operator()<WorkloadFactory>();
            }},
        program_factory);
}

// The cached workload's shared variables if a spec adapter built it, else nullptr.
template <typename mesh_device_operation_t, typename CachedMeshWorkload>
const typename mesh_device_operation_t::SpecSharedVariablesMap* spec_shared_variables_of(
    const CachedMeshWorkload& cached_mesh_workload) {
    if constexpr (std::is_same_v<
                      decltype(cached_mesh_workload.shared_variables),
                      typename mesh_device_operation_t::SpecSharedVariablesMap>) {
        return &cached_mesh_workload.shared_variables;
    } else {
        return nullptr;
    }
}

// Applies this invocation's runtime arguments to `cached_program_factory`'s workload, a workload the operation's
// program factory built for the same program cache key, and passes the workload to `use_workload`.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t, typename UseWorkload>
void use_cached_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    tt::tt_metal::program_cache::detail::CachedProgramFactory& cached_program_factory,
    const UseWorkload& use_workload) {
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }

    auto program_factory_index = cached_program_factory.program_factory_index;
    auto program_factory = map_index_to_variant(
        program_factory_index, mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args));

    // WorkloadFactory is unconstrained here because dispatch_to_mesh_workload_factory
    // already selects the correct adapted type via std::visit.
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(program_factory, [&]<typename WorkloadFactory>() {
        using cached_mesh_workload_t = typename WorkloadFactory::cached_mesh_workload_t;
        auto& cached_mesh_workload = cached_program_factory.cached_program.template get<cached_mesh_workload_t>();

        if (!graph_capture_blocks_dispatch()) {
            if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
                WorkloadFactory::apply_descriptor(
                    cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
            } else {
                WorkloadFactory::override_runtime_arguments(
                    cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
            }
        }

        use_workload(
            cached_mesh_workload.workload, spec_shared_variables_of<mesh_device_operation_t>(cached_mesh_workload));
    });
}

// Passes the mesh workload in `cached_program_factory`, which the operation's program factory built, and its shared
// variables (spec_shared_variables_of) to `fn`.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t, typename Fn>
void visit_cached_workload(CachedProgramFactory& cached_program_factory, const Fn& fn) {
    auto program_factory = map_index_to_variant(
        cached_program_factory.program_factory_index, typename mesh_device_operation_t::program_factory_t{});
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(program_factory, [&]<typename WorkloadFactory>() {
        using cached_mesh_workload_t = typename WorkloadFactory::cached_mesh_workload_t;
        auto& cached_mesh_workload = cached_program_factory.cached_program.template get<cached_mesh_workload_t>();
        fn(cached_mesh_workload.workload, spec_shared_variables_of<mesh_device_operation_t>(cached_mesh_workload));
    });
}

// The mesh workload in `cached_program_factory`, which the operation's program factory built.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
tt::tt_metal::distributed::MeshWorkload& get_cached_workload(CachedProgramFactory& cached_program_factory) {
    tt::tt_metal::distributed::MeshWorkload* workload = nullptr;
    visit_cached_workload<mesh_device_operation_t>(
        cached_program_factory,
        [&](tt::tt_metal::distributed::MeshWorkload& cached_workload,
            const typename mesh_device_operation_t::SpecSharedVariablesMap* /*spec_shared_variables*/) {
            workload = &cached_workload;
        });
    return *workload;
}

// Builds the operation's mesh workload with its program factory, type-erased with the factory's index as the program
// cache stores it.
template <DeviceOperationConcept mesh_device_operation_t>
CachedProgramFactory create_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device) {
    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

    auto program_factory = mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args);
    auto program_factory_index = program_factory.index();
    auto log_msg_func = [] {
        log_warning(
            tt::LogOp,
            "Tensors that are distributed across mesh device unevenly negatively affect Op dispatch performance.");
    };
    std::optional<CachedProgramFactory> created;
    // WorkloadFactory is unconstrained — dispatch_to_mesh_workload_factory resolves the type.
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(program_factory, [&]<typename WorkloadFactory>() {
        ttnn::MeshCoordinateRangeSet tensor_coords;
        if (mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
            // Fast path - a range covers the entire mesh.
            tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device->shape()));
        } else {
            // Slow path - iterate over coordinates and merge them into a range set one by one.
            log_msg_func();  // Work around for g++12 compiler bug
            for (const auto& coord :
                 mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
                tensor_coords.merge(ttnn::MeshCoordinateRange(coord, coord));
            }
        }
        created.emplace(
            create_mesh_workload_from_workload_factory<WorkloadFactory, mesh_device_operation_t>(
                operation_attributes, tensor_coords, tensor_args, tensor_return_value),
            program_factory_index);
    });
    return std::move(*created);
}

// Helper for creating and caching a mesh workload
template <DeviceOperationConcept mesh_device_operation_t, typename UseWorkload>
void create_and_cache_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    const ProgramCacheKey& program_key,
    const UseWorkload& use_workload) {
    auto created = create_mesh_workload<mesh_device_operation_t>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device);
    // Cached only after this use returns, so a use that throws leaves the cache unchanged.
    visit_cached_workload<mesh_device_operation_t>(created, use_workload);

    // Don't cache programs during NO_DISPATCH graph capture mode because
    // buffer addresses are invalid (address=0). Caching such programs would
    // cause issues when later running in NORMAL mode.
    // In NORMAL capture mode, the hook exists but is non-blocking, so caching is safe.
    bool should_cache = program_cache.is_enabled() && !graph_capture_blocks_dispatch();
    if (should_cache) {
        program_cache.insert(program_key, std::move(created));
    }
}

// Keep tracker-only context construction out of use_mesh_workload's
// cache-hit instruction path. This wrapper is called only for a cache miss when
// tracking was enabled at process startup.
template <DeviceOperationConcept mesh_device_operation_t, typename UseWorkload>
[[gnu::noinline]] void create_and_cache_mesh_workload_with_allocation_context(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    const ProgramCacheKey& program_key,
    const UseWorkload& use_workload) {
    auto op_name = get_operation_name<mesh_device_operation_t>(operation_attributes);
    std::string alloc_ctx =
        std::string(tt::tt_metal::kProgramCacheAllocationContextPrefix) + " " + std::string(op_name);
    for (const auto& [name, value] : ttsl::reflection::get_attributes(operation_attributes)) {
        alloc_ctx += " " + std::string(name) + "=" + value.to_string();
    }
    tt::tt_metal::AllocationContextGuard ctx_guard(alloc_ctx);
    create_and_cache_mesh_workload<mesh_device_operation_t>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device, program_cache, program_key, use_workload);
}

// The cache key is the 64-bit hash plus an exact canonical encoding; a hash collision is
// resolved by std::unordered_map's own operator== on the canonical part (issue #45821).
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
ProgramCacheKey compute_program_cache_key(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    ttnn::MeshDevice* mesh_device) {
    return mesh_device_operation_t::compute_mesh_workload_cache_key(
        mesh_device,
        ttsl::get_type_name<typename mesh_device_operation_t::device_operation_t>(),
        operation_attributes,
        tensor_args);
}

struct ProgramCacheLookup {
    ProgramCacheKey key;
    bool hit = false;
};

// Throws on a miss while cache misses are forbidden. With the cache disabled, returns an empty key and no hit.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
ProgramCacheLookup look_up_program_cache(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    ttnn::MeshDevice* mesh_device) {
    auto& program_cache = mesh_device->get_program_cache();
    ProgramCacheLookup lookup;
    if (!program_cache.is_enabled()) {
        return lookup;
    }
    lookup.key = compute_program_cache_key<mesh_device_operation_t>(operation_attributes, tensor_args, mesh_device);
    lookup.hit = program_cache.contains(lookup.key);
    if (!lookup.hit && !program_cache.cache_misses_allowed()) {
        auto op_name = get_operation_name<mesh_device_operation_t>(operation_attributes);
        TT_THROW("Device operation \"{}\": program cache miss occurred, but cache misses are forbidden", op_name);
    }
    return lookup;
}

// Passes the operation's mesh workload to `use_workload`: on a hit, the cached workload with this invocation's runtime
// arguments applied; on a miss, a newly created workload, which is cached once `use_workload` returns. The second
// argument is the workload's shared variables when a spec adapter built it, else nullptr.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t, typename UseWorkload>
void use_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    const ProgramCacheLookup& lookup,
    const UseWorkload& use_workload) {
    auto& program_cache = mesh_device->get_program_cache();
    if (lookup.hit) {
        use_cached_mesh_workload<mesh_device_operation_t>(
            operation_attributes, tensor_args, tensor_return_value, program_cache.get(lookup.key), use_workload);
    } else if (tt::tt_metal::trace_allocation_tracking_enabled()) [[unlikely]] {
        create_and_cache_mesh_workload_with_allocation_context<mesh_device_operation_t>(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            mesh_device,
            program_cache,
            lookup.key,
            use_workload);
    } else {
        create_and_cache_mesh_workload<mesh_device_operation_t>(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            mesh_device,
            program_cache,
            lookup.key,
            use_workload);
    }
}

// Tells graph capture which program factory the operation used.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void set_pending_program_factory(std::size_t program_factory_index, bool program_cache_hit) {
    auto program_factory =
        map_index_to_variant(program_factory_index, typename mesh_device_operation_t::program_factory_t{});
    const std::string_view factory_type = std::visit(
        [](auto&& alt) -> std::string_view { return ttsl::long_type_name<std::decay_t<decltype(alt)>>; },
        program_factory);
    ttnn::graph::GraphProcessor::set_pending_program_factory(
        std::string(factory_type), program_factory_index, program_cache_hit);
}

// A program `run` launches in place of the program cache's, and the kernel config buffer it runs from, if any.
struct CompiledLaunch {
    CompiledProgram& program;
    const tt::tt_metal::experimental::KernelConfigBuffer* kernel_config_buffer = nullptr;
};

// Main function to launch operations on mesh devices with special handling for MeshDeviceOperationAdapter. Launches
// `compiled`'s program when given (`run`), else the program cache's (`launch`).
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch_operation_with_adapter(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    const CompiledLaunch* compiled = nullptr) {
    // Skip if operation should be skipped
    if constexpr (HasSkipLaunch<mesh_device_operation_t>) {
        if (mesh_device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)) {
            mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
            return;
        }
    }

    ProgramCacheLookup lookup;
    if (compiled != nullptr) {
        // The compiled program serves these arguments only if the program cache would have served them from it.
        lookup.key = compute_program_cache_key<mesh_device_operation_t>(operation_attributes, tensor_args, mesh_device);
        TT_FATAL(
            lookup.key == compiled->program.key(),
            "Device operation \"{}\" cannot run the program compiled from \"{}\": its arguments have a different "
            "program cache key, so they need a different program",
            get_operation_name<mesh_device_operation_t>(operation_attributes),
            compiled->program.operation_name());
        lookup.hit = true;
    } else {
        lookup = look_up_program_cache<mesh_device_operation_t>(operation_attributes, tensor_args, mesh_device);
    }

    log_operation<mesh_device_operation_t>(
        mesh_device->id(), operation_attributes, tensor_args, lookup.key.hash, lookup.hit);

    const auto enqueue = [&](tt::tt_metal::distributed::MeshWorkload& workload,
                             const typename mesh_device_operation_t::SpecSharedVariablesMap* spec_shared_variables) {
        enqueue_mesh_workload<mesh_device_operation_t>(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            mesh_device,
            workload,
            spec_shared_variables,
            lookup.hit,
            compiled != nullptr ? compiled->kernel_config_buffer : nullptr);
    };
    if (compiled != nullptr) {
        use_cached_mesh_workload<mesh_device_operation_t>(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            compiled->program.cached_program_factory(),
            enqueue);
    } else {
        use_mesh_workload<mesh_device_operation_t>(
            operation_attributes, tensor_args, tensor_return_value, mesh_device, lookup, enqueue);
    }

    // Stash factory identity after adapter work so nested function_end events cannot consume it.
    if (ttnn::graph::GraphProcessor::has_active_instance()) {
        std::size_t program_factory_index = 0;
        if (compiled != nullptr) {
            program_factory_index = compiled->program.program_factory_index();
        } else {
            auto& program_cache = mesh_device->get_program_cache();
            // Prefer the cached index; select only when the miss was not inserted (NO_DISPATCH).
            program_factory_index =
                (program_cache.is_enabled() && program_cache.contains(lookup.key))
                    ? program_cache.get(lookup.key).program_factory_index
                    : mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args).index();
        }
        set_pending_program_factory<mesh_device_operation_t>(program_factory_index, lookup.hit);
    }
}

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void prepare_operation_with_adapter(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device) {
    if constexpr (HasSkipLaunch<mesh_device_operation_t>) {
        if (mesh_device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)) {
            mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
            return;
        }
    }

    const auto lookup = look_up_program_cache<mesh_device_operation_t>(operation_attributes, tensor_args, mesh_device);
    use_mesh_workload<mesh_device_operation_t>(
        operation_attributes,
        tensor_args,
        tensor_return_value,
        mesh_device,
        lookup,
        [&](tt::tt_metal::distributed::MeshWorkload& workload,
            const typename mesh_device_operation_t::SpecSharedVariablesMap* /*spec_shared_variables*/) {
            tt::tt_metal::experimental::program_preparation::prepare(*mesh_device, workload);
        });
}

template <DeviceOperationConcept device_operation_t>
void validate_input_tensors(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) {
    for (const auto& input_tensor_ref : input_tensors) {
        const auto& input_tensor = input_tensor_ref.get();
        TT_FATAL(is_device_tensor(input_tensor), "Device Operations expect device tensors as inputs");
        TT_FATAL(input_tensor.is_allocated(), "Input Tensor is not allocated");
    }

    // Whether the op accepts per-core allocation is a property of the op, not of any one tensor,
    // so ask it once rather than inside the loop above.
    if constexpr (!SupportsPerCoreAllocation<device_operation_t>) {
        const auto operation_name = get_operation_name<device_operation_t>(operation_attributes);
        for (size_t index = 0; index < input_tensors.size(); ++index) {
            validate_no_per_core_allocation(input_tensors[index].get(), operation_name, index);
        }
    }
}

// Restricts the outputs to the mesh coordinates the inputs occupy and assigns the output topologies.
template <DeviceOperationConcept device_operation_t>
void place_output_tensors(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    ttnn::MeshDevice* mesh_device,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    if (!mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
        tensor_return_value = mesh_device_operation_utils::filter_tensor_shards(
            mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device), tensor_return_value);
    }

    if (!input_tensors.empty()) [[likely]] {
        std::vector<tt::tt_metal::TensorTopology> custom_topologies;
        if constexpr (requires {
                          {
                              device_operation_t::compute_output_topologies(operation_attributes, tensor_args)
                          } -> std::same_as<std::vector<tt::tt_metal::TensorTopology>>;
                      }) {
            custom_topologies = device_operation_t::compute_output_topologies(operation_attributes, tensor_args);
        }

        mesh_device_operation_utils::update_output_tensor_topologies(
            tensor_return_value, input_tensors, std::move(custom_topologies));
    }
}

// The body of `launch` (compiled == nullptr) and `run`: launches the program cache's program or `compiled`'s.
template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_operation(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const CompiledLaunch* compiled) {
    std::vector<std::reference_wrapper<const Tensor>> input_tensors;
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);

    const auto operation_name = detail::get_operation_name<device_operation_t>(operation_attributes);
    // Everything below can throw: validation, output allocation and, for the circular-buffer /
    // L1 clash of #28836, program dispatch. The guard closes the tracked scope on those paths too,
    // marking it aborted, so the capture does not lose the failing op and misnest everything after
    // it.
    tt::tt_metal::internal::ScopedTrackedFunction tracked_function(operation_name, operation_attributes, input_tensors);

    detail::validate_input_tensors<device_operation_t>(operation_attributes, input_tensors);

    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);

    ttnn::MeshDevice* mesh_device = mesh_device_operation_utils::get_mesh_device(operation_attributes, tensor_args);
    if (compiled != nullptr) {
        TT_FATAL(
            mesh_device->id() == compiled->program.mesh_device_id(),
            "Device operation \"{}\" was compiled for MeshDevice {}, but its tensors are on MeshDevice {}",
            operation_name,
            compiled->program.mesh_device_id(),
            mesh_device->id());
        // A trace replays the program's kernel binaries from its workload. The program cache keeps the workloads that
        // `launch` traces alive, but nothing keeps a CompiledProgram's alive for the trace.
        const auto cq_id = ttnn::core::get_current_command_queue_id_for_thread();
        TT_FATAL(
            !ttnn::core::current_mesh_command_queue(*mesh_device, cq_id).trace_id().has_value(),
            "Device operation \"{}\" cannot run a compiled program while a trace is being captured",
            operation_name);
    }

    // TODO: #37267 - Remove this short-circuit once we have a better way to handle inactive MeshDevices.
    // Short-circuit for inactive MeshDevices (no-op). It is important this happens before any validation an op may
    // perform, as most of the MeshDevice calls will fail for inactive MeshDevices.
    if (mesh_device->is_remote_only()) {
        tracked_function.end(tensor_return_value);
        return tensor_return_value;
    }

    detail::place_output_tensors<device_operation_t>(
        operation_attributes, tensor_args, input_tensors, mesh_device, tensor_return_value);

    detail::launch_operation_with_adapter<MeshDeviceOperationAdapter<device_operation_t>>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device, compiled);

    tracked_function.end(tensor_return_value);
    return tensor_return_value;
}

/**
 * Launch an operation on a mesh device.
 *
 * @param operation_attributes The operation attributes.
 * @param tensor_args The tensor arguments.
 * @return Outputs are allocated but values are not immediately available till the operation is complete.
 */
template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    return launch_operation<device_operation_t>(operation_attributes, tensor_args, /*compiled=*/nullptr);
}

/**
 * Compile and finalize an operation without dispatching it.
 *
 * Validates the operation the same way `launch` does and throws if its program does not compile or does not fit the
 * kernel-configuration buffer. A successfully prepared workload is inserted into the program cache, so the next launch
 * of the same operation is a cache hit. An operation that skips launch is not compiled.
 */
template <DeviceOperationConcept device_operation_t>
void prepare(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    std::vector<std::reference_wrapper<const Tensor>> input_tensors;
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);
    detail::validate_input_tensors<device_operation_t>(operation_attributes, input_tensors);

    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    auto* mesh_device = mesh_device_operation_utils::get_mesh_device(operation_attributes, tensor_args);
    TT_FATAL(!mesh_device->get_device_ids().empty(), "Cannot prepare an operation for an inactive MeshDevice");
    detail::place_output_tensors<device_operation_t>(
        operation_attributes, tensor_args, input_tensors, mesh_device, tensor_return_value);
    detail::prepare_operation_with_adapter<MeshDeviceOperationAdapter<device_operation_t>>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device);
}

/**
 * Compile an operation into a CompiledProgram, without dispatching it or touching the program cache.
 *
 * Validates the operation as `launch` does, builds its mesh workload with its program factory, and compiles and lays
 * it out as its launch will (tt::tt_metal::experimental::program_preparation::compile). In slow dispatch the
 * kernel-configuration blocks are not checked against the KERNEL_CONFIG window here: `run` checks them against the
 * window, and the kernel binaries against the kernel config buffer when it runs from one. Throws for an operation that
 * skips launch with these arguments.
 */
template <DeviceOperationConcept device_operation_t>
CompiledProgram compile(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    using mesh_device_operation_t = MeshDeviceOperationAdapter<device_operation_t>;
    std::vector<std::reference_wrapper<const Tensor>> input_tensors;
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);
    detail::validate_input_tensors<device_operation_t>(operation_attributes, input_tensors);

    const auto operation_name = std::string(detail::get_operation_name<device_operation_t>(operation_attributes));
    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    auto* mesh_device = mesh_device_operation_utils::get_mesh_device(operation_attributes, tensor_args);
    TT_FATAL(!mesh_device->get_device_ids().empty(), "Cannot compile an operation for an inactive MeshDevice");
    if constexpr (HasSkipLaunch<mesh_device_operation_t>) {
        TT_FATAL(
            !mesh_device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value),
            "Device operation \"{}\" skips launch with these arguments, so it has no program to compile",
            operation_name);
    }
    detail::place_output_tensors<device_operation_t>(
        operation_attributes, tensor_args, input_tensors, mesh_device, tensor_return_value);

    auto key =
        detail::compute_program_cache_key<mesh_device_operation_t>(operation_attributes, tensor_args, mesh_device);
    auto cached_program_factory = detail::create_mesh_workload<mesh_device_operation_t>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device);
    CompiledProgram compiled(
        std::move(cached_program_factory),
        &detail::get_cached_workload<mesh_device_operation_t>,
        std::move(key),
        operation_name,
        mesh_device->id());
    tt::tt_metal::experimental::program_preparation::compile(*mesh_device, compiled.workload());
    return compiled;
}

/**
 * Launch `compiled_program` with these operation attributes and tensors, as `launch` launches an operation whose
 * program is cached: their runtime arguments are applied to the compiled workload, which is then enqueued with the same
 * bookkeeping (runtime ID, inspector, graph capture, profiler). `kernel_config_buffer`, when given, holds the Tensix
 * kernel binaries in place of the KERNEL_CONFIG window (create_kernel_config_buffer). Without one, a
 * CommandListBuilder recording on this thread records the workload as it records `launch`; with one, `run` throws
 * while recording, since a command list cannot hold a workload that runs from a kernel config buffer.
 *
 * The operation must be the one `compiled_program` was compiled from, with arguments of the same program cache key, on
 * the same MeshDevice.
 *
 * @return Outputs are allocated but values are not immediately available till the operation is complete.
 */
template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t run(
    CompiledProgram& compiled_program,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const tt::tt_metal::experimental::KernelConfigBuffer* kernel_config_buffer = nullptr) {
    if (kernel_config_buffer != nullptr) {
        detail::throw_if_recording_command_list("Running a compiled program from a kernel config buffer");
    }
    const CompiledLaunch compiled{compiled_program, kernel_config_buffer};
    return launch_operation<device_operation_t>(operation_attributes, tensor_args, &compiled);
}

// invoke is now deprecated - use launch_on_device directly
// Keeping for backwards compatibility with existing code
template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t invoke(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    return launch<device_operation_t>(operation_attributes, tensor_args);
}

}  // namespace detail

using ttnn::device_operation::detail::compile;
using ttnn::device_operation::detail::launch;
using ttnn::device_operation::detail::prepare;
using ttnn::device_operation::detail::run;

}  // namespace ttnn::device_operation
