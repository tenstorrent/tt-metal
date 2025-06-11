// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <random>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/indestructible.hpp>
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/program_cache.hpp>
#include <tracy/Tracy.hpp>
#include "tools/profiler/op_profiler.hpp"
#include <tt_stl/reflection.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"
#include <tt-metalium/distributed.hpp>
#include <type_traits>
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

namespace device_operation {

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

template <typename... Ts>
[[nodiscard]] std::variant<Ts...> map_index_to_variant(std::size_t i, std::variant<Ts...>) {
    assert(i < sizeof...(Ts));
    static constexpr std::variant<Ts...> table[] = {Ts{}...};
    return table[i];
}

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

// Returns `MeshCoordinateRangeSet` that covers a single coordinate (0,0).
inline const auto& get_unit_tensor_coords() {
    static tt::stl::Indestructible<ttnn::MeshCoordinateRangeSet> tensor_coords([]() {
        ttnn::MeshCoordinateRangeSet res;
        res.merge(ttnn::MeshCoordinateRange(ttnn::MeshCoordinate(0, 0), ttnn::MeshCoordinate(0, 0)));
        return res;
    }());
    return tensor_coords.get();
}

template <typename device_operation_t>
auto compute_program_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (DeviceOperationWithCustomProgramCacheConcept<device_operation_t>) {
        ZoneScopedN("Compute custom program hash");
        return device_operation_t::compute_program_hash(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Compute default program hash");
        return tt::stl::hash::hash_objects_with_default_seed(
            tt::stl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    }
}

template <typename device_operation_t>
tt::tt_metal::Program& create_or_get_program_from_cache(
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    bool program_cache_hit,
    auto program_hash,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    if (not program_cache_hit) {
        ZoneScopedN("Program Cache Miss");
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);

        tt::tt_metal::Program& program = std::visit(
            tt::stl::overloaded{
                [&program_cache,
                 &program_hash,
                 &operation_attributes,
                 &tensor_args,
                 &tensor_return_value,
                 program_factory_index = program_factory.index()]<ProgramFactoryConcept ProgramFactory>(
                    const ProgramFactory&) -> tt::tt_metal::Program& {
                    using cached_program_t =
                        decltype(ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value));
                    program_cache.insert(
                        program_hash,
                        CachedProgramFactory{
                            ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value),
                            program_factory_index});
                    auto& cached_program_factory = program_cache.get(program_hash);
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    return cached_program.program;
                },
                [&program_cache,
                 &program_hash,
                 &operation_attributes,
                 &tensor_args,
                 &tensor_return_value,
                 program_factory_index = program_factory.index()]<MeshWorkloadFactoryConcept WorkloadFactory>(
                    const WorkloadFactory&) -> tt::tt_metal::Program& {
                    using cached_mesh_workload_t = WorkloadFactory::cached_mesh_workload_t;
                    program_cache.insert(
                        program_hash,
                        CachedProgramFactory{
                            WorkloadFactory::create_mesh_workload(
                                operation_attributes, get_unit_tensor_coords(), tensor_args, tensor_return_value),
                            program_factory_index});
                    auto& cached_program_factory = program_cache.get(program_hash);
                    auto& cached_workload =
                        cached_program_factory.cached_program.template get<cached_mesh_workload_t>();
                    TT_FATAL(cached_workload.workload.get_programs().size() == 1, "Expected 1 program in workload.");
                    return cached_workload.workload.get_programs().begin()->second;
                },
            },
            program_factory);
        return program;
    } else {
        ZoneScopedN("Program Cache Hit");
        auto& cached_program_factory = program_cache.get(program_hash);
        auto program_factory_index = cached_program_factory.program_factory_index;

        using program_factory_variant_t =
            decltype(device_operation_t::select_program_factory(operation_attributes, tensor_args));
        auto program_factory = map_index_to_variant(program_factory_index, program_factory_variant_t{});

        tt::tt_metal::Program& program = std::visit(
            tt::stl::overloaded{
                [&cached_program_factory,
                 &operation_attributes,
                 &tensor_args,
                 &tensor_return_value]<ProgramFactoryConcept ProgramFactory>(
                    const ProgramFactory&) -> tt::tt_metal::Program& {
                    using cached_program_t =
                        decltype(ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value));
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    ProgramFactory::override_runtime_arguments(
                        cached_program, operation_attributes, tensor_args, tensor_return_value);
                    return cached_program.program;
                },
                [&cached_program_factory,
                 &operation_attributes,
                 &tensor_args,
                 &tensor_return_value]<MeshWorkloadFactoryConcept WorkloadFactory>(
                    const WorkloadFactory&) -> tt::tt_metal::Program& {
                    using cached_mesh_workload_t = WorkloadFactory::cached_mesh_workload_t;
                    auto& cached_workload =
                        cached_program_factory.cached_program.template get<cached_mesh_workload_t>();
                    WorkloadFactory::override_runtime_arguments(
                        cached_workload, operation_attributes, tensor_args, tensor_return_value);
                    TT_FATAL(cached_workload.workload.get_programs().size() == 1, "Expected 1 program in workload.");
                    return cached_workload.workload.get_programs().begin()->second;
                },
            },
            program_factory);
        return program;
    }
}

struct CheckDeviceBufferIsAllocated {
    std::size_t index = 0;

    void operator()(const Tensor& tensor) {
        if (not tensor.is_allocated()) {
            log_debug(tt::LogOp, "Tensor at index {} is not allocated", index);
        }
        index++;
    }
};

template <typename device_operation_t>
auto get_operation_name(const typename device_operation_t::operation_attributes_t& operation_attributes) {
    if constexpr (is_mesh_device_operation_adapter_v<device_operation_t>) {
        // For MeshAdapter operations, we recurse to get the name of the underlying device operation
        return get_operation_name<typename device_operation_t::device_operation_t>(operation_attributes);
    } else if constexpr (requires { device_operation_t::get_type_name(operation_attributes); }) {
        // TODO: remove this if statement once OldInfraDeviceOperation is removed
        return device_operation_t::get_type_name(operation_attributes);
    } else {
        return tt::stl::get_type_name<device_operation_t>();
    }
}

// GCC 12 has a bug that causes a segfault when using reflection to log tensors.
#if !defined(NDEBUG) && !defined(__GNUC__)

template <typename device_operation_t>
inline void log_operation(
    std::size_t device_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    tt::stl::hash::hash_t program_hash,
    bool program_cache_hit) {
    log_debug(
        tt::LogOp, "Launching Device Operation: \"{}\"", get_operation_name<device_operation_t>(operation_attributes));

    log_debug(tt::LogOp, "Program Hash: {}", program_hash);
    log_debug(tt::LogOp, "Program Cache Hit: {}", program_cache_hit);

    log_debug(tt::LogOp, "Attributes:");
    for (const auto& [key, value] : tt::stl::reflection::get_attributes(operation_attributes)) {
        log_debug(tt::LogOp, "\t{} = {}", key, value);
    }

    log_debug(tt::LogOp, "Tensors Args:");
    auto index = 0;
    tt::stl::reflection::visit_object_of_type<Tensor>(
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
    std::size_t device_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    tt::stl::hash::hash_t program_hash,
    bool program_cache_hit) {}

#endif

template <DeviceOperationConcept device_operation_t>
void launch_on_worker_thread(
    ttnn::QueueId cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value,
    tt::tt_metal::IDevice* device) {
    ZoneScopedN("TT_DNN_DEVICE_OP");

    auto device_operation_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();

    if constexpr (HasSkipLaunch<device_operation_t>) {
        if (device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)) {
            return;
        }
    }

    auto& program_cache = device->get_program_cache();

    auto program_hash = 0;
    bool program_cache_hit = false;

    auto is_program_cache_enabled = program_cache.is_enabled();
    if (is_program_cache_enabled) {
        program_hash = compute_program_hash<device_operation_t>(operation_attributes, tensor_args);
        program_cache_hit = program_cache.contains(program_hash);
        if (!program_cache_hit && !program_cache.cache_misses_allowed()) {
            auto op_name = get_operation_name<device_operation_t>(operation_attributes);
            TT_THROW("Device operation \"{}\": program cache miss occurred, but cache misses are forbidden", op_name);
        }
    }

    log_operation<device_operation_t>(device->id(), operation_attributes, tensor_args, program_hash, program_cache_hit);

    tt::stl::reflection::visit_object_of_type<Tensor>(CheckDeviceBufferIsAllocated{}, tensor_args);

    if (program_cache_hit) {
        ZoneScopedN("Validate on Program Cache Hit");
        device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Validate on Program Cache Miss");
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }

    const auto enqueue_or_launch_program = [=](tt::tt_metal::Program& program) {
        if (USE_FAST_DISPATCH) {
            ZoneScopedN("EnqueueProgram");
            auto& queue = device->command_queue(*cq_id);
            tt::tt_metal::EnqueueProgram(queue, program, false);
        } else {
            ZoneScopedN("LaunchProgram");
            tt::tt_metal::detail::LaunchProgram(device, program);
        }
    };

    if (is_program_cache_enabled) {
        auto& program = create_or_get_program_from_cache<device_operation_t>(
            program_cache, program_cache_hit, program_hash, operation_attributes, tensor_args, tensor_return_value);

        program.set_runtime_id(device_operation_id);

        tt::tt_metal::GraphTracker::instance().track_program(&program, device);
        if (tt::tt_metal::GraphTracker::instance().hook_program(&program)) {
            return;
        }

        enqueue_or_launch_program(program);

        TracyOpTTNNDevice(
            device_operation_t{},
            device_operation_id,
            device->id(),
            program,
            operation_attributes,
            tensor_args,
            tensor_return_value);

    } else {
        std::shared_ptr<tt::tt_metal::Program> program = std::visit(
            tt::stl::overloaded{
                [&]<ProgramFactoryConcept ProgramFactory>(
                    const ProgramFactory&) -> std::shared_ptr<tt::tt_metal::Program> {
                    auto cached_program =
                        ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
                    return std::make_shared<tt::tt_metal::Program>(std::move(cached_program.program));
                },
                [&]<MeshWorkloadFactoryConcept WorkloadFactory>(
                    const WorkloadFactory&) -> std::shared_ptr<tt::tt_metal::Program> {
                    auto cached_workload = WorkloadFactory::create_mesh_workload(
                        operation_attributes, get_unit_tensor_coords(), tensor_args, tensor_return_value);
                    TT_FATAL(cached_workload.workload.get_programs().size() == 1, "Expected 1 program in workload.");
                    return std::make_shared<tt::tt_metal::Program>(
                        std::move(cached_workload.workload.get_programs().begin()->second));
                }},
            device_operation_t::select_program_factory(operation_attributes, tensor_args));

        program->set_runtime_id(device_operation_id);

        tt::tt_metal::GraphTracker::instance().track_program(program.get(), device);
        if (tt::tt_metal::GraphTracker::instance().hook_program(program.get())) {
            return;
        }

        enqueue_or_launch_program(*program);

        TracyOpTTNNDevice(
            device_operation_t{},
            device_operation_id,
            device->id(),
            *program,
            operation_attributes,
            tensor_args,
            tensor_return_value);
    }
}

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void enqueue_mesh_workload(
    QueueId cq_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload) {
    mesh_device_operation_utils::set_runtime_id(workload);
    if (mesh_device_operation_utils::track_workload(workload, mesh_device)) {
        return;
    }
    {
        ZoneScopedN("EnqueueMeshWorkload");
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(*cq_id), workload, false);
    }

    TracyOpMeshWorkload(
        mesh_device, workload, mesh_device_operation_t{}, operation_attributes, tensor_args, tensor_return_value);
}

// Dispatches `fn` to `program_factory` through either the `MeshWorkloadFactoryConcept` directly, or through the adapted
// path for `ProgramFactoryConcept` factories.
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t, typename ProgramFactory, typename Fn>
void dispatch_to_mesh_workload_factory(const ProgramFactory& program_factory, const Fn& fn) {
    std::visit(
        tt::stl::overloaded{
            [&]<ProgramFactoryConcept T>(const T&) {
                // Adapt ProgramFactory to MeshWorkloadFactory concept.
                using AdaptedMeshWorkloadFactory = mesh_device_operation_t::template MeshWorkloadFactoryAdapter<T>;
                fn.template operator()<AdaptedMeshWorkloadFactory>();
            },
            [&]<MeshWorkloadFactoryConcept WorkloadFactory>(const WorkloadFactory&) {
                fn.template operator()<WorkloadFactory>();
            }},
        program_factory);
}

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void handle_mesh_adapter_cache_hit(
    QueueId cq_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {
    ZoneScopedN("Handle Mesh Adapter Cache Hit");
    mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);

    auto& cached_program_factory = program_cache.get(program_hash);
    auto program_factory_index = cached_program_factory.program_factory_index;
    auto program_factory = map_index_to_variant(
        program_factory_index, mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args));

    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(
        program_factory, [&]<MeshWorkloadFactoryConcept WorkloadFactory>() {
            using cached_mesh_workload_t = typename WorkloadFactory::cached_mesh_workload_t;
            auto& cached_mesh_workload = cached_program_factory.cached_program.template get<cached_mesh_workload_t>();

            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);

            enqueue_mesh_workload<mesh_device_operation_t>(
                cq_id,
                operation_attributes,
                tensor_args,
                tensor_return_value,
                mesh_device,
                cached_mesh_workload.workload);
        });
}

// Helper for creating and caching a mesh workload
template <DeviceOperationConcept mesh_device_operation_t>
void create_and_cache_mesh_workload(
    QueueId cq_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {
    ZoneScopedN("Handle Mesh Adapter Cache Miss");
    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

    auto program_factory = mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args);
    auto program_factory_index = program_factory.index();
    auto log_msg_func = []{
        log_warning(
            tt::LogOp,
            "Tensors that are distributed across mesh device unevenly negatively affect Op dispatch performance.");
    };
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(
        program_factory, [&]<MeshWorkloadFactoryConcept WorkloadFactory>() {
            using cached_mesh_workload_t = typename WorkloadFactory::cached_mesh_workload_t;

            ttnn::MeshCoordinateRangeSet tensor_coords;
            if (mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
                // Fast path - a range covers the entire mesh.
                tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device->shape()));
            } else {
                // Slow path - iterate over coordinates and merge them into a range set one by one.
                log_msg_func();  // Work around for g++12 compiler bug
                for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
                    tensor_coords.merge(ttnn::MeshCoordinateRange(coord, coord));
                }
            }
            auto cached_workload = WorkloadFactory::create_mesh_workload(
                operation_attributes, tensor_coords, tensor_args, tensor_return_value);

            if (program_cache.is_enabled()) {
                program_cache.insert(
                    program_hash, CachedProgramFactory{std::move(cached_workload), program_factory_index});
                auto& cached_program_factory = program_cache.get(program_hash);
                auto& workload = cached_program_factory.cached_program.template get<cached_mesh_workload_t>().workload;
                enqueue_mesh_workload<mesh_device_operation_t>(
                    cq_id, operation_attributes, tensor_args, tensor_return_value, mesh_device, workload);
            } else {
                enqueue_mesh_workload<mesh_device_operation_t>(
                    cq_id,
                    operation_attributes,
                    tensor_args,
                    tensor_return_value,
                    mesh_device,
                    cached_workload.workload);
            }
        });
}

// Main function to launch operations on mesh devices with special handling for MeshDeviceOperationAdapter
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch_operation_with_adapter(
    QueueId cq_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device) {
    ZoneScopedN("Launch With MeshDeviceAdapter");

    // Skip if operation should be skipped
    if constexpr (HasSkipLaunch<mesh_device_operation_t>) {
        if (mesh_device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)) {
            return;
        }
    }

    auto& program_cache = mesh_device->get_program_cache();
    auto program_hash = 0;
    bool program_cache_hit = false;

    auto is_program_cache_enabled = program_cache.is_enabled();
    if (is_program_cache_enabled) {
        // Use device_operation's compute_program_hash if available
        program_hash =
            mesh_device_operation_t::compute_mesh_workload_hash(mesh_device, operation_attributes, tensor_args);
        program_cache_hit = program_cache.contains(program_hash);
        if (!program_cache_hit && !program_cache.cache_misses_allowed()) {
            auto op_name = get_operation_name<mesh_device_operation_t>(operation_attributes);
            TT_THROW("Device operation \"{}\": program cache miss occurred, but cache misses are forbidden", op_name);
        }
    }

    log_operation<mesh_device_operation_t>(
        mesh_device->id(), operation_attributes, tensor_args, program_hash, program_cache_hit);

    tt::stl::reflection::visit_object_of_type<Tensor>(CheckDeviceBufferIsAllocated{}, tensor_args);

    if (program_cache_hit) {
        handle_mesh_adapter_cache_hit<mesh_device_operation_t>(
            cq_id, operation_attributes, tensor_args, tensor_return_value, mesh_device, program_cache, program_hash);
    } else {
        create_and_cache_mesh_workload<mesh_device_operation_t>(
            cq_id, operation_attributes, tensor_args, tensor_return_value, mesh_device, program_cache, program_hash);
    }
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_on_single_device(
    QueueId cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    ZoneScopedN("Launch Device Operation");

    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    if (!mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
        mesh_device_operation_utils::filter_tensor_shards(
            mesh_device_operation_utils::extract_tensor_coordinates(tensor_args), tensor_return_value);
    }

    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    if (auto mesh_device = first_tensor.mesh_device(); mesh_device != nullptr) {
        using MeshCompatibleOp = MeshDeviceOperationAdapter<device_operation_t>;
        launch_operation_with_adapter<MeshCompatibleOp>(
            cq_id, operation_attributes, tensor_args, tensor_return_value, mesh_device);
    } else {
        auto device = first_tensor.device();
        launch_on_worker_thread<device_operation_t>(
            cq_id, operation_attributes, tensor_args, tensor_return_value, device);
    }
    return tensor_return_value;
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t invoke(
    QueueId cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    ZoneScopedN("Run Device Operation");

    // TODO: Add GraphTracker::instance().track_device_operation to track device operations specifically?
    tt::tt_metal::GraphTracker::instance().track_function_start(
        get_operation_name<device_operation_t>(operation_attributes), operation_attributes, tensor_args);

    using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
    static_assert(not std::same_as<tensor_return_value_t, void>, "Operation return type cannot be \"void\"");

    // TODO: support the case when tensor args are empty? Or pass in the device as an argument in that case
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const auto& storage = first_tensor.storage();

    tensor_return_value_t tensor_return_value;

    TT_FATAL(std::holds_alternative<tt::tt_metal::DeviceStorage>(storage), "Unsupported storage type");
    tensor_return_value = detail::launch_on_single_device<device_operation_t>(cq_id, operation_attributes, tensor_args);

    // Should every output tensor be tracked?
    /*
    if (GraphTracker::instance().is_enabled()) {
        tensor_return_value = tt::stl::reflection::transform_object_of_type<Tensor>(tt::tt_metal::set_tensor_id,
    tensor_return_value);
    }
    */

    tt::tt_metal::GraphTracker::instance().track_function_end(tensor_return_value);
    return tensor_return_value;
}

}  // namespace detail

}  // namespace device_operation

}  // namespace ttnn
