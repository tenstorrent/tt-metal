// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <random>
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/program_cache.hpp>
#include <tracy/Tracy.hpp>
#include "tools/profiler/op_profiler.hpp"
#include <tt_stl/reflection.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"
#include <tt-metalium/distributed.hpp>
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

namespace device_operation {

template <typename T>
using CachedProgram = tt::tt_metal::program_cache::detail::ProgramAdapter<T>;

template <typename program_factory_t>
concept ProgramFactoryConcept = requires {
    [](const auto& operation_attributes, const auto& tensor_args, auto& tensor_return_value) {
        // Check that exactly one of create or create_at is implemented.
        constexpr bool has_create =
            requires { program_factory_t::create(operation_attributes, tensor_args, tensor_return_value); };
        constexpr bool has_create_at = requires {
            program_factory_t::create_at(
                operation_attributes, std::declval<ttnn::MeshCoordinate>(), tensor_args, tensor_return_value);
        };
        static_assert(has_create != has_create_at, "Program factory must implement exactly one of create or create_at");

        if constexpr (has_create) {
            auto cached_program = program_factory_t::create(operation_attributes, tensor_args, tensor_return_value);
            program_factory_t::override_runtime_arguments(
                cached_program, operation_attributes, tensor_args, tensor_return_value);
        } else if constexpr (has_create_at) {
            auto cached_program = program_factory_t::create_at(
                operation_attributes, std::declval<ttnn::MeshCoordinate>(), tensor_args, tensor_return_value);
            program_factory_t::override_runtime_arguments_at(
                cached_program,
                operation_attributes,
                std::declval<ttnn::MeshCoordinate>(),
                tensor_args,
                tensor_return_value);
        }
    };
};

template <typename device_operation_t>
concept HasComputeOutputSpecs = requires(device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    {op.compute_output_specs(operation_attributes, tensor_args)} -> std::same_as<typename device_operation_t::spec_return_value_t>;
};

template <typename device_operation_t>
concept DeviceOperationConcept = requires {
    [](const typename device_operation_t::operation_attributes_t& operation_attributes,
       const typename device_operation_t::tensor_args_t& tensor_args) {
        device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
        static_assert(std::same_as<
                      decltype(device_operation_t::create_output_tensors(operation_attributes, tensor_args)),
                      tensor_return_value_t>);

        const auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);
        std::visit(
            [](auto&& program_factory) {
                using program_factory_t = std::decay_t<decltype(program_factory)>;
                static_assert(ProgramFactoryConcept<program_factory_t>);
            },
            program_factory);
    };
} && HasComputeOutputSpecs<device_operation_t>;

template <typename device_operation_t>
concept DeviceOperationWithCustomProgramCacheConcept =
    DeviceOperationConcept<device_operation_t> &&
    requires(
        const typename device_operation_t::operation_attributes_t& operation_attributes,
        const typename device_operation_t::tensor_args_t& tensor_args) {
        { device_operation_t::compute_program_hash(operation_attributes, tensor_args)} -> std::convertible_to<tt::stl::hash::hash_t>;
    };

template <typename device_operation_t>
concept HasSkipLaunch = requires(
    device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    {
        device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)
    } -> std::convertible_to<bool>;
};

namespace detail {

template <typename... Ts>
[[nodiscard]] std::variant<Ts...> map_index_to_variant(std::size_t i, std::variant<Ts...>) {
    assert(i < sizeof...(Ts));
    static constexpr std::variant<Ts...> table[] = {Ts{}...};
    return table[i];
}

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

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
    auto& program_cache,
    auto program_cache_hit,
    auto program_hash,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    if (not program_cache_hit) {
        ZoneScopedN("Program Cache Miss");
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);

        tt::tt_metal::Program& program = std::visit(
            [&program_cache,
             &program_hash,
             &operation_attributes,
             &tensor_args,
             &tensor_return_value,
             program_factory_index = program_factory.index()]<typename ProgramFactory>(const ProgramFactory&) -> tt::tt_metal::Program& {
                if constexpr (requires { &ProgramFactory::create; }) {
                    using cached_program_t =
                        decltype(ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value));
                    program_cache.insert(
                        program_hash,
                        tt::tt_metal::program_cache::detail::CachedProgramFactory{
                            tt::tt_metal::program_cache::detail::ProgramAdapter(
                                ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value)),
                            program_factory_index});
                    auto& cached_program_factory = program_cache.get(program_hash);
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    return cached_program.program;
                } else {
                    using cached_program_t = decltype(ProgramFactory::create_at(
                        operation_attributes, ttnn::MeshCoordinate(0, 0), tensor_args, tensor_return_value));
                    program_cache.insert(
                        program_hash,
                        tt::tt_metal::program_cache::detail::CachedProgramFactory{
                            tt::tt_metal::program_cache::detail::ProgramAdapter(ProgramFactory::create_at(
                                operation_attributes, ttnn::MeshCoordinate(0, 0), tensor_args, tensor_return_value)),
                            program_factory_index});
                    auto& cached_program_factory = program_cache.get(program_hash);
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    return cached_program.program;
                }
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
            [&cached_program_factory,
             &operation_attributes,
             &tensor_args,
             &tensor_return_value]<typename ProgramFactory>(const ProgramFactory&) -> tt::tt_metal::Program& {
                if constexpr (requires { &ProgramFactory::create; }) {
                    using cached_program_t =
                        decltype(ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value));
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    ProgramFactory::override_runtime_arguments(
                        cached_program, operation_attributes, tensor_args, tensor_return_value);
                    return cached_program.program;
                } else {
                    using cached_program_t = decltype(ProgramFactory::create_at(
                        operation_attributes, std::declval<ttnn::MeshCoordinate>(), tensor_args, tensor_return_value));
                    auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                    ProgramFactory::override_runtime_arguments_at(
                        cached_program,
                        operation_attributes,
                        ttnn::MeshCoordinate(0, 0),
                        tensor_args,
                        tensor_return_value);
                    return cached_program.program;
                }
            },
            program_factory);
        return program;
    }
}

template <typename device_operation_t>
inline auto& create_or_get_meshworkload_from_cache(
    auto& program_cache,
    auto program_cache_hit,
    auto program_hash,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    uint64_t device_operation_id) {
    if (!program_cache_hit) {
        ZoneScopedN("Program Cache Miss");
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);
        auto program_factory_index = program_factory.index();

        auto& mesh_workload = std::visit(
            [&program_factory_index,
             &program_hash,
             &program_cache,
             &operation_attributes,
             &tensor_args,
             &tensor_return_value,
             &device_operation_id,
             &mesh_device]<typename ProgramFactory>(const ProgramFactory&) -> auto& {
                auto make_program = [&](const ttnn::MeshCoordinate& coord) {
                    if constexpr (requires { &ProgramFactory::create; }) {
                        auto cached_program =
                            ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
                        cached_program.program.set_runtime_id(device_operation_id);
                        return cached_program;
                    } else {
                        auto cached_program =
                            ProgramFactory::create_at(operation_attributes, coord, tensor_args, tensor_return_value);
                        cached_program.program.set_runtime_id(device_operation_id);
                        return cached_program;
                    }
                };

                tt::tt_metal::distributed::MeshWorkload mesh_workload;
                std::unordered_map<ttnn::MeshCoordinateRange, typename ProgramFactory::shared_variables_t>
                    coordinate_range_to_shared_variables;

                if (!mesh_device_operation_utils::uses_heterogenous_dispatch<device_operation_t, ProgramFactory>() &&
                    mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
                    auto cached_program = make_program(ttnn::MeshCoordinate(0, 0));
                    const ttnn::MeshCoordinateRange coordinate_range(mesh_device->shape());
                    mesh_workload.add_program(coordinate_range, std::move(cached_program.program));
                    coordinate_range_to_shared_variables[coordinate_range] = std::move(cached_program.shared_variables);
                } else {
                    for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
                        auto cached_program = make_program(coord);
                        const ttnn::MeshCoordinateRange coordinate_range(coord, coord);
                        mesh_workload.add_program(coordinate_range, std::move(cached_program.program));
                        coordinate_range_to_shared_variables[coordinate_range] =
                            std::move(cached_program.shared_variables);
                    }
                }

                auto cached_mesh_workload = tt::tt_metal::program_cache::detail::CachedMeshWorkload<
                    typename ProgramFactory::shared_variables_t>(
                    std::move(mesh_workload), std::move(coordinate_range_to_shared_variables));

                // Create a program adapter to wrap the cached mesh workload
                tt::tt_metal::program_cache::detail::ProgramAdapter<typename ProgramFactory::shared_variables_t>
                    adapter(std::move(cached_mesh_workload));

                // Insert the cached program factory into the cache
                program_cache.insert(
                    program_hash,
                    tt::tt_metal::program_cache::detail::CachedProgramFactory{
                        std::move(adapter), program_factory_index});

                // Return the mesh workload from the cached factory
                auto& cached_program_factory = program_cache.get(program_hash);
                // Get the program adapter from the cached factory
                auto& cached_adapter = cached_program_factory.cached_program
                                           .template get<tt::tt_metal::program_cache::detail::ProgramAdapter<
                                               typename ProgramFactory::shared_variables_t>>();
                return cached_adapter.get_cached_mesh_workload().workload;
            },
            program_factory);
        return mesh_workload;
    } else {
        ZoneScopedN("Program Cache Hit");
        auto& cached_program_factory = program_cache.get(program_hash);
        auto program_factory_index = cached_program_factory.program_factory_index;

        // Reconstruct the program factory variant based on the stored index
        using program_factory_variant_t =
            decltype(device_operation_t::select_program_factory(operation_attributes, tensor_args));
        auto program_factory = map_index_to_variant(program_factory_index, program_factory_variant_t{});

        // Use std::visit to override runtime arguments using the selected factory
        auto& mesh_workload = std::visit(
            [&program_factory_index,
             &operation_attributes,
             &tensor_args,
             &tensor_return_value,
             &device_operation_id,
             &mesh_device,
             &cached_program_factory]<typename ProgramFactory>(const ProgramFactory&) -> auto& {
                // Get the program adapter from the cached factory
                auto& adapter = cached_program_factory.cached_program.template get<
                    tt::tt_metal::program_cache::detail::ProgramAdapter<typename ProgramFactory::shared_variables_t>>();

                // Override runtime arguments through the adapter
                // TODO: override runtime args per coordinate
                if constexpr (requires { &ProgramFactory::override_runtime_arguments_at; }) {
                    ProgramFactory::override_runtime_arguments_at(
                        adapter, operation_attributes, ttnn::MeshCoordinate(0, 0), tensor_args, tensor_return_value);
                } else {
                    ProgramFactory::override_runtime_arguments(
                        adapter, operation_attributes, tensor_args, tensor_return_value);
                }

                // Set runtime ID for all programs
                for (auto& [_, program] : adapter.get_cached_mesh_workload().workload.get_programs()) {
                    program.set_runtime_id(device_operation_id);
                    tt::tt_metal::GraphTracker::instance().track_program(&program, mesh_device);
                }

                // Return the mesh workload from the cached factory
                return adapter.get_cached_mesh_workload().workload;
            },
            program_factory);
        return mesh_workload;
    }
}

struct CheckDeviceBufferIsAllocated {
    std::size_t index = 0;

    void operator()(const Tensor& tensor) {
        if (not tensor.is_allocated()) {
            tt::log_debug(tt::LogOp, "Tensor at index {} is not allocated", index);
        }
        index++;
    }
};

template <typename device_operation_t>
auto get_operation_name(const typename device_operation_t::operation_attributes_t& operation_attributes) {
    if constexpr (requires { device_operation_t::get_type_name(operation_attributes); }) {
        // TODO: remove this if statement once OldInfraDeviceOperation is removed
        return device_operation_t::get_type_name(operation_attributes);
    } else {
        return tt::stl::get_type_name<device_operation_t>();
    }
}


#ifdef DEBUG

template <typename device_operation_t>
inline void log_operation(
    std::size_t device_operation_id,
    std::size_t device_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    tt::stl::hash::hash_t program_hash,
    bool program_cache_hit) {
    tt::log_debug(
        tt::LogOp, "Launching Device Operation: \"{}\"",
        get_operation_name<device_operation_t>(operation_attributes));

    tt::log_debug(tt::LogOp, "Program Hash: {}", program_hash);
    tt::log_debug(tt::LogOp, "Program Cache Hit: {}", program_cache_hit);

    tt::log_debug(tt::LogOp, "Attributes:");
    for (const auto& [key, value] : tt::stl::reflection::get_attributes(operation_attributes)) {
        tt::log_debug(tt::LogOp, "\t{} = {}", key, value);
    }

    tt::log_debug(tt::LogOp, "Tensors Args:");
    auto index = 0;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&index](auto&& tensor) {
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
            index++;
        },
        tensor_args);

    tt::log_debug(tt::LogOp, "");
}


#else

template <typename device_operation_t>
inline void log_operation(
    std::size_t device_operation_id,
    std::size_t device_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    tt::stl::hash::hash_t program_hash,
    bool program_cache_hit) {}

#endif



template <DeviceOperationConcept device_operation_t>
void launch_on_worker_thread(
    ttnn::QueueId cq_id,
    uint64_t device_operation_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value,
    tt::tt_metal::IDevice* device) {
    ZoneScopedN("TT_DNN_DEVICE_OP");

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
    }

    log_operation<device_operation_t>(
            device_operation_id,
            device->id(),
            operation_attributes,
            tensor_args,
            program_hash,
            program_cache_hit
        );

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
        auto program = std::visit(
            [&]<typename ProgramFactory>(const ProgramFactory&) {
                if constexpr (requires { &ProgramFactory::create; }) {
                    auto cached_program =
                        ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
                    return std::make_shared<tt::tt_metal::Program>(std::move(cached_program.program));
                } else {
                    auto cached_program = ProgramFactory::create_at(
                        operation_attributes, ttnn::MeshCoordinate(0, 0), tensor_args, tensor_return_value);
                    return std::make_shared<tt::tt_metal::Program>(std::move(cached_program.program));
                }
            },
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

template <DeviceOperationConcept device_operation_t>
void launch_on_mesh_device(
    ttnn::QueueId cq_id,
    uint64_t device_operation_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* device) {
    ZoneScopedN("TT_DNN_DEVICE_OP");

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
    }

    log_operation<device_operation_t>(
        device_operation_id, device->id(), operation_attributes, tensor_args, program_hash, program_cache_hit);

    tt::stl::reflection::visit_object_of_type<Tensor>(CheckDeviceBufferIsAllocated{}, tensor_args);

    if (program_cache_hit) {
        ZoneScopedN("Validate on Program Cache Hit");
        device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Validate on Program Cache Miss");
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }

    const auto enqueue_mesh_workload = [=](tt::tt_metal::distributed::MeshWorkload& mesh_workload) {
        ZoneScopedN("EnqueueProgram");
        tt::tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(*cq_id), mesh_workload, false);
    };

    if (is_program_cache_enabled) {
        auto& mesh_workload = create_or_get_meshworkload_from_cache<device_operation_t>(
            program_cache,
            program_cache_hit,
            program_hash,
            operation_attributes,
            tensor_args,
            tensor_return_value,
            device,
            device_operation_id);
        enqueue_mesh_workload(mesh_workload);

        // Log the first program only
        TT_FATAL(mesh_workload.get_programs().size() >= 1, "Mesh workload must contain at least one program");
        const auto& first_program = mesh_workload.get_programs().begin()->second;
        TracyOpTTNNDevice(
            device_operation_t{},
            device_operation_id,
            device->id(),
            first_program,
            operation_attributes,
            tensor_args,
            tensor_return_value);

    } else {
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);
        constexpr bool has_create =
            requires { program_factory.create(operation_attributes, tensor_args, tensor_return_value); };

        bool tracker_hook_enabled = false;
        auto make_program = [&](const ttnn::MeshCoordinate& coord) {
            return std::visit(
                [&]<typename ProgramFactory>(const ProgramFactory&) {
                    std::shared_ptr<tt::tt_metal::Program> program;
                    if constexpr (requires { &ProgramFactory::create; }) {
                        auto cached_program =
                            ProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
                        program = std::make_shared<tt::tt_metal::Program>(std::move(cached_program.program));
                    } else {
                        auto cached_program =
                            ProgramFactory::create_at(operation_attributes, coord, tensor_args, tensor_return_value);
                        program = std::make_shared<tt::tt_metal::Program>(std::move(cached_program.program));
                    }
                    program->set_runtime_id(device_operation_id);
                    tt::tt_metal::GraphTracker::instance().track_program(program.get(), device);
                    tracker_hook_enabled = tt::tt_metal::GraphTracker::instance().hook_program(program.get());
                    return program;
                },
                program_factory);
        };

        tt::tt_metal::distributed::MeshWorkload mesh_workload;
        if (!mesh_device_operation_utils::uses_heterogenous_dispatch<device_operation_t, decltype(program_factory)>() &&
            mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
            auto program = make_program(ttnn::MeshCoordinate::zero_coordinate(device->shape().dims()));
            mesh_workload.add_program(ttnn::MeshCoordinateRange(device->shape()), std::move(*program));
        } else {
            for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
                auto program = make_program(coord);
                mesh_workload.add_program(ttnn::MeshCoordinateRange(coord, coord), std::move(*program));
            }
        }

        if (tracker_hook_enabled) {
            return;
        }

        enqueue_mesh_workload(mesh_workload);

        TT_FATAL(mesh_workload.get_programs().size() >= 1, "Mesh workload must contain at least one program");
        const auto& first_program = mesh_workload.get_programs().begin()->second;
        TracyOpTTNNDevice(
            device_operation_t{},
            device_operation_id,
            device->id(),
            first_program,
            operation_attributes,
            tensor_args,
            tensor_return_value);
    }
}

/**
 * @brief Concept that defines a device operation that has a mesh device adapter.
 *
 * This concept requires that the type satisfies both the DeviceOperationConcept
 * and the MeshDeviceOperationAdapterType concept. It represents operations that
 * can be executed across multiple devices in a mesh configuration using the
 * adapter pattern.
 */
template <typename device_operation_t>
concept DeviceOperationWithMeshDeviceAdapter =
    DeviceOperationConcept<device_operation_t> && MeshDeviceOperationAdapterType<device_operation_t>;

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void handle_mesh_adapter_cache_hit(
    QueueId cq_id,
    uint64_t device_operation_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {
    ZoneScopedN("MeshDeviceAdapter Cache Hit");
    mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);

    auto& cached_program_factory = program_cache.get(program_hash);
    auto program_factory_index = cached_program_factory.program_factory_index;

    using program_factory_variant_t =
        decltype(mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args));
    auto program_factory = map_index_to_variant(program_factory_index, program_factory_variant_t{});

    std::visit([&]<typename ProgramFactory>(const ProgramFactory&) {
        using shared_variables_t = typename ProgramFactory::shared_variables_t;

        // Get the adapter from cache with the correct shared variables type
        auto& adapter = cached_program_factory.cached_program.template get<
            tt::tt_metal::program_cache::detail::ProgramAdapter<shared_variables_t>>();

        // Override runtime arguments using the MeshDeviceOperationAdapter interface
        mesh_device_operation_t::override_mesh_runtime_arguments(
            adapter.get_cached_mesh_workload(),
            mesh_device,
            operation_attributes,
            tensor_args,
            tensor_return_value);

        // Set runtime ID for all programs
        for (auto& [_, program] : adapter.get_cached_mesh_workload().workload.get_programs()) {
            program.set_runtime_id(device_operation_id);
            tt::tt_metal::GraphTracker::instance().track_program(&program, mesh_device);
            if (tt::tt_metal::GraphTracker::instance().hook_program(&program)) {
                return;
            }
        }

        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(*cq_id), adapter.get_cached_mesh_workload().workload, false);

        if (!adapter.get_cached_mesh_workload().workload.get_programs().empty()) {
            const auto& first_program = adapter.get_cached_mesh_workload().workload.get_programs().begin()->second;
            TracyOpTTNNDevice(
                mesh_device_operation_t{},
                device_operation_id,
                mesh_device->id(),
                first_program,
                operation_attributes,
                tensor_args,
                tensor_return_value);
        }
    }, program_factory);
}

// Helper for creating and caching a mesh workload
template <DeviceOperationConcept mesh_device_operation_t>
void create_and_cache_mesh_workload(
    QueueId cq_id,
    uint64_t device_operation_id,
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {

    ZoneScopedN("MeshDeviceAdapter Cache Miss");

    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

    auto cached_workload = mesh_device_operation_t::create_mesh_workload(
        mesh_device,operation_attributes, tensor_args, tensor_return_value);

    // Set runtime ID for all programs
    for (auto& [_, program] : cached_workload.workload.get_programs()) {
        program.set_runtime_id(device_operation_id);
        tt::tt_metal::GraphTracker::instance().track_program(&program, mesh_device);
        if (tt::tt_metal::GraphTracker::instance().hook_program(&program)) {
            return;
        }
    }

    if (program_cache.is_enabled()) {
        auto program_factory = mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args);
        auto program_factory_index = program_factory.index();

        std::visit([&]<typename ConcreteFactory>(const ConcreteFactory&) {
            using concrete_shared_vars_t = typename ConcreteFactory::shared_variables_t;

            using namespace tt::tt_metal::program_cache::detail;
            auto cmw = CachedMeshWorkload<concrete_shared_vars_t>(
                std::move(cached_workload.workload),
                std::move(cached_workload.coordinate_range_to_shared_variables));

            ProgramAdapter<concrete_shared_vars_t> adapter(std::move(cmw));

            program_cache.insert(
                program_hash,
                CachedProgramFactory{std::move(adapter), program_factory_index});

            auto& cached_program_factory = program_cache.get(program_hash);
            auto& cached_adapter = cached_program_factory.cached_program.template get<
                tt::tt_metal::program_cache::detail::ProgramAdapter<concrete_shared_vars_t>>();

            // Enqueue the workload
            tt::tt_metal::distributed::EnqueueMeshWorkload(
                mesh_device->mesh_command_queue(*cq_id), cached_adapter.get_cached_mesh_workload().workload, false);

            // Log the first program for profiling
            const auto& programs = cached_adapter.get_cached_mesh_workload().workload.get_programs();
            if (!programs.empty()) {
                const auto& first_program = programs.begin()->second;
                TracyOpTTNNDevice(
                    mesh_device_operation_t{},
                    device_operation_id,
                    mesh_device->id(),
                    first_program,
                    operation_attributes,
                    tensor_args,
                    tensor_return_value);
            }
        }, program_factory);
    } else {
        // Enqueue the workload directly (no caching)
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(*cq_id), cached_workload.workload, false);

        // Log the first program for profiling
        if (!cached_workload.workload.get_programs().empty()) {
            const auto& first_program = cached_workload.workload.get_programs().begin()->second;
            TracyOpTTNNDevice(
                mesh_device_operation_t{},
                device_operation_id,
                mesh_device->id(),
                first_program,
                operation_attributes,
                tensor_args,
                tensor_return_value);
        }
    }
}

// Main function to launch operations on mesh devices with special handling for MeshDeviceOperationAdapter
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch_operation_with_adapter(
    QueueId cq_id,
    uint64_t device_operation_id,
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
        program_hash = mesh_device_operation_t::compute_mesh_workload_hash(mesh_device, operation_attributes, tensor_args);
        program_cache_hit = program_cache.contains(program_hash);
    }

    log_operation<mesh_device_operation_t>(
        device_operation_id, mesh_device->id(), operation_attributes, tensor_args, program_hash, program_cache_hit);

    tt::stl::reflection::visit_object_of_type<Tensor>(CheckDeviceBufferIsAllocated{}, tensor_args);

    if (program_cache_hit) {
        handle_mesh_adapter_cache_hit<mesh_device_operation_t>(
            cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value,
            mesh_device, program_cache, program_hash);
    } else {
        create_and_cache_mesh_workload<mesh_device_operation_t>(
            cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value,
            mesh_device, program_cache, program_hash);
    }
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_on_single_device(
    QueueId cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {

    // TODO(jchu): Revisit whether we should automatically use mesh adapter version if not already an adapter
    /*
    if constexpr (!is_mesh_device_operation_adapter<device_operation_t>::value) {
        using MeshCompatibleOp = MeshDeviceOperationAdapter<device_operation_t>;
        return launch_on_single_device<MeshCompatibleOp>(cq_id, operation_attributes, tensor_args);
    }
    */

    ZoneScopedN("Launch Device Operation");
    auto device_operation_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();

    // Create output tensor first
    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    if (auto mesh_device = first_tensor.mesh_device(); mesh_device != nullptr) {
        if constexpr (MeshDeviceOperationAdapterType<device_operation_t>) {
            launch_operation_with_adapter<device_operation_t>(
                cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value, mesh_device);
        } else {
            launch_on_mesh_device<device_operation_t>(
                cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value, mesh_device);
        }
    } else {
        auto device = first_tensor.device();
        launch_on_worker_thread<device_operation_t>(
            cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value, device);
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
    tt::tt_metal::GraphTracker::instance().track_function_start(get_operation_name<device_operation_t>(operation_attributes), operation_attributes, tensor_args);


    using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
    static_assert(not std::same_as<tensor_return_value_t, void>, "Operation return type cannot be \"void\"");

    // TODO: support the case when tensor args are empty? Or pass in the device as an argument in that case
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const auto& storage = first_tensor.get_storage();

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
