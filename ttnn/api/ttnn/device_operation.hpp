// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/indestructible.hpp>
#include "ttnn/tensor/tensor.hpp"
#include <unordered_map>

#include <tt-metalium/program_cache.hpp>
#include <tracy/Tracy.hpp>
#include "tools/profiler/op_profiler.hpp"
#include <tt_stl/reflection.hpp>
#include <tt_stl/concepts.hpp>
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

// Helper to create a mesh workload from a WorkloadFactory that may or may not
// provide create_mesh_workload. If missing, synthesize it from create_at.
template <typename WorkloadFactory, typename device_operation_t>
static auto create_mesh_workload_from_workload_factory(
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
            tt::stl::concepts::always_false_v<WorkloadFactory>,
            "WorkloadFactory must implement create_mesh_workload(operation_attributes, tensor_coords, tensor_args, "
            "tensor_return_value) or create_at(operation_attributes, mesh_coordinate, tensor_args, "
            "tensor_return_value, tensor_coords)");
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

// GCC 12 has a bug that causes a segfault when using reflection to log tensors in debug mode.
// If building with clang, allow this to be compiled
#if !defined(NDEBUG) && defined(__clang__)

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
    for ([[maybe_unused]] const auto& [key, value] : tt::stl::reflection::get_attributes(operation_attributes)) {
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

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void enqueue_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload) {
    mesh_device_operation_utils::set_runtime_id(workload);
    if (mesh_device_operation_utils::track_workload(workload, mesh_device)) {
        return;
    }
    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

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
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {
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
                operation_attributes, tensor_args, tensor_return_value, mesh_device, cached_mesh_workload.workload);
        });
}

// Helper for creating and caching a mesh workload
template <DeviceOperationConcept mesh_device_operation_t>
void create_and_cache_mesh_workload(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::program_cache::detail::ProgramCache& program_cache,
    tt::stl::hash::hash_t program_hash) {
    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

    auto program_factory = mesh_device_operation_t::select_program_factory(operation_attributes, tensor_args);
    auto program_factory_index = program_factory.index();
    auto log_msg_func = [] {
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
            auto cached_workload = create_mesh_workload_from_workload_factory<WorkloadFactory, mesh_device_operation_t>(
                operation_attributes, tensor_coords, tensor_args, tensor_return_value);

            if (program_cache.is_enabled()) {
                program_cache.insert(
                    program_hash, CachedProgramFactory{std::move(cached_workload), program_factory_index});
                auto& cached_program_factory = program_cache.get(program_hash);
                auto& workload = cached_program_factory.cached_program.template get<cached_mesh_workload_t>().workload;
                enqueue_mesh_workload<mesh_device_operation_t>(
                    operation_attributes, tensor_args, tensor_return_value, mesh_device, workload);
            } else {
                enqueue_mesh_workload<mesh_device_operation_t>(
                    operation_attributes, tensor_args, tensor_return_value, mesh_device, cached_workload.workload);
            }
        });
}

// Main function to launch operations on mesh devices with special handling for MeshDeviceOperationAdapter
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch_operation_with_adapter(
    const typename mesh_device_operation_t::operation_attributes_t& operation_attributes,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& tensor_return_value,
    ttnn::MeshDevice* mesh_device) {
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
            operation_attributes, tensor_args, tensor_return_value, mesh_device, program_cache, program_hash);
    } else {
        create_and_cache_mesh_workload<mesh_device_operation_t>(
            operation_attributes, tensor_args, tensor_return_value, mesh_device, program_cache, program_hash);
    }
}

// Default TensorTopology for output tensors is determined only by the input tensors with the highest distribution rank
// (highest number of dimensions). The output tensor will have the same distribution rank as these input tensors, taking
// the max strides of all input tensors. The placement for each distribution dimension will be Shard if at least one
// input tensor has a Shard placement for that dimension, otherwise it will be Replicate. Duplicate Shard placements are
// disallowed, Replicate is used instead. If two input tensors shard different tensor dimensions across the same
// distribution dimension, the earlier-seen shard dimension is kept.
template <DeviceOperationConcept device_operation_t>
std::pair<
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
get_output_placements_and_shape(
    const typename device_operation_t::tensor_args_t& tensor_args, const Tensor& first_tensor) {
    size_t max_distribution_rank = 0;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) {
            max_distribution_rank =
                std::max(max_distribution_rank, tensor.tensor_topology().distribution_shape().dims());
        },
        tensor_args);

    auto result_strides = tt::stl::SmallVector<uint32_t>(max_distribution_rank, 1);
    auto result_placements = tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>(
        max_distribution_rank, tt::tt_metal::distributed::MeshMapperConfig::Replicate{});
    std::unordered_set<int> shard_dims;
    bool dim_mismatch = false;

    // TODO: #25340 - Add back logging / validation. Currently, this results in a lot of log spam.
    constexpr bool kEnableLogging = false;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) {
            // Augment output tensor distribution shape with the max strides of all input tensors with the max
            // distribution rank
            const auto& tensor_distribution_shape = tensor.tensor_topology().distribution_shape();
            if (tensor_distribution_shape.dims() == max_distribution_rank) {
                for (int i = 0; i < std::min(result_strides.size(), tensor_distribution_shape.dims()); i++) {
                    result_strides[i] = std::max(result_strides[i], tensor_distribution_shape[i]);
                }

                const auto& tensor_placements = tensor.tensor_topology().placements();
                for (int i = 0; i < tensor_placements.size(); i++) {
                    tt::tt_metal::distributed::MeshMapperConfig::Placement output_placement = result_placements[i];
                    if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(
                            tensor_placements[i])) {
                        auto new_shard_placement =
                            std::get<tt::tt_metal::distributed::MeshMapperConfig::Shard>(tensor_placements[i]);

                        // Only shard if the tensor dimension is not already sharded
                        if (!shard_dims.contains(new_shard_placement.dim)) {
                            shard_dims.insert(new_shard_placement.dim);
                            if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(
                                    output_placement)) {
                                auto existing_shard_placement =
                                    std::get<tt::tt_metal::distributed::MeshMapperConfig::Shard>(output_placement);

                                // If a different tensor dim is sharded across this distribution dim, keep the
                                // earliest-seen shard dimension.
                                if (new_shard_placement.dim != existing_shard_placement.dim && kEnableLogging) {
                                    log_warning(
                                        tt::LogOp,
                                        "Output tensor cannot shard different tensor dimensions across the same "
                                        "distribution "
                                        "dimension: tensor dims {} (kept) and {} (ignored) across distribution dim {}",
                                        existing_shard_placement.dim,
                                        new_shard_placement.dim,
                                        i);
                                }
                                continue;
                            }
                            output_placement = new_shard_placement;
                        } else if (kEnableLogging) {
                            log_warning(
                                tt::LogOp,
                                "Duplicate tensor shard dimension {} across distribution dim {} replaced with "
                                "Replicate",
                                new_shard_placement.dim,
                                i);
                        }
                    }
                    result_placements[i] = output_placement;
                }
            } else {
                dim_mismatch = true;
            }
        },
        tensor_args);
    if (dim_mismatch && kEnableLogging) {
        log_warning(
            tt::LogOp,
            "Input tensors have different distribution ranks, only imputing output tensor topology with tensors that "
            "have the max distribution rank");
    }
    return {result_placements, tt::tt_metal::distributed::MeshShape(result_strides)};
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_on_device(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    if (!mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
        mesh_device_operation_utils::filter_tensor_shards(
            mesh_device_operation_utils::extract_tensor_coordinates(tensor_args), tensor_return_value);
    }

    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    auto mesh_device = first_tensor.device();
    auto [output_topology_placements, output_topology_shape] =
        detail::get_output_placements_and_shape<device_operation_t>(tensor_args, first_tensor);

    tensor_return_value = tt::stl::reflection::transform_object_of_type<Tensor>(
        [&output_topology_placements, &output_topology_shape](const Tensor& output_tensor) {
            auto topology = tt::tt_metal::TensorTopology(
                output_topology_shape, output_topology_placements, output_tensor.tensor_topology().mesh_coords());
            return output_tensor.with_tensor_topology(topology);
        },
        tensor_return_value);

    launch_operation_with_adapter<MeshDeviceOperationAdapter<device_operation_t>>(
        operation_attributes, tensor_args, tensor_return_value, mesh_device);
    return tensor_return_value;
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t invoke(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
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
    tensor_return_value = detail::launch_on_device<device_operation_t>(operation_attributes, tensor_args);

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
