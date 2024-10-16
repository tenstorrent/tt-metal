// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include "ttnn/tensor/tensor.hpp"

#include "tt_metal/impl/device/program_cache.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_stl/reflection.hpp"
#include "tt_metal/graph/graph_tracking.hpp"
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {

namespace device_operation {

template <typename T>
using CachedProgram = tt::tt_metal::program_cache::detail::CachedProgram<T>;

template <typename program_factory_t>
concept ProgramFactoryConcept = requires {
    [](const auto& operation_attributes, const auto& tensor_args, auto& tensor_return_value) {
        auto cached_program = program_factory_t::create(operation_attributes, tensor_args, tensor_return_value);
        program_factory_t::override_runtime_arguments(
            cached_program, operation_attributes, tensor_args, tensor_return_value);
    };
};

template <typename device_operation_t>
concept DeviceOperationConcept = requires {
    [](const typename device_operation_t::operation_attributes_t& operation_attributes,
       const typename device_operation_t::tensor_args_t& tensor_args) {
        device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

        using shape_return_value_t = typename device_operation_t::shape_return_value_t;
        static_assert(
            std::same_as<decltype(device_operation_t::compute_output_shapes(operation_attributes, tensor_args)),
                         shape_return_value_t>);

        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
        static_assert(
            std::same_as<decltype(device_operation_t::create_output_tensors(operation_attributes, tensor_args)),
                         tensor_return_value_t>);

        const auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);
        std::visit(
            [](auto&& program_factory) {
                using program_factory_t = std::decay_t<decltype(program_factory)>;
                static_assert(ProgramFactoryConcept<program_factory_t>);
            },
            program_factory);
    };
};

template <typename device_operation_t>
concept DeviceOperationWithCustomProgramCacheConcept =
    DeviceOperationConcept<device_operation_t> &&
    requires(const typename device_operation_t::operation_attributes_t& operation_attributes,
             const typename device_operation_t::tensor_args_t& tensor_args) {
        {
            device_operation_t::compute_program_hash(operation_attributes, tensor_args)
        } -> std::convertible_to<tt::stl::hash::hash_t>;
    };

namespace detail {
template <typename... Ts>
[[nodiscard]]
std::variant<Ts...> map_index_to_variant(std::size_t i, std::variant<Ts...>) {
    assert(i < sizeof...(Ts));
    static constexpr std::variant<Ts...> table[] = {Ts{}...};
    return table[i];
}

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

template <typename device_operation_t>
inline auto compute_program_hash(const typename device_operation_t::operation_attributes_t& operation_attributes,
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
inline auto& create_or_get_program_from_cache(
    auto& program_cache,
    auto program_cache_hit,
    auto program_hash,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    if (not program_cache_hit) {
        ZoneScopedN("Program Cache Miss");
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);

        auto& program = std::visit(
            [&program_cache,
             &program_hash,
             &operation_attributes,
             &tensor_args,
             &tensor_return_value,
             program_factory_index = program_factory.index()](auto&& program_factory) -> auto& {
                using program_factory_t = std::decay_t<decltype(program_factory)>;
                using cached_program_t =
                    decltype(program_factory_t::create(operation_attributes, tensor_args, tensor_return_value));
                program_cache.insert(
                    program_hash,
                    tt::tt_metal::program_cache::detail::CachedProgramFactory{
                        program_factory_t::create(operation_attributes, tensor_args, tensor_return_value),
                        program_factory_index});
                auto& cached_program_factory = program_cache.get(program_hash);
                auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();
                return cached_program.program;
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

        auto& program = std::visit(
            [&cached_program_factory, &operation_attributes, &tensor_args, &tensor_return_value](
                auto&& program_factory) -> auto& {
                using program_factory_t = std::decay_t<decltype(program_factory)>;

                using cached_program_t =
                    decltype(program_factory_t::create(operation_attributes, tensor_args, tensor_return_value));
                auto& cached_program = cached_program_factory.cached_program.template get<cached_program_t>();

                program_factory_t::override_runtime_arguments(
                    cached_program, operation_attributes, tensor_args, tensor_return_value);

                return cached_program.program;
            },
            program_factory);
        return program;
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
inline void log_operation(std::size_t device_operation_id,
                          std::size_t device_id,
                          const typename device_operation_t::operation_attributes_t& operation_attributes,
                          const typename device_operation_t::tensor_args_t& tensor_args,
                          tt::stl::hash::hash_t program_hash,
                          bool program_cache_hit) {
    tt::log_debug(
        tt::LogOp, "Launching Device Operation: \"{}\"", get_operation_name<device_operation_t>(operation_attributes));

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
inline void log_operation(std::size_t device_operation_id,
                          std::size_t device_id,
                          const typename device_operation_t::operation_attributes_t& operation_attributes,
                          const typename device_operation_t::tensor_args_t& tensor_args,
                          tt::stl::hash::hash_t program_hash,
                          bool program_cache_hit) {}

#endif

template <DeviceOperationConcept device_operation_t>
void launch_on_worker_thread(auto cq_id,
                             auto device_operation_id,
                             const auto& operation_attributes,
                             const auto& tensor_args,
                             auto& tensor_return_value,
                             auto& device) {
    ZoneScopedN("TT_DNN_DEVICE_OP");

    auto& program_cache = device->program_cache;

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

    const auto enqueue_or_launch_program = [=](Program& program) {
        if (USE_FAST_DISPATCH) {
            ZoneScopedN("EnqueueProgram");
            auto& queue = device->command_queue(cq_id);
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

        GraphTracker::instance().track_program(&program);
        if (GraphTracker::instance().hook_program(&program)) {
            return;
        }

        enqueue_or_launch_program(program);

        TracyOpTTNNDevice(device_operation_t{},
                          device_operation_id,
                          device->id(),
                          program,
                          operation_attributes,
                          tensor_args,
                          tensor_return_value);

    } else {
        auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);

        auto program = std::visit(
            [&](auto&& program_factory) {
                using program_factory_t = std::decay_t<decltype(program_factory)>;
                return std::make_shared<tt::tt_metal::Program>(
                    program_factory_t::create(operation_attributes, tensor_args, tensor_return_value).program);
            },
            program_factory);

        program->set_runtime_id(device_operation_id);

        GraphTracker::instance().track_program(program.get());
        if (GraphTracker::instance().hook_program(program.get())) {
            return;
        }

        enqueue_or_launch_program(*program);

        TracyOpTTNNDevice(device_operation_t{},
                          device_operation_id,
                          device->id(),
                          *program,
                          operation_attributes,
                          tensor_args,
                          tensor_return_value);
    }
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_on_single_device(
    uint8_t cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    ZoneScopedN("Launch Device Operation");
    auto device_operation_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();

    // Create output tensor first
    auto tensor_return_value = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
    auto device = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args).device();
    launch_on_worker_thread<device_operation_t>(
        cq_id, device_operation_id, operation_attributes, tensor_args, tensor_return_value, device);
    return tensor_return_value;
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_args_t get_shard_tensor_args(
    std::size_t index,
    auto device,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    auto get_shard = [device](const auto& tensor) {
        auto& storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        return Tensor{DeviceStorage{storage.get_buffer_for_device(device)},
                      storage.get_tensor_shape_for_device(device),
                      tensor.get_dtype(),
                      tensor.get_layout()};
    };
    return tt::stl::reflection::transform_object_of_type<Tensor>(get_shard, tensor_args);
}

static Tensor make_tensor_return_value_from_shards(auto& old_storage, std::vector<Tensor>& output_shards) {
    return distributed::create_multi_device_tensor(output_shards, StorageType::MULTI_DEVICE, old_storage.strategy);
}

static std::vector<Tensor> make_tensor_return_value_from_shards(auto& old_storage,
                                                                std::vector<std::vector<Tensor>>& output_shards) {
    auto& first_shard = output_shards[0];

    std::vector<Tensor> output;
    output.reserve(first_shard.size());

    for (auto index = 0; index < first_shard.size(); index++) {
        std::vector<Tensor> tensors;
        for (auto shard_index = 0; shard_index < output_shards.size(); shard_index++) {
            tensors.push_back(output_shards[shard_index][index]);
        }
        output.push_back(make_tensor_return_value_from_shards(old_storage, tensors));
    }
    return output;
}

static std::vector<std::optional<Tensor>> make_tensor_return_value_from_shards(
    auto& old_storage,
    std::vector<std::vector<std::optional<Tensor>>>& output_shards) {
    auto& first_shard = output_shards[0];

    std::vector<std::optional<Tensor>> output;
    output.reserve(first_shard.size());

    for (auto index = 0; index < first_shard.size(); index++) {
        if (not first_shard[index].has_value()) {
            output.push_back(std::nullopt);
            continue;
        }
        std::vector<Tensor> tensors;
        for (auto shard_index = 0; shard_index < output_shards.size(); shard_index++) {
            tensors.push_back(output_shards[shard_index][index].value());
        }
        output.push_back(make_tensor_return_value_from_shards(old_storage, tensors));
    }
    return output;
}

template <typename T>
static T make_tensor_return_value_from_shards(auto& old_storage, std::vector<T>& output_shards) {
    // TODO: add logic to handle all types we want to support generically
    TT_THROW("make_tensor_return_value_from_shards is not implemented for this type. Please add an overload");
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch_on_multi_device(
    uint8_t cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    ZoneScopedN("Launch Multi Device Operation");

    using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;

    // TODO: support the case when tensor args are empty? Or pass in the device as an argument in that case
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const auto& storage = std::get<tt::tt_metal::MultiDeviceStorage>(first_tensor.get_storage());
    using storage_t = std::remove_cvref_t<decltype(storage)>;

    auto num_shards = storage.num_buffers();

    std::vector<tensor_return_value_t> outputs;
    outputs.reserve(num_shards);

    bool launch_shards_in_parallel = false;
    if (launch_shards_in_parallel) {
        std::vector<std::future<tensor_return_value_t>> shard_futures;
        shard_futures.reserve(num_shards);

        // Launch each shard
        for (auto shard_index = 0; shard_index < num_shards; shard_index++) {
            shard_futures.emplace_back(std::async(
                std::launch::async, [cq_id, operation_attributes, tensor_args, shard_index, storage]() mutable {
                    auto device = storage.get_buffer_for_device_id(shard_index)->device();
                    auto shard_tensor_args =
                        get_shard_tensor_args<device_operation_t>(shard_index, device, tensor_args);
                    return launch_on_single_device<device_operation_t>(cq_id, operation_attributes, shard_tensor_args);
                }));
        }

        // Combine shards into a multi-device storage
        for (auto& shard_future : shard_futures) {
            outputs.push_back(shard_future.get());
        }
    } else {
        for (auto shard_index = 0; shard_index < num_shards; shard_index++) {
            auto device = storage.get_buffer_for_device_id(shard_index)->device();
            auto shard_tensor_args = get_shard_tensor_args<device_operation_t>(shard_index, device, tensor_args);
            outputs.push_back(
                launch_on_single_device<device_operation_t>(cq_id, operation_attributes, shard_tensor_args));
        }
    }

    return make_tensor_return_value_from_shards(storage, outputs);
}

template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t invoke(
    uint8_t cq_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    ZoneScopedN("Run Device Operation");

    // TODO: Add GraphTracker::instance().track_device_operation to track device operations specifically?
    GraphTracker::instance().track_function_start(
        get_operation_name<device_operation_t>(operation_attributes), operation_attributes, tensor_args);

    using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
    static_assert(not std::same_as<tensor_return_value_t, void>, "Operation return type cannot be \"void\"");

    // TODO: support the case when tensor args are empty? Or pass in the device as an argument in that case
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const auto& storage = first_tensor.get_storage();

    auto tensor_return_value = std::visit(
        [&cq_id, &operation_attributes, &tensor_args](auto&& storage) -> tensor_return_value_t {
            using storage_t = std::remove_cvref_t<decltype(storage)>;
            if constexpr (std::is_same_v<storage_t, tt::tt_metal::DeviceStorage>) {
                return detail::launch_on_single_device<device_operation_t>(cq_id, operation_attributes, tensor_args);
            } else if constexpr (std::is_same_v<storage_t, tt::tt_metal::MultiDeviceStorage>) {
                return detail::launch_on_multi_device<device_operation_t>(cq_id, operation_attributes, tensor_args);
            } else {
                TT_THROW("Unsupported storage type");
            }
        },
        storage);

    // Should every output tensor be tracked?
    /*
    if (GraphTracker::instance().is_enabled()) {
        tensor_return_value = tt::stl::reflection::transform_object_of_type<Tensor>(tt::tt_metal::set_tensor_id,
    tensor_return_value);
    }
    */

    GraphTracker::instance().track_function_end(tensor_return_value);
    return tensor_return_value;
}

}  // namespace detail

}  // namespace device_operation

}  // namespace ttnn
