// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <ttnn/decorators.hpp>

namespace ttnn::operations::rotate {

struct RotateDeviceOperation {
    struct operation_attributes_t {
        const float angle;
        const std::optional<std::tuple<float, float>> center;
        const float fill;
        const bool expand;
        const std::string interpolation_mode;
        const MemoryConfig memory_config;

        static constexpr auto attribute_names =
            std::forward_as_tuple("angle", "center", "fill", "expand", "interpolation_mode", "memory_config");
        auto attribute_values() const {
            return std::forward_as_tuple(angle, center, fill, expand, interpolation_mode, memory_config);
        }
    };

    struct tensor_args_t {
        const Tensor& input;

        static constexpr auto attribute_names = std::forward_as_tuple("input");
        auto attribute_values() const { return std::forward_as_tuple(input); }
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct NearestProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            std::size_t num_cores{};
            std::size_t num_cores_y{};
            bool is_sharded{};
            std::vector<CoreCoord> logical_cores;
            tt::tt_metal::CBHandle input_cb_handle{};
            tt::tt_metal::CBHandle output_cb_handle{};
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    struct BilinearProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            std::size_t num_cores{};
            std::size_t num_cores_y{};
            bool is_input_sharded{};
            bool is_output_sharded{};
            std::vector<CoreCoord> logical_cores;
            tt::tt_metal::CBHandle output_cb_handle{};
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<NearestProgramFactory, BilinearProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        float angle,
        const std::optional<std::tuple<float, float>>& center,
        float fill,
        bool expand,
        const std::string& interpolation_mode,
        const std::optional<MemoryConfig>& memory_config);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::rotate

namespace ttnn::prim {
constexpr auto rotate =
    ttnn::register_operation<"ttnn::prim::rotate", ttnn::operations::rotate::RotateDeviceOperation>();
}  // namespace ttnn::prim
