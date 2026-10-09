// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/concat/device/concat_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

ConcatDeviceOperation::program_factory_t ConcatDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.input_tensors.empty(), "ConcatDeviceOperation: input_tensors cannot be empty");
    return ConcatProgramFactory{};
}

ttsl::hash::hash_t ConcatDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Aliasing signature: for each input, the position of the first input backed by the same
    // MeshTensor. The Metal 2.0 spec-factory cache path (resolve_bindings, in mesh_device_operation
    // _adapter.hpp) records each input's argument position by first MeshTensor-address match and
    // freezes that table into the cache entry, so concat([x, x]) binds both inputs to position 0.
    // Folding this partition into the hash gives aliased and distinct call patterns separate cache
    // entries: a later concat([a, b]) of the same spec then misses and rebinds both inputs, instead
    // of hitting the [x, x] entry and silently reading the first input twice. (Same fix as the
    // original op's.)
    const auto& input_tensors = tensor_args.input_tensors;
    std::vector<uint32_t> input_alias_signature(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        input_alias_signature[i] = i;
        for (uint32_t j = 0; j < i; ++j) {
            if (&input_tensors[j].mesh_tensor() == &input_tensors[i].mesh_tensor()) {
                input_alias_signature[i] = j;
                break;
            }
        }
    }

    return tt::tt_metal::operation::hash_operation<ConcatDeviceOperation>(
        operation_attributes, tensor_args, input_alias_signature);
}

void ConcatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "need 1 or more tensors");
    TT_FATAL(
        input_tensors.size() <= max_inputs_per_concat_program,
        "ttnn.experimental.quasar.concat: {} inputs exceed the {} one program can take (DM reader stack); the host "
        "op concatenates longer lists in batches",
        input_tensors.size(),
        max_inputs_per_concat_program);

    const auto& first_input = input_tensors[0];
    auto shape_first = first_input.logical_shape();
    const uint32_t rank = shape_first.rank();
    TT_FATAL(args.dim < rank, "ConcatDeviceOperation dim specified is larger than input tensor rank.");
    shape_first[args.dim] = 0;
    const bool shard_first = first_input.is_sharded();
    const bool rm_layout = first_input.layout() == Layout::ROW_MAJOR;

    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        const auto& in_ref = input_tensors[i];
        TT_FATAL(in_ref.buffer(), "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.device(), "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.device() == first_input.device(), "Operands to concat need to be on the same device.");
        TT_FATAL(in_ref.layout() == first_input.layout(), "All Tensors should have same layouts.");
        TT_FATAL(in_ref.dtype() == first_input.dtype(), "All Tensors should have same dtypes.");
        auto curr_shape = in_ref.logical_shape();
        TT_FATAL(curr_shape.rank() == shape_first.rank(), "Input tensor ranks must be equal");
        curr_shape[args.dim] = 0;
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        TT_FATAL(in_ref.is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
        if (shard_first) {
            TT_FATAL(in_ref.shard_spec().has_value(), "Sharded tensors must have a shard spec.");
            TT_FATAL(
                in_ref.shard_spec().value().grid == first_input.shard_spec().value().grid,
                "Sharded tensors must have the same grid.");
            TT_FATAL(
                in_ref.memory_config().memory_layout() == first_input.memory_config().memory_layout(),
                "Sharded tensors must have the same memory layout.");
            TT_FATAL(
                in_ref.shard_spec().value().orientation == first_input.shard_spec().value().orientation,
                "Sharded tensors must have the same shard orientation.");
        }
        // The factory concatenates whole pages. A tile-padded concat dim is only expressible that way
        // on the last input, whose padding tiles become the output's; the host op untilizes, concats
        // row-major and retilizes everything else (the original's tiled-unaligned factory is not
        // ported).
        if (!rm_layout && i + 1 < input_tensors.size() && args.dim + 2 >= rank) {
            TT_FATAL(
                in_ref.logical_shape()[args.dim] == in_ref.padded_shape()[args.dim],
                "ttnn.experimental.quasar.concat: input {} is tile-padded on concat dim {} ({} logical, {} padded); "
                "only the last input may be",
                i,
                args.dim,
                in_ref.logical_shape()[args.dim],
                in_ref.padded_shape()[args.dim]);
        }
        // A width concat lays each input's row next to the previous one inside the staging page, so
        // every input row has to end on an alignment boundary.
        if (rm_layout && args.dim + 1 == rank) {
            TT_FATAL(
                (in_ref.padded_shape()[args.dim] * in_ref.element_size()) % in_ref.buffer()->alignment() == 0,
                "Current concat implementation requires aligned last dim when concatting on last dim");
        }
    }

    // ROW_MAJOR pages are whole rows only for interleaved and height-sharded buffers; width-, block-
    // and ND-sharded ones page by shard width, which the factory cannot assemble rows from. The host
    // op stages those through interleaved tensors.
    auto rm_pages_are_rows = [](const MemoryConfig& memory_config) {
        return !memory_config.is_sharded() || memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    };
    if (rm_layout) {
        TT_FATAL(
            rm_pages_are_rows(first_input.memory_config()) && rm_pages_are_rows(args.output_mem_config),
            "ttnn.experimental.quasar.concat: ROW_MAJOR {} input / {} output page by shard width; only interleaved "
            "and height-sharded ROW_MAJOR tensors are supported by the device op",
            first_input.memory_config().memory_layout(),
            args.output_mem_config.memory_layout());
    }
}

tt::tt_metal::TensorSpec ConcatDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& ref_in_tensor = tensor_args.input_tensors.at(0);
    ttnn::Shape shape_out = ref_in_tensor.logical_shape();
    shape_out[args.dim] = 0;
    for (const Tensor& in_ref : tensor_args.input_tensors) {
        ttnn::Shape curr_shape = in_ref.logical_shape();
        shape_out[args.dim] += curr_shape[args.dim];
    }

    return tt::tt_metal::TensorSpec(
        shape_out, TensorLayout(ref_in_tensor.dtype(), PageConfig(ref_in_tensor.layout()), args.output_mem_config));
}

Tensor ConcatDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensors[0].device());
}

ConcatDeviceOperation::tensor_return_value_t concat(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    using OperationType = ConcatDeviceOperation;
    uint32_t normalized_dim = input_tensors[0].logical_shape().get_normalized_index(dim);
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = normalized_dim,
            .output_mem_config = output_mem_config,
            .sub_core_grids = sub_core_grids,
        },
        OperationType::tensor_args_t{.input_tensors = input_tensors});
}

}  // namespace ttnn::prim::qsr
