// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

ttnn::Shape squeeze_shape_to_ND(const ttnn::Shape& output_shape, uint32_t);

ttnn::Shape squeeze_shape_to_4D(const ttnn::Shape& output_shape);
ttnn::Shape squeeze_shape_to_3D(const ttnn::Shape& output_shape);

ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor);
ttnn::Shape unsqueeze_shape_to_3D(const ttnn::Shape& shape);
ttnn::Shape unsqueeze_shape_to_4D(const ttnn::Shape& shape);

ttnn::Shape unsqueeze_shape_to_nd(const ttnn::Shape& shape, uint32_t n);

ttnn::Shape squeeze_or_unsqueeze_shape_to_ND(const ttnn::Shape& shape, uint32_t n);

uint32_t get_estimated_size_of_cbs(
    const Tensor& input_tensor_a,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t num_tiles_per_row);

uint32_t get_max_l1_space(const Tensor& input_tensor_a);

bool is_enough_space(
    const Tensor& input_tensor_a,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t num_tiles_per_row);

ttnn::Tensor pad_to_tile_vol(
    QueueId queue_id,
    const ttnn::Tensor& tensor,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config);

uint32_t wrap_index(int index, int size);

template <typename OpOutputType, typename... OpInputTypes>
struct MassagedOperationParams {
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    PredicateFunc predicate;           // Function to determine if formatting should be applied
    PreTransformFunc pre_transform;    // Function to pre-process input arguments
    PostTransformFunc post_transform;  // Function to post-process the operation output
    OpType operation;                  // The main operation to be performed
};

template <typename OpOutputType, typename... OpInputTypes>
class MassagedOperation {
public:
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    // post transform takes the output and optionally the args; it may use
    // the args in order to know if it needs to post process the output.
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...> params) :
        predicate_(params.predicate),
        pre_transform_(params.pre_transform),
        post_transform_(params.post_transform),
        operation_(params.operation) {}

    inline bool should_format(OpInputTypes... args) const { return predicate_(args...); }

    inline OwnedArgsType pre_format(OpInputTypes... args) const { return pre_transform_(args...); }

    inline OpOutputType post_format(OpOutputType output) const { return post_transform_(output); }

    inline OpOutputType operator()(OpInputTypes... args) const {
        if (should_format(args...)) {
            auto formatted_input = pre_format(args...);
            auto op_output = std::apply(operation_, formatted_input);
            return post_format(op_output);
        }
        return operation_(args...);
    }

    MassagedOperation sequence(const MassagedOperation& other) {
        std::shared_ptr<bool> t1_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t2_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t1_then_t2_required = std::make_shared<bool>(false);

        auto merged_predicate =
            [p1 = this->predicate_, p2 = other.predicate_, t1_required, t2_required](OpInputTypes... args) -> bool {
            if (p1(args...)) {
                *t1_required = true;
            }
            if (p2(args...)) {
                *t2_required = true;
            }
            return *t1_required or *t2_required;
        };

        auto merged_pre_transform = [t1 = this->pre_transform_,
                                     t2 = other.pre_transform_,
                                     p1 = this->predicate_,
                                     p2 = other.predicate_,
                                     t1_required,
                                     t2_required,
                                     t1_then_t2_required](OpInputTypes... args) -> OwnedArgsType {
            if (*t1_required) {
                auto transformed_args = t1(args...);
                if (std::apply(p2, transformed_args)) {
                    *t1_then_t2_required = true;
                    return std::apply(t2, transformed_args);
                }
                return transformed_args;
            } else if (*t2_required) {
                return t2(args...);
            } else {
                return std::make_tuple(args...);
            }
        };

        auto merged_post_transform =
            [t1 = this->post_transform_, t2 = other.post_transform_, t1_then_t2_required, t1_required, t2_required](
                OpOutputType output) -> OpOutputType {
            if (*t1_then_t2_required) {
                // we go backwards for post-transform
                auto t2_output = t2(output);
                auto t1_output = t1(t2_output);
                return t1_output;
            } else if (*t1_required) {
                return t1(output);
            } else if (*t2_required) {
                return t2(output);
            } else {
                return output;
            }
        };

        return MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...>{
            .predicate = merged_predicate,
            .pre_transform = merged_pre_transform,
            .post_transform = merged_post_transform,
            .operation = this->operation_});
    }

    // getters for all private members
    PredicateFunc get_predicate() const { return predicate_; }
    PreTransformFunc get_pre_transform() const { return pre_transform_; }
    PostTransformFunc get_post_transform() const { return post_transform_; }
    OpType get_operation() const { return operation_; }

    // setters for all private members
    void set_predicate(PredicateFunc predicate) { predicate_ = predicate; }
    void set_pre_transform(PreTransformFunc pre_transform) { pre_transform_ = pre_transform; }
    void set_post_transform(PostTransformFunc post_transform) { post_transform_ = post_transform; }
    void set_operation(OpType operation) { operation_ = operation; }

private:
    PredicateFunc predicate_;
    PreTransformFunc pre_transform_;
    PostTransformFunc post_transform_;
    OpType operation_;
};

ttnn::Shape compute_padded_shape(
    const ttnn::Shape& logical_shape,
    uint32_t tile_height = tt::constants::TILE_HEIGHT,
    uint32_t tile_width = tt::constants::TILE_WIDTH);

enum class ShardStrategy { BLOCK, HEIGHT, WIDTH };

// Helper function for creating a sharded memory configuration for a tensor
// based on its logical shape, a shard strategy and orientation, and a core
// grid. Optionally, you may pass a preferred shard shape to use. If not
// provided, the shard shape will be inferred from the tensor shape and the
// shard strategy.
ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape = std::nullopt,
    const tt::tt_metal::Layout& layout = tt::tt_metal::Layout::ROW_MAJOR);

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord);

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
