// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace ttnn {
namespace operations {
namespace data_movement {
    ttnn::Tensor pad_to_tile_vol(uint8_t queue_id,
                                 const ttnn::Tensor& tensor,
                                 const float value,
                                 const bool use_multicore,
                                 const std::optional<MemoryConfig>& memory_config) {
        auto logical_shape = tensor.get_logical_shape();
        auto padded_shape = tensor.get_padded_shape();
        auto rank = tensor.get_shape().rank();
        if (padded_shape.volume() % tt::constants::TILE_HW != 0) {
            TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

            auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
            auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
            uint32_t num_non_hw_dims = rank - 2u;
            auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0,0});
            padding_vec.reserve(rank);
            padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
            padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

            constexpr bool pad_use_multicore = true;
            auto padded_output = ttnn::pad(queue_id,
                                            tensor,
                                            padding_vec,
                                            value,
                                            use_multicore,
                                            memory_config);
            return padded_output;
        }
        return tensor;
    }

    template<typename OpOutputType, typename... OpInputTypes>
    struct MassagedOperationParams {
        using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
        using PredicateFunc = std::function<bool(OpInputTypes...)>;
        using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using PostTransformFuncWithArgs = std::function<OpOutputType(const OpOutputType&, OpInputTypes...)>;
        using PostTransformFunctWithoutArgs = std::function<OpOutputType(const OpOutputType&)>;
        using PostTransformFunc = std::variant<PostTransformFuncWithArgs, PostTransformFunctWithoutArgs>;
        using PrecompFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using OpType = std::function<OpOutputType(OpInputTypes...)>;

        PredicateFunc predicate;      // Function to determine if formatting should be applied
        PreTransformFunc pre_transform;  // Function to pre-process input arguments
        PostTransformFunc post_transform;  // Function to post-process the operation output
        PrecompFunc precomp;           // Function for fixing up input arguments after preprocessing has been applied
        OpType operation;              // The main operation to be performed
    };

    template<typename OpOutputType, typename... OpInputTypes>
    class MassagedOperation {
    public:
        using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
        using PredicateFunc = std::function<bool(OpInputTypes...)>;
        using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        // post transform takes the output and optionally the args; it may use
        // the args in order to know if it needs to post process the output.
        using PostTransformFuncWithArgs = std::function<OpOutputType(const OpOutputType&, OpInputTypes...)>;
        using PostTransformFunctWithoutArgs = std::function<OpOutputType(const OpOutputType&)>;
        using PostTransformFunc = std::variant<PostTransformFuncWithArgs, PostTransformFunctWithoutArgs>;
        using PrecompFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using OpType = std::function<OpOutputType(OpInputTypes...)>;

        MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...> params)
            : predicate_(params.predicate)
            , pre_transform_(params.pre_transform)
            , post_transform_(params.post_transform)
            , precomp_(params.precomp)
            , operation_(params.operation) {}

        inline bool should_format(OpInputTypes... args) const {
            return predicate_(args...);
        }

        inline OwnedArgsType pre_format(OpInputTypes... args) const {
            return pre_transform_(args...);
        }

        inline OpOutputType post_format(OpOutputType output, OpInputTypes... args) const {
            return std::visit([&output, &args...](auto&& f) -> OpOutputType {
                if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                    return f(output, args...);
                } else {
                    return f(output);
                }
            }, post_transform_);
        }

        inline OwnedArgsType precomp(OpInputTypes... args) const {
            return precomp_(args...);
        }

        inline OpOutputType operator()(OpInputTypes... args) const {
            if (should_format(args...)) {
                auto formatted_input = pre_format(args...);
                auto precomped = std::apply(precomp_,formatted_input);
                return post_format(std::apply(operation_, precomped), args...);
            }
            return operation_(args...);
        }

        MassagedOperation sequence(const MassagedOperation& other) {
            auto merged_predicate = [p1 = this->predicate_,
                                     p2 = other.predicate_](OpInputTypes... args) -> bool {
                return p1(args...) or p2(args...);
            };
            auto merged_pre_transform = [t1 = this->pre_transform_,
                                         t2 = other.pre_transform_,
                                         p1 = this->predicate_,
                                         p2 = other.predicate_](OpInputTypes... args) -> OwnedArgsType {
                // this is ugly, but I couldn't find a way around it since
                // the OpInputTypes may contain consts/refs.
                if (p1(args...) && std::apply(p2, t1(args...))) {
                    return std::apply(t2, t1(args...));
                } else if (p1(args...)) {
                    return t1(args...);
                } else if (p2(args...)) {
                    return t2(args...);
                } else {
                    return std::make_tuple(args...);
                }
            };

            auto merged_post_transform = [t1 = this->post_transform_,
                                          t2 = other.post_transform_,
                                          p1 = this->predicate_,
                                          p2 = other.predicate_,
                                          this_pretransform = this->pre_transform_](OpOutputType output, OpInputTypes... args) -> OpOutputType {

                OpOutputType transformed_output = output;
                if (p1(args...)) {
                    // for post-transformation, we need to go in reverse order
                    auto pretransformed_args = this_pretransform(args...);
                    if (std::apply(p2, pretransformed_args)) {
                        transformed_output = std::visit([&transformed_output, &pretransformed_args](auto&& f) -> OpOutputType {
                            if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                                return std::apply(f, std::tuple_cat(std::make_tuple(transformed_output), pretransformed_args));
                            } else {
                                return f(transformed_output);
                            }
                        }, t2);
                    }
                    transformed_output = std::visit([&transformed_output, &args...](auto&& f) -> OpOutputType {
                        if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                            return f(transformed_output, args...);
                        } else {
                            return f(transformed_output);
                        }
                    }, t1);
                }
                else if (p2(args...)) {
                    transformed_output = std::visit([&transformed_output, &args...](auto&& f) -> OpOutputType {
                        if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                            return f(transformed_output, args...);
                        } else {
                            return f(transformed_output);
                        }
                    }, t2);
                }
                return transformed_output;
            };

            auto merged_precomp = [pc1 = this->precomp_,
                                   pc2 = other.precomp_,
                                   p1 = this->predicate_,
                                   p2 = other.predicate_,
                                   t1 = this->pre_transform_](OpInputTypes... args) -> OwnedArgsType {
                if (p1(args...) && std::apply(p2, t1(args...))) {
                    return std::apply(pc2, pc1(args...));
                }
                else if (p1(args...)) {
                    return pc1(args...);
                }
                else if (p2(args...)) {
                    return pc2(args...);
                }
                return std::make_tuple(args...);
            };

            return MassagedOperation(
                MassagedOperationParams<OpOutputType, OpInputTypes...>{
                    .predicate = merged_predicate,
                    .pre_transform = merged_pre_transform,
                    .post_transform = merged_post_transform,
                    .precomp = merged_precomp,
                    .operation = this->operation_
                }
            );
        }

        // getters for all private members
        PredicateFunc get_predicate() const { return predicate_; }
        PreTransformFunc get_pre_transform() const { return pre_transform_; }
        PostTransformFunc get_post_transform() const { return post_transform_; }
        PrecompFunc get_precomp() const { return precomp_; }
        OpType get_operation() const { return operation_; }

        // setters for all private members
        void set_predicate(PredicateFunc predicate) { predicate_ = predicate; }
        void set_pre_transform(PreTransformFunc pre_transform) { pre_transform_ = pre_transform; }
        void set_post_transform(PostTransformFunc post_transform) { post_transform_ = post_transform; }
        void set_precomp(PrecompFunc precomp) { precomp_ = precomp; }
        void set_operation(OpType operation) { operation_ = operation; }

    private:
        PredicateFunc predicate_;
        PreTransformFunc pre_transform_;
        PostTransformFunc post_transform_;
        PrecompFunc precomp_;
        OpType operation_;
    };
}
}
}
