// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/operation.hpp"
#include "ttnn/operations/eltwise/lazy/overload.hpp"

namespace ttnn::operations::lazy {

struct UnaryFn : OverloadsFor<UnaryFn, ExpressionView> {
    Unary operation;

    [[nodiscard]] Function operator()(ExpressionView first) const;
    using OverloadsFor<UnaryFn, ExpressionView>::operator();
};

struct UnaryWithParamFn : OverloadsFor<UnaryWithParamFn, ExpressionView, Param> {
    UnaryWithParam operation;

    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    using OverloadsFor<UnaryWithParamFn, ExpressionView, Param>::operator();
};

struct RUnaryWithParamFn : OverloadsFor<RUnaryWithParamFn, Param, ExpressionView> {
    UnaryWithParam operation;

    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;
    using OverloadsFor<RUnaryWithParamFn, Param, ExpressionView>::operator();
};

struct BinaryFn : OverloadsFor<BinaryFn, ExpressionView, ExpressionView> {
    Binary operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;
    using OverloadsFor<BinaryFn, ExpressionView, ExpressionView>::operator();
};

struct RBinaryFn : OverloadsFor<RBinaryFn, ExpressionView, ExpressionView> {
    Binary operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;
    using OverloadsFor<RBinaryFn, ExpressionView, ExpressionView>::operator();
};

struct TernaryFn : OverloadsFor<TernaryFn, ExpressionView, ExpressionView, ExpressionView> {
    Ternary operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second, ExpressionView third) const;
    using OverloadsFor<TernaryFn, ExpressionView, ExpressionView, ExpressionView>::operator();
};

struct CompareFn : ttsl::overloaded<
                       OverloadsFor<CompareFn, ExpressionView, ExpressionView>,
                       OverloadsFor<CompareFn, ExpressionView, Param>,
                       OverloadsFor<CompareFn, Param, ExpressionView>> {
    UnaryFn operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;

    using OverloadsFor<CompareFn, ExpressionView, ExpressionView>::operator();
    using OverloadsFor<CompareFn, ExpressionView, Param>::operator();
    using OverloadsFor<CompareFn, Param, ExpressionView>::operator();
};

struct OverloadedBinaryFn : ttsl::overloaded<UnaryWithParamFn, RUnaryWithParamFn, BinaryFn> {};

struct OverloadedRBinaryFn : ttsl::overloaded<UnaryWithParamFn, RUnaryWithParamFn, RBinaryFn> {};

struct DivFn : ttsl::overloaded<UnaryWithParamFn, BinaryFn, OverloadsFor<DivFn, Param, ExpressionView>> {
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;

    using UnaryWithParamFn::operator();
    using BinaryFn::operator();
    using OverloadsFor<DivFn, Param, ExpressionView>::operator();
};

struct RDivFn : ttsl::overloaded<RUnaryWithParamFn, RBinaryFn, OverloadsFor<RDivFn, ExpressionView, Param>> {
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;

    using RUnaryWithParamFn::operator();
    using RBinaryFn::operator();
    using OverloadsFor<RDivFn, ExpressionView, Param>::operator();
};

struct LogicalBinaryFn : ttsl::overloaded<
                             OverloadsFor<LogicalBinaryFn, ExpressionView, ExpressionView>,
                             OverloadsFor<LogicalBinaryFn, ExpressionView, Param>,
                             OverloadsFor<LogicalBinaryFn, Param, ExpressionView>> {
    OverloadedBinaryFn operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;

    using OverloadsFor<LogicalBinaryFn, ExpressionView, ExpressionView>::operator();
    using OverloadsFor<LogicalBinaryFn, ExpressionView, Param>::operator();
    using OverloadsFor<LogicalBinaryFn, Param, ExpressionView>::operator();
};

struct WhereFn : ttsl::overloaded<
                     TernaryFn,
                     OverloadsFor<WhereFn, ExpressionView, ExpressionView, Param>,
                     OverloadsFor<WhereFn, ExpressionView, Param, ExpressionView>,
                     OverloadsFor<WhereFn, ExpressionView, Param, Param>> {
    [[nodiscard]] Function operator()(ExpressionView condition, ExpressionView input, Param other) const;
    [[nodiscard]] Function operator()(ExpressionView condition, Param input, ExpressionView other) const;
    [[nodiscard]] Function operator()(ExpressionView condition, Param input, Param other) const;

    using TernaryFn::operator();
    using OverloadsFor<WhereFn, ExpressionView, ExpressionView, Param>::operator();
    using OverloadsFor<WhereFn, ExpressionView, Param, ExpressionView>::operator();
    using OverloadsFor<WhereFn, ExpressionView, Param, Param>::operator();
};

struct Atan2Fn : ttsl::overloaded<
                     OverloadsFor<Atan2Fn, ExpressionView, ExpressionView>,
                     OverloadsFor<Atan2Fn, ExpressionView, Param>,
                     OverloadsFor<Atan2Fn, Param, ExpressionView>> {
    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;

    using OverloadsFor<Atan2Fn, ExpressionView, ExpressionView>::operator();
    using OverloadsFor<Atan2Fn, ExpressionView, Param>::operator();
    using OverloadsFor<Atan2Fn, Param, ExpressionView>::operator();
};

inline constexpr UnaryFn recip{.operation = Unary::RECIP};
inline constexpr UnaryFn negative{.operation = Unary::NEGATIVE};
inline constexpr UnaryFn exp{.operation = Unary::EXP};

inline constexpr UnaryFn eqz{.operation = Unary::EQZ};
inline constexpr UnaryFn gez{.operation = Unary::GEZ};
inline constexpr UnaryFn gtz{.operation = Unary::GTZ};
inline constexpr UnaryFn lez{.operation = Unary::LEZ};
inline constexpr UnaryFn ltz{.operation = Unary::LTZ};
inline constexpr UnaryFn nez{.operation = Unary::NEZ};

inline constexpr UnaryFn logical_not{.operation = Unary::LOGICAL_NOT};

inline constexpr UnaryFn atan{.operation = Unary::ATAN};

inline constexpr CompareFn eq{.operation = lazy::eqz};
inline constexpr CompareFn ge{.operation = lazy::gez};
inline constexpr CompareFn gt{.operation = lazy::gtz};
inline constexpr CompareFn le{.operation = lazy::lez};
inline constexpr CompareFn lt{.operation = lazy::ltz};
inline constexpr CompareFn ne{.operation = lazy::nez};

inline constexpr OverloadedBinaryFn add{
    UnaryWithParamFn{.operation = UnaryWithParam::ADD},
    RUnaryWithParamFn{.operation = UnaryWithParam::ADD},
    BinaryFn{.operation = Binary::ADD},
};

inline constexpr OverloadedBinaryFn sub{
    UnaryWithParamFn{.operation = UnaryWithParam::SUB},
    RUnaryWithParamFn{.operation = UnaryWithParam::RSUB},
    BinaryFn{.operation = Binary::SUB},
};

inline constexpr OverloadedRBinaryFn rsub{
    UnaryWithParamFn{.operation = UnaryWithParam::RSUB},
    RUnaryWithParamFn{.operation = UnaryWithParam::SUB},
    RBinaryFn{.operation = Binary::SUB},
};

inline constexpr OverloadedBinaryFn mul{
    UnaryWithParamFn{.operation = UnaryWithParam::MUL},
    RUnaryWithParamFn{.operation = UnaryWithParam::MUL},
    BinaryFn{.operation = Binary::MUL},
};

inline constexpr OverloadedBinaryFn pow{
    UnaryWithParamFn{.operation = UnaryWithParam::POWER},
    RUnaryWithParamFn{.operation = UnaryWithParam::RPOW},
    BinaryFn{.operation = Binary::POWER},
};

inline constexpr OverloadedRBinaryFn rpow{
    UnaryWithParamFn{.operation = UnaryWithParam::RPOW},
    RUnaryWithParamFn{.operation = UnaryWithParam::POWER},
    RBinaryFn{.operation = Binary::POWER},
};

inline constexpr DivFn div{
    UnaryWithParamFn{.operation = UnaryWithParam::DIV},
    BinaryFn{.operation = Binary::DIV},
};

inline constexpr RDivFn rdiv{
    RUnaryWithParamFn{.operation = UnaryWithParam::DIV},
    RBinaryFn{.operation = Binary::DIV},
};

inline constexpr LogicalBinaryFn logical_and{.operation = lazy::mul};
inline constexpr LogicalBinaryFn logical_or{.operation = lazy::add};
inline constexpr LogicalBinaryFn logical_xor{.operation = lazy::sub};

inline constexpr WhereFn where{
    TernaryFn{.operation = Ternary::WHERE},
};

inline constexpr Atan2Fn atan2{};

}  // namespace ttnn::operations::lazy
