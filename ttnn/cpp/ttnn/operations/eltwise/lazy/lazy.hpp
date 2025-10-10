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

struct TernaryFn : OverloadsFor<TernaryFn, ExpressionView, ExpressionView, ExpressionView> {
    Ternary operation;

    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second, ExpressionView third) const;
    using OverloadsFor<TernaryFn, ExpressionView, ExpressionView, ExpressionView>::operator();
};

struct CompareFn : ttsl::overloaded<
                       OverloadsFor<CompareFn, ExpressionView, Param>,
                       OverloadsFor<CompareFn, Param, ExpressionView>,
                       OverloadsFor<CompareFn, ExpressionView, ExpressionView>> {
    Unary operation;

    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;

    using OverloadsFor<CompareFn, ExpressionView, Param>::operator();
    using OverloadsFor<CompareFn, Param, ExpressionView>::operator();
    using OverloadsFor<CompareFn, ExpressionView, ExpressionView>::operator();
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

inline constexpr CompareFn eq{.operation = Unary::EQZ};
inline constexpr CompareFn ge{.operation = Unary::GEZ};
inline constexpr CompareFn gt{.operation = Unary::GTZ};
inline constexpr CompareFn le{.operation = Unary::LEZ};
inline constexpr CompareFn lt{.operation = Unary::LTZ};
inline constexpr CompareFn ne{.operation = Unary::NEZ};

inline constexpr ttsl::overloaded add{
    UnaryWithParamFn{.operation = UnaryWithParam::ADD},
    RUnaryWithParamFn{.operation = UnaryWithParam::ADD},
    BinaryFn{.operation = Binary::ADD},
};

inline constexpr ttsl::overloaded sub{
    UnaryWithParamFn{.operation = UnaryWithParam::SUB},
    RUnaryWithParamFn{.operation = UnaryWithParam::RSUB},
    BinaryFn{.operation = Binary::SUB},
};

inline constexpr ttsl::overloaded rsub{
    UnaryWithParamFn{.operation = UnaryWithParam::RSUB},
    RUnaryWithParamFn{.operation = UnaryWithParam::SUB},
    RBinaryFn{.operation = Binary::SUB},
};

inline constexpr ttsl::overloaded mul{
    UnaryWithParamFn{.operation = UnaryWithParam::MUL},
    RUnaryWithParamFn{.operation = UnaryWithParam::MUL},
    BinaryFn{.operation = Binary::MUL},
};

inline constexpr DivFn div{
    UnaryWithParamFn{.operation = UnaryWithParam::DIV},
    BinaryFn{.operation = Binary::DIV},
};

inline constexpr RDivFn rdiv{
    RUnaryWithParamFn{.operation = UnaryWithParam::DIV},
    RBinaryFn{.operation = Binary::DIV},
};

inline constexpr ttsl::overloaded power{
    UnaryWithParamFn{.operation = UnaryWithParam::POWER},
    BinaryFn{.operation = Binary::POWER},
};

inline constexpr ttsl::overloaded where{
    TernaryFn{.operation = Ternary::WHERE},
};

}  // namespace ttnn::operations::lazy
