// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/overload.hpp"

namespace ttnn::operations::lazy {

struct UnaryFn : OverloadsFor<UnaryFn, ExpressionView> {
    Unary operation;

    [[nodiscard]] Function operator()(ExpressionView first) const;
    using OverloadsFor<UnaryFn, ExpressionView>::operator();
};

struct UnaryParamFn : OverloadsFor<UnaryParamFn, ExpressionView, Param> {
    Unary operation;

    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    using OverloadsFor<UnaryParamFn, ExpressionView, Param>::operator();
};

struct ParamUnaryFn : OverloadsFor<ParamUnaryFn, Param, ExpressionView> {
    Unary operation;

    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;
    using OverloadsFor<ParamUnaryFn, Param, ExpressionView>::operator();
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

struct DivFn : ttsl::overloaded<
                   OverloadsFor<DivFn, ExpressionView, Param>,
                   OverloadsFor<DivFn, Param, ExpressionView>,
                   OverloadsFor<DivFn, ExpressionView, ExpressionView>> {
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;

    using OverloadsFor<DivFn, ExpressionView, Param>::operator();
    using OverloadsFor<DivFn, Param, ExpressionView>::operator();
    using OverloadsFor<DivFn, ExpressionView, ExpressionView>::operator();
};

struct RDivFn : ttsl::overloaded<
                    OverloadsFor<RDivFn, ExpressionView, Param>,
                    OverloadsFor<RDivFn, Param, ExpressionView>,
                    OverloadsFor<RDivFn, ExpressionView, ExpressionView>> {
    [[nodiscard]] Function operator()(ExpressionView first, Param second) const;
    [[nodiscard]] Function operator()(Param first, ExpressionView second) const;
    [[nodiscard]] Function operator()(ExpressionView first, ExpressionView second) const;

    using OverloadsFor<RDivFn, ExpressionView, Param>::operator();
    using OverloadsFor<RDivFn, Param, ExpressionView>::operator();
    using OverloadsFor<RDivFn, ExpressionView, ExpressionView>::operator();
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

inline constexpr ttsl::overloaded power{
    UnaryParamFn{.operation = Unary::POWER},
};

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
    UnaryParamFn{.operation = Unary::ADD},
    ParamUnaryFn{.operation = Unary::ADD},
    BinaryFn{.operation = Binary::ADD},
};

inline constexpr ttsl::overloaded sub{
    UnaryParamFn{.operation = Unary::SUB},
    ParamUnaryFn{.operation = Unary::RSUB},
    BinaryFn{.operation = Binary::SUB},
};

inline constexpr ttsl::overloaded rsub{
    UnaryParamFn{.operation = Unary::RSUB},
    ParamUnaryFn{.operation = Unary::SUB},
    RBinaryFn{.operation = Binary::SUB},
};

inline constexpr ttsl::overloaded mul{
    UnaryParamFn{.operation = Unary::MUL},
    ParamUnaryFn{.operation = Unary::MUL},
    BinaryFn{.operation = Binary::MUL},
};

inline constexpr DivFn div{};

inline constexpr RDivFn rdiv{};

}  // namespace ttnn::operations::lazy
