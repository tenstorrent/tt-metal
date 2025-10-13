// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_stl/assert.hpp"
#include "ttnn/operations/eltwise/lazy/lazy.hpp"

namespace ttnn::operations::lazy {

Function UnaryFn::operator()(ExpressionView first) const { return lazy::defer(operation, first).value(); }

Function UnaryWithParamFn::operator()(ExpressionView first, Param second) const {
    return lazy::defer(operation, first, second).value();
}

Function RUnaryWithParamFn::operator()(Param first, ExpressionView second) const {
    return lazy::defer(operation, second, first).value();
}

Function BinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::defer(operation, first, second).value();
}

Function RBinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::defer(operation, second, first).value();
}

Function TernaryFn::operator()(ExpressionView first, ExpressionView second, ExpressionView third) const {
    return lazy::defer(operation, first, second, third).value();
}

Function DivFn::operator()(Param first, ExpressionView second) const { return mul(first, recip(second)); }

Function RDivFn::operator()(ExpressionView first, Param second) const { return div(second, first); }

Function CompareFn::operator()(ExpressionView first, Param second) const {
    return lazy::defer(operation, sub(first, second)).value();
}

Function CompareFn::operator()(Param first, ExpressionView second) const {
    return lazy::defer(operation, sub(first, second)).value();
}

Function CompareFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::defer(operation, sub(first, second)).value();
}

Function PowerFn::operator()(float first, ExpressionView second) const {
    TT_FATAL(first > 0, "base {} must be positive for real-valued expression", first);
    return exp(mul(std::logf(first), second));
}

}  // namespace ttnn::operations::lazy
