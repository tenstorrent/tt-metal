// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/pimpl.hpp>

namespace ttsl {

// WidgetImpl is only forward-declared here; its full definition lives in widget.cpp.
class WidgetImpl;

// Widget follows the pimpl idiom: it derives from PimplBase<WidgetImpl> and forward-declares its
// copy/move special members here; they are defined (= default) out-of-line in widget.cpp, where
// WidgetImpl is complete. WidgetImpl remains an INCOMPLETE TYPE in this header -- and therefore in
// any consumer TU that only includes this header, such as test_pimpl.cpp -- which is exactly the
// header/.cpp (and, in a real build, linker) boundary the pimpl idiom exists to support.
// widget.hpp + test_pimpl.cpp form one such TU; widget.hpp + widget.cpp form the other, where
// WidgetImpl is complete (see test_indirect_util/CMakeLists.txt for how the two are compiled and
// linked separately).
class Widget : public PimplBase<WidgetImpl> {
public:
    explicit Widget(int value);

    ~Widget();
    Widget(const Widget&);
    Widget& operator=(const Widget&);
    Widget(Widget&&) noexcept;
    Widget& operator=(Widget&&) noexcept;

    int value() const;
    void set_value(int value);
};

// Move-only sibling of Widget: mirrors the common case (e.g. MeshTensor), where copy is deleted
// and WidgetImpl's copy constructor is never instantiated.
class MoveOnlyWidget : public PimplBase<WidgetImpl> {
public:
    explicit MoveOnlyWidget(int value);

    ~MoveOnlyWidget();
    MoveOnlyWidget(const MoveOnlyWidget&) = delete;
    MoveOnlyWidget& operator=(const MoveOnlyWidget&) = delete;
    MoveOnlyWidget(MoveOnlyWidget&&) noexcept;
    MoveOnlyWidget& operator=(MoveOnlyWidget&&) noexcept;

    int value() const;
};

}  // namespace ttsl
