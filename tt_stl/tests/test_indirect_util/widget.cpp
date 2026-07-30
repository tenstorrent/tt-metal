// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "widget.hpp"

namespace ttsl {

// Full WidgetImpl definition -- only visible in this translation unit.
class WidgetImpl {
public:
    int value;

    explicit WidgetImpl(int v) : value(v) {}
};

Widget::Widget(int value) : PimplBase(std::in_place, value) {}
Widget::~Widget() = default;
Widget::Widget(const Widget&) = default;
Widget& Widget::operator=(const Widget&) = default;
Widget::Widget(Widget&&) noexcept = default;
Widget& Widget::operator=(Widget&&) noexcept = default;

int Widget::value() const { return impl().value; }
void Widget::set_value(int value) { impl().value = value; }

MoveOnlyWidget::MoveOnlyWidget(int value) : PimplBase(std::in_place, value) {}
MoveOnlyWidget::~MoveOnlyWidget() = default;
MoveOnlyWidget::MoveOnlyWidget(MoveOnlyWidget&&) noexcept = default;
MoveOnlyWidget& MoveOnlyWidget::operator=(MoveOnlyWidget&&) noexcept = default;

int MoveOnlyWidget::value() const { return impl().value; }

}  // namespace ttsl
