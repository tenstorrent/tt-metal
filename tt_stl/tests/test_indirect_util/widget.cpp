// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "widget.hpp"

namespace ttsl {

// Full Widget definition - only visible in this translation unit
// This is the COMPLETE TYPE that is incomplete in widget.hpp
class Widget {
public:
    int value;

    Widget() : value(0) {}
    explicit Widget(int v) : value(v) {}
};

// Factory function implementations
// These create indirect<Widget> where Widget is complete
indirect<Widget> createWidget() { return indirect<Widget>(Widget()); }

indirect<Widget> createWidget(int value) { return indirect<Widget>(Widget(value)); }

int getWidgetValue(const indirect<Widget>& w) { return w->value; }

void setWidgetValue(indirect<Widget>& w, int value) { w->value = value; }

}  // namespace ttsl
