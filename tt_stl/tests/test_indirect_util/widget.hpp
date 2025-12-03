// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indirect.hpp>

namespace ttsl {

// Forward declaration only - Widget is an INCOMPLETE TYPE in this header
// The full definition is in widget.cpp
// This allows testing indirect<Widget> with an incomplete type
class Widget;

// Factory functions to create and manipulate Widget
// These return/accept indirect<Widget> where Widget is incomplete in this header
// but complete in widget.cpp where these functions are defined
indirect<Widget> createWidget();
indirect<Widget> createWidget(int value);
int getWidgetValue(const indirect<Widget>& w);
void setWidgetValue(indirect<Widget>& w, int value);

}  // namespace ttsl
