// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <tt_stl/reflection.hpp>

// This header provides a convenient macro for defining serialization for types
// that use tt::stl::reflection for their operator<< implementation

#define TTNN_DEFINE_REFLECTION_SERIALIZATION(TYPE)                         \
    inline std::ostream& operator<<(std::ostream& os, const TYPE& value) { \
        tt::stl::reflection::operator<<(os, value);                        \
        return os;                                                         \
    }

// Macro to register a type with the graph argument serializer
// This should be called in the initialize() function
#define TTNN_REGISTER_GRAPH_TYPE(SERIALIZER, TYPE) SERIALIZER.register_type<TYPE>()
