# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Library for PImpl test utilities
# This library is compiled separately to ensure that indirect
# can properly handle incomplete types (PImpl pattern)
add_library(test_indirect_util STATIC)

target_sources(
    test_indirect_util
    PRIVATE
        widget.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
        FILES widget.hpp
)

target_link_libraries(test_indirect_util PUBLIC TT::STL)
