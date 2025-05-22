// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <tt_stl/span.hpp>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <cstdint>

namespace tt::stl {
namespace {

using ::testing::Eq;
using ::testing::Pointwise;

TEST(SpanTest, AsBytes) {
    std::vector<uint32_t> src = {1, 2, 3, 4, 5};
    std::vector<uint32_t> dst = {0, 0, 0, 0, 0};

    ASSERT_EQ(src.size(), dst.size());

    auto src_span = tt::stl::make_const_span(src);
    auto dst_span = tt::stl::make_span(dst);

    auto src_bytes = tt::stl::as_bytes(src_span);
    auto dst_writable_bytes = tt::stl::as_writable_bytes(dst_span);

    EXPECT_EQ(src_bytes.size(), src.size() * sizeof(uint32_t));
    EXPECT_EQ(dst_writable_bytes.size(), dst.size() * sizeof(uint32_t));

    for (size_t i = 0; i < src_bytes.size(); ++i) {
        dst_writable_bytes[i] = src_bytes[i];
    }

    EXPECT_THAT(src, Pointwise(Eq(), dst));
}

}  // namespace
}  // namespace tt::stl
