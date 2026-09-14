// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared spellings for host-emitted names and device token concatenation.
#define TT_MCAST_SPEC_STEM _mcast_
#define TT_MCAST_SPEC_JOIN_IMPL(a, b, c) a##b##c
#define TT_MCAST_SPEC_JOIN(a, b, c) TT_MCAST_SPEC_JOIN_IMPL(a, b, c)
#define TT_MCAST_SPEC_NAME(prefix, field) TT_MCAST_SPEC_JOIN(prefix, TT_MCAST_SPEC_STEM, field)
#define TT_MCAST_SPEC_STRING_IMPL(value) #value
#define TT_MCAST_SPEC_STRING(value) TT_MCAST_SPEC_STRING_IMPL(value)
#define TT_MCAST_SPEC_METADATA(F, prefix) \
    F(prefix, rotating_span)              \
    F(prefix, rectangle_capacity)         \
    F(prefix, ack_count)                  \
    F(prefix, uniform_remote_count)       \
    F(prefix, uniform_loopback_count)     \
    F(prefix, sender_mcast_mode)          \
    F(prefix, has_remote_receivers)       \
    F(prefix, flags)
