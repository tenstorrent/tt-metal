// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared spellings for host-emitted names and device token concatenation.
#define TT_MCAST_METAL2_STEM _mcast_
#define TT_MCAST_METAL2_JOIN_IMPL(a, b, c) a##b##c
#define TT_MCAST_METAL2_JOIN(a, b, c) TT_MCAST_METAL2_JOIN_IMPL(a, b, c)
#define TT_MCAST_METAL2_NAME(prefix, field) TT_MCAST_METAL2_JOIN(prefix, TT_MCAST_METAL2_STEM, field)
#define TT_MCAST_METAL2_STRING_IMPL(value) #value
#define TT_MCAST_METAL2_STRING(value) TT_MCAST_METAL2_STRING_IMPL(value)
