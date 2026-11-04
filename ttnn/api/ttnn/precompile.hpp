// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {
class CompileSession;
}  // namespace tt::tt_metal::experimental

namespace ttnn::precompile {

// Starts a collect window on the calling thread. Ops still build their programs, but nothing runs: dispatch
// is blocked by a NO_DISPATCH graph capture, and every program built is handed to session (its kernels
// queued for building) instead of being compiled and enqueued. Outputs are meaningless inside the window.
// Must not be called inside another graph capture.
//
// real_alloc=true lets the allocator assign real buffer addresses, so kernels that bake addresses into
// compile-time arguments are built exactly as the real run will need them; it costs the device memory the
// ops would use. real_alloc=false mocks every buffer at address 0.
void begin_collect(tt::tt_metal::experimental::CompileSession& session, bool real_alloc);

// Ends the collect window started by begin_collect.
void end_collect(tt::tt_metal::experimental::CompileSession& session);

}  // namespace ttnn::precompile
