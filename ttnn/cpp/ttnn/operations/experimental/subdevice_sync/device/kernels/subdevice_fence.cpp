// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// subdevice_fence: an empty one-core program. Its only effect is its slot in the issue order: its go signal waits for
// every earlier program on its sub-device, and every later go of the same CQ (any sub-device) waits behind it.
void kernel_main() {}
