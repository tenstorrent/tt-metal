// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fabric_ring_exchange idle core: launches and exits. Puts the op's compute-core count into the program so the empty
// variant measures the fixed cost of a program of that size (launch, teardown) with no work and no fabric.

void kernel_main() {}
