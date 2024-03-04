# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl


def enable_program_cache():
    ttl.program_cache.enable()


def disable_and_clear_program_cache():
    ttl.program_cache.disable_and_clear()
