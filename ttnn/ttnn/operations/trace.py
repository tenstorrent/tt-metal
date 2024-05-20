# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn

THIS_MODULE = sys.modules[__name__]

__all__ = []


begin_trace_capture = ttnn.register_operation(name="ttnn.begin_trace_capture")(
    ttnn._ttnn.operations.core.begin_trace_capture
)

end_trace_capture = ttnn.register_operation(name="ttnn.end_trace_capture")(ttnn._ttnn.operations.core.end_trace_capture)

execute_trace = ttnn.register_operation(name="ttnn.execute_trace")(ttnn._ttnn.operations.core.execute_trace)

release_trace = ttnn.register_operation(name="ttnn.release_trace")(ttnn._ttnn.operations.core.release_trace)
