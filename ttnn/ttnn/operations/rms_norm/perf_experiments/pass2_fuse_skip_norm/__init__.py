# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Intentionally empty (no imports): this slug is a regular package under the perf_experiments
# NAMESPACE package (which must have NO __init__.py, mirroring ttnn/ttnn/operations/examples/,
# so the ttnn.operations auto-import crawl in operations/__init__.py never descends here). An
# __init__.py that does relative imports would break `import ttnn` when perf_experiments is a
# regular package — keep this one importless.
