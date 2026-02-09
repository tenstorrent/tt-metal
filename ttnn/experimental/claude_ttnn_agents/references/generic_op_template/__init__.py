# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic Op Template

Copy this folder to: ttnn/ttnn/operations/<your_op_name>/
Then rename all 'template_op' references to your operation name.

After copying, import as:
    from ttnn.operations.<your_op_name> import <your_op_name>
"""

from .template_op import template_op

__all__ = ["template_op"]
