# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deprecated names of renamed ``ttnn.experimental`` operations.

The heads operations lost their ``nlp_`` prefix. Each old name still works: it emits a
DeprecationWarning and forwards to the new operation. Remove an entry here, and the
matching forwarding header under ttnn/cpp/ttnn/operations/experimental/transformer/nlp_*/,
one release after the rename landed.
"""

import sys
import warnings

RENAMED_EXPERIMENTAL_OPERATIONS = {
    "nlp_concat_heads": "concat_heads",
    "nlp_concat_heads_decode": "concat_heads_decode",
    "nlp_concat_heads_boltz": "concat_heads_boltz",
    "nlp_create_qkv_heads": "create_qkv_heads",
    "nlp_create_q_heads_split": "create_q_heads_split",
    "nlp_create_qkv_heads_decode": "create_qkv_heads_decode",
    "nlp_create_qkv_heads_falcon7b": "create_qkv_heads_falcon7b",
    "nlp_create_qkv_heads_vit": "create_qkv_heads_vit",
    "nlp_create_qkv_heads_segformer": "create_qkv_heads_segformer",
    "nlp_create_qkv_heads_boltz": "create_qkv_heads_boltz",
    "nlp_kv_cache_load_slice": "kv_cache_load_slice",
}


def _deprecated_alias(experimental, old_name, new_name):
    def alias(*args, **kwargs):
        warnings.warn(
            f"ttnn.experimental.{old_name} was renamed to ttnn.experimental.{new_name}; "
            "the old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(experimental, new_name)(*args, **kwargs)

    alias.__name__ = old_name
    alias.__qualname__ = old_name
    alias.__doc__ = f"Deprecated alias of ttnn.experimental.{new_name}."
    return alias


def _install():
    experimental = sys.modules["ttnn.experimental"]
    for old_name, new_name in RENAMED_EXPERIMENTAL_OPERATIONS.items():
        if not hasattr(experimental, new_name):
            raise AttributeError(f"ttnn.experimental.{new_name} is missing; the alias table is stale")
        setattr(experimental, old_name, _deprecated_alias(experimental, old_name, new_name))


_install()
