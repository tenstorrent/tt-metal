# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def _fill_cache_for_user_validate_input_tensors(operation_name, input_tensor, kv_input_tensor=None, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        kv_input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.kv_cache.fill_cache_for_user_",
    validate_input_tensors=_fill_cache_for_user_validate_input_tensors,
)
def fill_cache_for_user_(
    cache: ttnn.Tensor,
    user_entry: ttnn.Tensor,
    user_index: int,
) -> ttnn.Tensor:
    """
    fill_cache_for_user_(cache: ttnn.Tensor, user_entry: ttnn.Tensor, user_index: int) -> ttnn.Tensor

    Fills the cache with the user_entry at the user_index.
    """
    return ttnn.experimental.tensor.fill_cache(cache, user_entry, user_index)


def _update_cache_for_token_validate_input_tensors(operation_name, input_tensor, kv_input_tensor=None, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        kv_input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.kv_cache.update_cache_for_token_",
    validate_input_tensors=_update_cache_for_token_validate_input_tensors,
)
def update_cache_for_token_(
    cache: ttnn.Tensor,
    token_entry: ttnn.Tensor,
    token_index: int,
) -> ttnn.Tensor:
    """
    update_cache_for_token_(cache: ttnn.Tensor, token_entry: ttnn.Tensor, token_index: int) -> ttnn.Tensor

    Updates the cache with the token_entry at the token_index.


    """
    return ttnn.experimental.tensor.update_cache(cache, token_entry, token_index)
