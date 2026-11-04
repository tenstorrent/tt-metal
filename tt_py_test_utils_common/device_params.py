# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger


def get_updated_device_params(device_params):
    import ttnn

    new_device_params = device_params.copy()

    dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
    dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
    fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

    if ttnn.device.is_blackhole():
        # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
        fabric_config = new_device_params.get("fabric_config", None)
        if not (fabric_config and fabric_tensix_config):
            # When not both are set, force COL dispatch
            if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                logger.warning(
                    "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                )
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL
        elif fabric_config and fabric_tensix_config:
            logger.warning(
                f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
            )

    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis, fabric_tensix_config)
    new_device_params["dispatch_core_config"] = dispatch_core_config

    return new_device_params
