# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Modified VIT 2CQ test that stresses the copy/stall path to amplify ND failure.
#
# Changes vs original:
#   - copies_per_iteration: multiple copy_host_to_device_tensor calls per iteration
#     (each triggers add_dispatch_wait_with_prefetch_stall = 1 stall point)
#   - More iterations: 2000 measurement (vs 1000)
#   - No perf assertions: we only care about provoking the fetch-queue hang
#
# Run: pytest vit_n300/test_vit_2cq_copy_stress.py -v -s
# Or via: ./vit_n300/stress_test_vit_n300.sh (after setting TEST_FILE)

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.vision.classification.vit.common.tests.vit_test_infra import create_test_infra

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_trace_2cq_copy_stress(
    device,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
    copies_per_iteration,
):
    """Same structure as run_trace_2cq_model but with multiple copies per iteration."""
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    first_op_event = ttnn.record_event(device, 0)
    read_event = ttnn.record_event(device, 1)

    # JIT
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_op_event = ttnn.record_event(device, 0)
    test_infra.run()
    output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    last_op_event = ttnn.record_event(device, 0)

    # Capture trace
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_op_event = ttnn.record_event(device, 0)

    spec = test_infra.input_tensor.spec
    input_trace_addr = test_infra.input_tensor.buffer_address()
    test_infra.output_tensor.deallocate(force=True)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)

    # Warmup
    outputs = []
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    for iter in range(num_warmup_iterations):
        ttnn.wait_for_event(0, write_event)
        input_l1_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_l1_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        for _ in range(copies_per_iteration):
            ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
            copy_done = ttnn.record_event(device, 1)
            ttnn.wait_for_event(1, copy_done)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    # Measurement loop
    for iter in range(num_measurement_iterations):
        ttnn.wait_for_event(0, write_event)
        input_l1_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_l1_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.to_memory_config(test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        for _ in range(copies_per_iteration):
            ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
            copy_done = ttnn.record_event(device, 1)
            ttnn.wait_for_event(1, copy_done)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="stop")
    ttnn.release_trace(device, trace_id)


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1753088}], indirect=True
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("copies_per_iteration", [10])  # 10 stalls per iteration vs 1 in original
def test_vit_2cq_copy_stress(device, batch_size, copies_per_iteration, is_single_card_n300):
    """Stress test: many copies per iteration to amplify prefetcher/dispatch stall race."""
    if not is_single_card_n300:
        pytest.skip("Copy-stress test targets N300; skip on N150")
    torch.manual_seed(0)

    test_infra = create_test_infra(device, batch_size, use_random_input_tensor=True)
    ttnn.synchronize_device(device)

    num_warmup_iterations = 100
    num_measurement_iterations = 2000  # vs 1000 in original

    total_stalls = (num_warmup_iterations + num_measurement_iterations) * copies_per_iteration
    logger.info(
        f"Copy stress: {copies_per_iteration} copies/iter × {num_measurement_iterations} iters "
        f"= {total_stalls} stall points (vs ~1100 in original)"
    )

    run_trace_2cq_copy_stress(
        device,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        copies_per_iteration,
    )
