.. _Using ttnn:

Using ttnn
##########

.. note::
   These basic snippets currently work on Grayskull only. We are working on
   updating the API for other architectures, like Wormhole.

Basic Examples
**************


1. Converting from and to torch tensor
--------------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/convert_to_from_torch.py
   :language: python


2. Running an operation on the device
--------------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/run_op_on_device.py
   :language: python


3. Using __getitem__ to slice the tensor
----------------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/get_item.py
   :language: python


4. Enabling program cache
--------------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/program_cache.py
   :language: python


5. Debugging intermediate tensors
---------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/debugging_intermediate_tensors.py
   :language: python


6. Tracing the graph of operations
----------------------------------

.. note::
   This basic snippet is under construction, and may not work on all hardware architectures.

.. code-block:: python

    import torch
    import ttnn

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    with ttnn.tracer.trace():
        torch_input_tensor = torch.rand(32, 32, dtype=torch.float32)
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.exp(input_tensor)
        torch_output_tensor = ttnn.to_torch(output_tensor)
    ttnn.tracer.visualize(torch_output_tensor, file_name="exp_trace.svg")

    ttnn.close_device(device)


7. Using tt_lib operation in ttnn
---------------------------------

`tt_lib` operations are missing some of the features of ttnn operations such as graph tracing and in order to support these features, ttnn provides a different to call `tt_lib` operations that enabled the missing features.

`tt_lib` operations are missing some of the features of ttnn operations such as graph tracing and in order to support these features, ttnn provides a different to call `tt_lib` operations that enabled the missing features.

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/using_tt_lib.py
   :language: python


8. Enabling Logging
-------------------

.. code-block:: bash

    # To print currently executing ttnn operations
    export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'

    # To print the currently executing ttnn and tt_lib operation and its input tensors to stdout
    export TT_METAL_LOGGER_TYPES=Op
    export TT_METAL_LOGGER_LEVEL=Debug

Logging is not a substitute for profiling.
Please refer to :doc:`Profiling ttnn Operations </ttnn/profiling_ttnn_operations>` for instructions on how to profile operations.


.. note::

    The logging is only available when compiling with CONFIG=assert or CONFIG=debug.



9. Supported Python Operators
-----------------------------

.. note::
   This basic snippet is under construction, and may not work on all hardware architectures.

.. code-block:: python

    import ttnn

    input_tensor_a: ttnn.Tensor = ttnn.from_torch(torch.rand(2, 4), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b: ttnn.Tensor = ttnn.from_torch(torch.rand(2, 4), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Add (supports broadcasting)
    input_tensor_a + input_tensor_b

    # Subtract (supports broadcasting)
    input_tensor_a - input_tensor_b

    # Multiply (supports broadcasting)
    input_tensor_a - input_tensor_b

    # Matrix Multiply
    input_tensor_a @ input_tensor_b

    # Equals
    input_tensor_a == input_tensor_b

    # Not equals
    input_tensor_a != input_tensor_b

    # Greater than
    input_tensor_a > input_tensor_b

    # Greater than or equals
    input_tensor_a >= input_tensor_b

    # Less than
    input_tensor_a < input_tensor_b

    # Less than or equals
    input_tensor_a <= input_tensor_b



10. Changing the string representation of the tensor
----------------------------------------------------

.. note::
   This basic snippet is under construction, and may not work on all hardware architectures.

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/string_repr.py
   :language: python


11. Visualize using Web Browser
-------------------------------

.. note::
   This basic snippet is under construction, and may not work on all hardware architectures.

Set the following environment variables as needed

.. code-block:: bash

    # enable_fast_runtime_mode - This has to be disabled to enable logging
    # enable_logging - Synchronize main thread after every operation and log the operation
    # report_name (optional) - Name of the report used by the visualizer. If not provided, then no data will be dumped to disk
    # enable_detailed_buffer_report (if report_name is set) - Enable to visualize the detailed buffer report after every operation
    # enable_graph_report (if report_name is set) - Enable to visualize the graph after every operation
    # enable_detailed_tensor_report (if report_name is set) - Enable to visualize the values of input and output tensors of every operation
    # enable_comparison_mode (if report_name is set) - Enable to test the output of operations against their golden implementaiton


     # If running a pytest that is located inside of tests/ttnn, use this config (unless you want to override "report_name" manually)
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_logging": true,
        "enable_graph_report": false,
        "enable_detailed_buffer_report": false,
        "enable_detailed_tensor_report": false,
        "enable_comparison_mode": false
    }'

    # Otherwise, use this config and make sure to set "report_name"
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_logging": true,
        "report_name": "<name of the run in the visualizer>",
        "enable_graph_report": false,
        "enable_detailed_buffer_report": false,
        "enable_detailed_tensor_report": false,
        "enable_comparison_mode": false
    }'

    # Additionally, a json file can be used to override the config values
    export TTNN_CONFIG_PATH=<path to the file>


Run the code. i.e.:

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/visualizer_example.py
   :language: python

Open the visualizer by running the following command:

.. code-block:: bash

    python ttnn/visualizer/app.py



12. Register pre- and/or post-operation hooks
---------------------------------------------

.. note::
   This basic snippet is under construction, and may not work on all hardware architectures.

.. code-block:: python

    import torch
    import ttnn

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch_input_tensor = torch.rand((1, 32, 64), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    def pre_hook_to_print_args_and_kwargs(operation, args, kwargs):
        print(f"Pre-hook called for {operation}. Args: {args}, kwargs: {kwargs}")

    def post_hook_to_print_output(operation, args, kwargs, output):
        print(f"Post-hook called for {operation}. Output: {output}")

    with ttnn.register_pre_operation_hook(pre_hook_to_print_args_and_kwargs), ttnn.register_post_operation_hook(post_hook_to_print_output):
        ttnn.exp(input_tensor) * 2 + 1

    ttnn.close_device(device)



13. Query all operations
------------------------

.. code-block:: python

    import ttnn
    ttnn.query_registered_operations()


14. Falling back to torch
-------------------------

.. code-block:: python

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/falling_back_to_torch.py
   :language: python


15. Capturing graph of C++ functions, buffer allocations, etc
-------------------------------------------------------------

.. literalinclude:: ../../../../ttnn/ttnn/examples/usage/graph_capture.py
   :language: python
