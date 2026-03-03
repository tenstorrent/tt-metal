tt-triage
=========

Overview
--------

tt-triage is a tool that provides insights into Metal execution. It is designed to be easily extensible.
It consists of data provider scripts that gather data from various sources (such as Inspector, tt-exalens, etc.) and
analysis scripts that process data supplied by the data provider scripts.
Since most of the scripts use Inspector data, before using tt-triage, ensure that you have enabled Inspector in your Metal host runtime.

Running
-------

To run tt-triage, execute the ``tools/tt-triage.py`` script using Python 3.10 or newer.

Example use with tt-metal:

.. code-block:: bash

    export TT_METAL_HOME=~/work/tt-metal
    ./build_metal.sh --build-programming-examples
    build/programming_examples/matmul_multi_core
    triage


Arguments
---------

You can run ``tt-triage --help`` to see the available options.
Some notable options include:

 - ``--initialize-with-noc1``: Initializes the debugger context with NOC1 enabled (default: False). Useful if NOC0 is not functioning.
 - ``--dev=<device_id>``: Specifies the device ID to analyze (default: ``in_use``). You can provide multiple ``--dev`` options to analyze several devices. Use ``all`` to analyze every device.
 - ``--remote-exalens``: Connects to a remote exalens server (default: False). This is helpful if tt-triage is fails during UMD initialization. If you encounter errors indicating UMD cannot be initialized, try this option. Open a new terminal and run ``tt-exalens --server`` before starting your workload. Then, in your original terminal, run ``tt-triage --remote-exalens``.

Extending
---------

Take a look at the ``tools/triage/tt-triage.md`` file for instructions on how to extend tt-triage with new data providers and analysis scripts.
