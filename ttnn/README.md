# TT-NN
TT-NN is an open-source C++ and Python library of neural network operations, built on top of the TT-Metallium programming model.
It provides a PyTorch-like interface for running machine learning workloads on Tenstorrent AI accelerators, and serves as the primary high-level API for developing and optimizing ML models on Tenstorrent hardware.

## Purpose
### Enable ML frameworks targeting Tenstorrent hardware
Developers have access to a vast library of existing models implemented in PyTorch, JAX, TensorFlow, and other frameworks.
TT-NN offers a collection of operations and reusable building blocks to support the development of ML compilers and framework backends targeting Tenstorrent hardware.

→ See framework integrations: [Forge Compiler](https://github.com/tenstorrent/tt-forge) [PyTorch 2.0 TT-NN Backend](https://github.com/tenstorrent/pytorch2.0_ttnn)

### Manual bringup and optimization of ML models
When performance is critical, developers need fine-grained control.
Existing ML frameworks and compilers don’t fully expose the capabilities of our hardware.
TT-NN lets developers work with familiar high-level operations while tapping into hardware-specific optimizations when necessary.
This includes options to specify data format for mixed-precision, tensor layout and distribution, op fusion and other operation specific settings.

→ See production-ready model examples: [Model Zoo](https://tenstorrent.com/developers) <br>
→ See some model bringup guides: [General](https://github.com/tenstorrent/tt-metal/blob/0c0528ef1b395127c535edc0152a3ba2558fcfa4/tech_reports/ttnn/TTNN-model-bringup.md) | [LLMs](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) | [CNNs](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md)

## Key Features
* **High-Level Neural Network Operations:**<br> Optimized implementations of key neural net components: matrix multiplication, convolution, attention mechanisms, data movement, collective communications (CCLs), element-wise ops, reductions, losses, pooling, and more. APIs are PyTorch-style but expose hardware-specific options.
* **Tensor Library:**<br> A flexible tensor abstraction for managing multidimensional arrays across host and device. Developers can precisely control data layout across a cluster of Tenstorrent accelerators via Tensor APIs.
* **Native Multi-Device Support:**<br> TT-NN seamlessly virtualizes multiple Tenstorrent devices into a single logical unit, enabling effortless scaling across device clusters.

## Getting Started
Assuming you completed the installation of the hardware and drivers, you can simply install TT-NN from PyPi
```
pip install ttnn
```
To check that installation was successful run a simple script
```
import ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.full([5, 5, 5], fill_value=1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.full([1], fill_value=2.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = a * b
print(c)
```

There is a collection of tutorials written with [Jupyter Notebooks](https://jupyter.org/install) to help you ramp up your skillset for using TT-NN.
These notebooks can be found under [ttnn/tutorials](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials).

From within the `ttnn/tutorials` directory, launch the notebooks with:
```
jupyter lab --no-browser --port=8888
```

## For Contributors

### Linking TT-NN with your C++ projects
We are actively working to make it easy for you to consume TT-NN in your own projects.
Current best example can be found in the training framework here:
* Must manually align [3rdparty dependencies via CPM](https://github.com/tenstorrent/tt-metal/tree/main/tt-train/cmake)
* Must manually align [includepath and other options](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/CMakeLists.txt#L81-L154)

### FAQ
#### Where are the tests for ttnn?
All tests can be found under tests/ttnn

#### What is the difference between each kind of test?
 * tests/ttnn/integration_tests
     * Demonstrates the inference models built with ttnn
 * tests/ttnn/sweep_tests
     * Used to check coverage of what is and what is NOT supported
     * Tests can be added well before the actual implementation is finished
     * These tests do not block the continuous integration pipeline
     * They are built so that their results can be uniformly reported
     * Can be run with pytest although they follow a strict format when built
 * tests/ttnn/unit_test
     * These are traditional unit tests written with pytest
     * Failures on these tests will cause alarms if code is merged into main

#### Why do the sweep tests use a dictionary for all the combinations of input and then use a special run method?  Could you not have done this with a traditional pytest instead?
The primary reason was because we needed a way to create a consolidated report per operation in the form of a csv file.  The idea was that each operation would get its own python file where all the test combinations are handled by a single run method.  Each permutation of the input combinations would become the header for the resulting csv which is then uploaded and reported on.

How do I run sweep tests with pytest?
 To run all of the sweep tests for a given python operation file:
  * `pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>`
  * Example for matmul: `pytest /home/ubuntu/git/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul`
 To run just one sample combination for an operation:
   * `pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>[<operation>.py-<index-of-test-instance>]`
   * Example for matmul: `pytest /home/ubuntu/git/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul[matmul.py-0]`

#### What if my device hangs?
Reset with `tt-smi -tr 0` where 0 represents the device id.
Be sure that you have a clean build with the latest code update.  Updating code without rebuilding is often the source of the issue.

#### How do I dump the logs of operations from the TT device?
You can add one or both of these environment variables
  *   `export TT_LOGGER_TYPES=Op`
  *   `export TT_LOGGER_LEVEL=DEBUG`
In addition, you can add the following environment variable to print currently executing ttnn operations. This makes every op blocking, ensuring that what is printed is actually executing. Otherwise, logging may not be representative of where the error occurs. Note: you may want to omit this when using gdb since there may be interactions with gdb.
  * `export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'`
      * `enable_fast_runtime_mode`: When turned on, op validation is always skipped
      * `enable_logging`: Turns on ttnn logging feature for ops, which makes every op blocking

#### How to setup TT-NN tests from VSCode
* Add the Makefile Tools extension to vscode
* Add the Python extension to vscode
* Update settings.json to make vscode aware of the pytests
    * ```
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": [
            "tests/ttnn"
        ],
        "python.autoComplete.extraPaths": [
            "${workspaceFolder}/tests/ttnn"
        ],
        "python.analysis.extraPaths": [
            "${workspaceFolder}/tests/ttnn"
        ],
    ```

#### How to launch C++ example code from VSCode
* Add the Makefile Tools extension
* Be sure to build with `make tests/tt_eager`
* Update launch.json to debug the code sample you want to run.  For example if you want to run test_bert, your update to launch.json might look like:
    ```
            {
                "name": "test_bert",
                "type": "cppdbg",
                "request": "launch",
                "args": [],
                "stopAtEntry": false,
                "externalConsole": false,
                "cwd": "${workspaceFolder}/build",
                "program": "${workspaceFolder}/build/test/tt_eager/integration_tests/test_bert",
                "MIMode": "gdb",
                "miDebuggerPath": "gdb",
                "setupCommands": [
                    {
                        "description": "Enable pretty-printing for gdb",
                        "text": "-enable-pretty-printing",
                        "ignoreFailures": true
                    }
                ]
            },

    ```
 * Debug with vscode by launching it from the "Run and Debug"

#### How to debug from python and C++ at the same time within vscode
* Make sure to build tt-metal in Debug. You can use `./build_metal.sh --debug`
* Allow your process to attach to the debugger.
    * `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
    * When done you can reverse this with... `echo 1 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
* Add a task in launch.json for debugging your python code.  For example, you might have something like this below...
    ```
    {
      "name": "Python: test_matmul",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/build/python_env/bin/pytest",
      "args": [
        "${workspaceFolder}/tests/ttnn/test_matmul.py",
        "-k",
        "test_matmul"
      ],
      "justMyCode": false,
      "stopOnEntry": false,
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "internalConsoleOptions": "openOnSessionStart",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    ```
* Additionally, add the following to launch.json (note that the name "C++: Attach to Python" will be used to find this task in the next step)
    ```
    {
      "name": "C++: Attach to Python",
      "type": "cppdbg",
      "request": "attach",
      "program": "${workspaceFolder}/build/python_env/bin/python3",
      "MIMode": "gdb",
      "miDebuggerPath": "gdb",
      "processId": "296907"
    },
    ```
* Wherever you want to debug the python code, add the function call update_process_id() from utils_for_testing.py
    * When you run your python test in the vscode debugger (for example, launching the test_matmul above), this method will update the "processId" of the task "C++: Attach to Python" in the .vscode/launch.json
    * Use a breakpoint in your python code to halt execution to give you time to attach the C++ debugger.
    * Run the "C++: Attach to Python" which will now have the processId updated that was being used by the debugger.
    * Remember to have a breakpoint in the C++ code that you want to debug as well.
    * Finally, continue from the python breakpoint to begin debugging your C++ calls.
    * Note, the function update_process_id() is defined in utils_for_testing.py as...
        ```
        def update_process_id():
            print(f"Debugging PID: {os.getpid()}")
            cwd = os.getcwd()
            launch_json_path = os.path.join(cwd, ".vscode", "launch.json")
            with open(launch_json_path, "r") as f:
                launch_data = json.load(f)

            for config in launch_data.get("configurations", []):
                if config.get("name") == "C++: Attach to Python":
                    config["processId"] = str(os.getpid())
                    break

            with open(launch_json_path, "w") as f:
                json.dump(launch_data, f, indent=4)

        ```
