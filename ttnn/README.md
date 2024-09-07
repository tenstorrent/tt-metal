## TTNN

TTNN is a Python library that provides a launching point for learning the APIs within ``TT-METAL``.
The TTNN library assumes the user is familiar with [PyTorch](https://pytorch.org) and provides
operations that easily translate PyTorch tensors to and from ``ttnn.Tensor``(s).   This library is an application programming interface intended for Tenstorrent device operations with a primary dependency on the Python library tt_lib within the tt_eager subproject.  This code has been tested with PyTorch 1.13.

We trust that this library will be a valuable guide to helping you on your journey to take full advantage of our devices!

#### Please learn the API using our Jupyter Notebook tutorials
* There is a collection of tutorials written with Jupyter Notebooks to help you ramp up your skillset for using `tt-metal`. These
notebooks can be found under https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials.
* These tutorials assume you already have a machine set up with either a grayskull or wormhole device available and that you have successfully
followed the instructions for [installing and building the software with the development Python environment](https://github.com/tenstorrent/tt-metal/blob/main/README.md).
* From within the `ttnn/tutorials` directory, launch the notebooks with: `jupyter lab --no-browser --port=8888`


#### Here are a few key differences with the operations we currently support compared to their equivalents in PyTorch
* shape
    * Unlike PyTorch, the shape class maintains both the intended shape and the shape including padding.
    * When moving between row major and tile layout via the ttnn.to_layout(...) function, the last two dimensions are automatically padded and unpadded as necessary.
* matmul
    * The tensors must be moved to device before the operation can be done.
    * The last two dimensions must be a multiple of 32 if the multiplication is to work on the device.  For example, a tensor with shape (1, 1, 3, 4) would not be successfully multiplied to another tensor with tile layout (ttnn.TILE_LAYOUT).  For these reason the to_layout function will automatically add padding to the last two dimensions so that the height and width dimensions are a multiples of 32 and ready for a device multiplication.
    * Results from a matmul will not have the exact same precision. The order of operations are likely to be different even when comparing dytypes of bfloat16.  Instead of PyTorch allclose, we often use a pearson correlation coefficient to verify the results.
    * The dot product is not fully supported yet and unfortunately returns a Tensor with a shape.
* add
    * The tensors must be moved to device before the operation can be done.
* subtract.
    * Broadcasting is only supported in the last two dimensions (i.e. the height and width dimensions).
    * The tensors must be moved to device before the operation can be done.
* reshape
    * The last two dimensions in the reshape must be a multiple of 32 when using the tensor on a device.
    * When converting a tt tensor to a PyTorch tensor, the 3rd and 4th dimensions must be a multiple of 32 or the operation will default to using the PyTorch implementation.
* transpose
    * There is no transpose method.  Please use permute instead.
* permute
    * When using the ttnn library, the operation requires the first parameter to be the tensor and the next to be the new order of dimensions within a parenthesis.

#### Frequently asked questions
* Where are the tests for ttnn?
    * All tests can be found under tests/ttnn
* Tell me the differences between each kind of test under tests/ttnn?
    * tests/ttnn/integration_tests
        * Demonstrates the inference models built with ttnn
    * tests/ttnn/sweep_tests
        * Used to check coverage of what is and what is NOT supported
        * Tests can be added well before the actuall implementation is finished
        * These tests do not block the continuous integration pipeline
        * They are built so that their results can be uniformly reported
        * Can be run with pytest although they follow a strict format when built
    * tests/ttnn/unit_test
        * These are traditional unit tests written with pytest
        * Failures on these tests will cause alarms if code is merged into main
* Why do the sweep tests use a dictionary for all the combinations of input and then use a special run method?  Could you not have done this with a traditional pytest instead?
    * The primary reason was because we needed a way to create a consolidated report per operation in the form of a csv file.  The idea was that each operation would get its own python file where all the test combinations are handled by a single run method.  Each permutation of the input combinations would become the header for the resulting csv which is then uploaded and reported on.
* How do I run sweep tests with pytest?
    * To run all of the sweep tests for a given python operation file:
        * `pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>`
        * Example for matmul: `pytest /home/ubuntu/git/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul`
    * To run just one sample combination for an operation:
        * `pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>[<operation>.py-<index-of-test-instance>]`
        * Example for matmul: `pytest /home/ubuntu/git/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul[matmul.py-0]`
* What if my device hangs?
    * Be sure that you have a clean build with the latest code update.  Updating code without rebuilding is often the source of the issue.
    * If you have a clean build, you can reset the board on the command line using the tt-smi command and the device id with: `tt-smi -tr 0` where 0 represents the device id.
* What types are supported on device?
    * We currently support ttnn.bfloat16, ttnn.bfloat8 and ttnn.uint32.
* What shapes are supported on device?
    * The last dimension of the shape multiplied by the number of bytes of the sizeof the dataype must be a multiple of four.  For example, ttnn.bloat16 would need to have the last dimension be even for a tensor using ttnn.ROW_MAJOR_LAYOUT.  For ttnn.TILE_LAYOUT the to_layout operation will automatically do padding to ensure the last two dimensions (height and width) are multiples of 32.
* Is slicing available?
    * Slicing is supported.  At the moment this feature falls back to using PyTorch slicing on the host.
    * Example:
        * tensor1 = ttnn.from_torch(torch.randn(3,3))
        * print(tensor1[:1])
* Why are the results from operations like add and matmul not the same precision and require a pearson correlation coefficient comparison?
    * Results for operations are different because the order of floating point operations is different between CPU and the TT device.  A similar issue would arise when comparing cpu and gpu operations.
* How do I create a tensor of all zeros that is not on device and the height and width do not have to be multiples of 32?
    * Use PyTorch to achieve this.
        * tensor = ttnn.from_torch(torch.zeros(3,3))
        * print(tensor)
* How do I dump the logs of operations from the TT device?
    * You can add one or both of these environment variables
        *   `export TT_METAL_LOGGER_TYPES=Op`
        *   `export TT_METAL_LOGGER_LEVEL=DEBUG`
    * In addition, you can add the following environment variable to print currently executing ttnn operations. This makes every op blocking, ensuring that what is printed is actually executing. Otherwise, logging may not be representative of where the error occurs. Note: you may want to omit this when using gdb since there may be interactions with gdb.
        * `export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'`
            * `enable_fast_runtime_mode`: When turned on, op validation is always skipped
            * `enable_logging`: Turns on ttnn logging feature for ops, which makes every op blocking
* What is the format for git commit messages?
    * As mentioned in other documentation, the use of the '#' symbol to identify an issue request number is expected on each commit message.
        * For example your git commit message might be: "#4003: Your message here" for github issue 4003.
    * Consider using: `git config --global core.commentChar '>'`

#### Steps to setup ttnn tests from vscode
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

#### Steps to launch the tt-eager example code from within vscode
* Add the Makefile Tools extension
* Be sure to build with `make tests/tt_eager`
* Update launch.json to debug the code sample you want to run.  For example if you want to run test_bert, your update to launch.json might look like like:
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
* `export CONFIG=debug` within your virtual environment and run `make build`
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
