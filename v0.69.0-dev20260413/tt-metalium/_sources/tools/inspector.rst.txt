Inspector
=========

Overview
--------

The Inspector is a tool that provides insights into Metal host runtime. It is designed to be on by default and to have
minimal impact on performance.
It consists of two components: one that logs necessary data to do investigation and one that allows clients to connect
and query Metal host runtime data.

Enabling
--------

Configure the Inspector by setting the following environment variables:

.. code-block::

   export TT_METAL_INSPECTOR=1                                  # optional: enable/disable the Inspector. Default is `1` (enabled).
   export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=1      # optional: enable/disable stopping execution if the Inspector is not initialized properly. Default is `0` (disabled).
   export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0         # optional: enable/disable warnings on logging write exceptions (like disk out of space). Default is `1` (enabled).
   export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:50051 # optional: set the address of the Inspector RPC server. Default is `localhost:50051`.
   export TT_METAL_INSPECTOR_RPC=1                              # optional: enable/disable the Inspector RPC server. Default is `1` (enabled).

Enabling the Inspector will override `TT_METAL_RISCV_DEBUG_INFO` and debugging info will be generated for riscv elfs.
You can also use unix sockets for the RPC server by setting `TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS` to a unix socket path,
e.g. `unix:/tmp/inspector_socket`.

Extending
---------

To add new RPC functionality, review the ``Inspector`` interface in the ``tt_metal/impl/debug/inspector/rpc.capnp`` file.
Add your new method to this interface.
This will generate a corresponding callback that you can attach by calling ``Inspector::get_rpc_server().setYourNewMethodCallback(...)``.
Next, implement your callback to handle the request and send a response to the client.

Examples
________

Example changes in ``tt_metal/impl/debug/inspector/rpc.capnp``:

.. code-block:: capnp

   interface Inspector {
       getHelloWorld @0 () -> (message :Text);

       sumTwoNumbers @1 (a :Int32, b :Int32) -> (result :Int32);

       ... other methods ...
   }

Example changes in your Metal host runtime code:

.. code-block:: cpp

   #include "tt_metal/impl/debug/inspector/inspector.hpp"
   #include "impl/debug/inspector/rpc_server_generated.hpp"

   // In your initialization code
   Inspector::get_rpc_server().setGetHelloWorldCallback(
       [](auto result) {
           result.setMessage("Hello, World!");
       });
   Inspector::get_rpc_server().setSumTwoNumbersCallback(
       [](auto params, auto result) {
           result.setResult(params.getA() + params.getB());
       });

Using in tt-triage
__________________

After adding your new RPC method, you can use it in the ``tt-triage`` tool to fetch data from the Metal host runtime.
The tool automatically generates code to call your new RPC method.
In your script, add a dependency to the inspector data script:

.. code-block::

   from inspector_data import run as get_inspector_data, InspectorData

   script_config = ScriptConfig(
      depends=["inspector_data"],
   )

   def run(args, context: Context):
       inspector_data = get_inspector_data(args, context)

You can then call your new RPC method:

.. code-block:: python

       hello_world = inspector_data.getHelloWorld()
       print(hello_world.message)

       sum_result = inspector_data.sumTwoNumbers(3, 5)
       print(f"Sum of 3 and 5 is {sum_result.result}")

Limitations
-----------

Inspector data is designed to be available even after the Metal host runtime exits.
To achieve this, data is serialized to disk in the ``generated/inspector`` directory.
Methods of the ``Inspector`` interface that do not require arguments are automatically serialized during Metal runtime exit.
If you add a method that requires arguments, you must implement serialization and deserialization for that data yourself.
Serialization should be implemented in the Metal host runtime code, and deserialization in the ``tt-triage`` tool.

It is acceptable to add methods that require arguments and query Metal host runtime state during execution, and use this data in
``tt-triage`` scripts to provide insights into the system's state during execution.
If you run a ``tt-triage`` script that requires data which is not serialized, you will receive an error indicating that script execution is blocked due to a dependency failure.
