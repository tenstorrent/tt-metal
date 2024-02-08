export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=`pwd`
export PYTHONPATH=`pwd`
export TT_METAL_ENV=dev
make build && source build/python_env/bin/activate
