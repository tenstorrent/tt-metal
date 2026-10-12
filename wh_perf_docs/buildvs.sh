#!/bin/bash
# build /tmp/vsim/versim-wormhole-b0: fmarkovic's versim with a Verilator error limit, so an RTL assertion does not $stop the run
set -e
R=/proj_sw/user_dev/fmarkovic/tt-umd-simulators; B=$R/build; O=/tmp/vsim; mkdir -p $O/obj
blk=$(sed -n 8786,8830p $B/build.ninja)
DEF=$(echo "$blk" | grep -m1 "^  DEFINES" | sed 's/^  DEFINES = //')
INC=$(echo "$blk" | grep -m1 "^  INCLUDES" | sed 's/^  INCLUDES = //')
LIBS=$(echo "$blk" | grep -m1 "^  LINK_LIBRARIES" | sed 's/^  LINK_LIBRARIES = //' | sed "s| _deps/nanomsg-build| $B/_deps/nanomsg-build|g")
LP=$(echo "$blk" | grep -m1 "^  LINK_PATH" | sed 's/^  LINK_PATH = //')
cp $R/versim/wormhole_b0/tt_versim_device.cpp $O/tt_versim_device.cpp
sed -i 's|std::vector<std::string> plusargs = {};|std::vector<std::string> plusargs = {"+verilator+error+limit+1000000"};|' $O/tt_versim_device.cpp
grep -n "error+limit" $O/tt_versim_device.cpp
CXX=$(command -v clang++-20 || command -v clang++ || echo g++)
FL="-O2 -std=gnu++17 -Wno-deprecated-declarations -Wno-unused-value -w"
$CXX $DEF $INC -I$R/versim/wormhole_b0 -I$R/versim $FL -c $R/versim/versim.cpp -o $O/obj/versim.o
$CXX $DEF $INC -I$R/versim/wormhole_b0 -I$R/versim $FL -c $O/tt_versim_device.cpp -o $O/obj/dev.o
$CXX -O2 $O/obj/versim.o $O/obj/dev.o -o $O/versim-wormhole-b0.new $LP $LIBS
mv $O/versim-wormhole-b0.new $O/versim-wormhole-b0
P=$B/versim-wormhole-b0; cp $P/soc_descriptor.yaml $O/; cp $P/run.sh $O/run.sh
ls -la $O
