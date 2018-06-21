#!/bin/bash

# takes past time steps and action distribution, outputting next pose
extra_args=""
if [ ! -z "$@" ]; then
    if [ $* == "--debug" ]; then
        extra_args="$extra_args -m ipdb"
    fi
fi
exec python2.7 $extra_args train.py -vm LR -infm structured -ds 50 \
    -dh 50 -uid ikeadb-no-acts -sfreq 5 -ar 1000 \
    -reload chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP25-params.npz \
    -params chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-config.pkl
