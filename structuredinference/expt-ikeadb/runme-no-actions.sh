#!/bin/bash

# takes past time steps and action distribution, outputting next pose
extra_args=""
if [ ! -z "$@" ]; then
    if [ $* == "--debug" ]; then
        extra_args="$extra_args -m ipdb"
    fi
fi
exec python2.7 $extra_args train.py -vm LR -infm structured -ds 50 \
    -dh 50 -uid ikeadb-no-acts -sfreq 5 -ar 1000
