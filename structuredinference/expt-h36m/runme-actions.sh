#!/bin/bash

# uses past time steps (L) and an action flag at each iteration (cond)
extra_args=""
if [ ! -z "$@" ]; then
    if [ $* == "--debug" ]; then
        extra_args="$extra_args -m ipdb"
    fi
fi
exec python2.7 $extra_args train.py -vm L -cond -infm structured -ds 10 -dh 50 -uid past-act-cond
