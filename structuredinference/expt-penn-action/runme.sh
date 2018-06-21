#!/bin/bash

# takes past time steps and action distribution, outputting next pose
extra_args=""
if [ ! -z "$@" ]; then
    if [ $* == "--debug" ]; then
        extra_args="$extra_args -m ipdb"
    fi
fi
# TODO: Should I try to have a larger latent representation? The classifier
# isn't doing a very good job on the normal setup. That said, it might be a
# better idea just to use more of the information I have (instead of samples
# from latent transition distribution, concatenate mean, covariance, and
# deterministic LSTM state).
exec python2.7 $extra_args train.py -vm L -infm structured -ds 10 -dh 50 -uid penn-acts
