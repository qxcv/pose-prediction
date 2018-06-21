#!/bin/sh

# like runme.sh, but only uses past (L) instead of future (R) inputs; also
# conditions on previous pose
exec python2.7 train.py -vm L -infm structured -ds 10 -dh 50 -uid past-only-cond-emis -etype conditional
