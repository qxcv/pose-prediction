# Notes on Human3.6M

These notes are copied from the H3.6M dataset-loading code in my
[old repo](https://github.com/qxcv/joint-regressor).

## Format

Data is distributed either as subject-specific archives (one archive for
video and one archive for poses, per subject) or as activity-specific
archives. I'm downloading the subject-specific archives.

Each video archive has the name `VideosSubjectSpecific_<N>.tgz`, and
contains only the directory `S<M>/Videos/` (where `M != N` in some cases).
Each video has format `<action>.<camera ID>.mp4`. Some of the cameras are
forward-facing and some are not.

Each pose archive has the name `PosesD2_PositionsSubjectSpecific_3.tgz`
and has just the directory `S<M>/MyPoseFeatures/D2_Positions/`. The files
in that directory are named `<action>.<camera ID>.cdf`, and are in
CDF 3.3 format (*not* NetCDF or HDF5, despite both of those formats being
vastly more popular).

## Camera IDs

- 54138969: aft port
- 55011271: fwd port
- 58860488: rear stbd
- 60457274: fwd stbd

## Action and subject IDs

There are 52 action IDs in total:

```
_ALL, _ALL 1, Directions, Directions 1, Directions 2, Discussion,
Discussion 1, Discussion 2, Discussion 3, Eating, Eating 1, Eating 2,
Greeting, Greeting 1, Greeting 2, Phoning, Phoning 1, Phoning 2, Phoning
3, Photo, Photo 1, Photo 2, Posing, Posing 1, Posing 2, Purchases,
Purchases 1, Sitting, Sitting 1, Sitting 2, SittingDown, SittingDown 1,
SittingDown 2, Smoking, Smoking 1, Smoking 2, TakingPhoto, TakingPhoto 1,
Waiting, Waiting 1, Waiting 2, Waiting 3, WalkDog, WalkDog 1, Walking,
Walking 1, Walking 2, WalkingDog, WalkingDog 1, WalkTogether,
WalkTogether 1, WalkTogether 2
```

Per
[this paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Fragkiadaki_Recurrent_Network_Models_ICCV_2015_paper.pdf),
I should use subject 5 as test subject and all others as train
subjects. Will probably use all cameras (although I have a strong
suspicion that will cause things to asplode hard; might need to fix to
fwd {port, stbd} if it all goes south).

## Joint information

```
1: Pelvis --> Dupe of 12?
2: Hip (right)
3: Knee (right)
4: Ankle (right)
5: Foot arch (right)
6: Toes (right)
7: Hip (left)
8: Knee (left)
9: Ankle (left)
10: Foot arch (left)
11: Toes (left)
12: Pelvis --> Dupe of 1?
13: Torso
14: Base of neck --> Dupe of 17, 25?
15: Head low
16: Head high
17: Base of neck --> Dupe of 14, 25?
18: Shoulder (left)
19: Elbow (left)
20: Wrist (left) --> Dupe of 21?
21: Wrist (left) --> Dupe of 20?
22: ?? hand (left)
23: ?? hand (left) --> Dupe of 24? Unreliable?
24: ?? hand (left) --> Dupe of 23? Unreliable?
25: Base of neck --> Dupe of 14, 17?
26: Shoulder (right)
27: Elbow (right)
28: Wrist (right) --> Dupe of 29?
29: Wrist (wright) --> Dupe of 29?
30: ?? hand (right) --> Unreliable?
31: ?? hand (right) --> Dupe of 32? Unreliable?
32: ?? hand (right) --> Dupe of 31? Unreliable?
```