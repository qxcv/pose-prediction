# Pose prediction code

<p align="center">
  <a href="http://users.cecs.anu.edu.au/~u5568237/ikea/" target="_blank">
    <img src="images/ikea-fa-teaser.jpg"
        title="Sample of images from Ikea Furniture Assembly dataset"
        alt="A grid of frames from videos of people screwing legs into Ikea tables" />
  </a>
</p>

This repository contains the code for "[Human Pose Prediction via Deep Markov
Models](https://arxiv.org/pdf/1707.09240.pdf)" (Toyer et al., DICTA'17). The
instructions below explain how to preprocess our [Ikea Furniture
Assembly](http://users.cecs.anu.edu.au/~u5568237/ikea/) dataset and train both
an LSTM baseline and a deep Markov model to predict poses on that dataset.

## Installing dependencies

This repository has several dependencies on [PyPI](https://pypi.org/), as listed
in `requirements.txt`. If you have [the `virtualenv` and `virtualenvwrapper`
package management tools
installed](http://exponential.io/blog/2015/02/10/install-virtualenv-and-virtualenvwrapper-on-ubuntu/),
then you can install the following dependencies in an isolated development
environment using the following commands in a shell:

    mkvirtualenv -p "$(which python2)" pose-prediction-env
    pip install -r requirements.txt
    
(unless stated otherwise, it's assumed that all commands, including the ones
above, are run from the root of this repository)

The installed dependencies will only be usable from within the virtual
environment. Whenever you open a new shell & want to use this code, you'll need
to remember to execute `workon pose-prediction-env` to re-renter the
environment. That will prepend the name of the environment to your shell prompt,
like this:

    $ workon pose-prediction-env
    (pose-prediction-env)$ python -c 'print("now we can run Python, etc., with correct deps")'

As a final setup step, we'll make a directory to place all of our results in:

    mkdir ikea-fa-results 

## Obtaining the data

These instructions assume that you want to train a model to predict poses on
Ikea Furniture Assembly dataset. If you don't have the dataset already, then you
can download it using the following commands:

    wget 'http://users.cecs.anu.edu.au/~u5568237/ikea/ikea-fa-release-data.zip'
    unzip -j ikea-fa-release-data.zip ikea-fa-release-data/processed-python-data/ikea_action_data.h5 -d .
    
This will create a file named `ikea_action_data.h5` in the current directory
(`md5sum` is 92b334d368b798d5456401014aaf21c6). Note that most of the data in
`ikea-fa-release-data.zip` is _videos_, which you will probably want later, but
which you don't actually _need_ just to train some prediction models. If you
just want the `.h5` file for the purpose of following these instructions, then
you can get it from [this
link](https://mega.nz/#!ULwV1KYK!E2IxcLk3QaX3wMeMmSO5xHOMfcXZ2guYUU1Ni6KB77I).
You can [also download a tiny sample
file](https://mega.nz/#!hORy0Y7B!8ZvQ9ifg_QxP_y5SpdpDNFy9yR5OGRF5DXDWoi5xXF8)
(700kb vs. 20mb for original) that has only subset of the pose data, and which
will require much less memory at train time; this option is recommended if you
don't have a server with a lot of memory.

### Datasets other than Ikea FA

Most other datasets supported by this code---including Human3.6M (H3.6M), the
NTU RGBD dataset, the Penn Action dataset, and so on---require an additional
conversion step before you can use them. Specifically, you need to use one of
the `convert_<dataset>.py` scripts to convert from the original dataset format
to a uniform HDF5-based format that makes it easy for other tools in this
repository to get at pose data. Refer to the inbuilt help for each script to see
how to do this (e.g. `python convert_ntu.py --help`).

## Training & evaluating an LSTM baseline

In our paper, we compared against several sequence regression baselines. You can
train and test one such baseline as follows:

```bash
# choose lstm, lstm3lr, or erd
BASELINE=lstm
# this will train the model; once it has trained for a while (probably a few
# hours, or until the displayed loss stops going down), you can interrupt it
# with Ctrl+C to stop (there's no early-stopping IIRC)
python basic_lstm_baselines.py "$BASELINE" ikea_action_data.h5 ikea-fa-results/baselines/
# re-running the script will produce results
python basic_lstm_baselines.py "$BASELINE" ikea_action_data.h5 ikea-fa-results/baselines/
```

`results_${BASELINE}.h5` contains both ground truth and predicted poses for the
Ikea FA test set. You can calculate statistics for such a file using
`stats_calculator.py`, e.g.

```bash
python stats_calculator.py --output_dir ./ikea-fa-results/csv/ \
  "./ikea-fa-results/baselines/results_${BASELINE}.h5"
```

This will write some PCK statistics to CSV files in `ikea-fa-results/csv`.

## Training & evaluating a deep Markov model

Code for the deep Markov models is in the `structuredinference` directory. A DMM
can be trained using the following commands:

```
# if using another dataset, replace "ikeadb" with the dataset name (e.g.
# ntu-2d, h36m-2d, etc.)
cd structuredinference/expt-ikeadb
cp ../../ikea_action_data.h5 ./
bash runme-no-actions.sh
```

If you've just followed the instructions up to this point, then the above
command will train on the CPU using Theano. To make Theano train the network on
a GPU, you'll have to [install
`libgpuarray`](http://deeplearning.net/software/libgpuarray/installation.html**.
In either case, the network will require quite a bit of memory to train, since
it keeps all data in memory at once; the code was developed on a GPU server with
128GB of memory, which should be more than adequate for training. If you run
into memory errors, then you may want to try again with the [tiny sample
dataset](https://mega.nz/#!hORy0Y7B!8ZvQ9ifg_QxP_y5SpdpDNFy9yR5OGRF5DXDWoi5xXF8)
linked in the "Obtaining the data" section above.

As the network trains, the command executed above will periodically output a
series of update messages like this:

```
<<Bnum: 0, Batch Bound: 1.3835, |w|: 157.8771, |dw|: 1.0000, |w_opt|: 0.0000>>
<<-veCLL:28223.2433, KL:10996.1426, anneal:0.0100, l1:0.0000>>
<<Bnum: 10, Batch Bound: 0.9274, |w|: 157.9926, |dw|: 1.0000, |w_opt|: 0.1953>>
<<-veCLL:18792.7172, KL:10044.8623, anneal:0.0200, l1:0.0000>>
<<Bnum: 20, Batch Bound: 1.0508, |w|: 158.1243, |dw|: 1.0000, |w_opt|: 0.1990>>
<<-veCLL:21316.7489, KL:6796.2031, anneal:0.0300, l1:0.0000>>
…
<<-veCLL:7437.7972, KL:2064.8850, anneal:1.0000, l1:0.0000>>
<<Bnum: 1310, Batch Bound: 0.8889, |w|: 168.7077, |dw|: 1.0000, |w_opt|: 0.2590>>
<<-veCLL:16589.0754, KL:1615.3231, anneal:1.0000, l1:0.0000>>
<<(Ep 0) Bound: 1.0867 [Took 2697.6768 seconds] >>
<<Saving at epoch 0>>
<<Saved model (./chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP0-params) 
      opt (./chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP0-optParams) weights>>
```

The message at the end signifies the end of an epoch. After this message the
model will be evaluated & the results printed to stdout too.

Once the validation loss printed out at the end of an epoch stops going down,
you should interrupt the training script with Ctrl+C & evaluate the model. The
relevant model files are in `chkpt-ikeadb`; the directory contents will look
something like this:

```
(pose-prediction-env)$ ls -l chkpt-ikeadb/
total 69156
-rw-r--r-- 1 user user     1464 Jan 24 09:47 DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-config.pkl
-rw-r--r-- 1 user user 47199544 Jan 24 10:32 DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP0-optParams.npz
-rw-r--r-- 1 user user 23599946 Jan 24 10:32 DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP0-params.npz
```

The `*-config.pkl` file contains information related to the structure of the
network. The `*-EP<N>-optParams.npz` and `*-EP<N>-params.npz` files contain
actual weights from epoch `N`. To get results, we'll have to take the model
configuration and the `-params.npz` file for the most recent epoch & do the
following:

```bash
# I assume you are running this from the expt-ikeadb directory
python ../common_pp/make_eval_results.py \
  ./runme-no-actions.sh \
  "chkpt-ikeadb/<MODEL>-config.pkl" \
  "chkpt-ikeadb/<MODEL>-EP<N>-params.npz" \
  results_dkf.h5
```

After replacing `MODEL` and `N` with the values shown above, my command was:

```
python ../common_pp/make_eval_results.py \
  ./runme-no-actions.sh \
  "chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-config.pkl" \
  "chkpt-ikeadb/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+03-rv-5_0000e-02-nade-False-nt-5000-cond-False-ikeadb-no-acts-EP0-params.npz" \
  results_dkf.h5
```

That will give you a `results_dkf.h5` file (in the current directory) that you
can calculate statistics for in the same way as the `results_${BASELINE}.h5`
files described above. Concretely, from the `expt-ikeadb` directory, you can do this:


```bash
python ../../stats_calculator.py \
  --output_dir ../../ikea-fa-results/csv/ \
  results_dkf.h5
```

I usually put all my `results_*.h5` files in the same directory and run
`stats_calculator.py` on all of them in one go.

```





























```
    
# Other notes from my personal wiki

*(these notes from my personal wiki are included from posterity; they may help
with producing new plots)*

How to make videos from DKF predictions: at the moment, my code for making
videos from DKF predictions is in the `structuredinference` directory. You can
run it with the following sequence of commands:

```bash
cd ~/repos/structuredinference/expt-ikeadb
source activate pose-prediction  # or `workon` or whatever
./make_eval_videos.py results_dkf.h5 --vid-dir some-dest-dir
```

It will select a random sequence each time, so re-run a few times to plot a good
range.

Making statistics and plotting PCK: it works something like this:

```bash
cd /path/to/pose-prediction/

# repeat this call as necessary for all your baselines
# you may need --max-thresh to pick a threshold for (normalised) comparisons
# e.g. ikea works well with --max-thresh 0.1, NTU works well with --max-thresh 1
./stats_calculator.py --output_dir ikea_baselines/ ikea_baselines/_zero_velocity.h5

# the next two make actual plots that appear inthe paper
./plot_pck.py --stats-dir ~/etc/pp-baselines/2017-05-02/stats/ --methods dkf srnn erd lstm lstm3lr zero_velocity --method-names DKF SRNN ERD LSTM LSTM3LR "Zero-velocity" --parts elbows shoulders wrists --save plot.pdf --fps 16 --times 1 10 25 50 --no-thresh-px --dims 6 8 && mv plot.pdf plot-xtype-thresh.pdf
# maybe this one has time on the x-axis, and the above has a threshold?
./plot_pck.py --stats-dir ~/etc/pp-baselines/2017-05-02/stats/ --methods dkf srnn erd lstm lstm3lr zero_velocity --method-names DKF SRNN ERD LSTM LSTM3LR "Zero-velocity" --parts elbows shoulders wrists --save plot.pdf --legend-below --xtype time --fps 16 && mv plot.pdf plot-xtype-time.pdf
```
