#!/usr/bin/env python3
"""Predict a sequence of poses with a recurrent GAN. I don't anticipate that
this will work."""

from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Dense, Activation, LSTM, Input, RepeatVector, \
    Bidirectional, Concatenate
from keras.optimizers import RMSprop
from keras.utils.generic_utils import Progbar
import numpy as np
from os import path, makedirs, listdir
import re

WGAN = True  # change this at will
# TODO: should train to convergence, not just for fixed number of iterations
DISCS_PER_GEN = 30 if WGAN else 5


def make_generator(in_length, out_length, noise_dim, pose_size):
    # we start by encoding the input sequence with a double LSTM
    x = in_poses = Input(shape=(in_length, pose_size))
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    # copy it, join it with some noise
    x = RepeatVector(out_length)(x)
    in_noise = Input(shape=(out_length, noise_dim))
    x = Concatenate()([x, in_noise])
    x = LSTM(256, return_sequences=True)(x)
    out_layer = LSTM(pose_size, return_sequences=True)(x)

    model = Model(inputs=[in_poses, in_noise], outputs=[out_layer])

    return model


def make_discriminator(cond_seq_length, gan_seq_length, pose_size):
    # as above, but we're returning something different
    x = in_cond = Input(shape=(cond_seq_length, pose_size))
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    x = RepeatVector(gan_seq_length)(x)
    in_gan = Input(shape=(gan_seq_length, pose_size))
    x = Concatenate()([x, in_gan])
    # XXX: bidirectional LSTM is probably a terrible idea because it creates
    # representational asymmetry between generator and discriminator
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256)(x)
    x = Dense(1)(x)
    if WGAN:
        out_layer = x
    else:
        out_layer = Activation('sigmoid')(x)

    model = Model(inputs=[in_cond, in_gan], outputs=[out_layer])

    return model


def base_wasserstein_loss(u, v):
    return K.mean(K.batch_dot(u, v))

def make_wasserstein_loss(lam):
    # for example of improved WGAN in Keras, see
    # https://github.com/farizrahman4u/keras-contrib/blob/master/examples/improved_wgan.py
    def wasserstein_loss(u, v):
        base_loss = base_wasserstein_loss(u, v)
        return base_loss + lam * grad_penalty
    return wasserstein_loss


class GANTrainer:
    def __init__(self,
                 in_length,
                 out_length,
                 pose_size,
                 noise_dim=32,
                 d_lr=0.000001,
                 g_lr=0.000001):
        self.in_length = in_length
        self.out_length = out_length
        self.noise_dim = noise_dim

        self.discriminator = make_discriminator(in_length, out_length,
                                                pose_size)
        # copy is read-only; it doesn't get compiled
        self.discriminator_copy = make_discriminator(in_length, out_length,
                                                     pose_size)
        self.discriminator_copy.trainable = False
        disc_opt = RMSprop(lr=d_lr, clipnorm=1.0)
        self.discriminator.compile(
            disc_opt, 'binary_crossentropy', metrics=['binary_accuracy'])

        self.generator = make_generator(in_length, out_length, noise_dim,
                                        pose_size)
        # this looks unnecessary (and compile() arguments make no sense anyway)
        # self.generator.compile('sgd', 'mae')

        in_cond = Input(shape=(in_length, pose_size))
        in_noise = Input(shape=(out_length, noise_dim))
        gan_out = self.generator([in_cond, in_noise])
        disc_out = self.discriminator_copy([in_cond, gan_out])
        nested = Model(inputs=[in_cond, in_noise], outputs=[disc_out])
        gen_opt = RMSprop(lr=g_lr, clipnorm=1.0)
        if WGAN:
            # The objective will be different for WGAN
            # see https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/WassersteinGAN/src/model/models_WGAN.py#L18-L21
            # WGAN loss can easily be implemented using +1/-1 labels and a dot
            # product objective.
            assert False, "this is broken, still need gradient penalty"
            nested.compile(gen_opt, wasserstein_loss)
        else:
            nested.compile(
                gen_opt, 'binary_crossentropy', metrics=['binary_accuracy'])
        self.nested_generator = nested

        self.num_disc_steps = 0
        self.num_gen_steps = 0

    def update_disc_copy(self):
        source = self.discriminator
        dest = self.discriminator_copy

        assert len(source.layers) == len(dest.layers)
        for dest_layer, source_layer in zip(dest.layers, source.layers):
            dest_layer.set_weights(source_layer.get_weights())

    def make_noise(self, num):
        return np.random.randn(num, self.out_length, self.noise_dim)

    def gen_train_step(self, batch_X):
        self.discriminator.trainable = False
        batch_size = len(batch_X)
        # true poses get 1s, so that's our target here!
        labels = [1] * batch_size
        noise = self.make_noise(batch_size)
        self.update_disc_copy()
        self.num_gen_steps += 1
        # we do sanity check to ensure discriminator weights unchanged
        pre_weights = self.discriminator_copy.get_weights()
        rv = self.nested_generator.train_on_batch([batch_X, noise], labels)
        post_weights = self.discriminator_copy.get_weights()
        assert all(np.all(a == b) for a, b in zip(pre_weights, post_weights))
        return rv

    def disc_train_step(self, true_batch_X, true_batch_Y):
        self.discriminator.trainable = True
        # we will replace the first half of the batch with random junk
        junk_count = len(true_batch_X) // 2
        poses = self.generate_poses(true_batch_X[:junk_count])
        # fakes get a zero (GAN) or -1 (WGAN), true values get a 1
        labels = np.ones((len(true_batch_X), ))
        if WGAN:
            labels[:junk_count] = -1
        else:
            labels[:junk_count] = 0
        gen_poses = np.concatenate([poses, true_batch_Y[junk_count:]], axis=0)
        self.num_disc_steps += 1
        # get back loss
        return self.discriminator.train_on_batch([true_batch_X, gen_poses],
                                                 labels)

    # TODO: update {gen,disc} val so that they still work

    # def disc_val(self, val_data, batch_size):
    #     fakes = self.generate_poses(len(val_data))
    #     labels = np.array([1] * len(val_data) + [0] * len(fakes))
    #     data = np.concatenate([val_data, fakes])
    #     rv = self.discriminator.evaluate(data, labels, batch_size=batch_size)

    #     # The :100 thing was originally so that we would not run out of
    #     # memory when saving some quantity with TensorBoard. Not sure whether
    #     # it still matters.
    #     fit_indices = np.random.permutation(len(data))[:100]
    #     fit_data = (data[fit_indices], labels[fit_indices])
    #     self.discriminator.fit(
    #         *fit_data,
    #         batch_size=batch_size,
    #         nb_epoch=1,
    #         verbose=0,
    #         validation_data=fit_data)

    #     return rv

    # def gen_val(self, num_poses, batch_size):
    #     noise = self.make_noise(num_poses)
    #     labels = [1] * num_poses
    #     self.update_disc_copy()

    #     rv = self.nested_generator.evaluate(
    #         noise, labels, batch_size=batch_size)
    #     self.update_disc_copy()

    #     # Same trick we pull in disc_val
    #     fit_data = (noise[:100], labels[:100])
    #     self.nested_generator.fit(
    #         *fit_data,
    #         batch_size=batch_size,
    #         nb_epoch=1,
    #         verbose=0,
    #         validation_data=fit_data)

    #     return rv

    def generate_poses(self, cond_poses, batch_size=256):
        return self.generator.predict(
            [cond_poses, self.make_noise(len(cond_poses))],
            batch_size=batch_size)

    def save(self, dest_dir):
        """Save generator and discriminator to some path"""
        try:
            makedirs(dest_dir)
        except FileExistsError:
            pass
        suffix = '-%d-%d.h5' % (self.num_gen_steps, self.num_disc_steps)
        gen_path = path.join(dest_dir, 'gen' + suffix)
        disc_path = path.join(dest_dir, 'disc' + suffix)
        self.discriminator.save(disc_path)
        self.generator.save(gen_path)


def random_subset(count, array, *arrays):
    """Choose a random set of 'count' rows."""
    indices = np.random.permutation(len(array))[:count]
    fst = array[indices]
    if not arrays:
        # return single array, not tuple
        return fst
    # otherwise, do the same to all other arrays and return as tuple
    return (fst, ) + tuple(a[indices] for a in arrays)


def train_model(train_X,
                train_Y,
                val_X,
                val_Y,
                mask_value=0.0,
                noise_dim=32,
                save_path='./best-gan-weights'):
    assert train_X.ndim == 3, train_X.shape
    total_X, in_length, _ = train_X.shape
    _, out_length, out_shape = train_Y.shape
    trainer = GANTrainer(in_length, out_length, out_shape, noise_dim)
    epochs = 0
    # we could do way larger batches, but they probably won't help us converge
    # faster
    batch_size = 64
    epoch_size = 20000

    print('Training generator')

    while True:
        # Only use a subset of data in each epoch (otherwise epochs go forever)
        copy_X, copy_Y = random_subset(epoch_size, train_X, train_Y)
        total_X, _, _ = copy_X.shape
        epochs += 1
        print('Epoch %d' % epochs)
        bar = Progbar(total_X)
        bar.update(0)
        epoch_fetched = 0

        while epoch_fetched < total_X:
            # Fetch some ground truth to train the discriminator
            for i in range(DISCS_PER_GEN):
                if epoch_fetched >= total_X:
                    break
                fetched_X = copy_X[epoch_fetched:epoch_fetched + batch_size]
                fetched_Y = copy_Y[epoch_fetched:epoch_fetched + batch_size]
                dloss, dacc = trainer.disc_train_step(fetched_X, fetched_Y)
                epoch_fetched += len(fetched_X)
                bar.update(
                    epoch_fetched, values=[('d_loss', dloss), ('d_acc', dacc)])

            if epoch_fetched >= total_X:
                break

            # Train the generator on some subset (don't worry about loss)
            fetched_X = copy_X[epoch_fetched:epoch_fetched + batch_size]
            trainer.gen_train_step(fetched_X)
            epoch_fetched += len(fetched_X)

        # # End of an epoch, so let's validate models (kind of useless for GAN)
        # print('\nValidating')
        # # Again, use a small subset of 5K points
        # to_val = random_subset(val_X, 5000)
        # disc_loss, disc_acc = trainer.disc_val(to_val, batch_size)
        # gen_loss, gen_acc = trainer.gen_val(100, batch_size)
        # print('\nDisc loss/acc:   %g/%g' % (disc_loss, disc_acc))
        # print('Gen loss/acc:    %g/%g' % (gen_loss, gen_acc))

        # Sometimes we save a model
        # TODO: do proper validation on prediction accuracy, and only save
        # models when the generator improves in predictive accuracy
        if not (epochs - 1) % 5:
            print('Saving model to %s' % save_path)
            trainer.save(save_path)

    return trainer.generator


class Predictor:
    def __init__(self, orig_model, mask_value):
        # initial
        self.orig_model = orig_model
        self.pose_size = self.orig_model.input_shape[0][-1]
        self.noise_dim = self.orig_model.input_shape[1][-1]
        self._rebuild_model(1, 1)

    def _rebuild_model(self, in_steps, out_steps):
        """Instantiate new GAN model, then copy across weights. Keras may be
        able to do this with statefulness, but the stacked LSTM thing makes me
        suspect it's just too hard."""
        print('Rebuilding GAN (in=%d, out=%d)' % (in_steps, out_steps))
        new_gen = make_generator(in_steps, out_steps, self.noise_dim,
                                 self.pose_size)
        new_orig_pairs = zip(new_gen.layers, self.orig_model.layers)
        for new_layer, orig_layer in new_orig_pairs:
            new_layer.set_weights(orig_layer.get_weights())
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.cached_model = new_gen

    def __call__(self, in_tensor, steps_to_predict):
        assert in_tensor.ndim == 3, \
            "Expecting N*T*D tensor, got %s" % (in_tensor.shape,)
        in_steps = in_tensor.shape[1]
        if in_steps != self.in_steps \
           or steps_to_predict != self.out_steps:
            self._rebuild_model(in_steps, steps_to_predict)
        in_noise = np.random.randn(in_tensor.shape[0], steps_to_predict,
                                   self.noise_dim)
        out_tensor = self.cached_model.predict([in_tensor, in_noise])
        return out_tensor


def make_model_predict(trained_model, mask_value):
    return Predictor(trained_model, mask_value)


_gen_re = re.compile(r'^gen-(?P<gen_steps>\d+)-(?P<disc_steps>\d+)\.h5$')


def load_eval_model(model_dir, *args, **kwargs):
    gen_steps = {}
    for name in listdir(model_dir):
        match = _gen_re.match(name)
        if match is None:
            continue
        gscount = int(match.groupdict()['gen_steps'])
        full_path = path.join(model_dir, name)
        gen_steps[full_path] = gscount
    # get most recent model (maybe not a good idea? really should validate!)
    best_path = max(gen_steps.keys(), key=gen_steps.get)
    return load_model(best_path)
