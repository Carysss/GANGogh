
import os, sys
sys.path.append(os.getcwd())

import tflib.save_images
import tflib.wikiartGenre
import tflib.plot

from GANgogh import *

from tflib.wikiartGenre import get_style

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

ITERS = 2000 #200000  # How many iterations to train for
PREITERATIONS = 20# 2000  # Number of preiteration training cycles to run

def main():

    Generator, Discriminator = GeneratorAndDiscriminator()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
        all_real_label_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, CLASSES])

        generated_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, CLASSES])
        sample_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, CLASSES])

        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        split_real_label_conv = tf.split(all_real_label_conv, len(DEVICES))
        split_generated_labels_conv = tf.split(generated_labels_conv, len(DEVICES))
        split_sample_labels_conv = tf.split(sample_labels_conv, len(DEVICES))

        gen_costs, disc_costs = [], []

        for device_index, (device, real_data_conv, real_label_conv) in enumerate(
                zip(DEVICES, split_real_data_conv, split_real_label_conv)):

            with tf.device(device):
                real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                       [BATCH_SIZE // len(DEVICES), OUTPUT_DIM])
                real_labels = tf.reshape(real_label_conv, [BATCH_SIZE // len(DEVICES), CLASSES])

                generated_labels = tf.reshape(split_generated_labels_conv, [BATCH_SIZE // len(DEVICES), CLASSES])
                sample_labels = tf.reshape(split_sample_labels_conv, [BATCH_SIZE // len(DEVICES), CLASSES])

                fake_data, fake_labels = Generator(BATCH_SIZE // len(DEVICES), CLASSES, generated_labels)

                # set up discrimnator results

                disc_fake, disc_fake_class = Discriminator(fake_data, CLASSES)
                disc_real, disc_real_class = Discriminator(real_data, CLASSES)

                prediction = tf.argmax(disc_fake_class, 1)
                correct_answer = tf.argmax(fake_labels, 1)
                equality = tf.equal(prediction, correct_answer)
                genAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

                prediction = tf.argmax(disc_real_class, 1)
                correct_answer = tf.argmax(real_labels, 1)
                equality = tf.equal(prediction, correct_answer)
                realAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                gen_cost_test = -tf.reduce_mean(disc_fake)
                disc_cost_test = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                generated_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class,
                                                                                              labels=fake_labels))

                real_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class,
                                                                                         labels=real_labels))
                gen_cost += generated_class_cost
                disc_cost += real_class_cost

                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE // len(DEVICES), 1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha * differences)
                gradients = tf.gradients(Discriminator(interpolates, CLASSES)[0], [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                disc_cost += LAMBDA * gradient_penalty

                real_class_cost_gradient = real_class_cost * 50 + LAMBDA * gradient_penalty

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                 var_list=lib.params_with_name(
                                                                                                     'Generator'),
                                                                                                 colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=lib.params_with_name(
                                                                                                      'Discriminator.'),
                                                                                                  colocate_gradients_with_ops=True)
        class_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(real_class_cost_gradient,
                                                                                                   var_list=lib.params_with_name(
                                                                                                       'Discriminator.'),
                                                                                                   colocate_gradients_with_ops=True)
        # For generating samples

        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE // len(DEVICES)
            all_fixed_noise_samples.append(Generator(n_samples, CLASSES, sample_labels, noise=fixed_noise[
                                                                                              device_index * n_samples:(
                                                                                                                                   device_index + 1) * n_samples])[
                                               0])
            if tf.__version__.startswith('1.'):
                all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
            else:
                all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)


        def generate_image(iteration):
            for i in range(CLASSES):
                curLabel = genRandomLabels(BATCH_SIZE, CLASSES, condition=i)
                samples = session.run(all_fixed_noise_samples, feed_dict={sample_labels: curLabel})
                samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
                lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)),
                                            'generated/samples_{}_{}.png'.format(get_style(i), iteration))


        # Dataset iterator
        train_gen, dev_gen = lib.wikiartGenre.load(BATCH_SIZE, CLASSES)


        def softmax_cross_entropy(logit, y):
            return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


        def inf_train_gen():
            while True:
                for (images, labels) in train_gen():
                    yield images, labels


        _sample_labels = genRandomLabels(BATCH_SIZE, CLASSES)
        # Save a batch of ground-truth samples
        _x, _y = next(train_gen())
        _x_r = session.run(real_data, feed_dict={all_real_data_conv: _x})
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, 3, 64, 64)), 'generated/samples_groundtruth.png')

        session.run(tf.initialize_all_variables(), feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
        gen = train_gen()

        for iterp in range(PREITERATIONS):
            _data, _labels = next(gen)
            _, accuracy = session.run([disc_train_op, realAccuracy],
                                      feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels,
                                                 generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
            if iterp % 100 == 99:
                print('pretraining accuracy: ' + str(accuracy))

        for iteration in range(ITERS):
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op, feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
            # Train critic
            disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data, _labels = next(gen)
                _disc_cost, _disc_cost_test, class_cost_test, gen_class_cost, _gen_cost_test, _genAccuracy, _realAccuracy, _ = session.run(
                    [disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy,
                     realAccuracy, disc_train_op], feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels,
                                                              generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('wgan train disc cost', _disc_cost_test)
            lib.plot.plot('train class cost', class_cost_test)
            lib.plot.plot('generated class cost', gen_class_cost)
            lib.plot.plot('gen cost cost', _gen_cost_test)
            lib.plot.plot('gen accuracy', _genAccuracy)
            lib.plot.plot('real accuracy', _realAccuracy)

            if (iteration % 100 == 99 and iteration < 1000) or iteration % 1000 == 999:
                t = time.time()
                dev_disc_costs = []
                images, labels = next(dev_gen())
                _dev_disc_cost, _dev_disc_cost_test, _class_cost_test, _gen_class_cost, _dev_gen_cost_test, _dev_genAccuracy, _dev_realAccuracy = session.run(
                    [disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy,
                     realAccuracy], feed_dict={all_real_data_conv: images, all_real_label_conv: labels,
                                               generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
                dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                lib.plot.plot('wgan dev disc cost', _dev_disc_cost_test)
                lib.plot.plot('dev class cost', _class_cost_test)
                lib.plot.plot('dev generated class cost', _gen_class_cost)
                lib.plot.plot('dev gen  cost', _dev_gen_cost_test)
                lib.plot.plot('dev gen accuracy', _dev_genAccuracy)
                lib.plot.plot('dev real accuracy', _dev_realAccuracy)

#            if iteration % 1000 == 999:
            if iteration % 100 == 99:
                generate_image(iteration)
                # Can add generate_good_images method in here if desired

            if (iteration < 10) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

if __name__ == '__main__':
    main()

