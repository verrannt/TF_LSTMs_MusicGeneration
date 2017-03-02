'''

Python implementation of a recurrent neural network for generation of music
sequences using LSTM cells in Tensorflow.

@authors: Leonhard von Heinz, Leon Schmid, Pascal Schröder
@date: 28.02.2017
@version: 0.8

Notes:
Since we use parts of the non-stable tensorflow.contrib library future
functionality can not be guaranteed.

'''

import sklearn.preprocessing as prep
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------


# - GENERAL
TRAINING_PHASE = True                   # specialize if training or testing phase
input_numbers  = 60719                  # how many samples we have in our dataset
frame_rate     = 22050                  # we are using 22,05kHz samples

# - PARAMETERS FOR TRAINING
number_of_epochs       = 3              # number of training epochs
number_of_mini_batches = 150            # how many mini batches we want to feed in one epoch
mini_batch_size        = 40             # how many sound samples we want to feed in one mini batch
time_steps             = 50             # in how many timesteps we want to split one sound sample
eval_time_steps        = 2              # how many timesteps of the input we want to use for backpropagation
store_directory         = "wav-snippets"# directory in which to look for files
weights_directory       = "./weights/"  # directory in which to store session weights

# - PARAMETERS FOR NETWORK DEFINITION
learning_rate          = 0.001          # initial learning rate for the backpropagation
learning_adapt_rate    = 5              # scale for how fast the learning rate will adapt
segment_length         = 10             # samples are 10 seconds long
n_in_time_steps        = time_steps-1   # save for backpropagation
n_units                = 2048           # number of LSTM units
n_input  = frame_rate*segment_length/time_steps*2       # datapoints in 1/5th of a second for 22.05kHz samples
n_output = int(frame_rate*segment_length/time_steps)*2  # size of the linear layer output vector

# - FEEDS FOR THE NETWORK
x = tf.placeholder(tf.float32,          # network data input
    (None, n_in_time_steps, n_input))
y = tf.placeholder(tf.float32,          # training/backprop input
    (None, eval_time_steps, n_output))


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------


def fourierTransform(input_signal):
    '''
        creates the fourier transform from pcm input and normalizes it using
        the sklearn preprocessing library
        ---
        @params
        input_signal: array containing the sound data to be processed
        @returns
        fourier: array containing the real parts of the fourier transform
            in its first half, imag parts in its seconds half
    '''
    # use numpy's fourier transform library to calculate fourier transform
    fourier_signal = np.fft.fft(input_signal)
    # put real part into the first half, imag part in the second half
    fourier = np.concatenate((fourier_signal.real, fourier_signal.imag))
    # decrease to values that float32 can handle and cast to float32
    fourier = (fourier/np.amax(fourier)).astype('float32')
    # normalize using sklearn
    fourier = prep.normalize([fourier])
    # reshape to fit network demands
    fourier = fourier.reshape((1,-1)).reshape(-1)
    return fourier

def invFourierTransform(output_signal):
    '''
        creates the inverse fourier transform from float32 input array
        ---
        @params
        output_signal: array containing the sound data to be processed
            first half contains real parts, seconds half contains imag parts
        @returns
        output_signal.real: array containing the real parts of the audio signal
    '''
    output_signal = output_signal / np.amax(np.absolute(output_signal))
    (myreals, myimags) = np.split(output_signal,2)
    output_signal = np.vectorize(complex)(myreals, myimags)
    output_signal = np.real(np.fft.ifft(output_signal))
    output_signal = output_signal / np.amax(np.absolute(output_signal))
    output_signal = (output_signal*127)+128
    output_signal = output_signal.astype('uint8')
    return output_signal.real

def generateBatch(mini_batch_size, is_used_vector):
    '''
        generates a random input and validation batch from a defined file
        directory to be fed into the network
        ---
        @params
        mini_batch_size: the size of the mini batch we want to generate
        is_used_vector: array of size "input_numbers" containing bools depicting
            whether the sound sample at index has already been used for training
            True: has not been used; False: has been used already
        @returns
        batch_x: python array containing input values for the network
        batch_y: python array containing values for backpropagation
        is_used_vector: altered state of the is_used_vector
    '''
    batch_x, batch_y = [], []
    # For each element to be used in this batch
    for i in range(mini_batch_size):
        while True:
            # we want a ranodom sound sample from the dataset
            random_index = np.random.randint(0,input_numbers)
            # iff it has not been used before.
            if is_used_vector[random_index]:
                one_sound_sample = getSoundSample(
                    store_directory, "musicdata_{}.wav".format(random_index))
                # Need to apply fourier transform timestep-wise
                for i, pcm in enumerate(one_sound_sample):
                    one_sound_sample[i] = fourierTransform(pcm)
                # From that sample we want all but the last timestep as network input
                sample_part_for_network  = one_sound_sample[:-1]
                # And the last $eval_time_steps will be used for backpropagation
                sample_part_for_backprop = one_sound_sample[-eval_time_steps:]
                batch_x.append(sample_part_for_network)
                batch_y.append(sample_part_for_backprop)
                # Note that this sample has been used now
                is_used_vector[random_index] = False
                break
    return batch_x, batch_y, is_used_vector

def getSoundSample(store_directory, sample_name):
    '''
        reads a sound sample into python using scipy and segments it into
        segments of desired length
        ---
        @params
        store_directory: str, the directory in which to look for the file to import
        sample_name: str, the name of the file to import
        @returns
        one_sound_sample: the segmented sound sample as numpy array
    '''
    one_sound_sample = wavfile.read(
        "./{}/{}".format(store_directory, sample_name))[1]
    one_sound_sample = one_sound_sample.astype("float32")
    one_sound_sample = (one_sound_sample/128)-0.5
    one_sound_sample = np.array_split(one_sound_sample, time_steps)
    return one_sound_sample


# ------------------------------------------------------------------------------
# DATA FLOW GRAPH
# ------------------------------------------------------------------------------


# Define our LSTM cell from tensorflow's contrib library with variable number of units
cell = tf.contrib.rnn.BasicLSTMCell(n_units)

# Define the zero state of the cell
initial_state = cell.zero_state(mini_batch_size, tf.float32)

# Launch dynamic RNN Network with specified cell and initial state
# We use time_major=False because we would need to transpose the input on our own otherwise
rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(cell, x,
                            initial_state=initial_state, time_major=False)

# Get the last $eval_time_steps timestep(s) for training
rnn_outputs_on_last_t_step =  tf.slice(
                            rnn_outputs,
                            [0, n_in_time_steps - (1+eval_time_steps), 0],
                            [mini_batch_size, eval_time_steps, n_units])

# Project output from rnn output size to n_output
final_projection = lambda z: layers.linear(z, num_outputs=n_output,
                            activation_fn=tf.nn.sigmoid)

# Apply projection to every time step
predicted = tf.map_fn(final_projection, rnn_outputs_on_last_t_step)

# Error and backprop
error = tf.nn.l2_loss(tf.subtract(tf.abs(y),tf.abs(predicted)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

# Prediction error and accuracy
accuracy = tf.reduce_mean(tf.subtract(tf.abs(y),tf.abs(predicted)))


#-------------------------------------------------------------------------------
# RUN THE NETWORK
#-------------------------------------------------------------------------------


if TRAINING_PHASE:

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Iterate through all epochs
        for epoch in range(number_of_epochs):

            # Reset the is_used_vector to state that no sample has been used yet
            is_used_vector = np.ones(input_numbers+1).astype(bool)

            # Array variables we use for testing the network
            error_function = np.zeros(number_of_mini_batches)
            accuracy_function = np.zeros(number_of_mini_batches)
            learning_rate_array = np.zeros(number_of_mini_batches)

            # Training loop
            for mini_batch_number in range(number_of_mini_batches):
                # Generate an input and backprop batch and update the is_used_vector
                batch_x, batch_y, is_used_vector = generateBatch(mini_batch_size, is_used_vector)

                training_accuracy, prediction_error, _ = session.run(
                    [accuracy,
                    error,
                    train_step],
                    feed_dict = {x: batch_x, y: batch_y})

                # Adapt learning rate
                if mini_batch_number > 10:
                    learning_rate = np.mean(accuracy_function[-3:])/learning_adapt_rate
                    if ((epoch > 0) | (mini_batch_number > 10)):
                        if (np.absolute(np.absolute(learning_rate_array[mini_batch_number]) - np.absolute(learning_rate_array[mini_batch_number-1])) > (np.absolute(learning_rate_array[mini_batch_number-1]) / 10)):
                            learning_rate = (((2*learning_rate_array[mini_batch_number-1])+learning_rate_array[mini_batch_number])/3)
                    if learning_rate < 0.000001:
                        learning_rate = 0.000001

                # Feed into arrays for plotting
                learning_rate_array[mini_batch_number] = learning_rate
                error_function[mini_batch_number] = prediction_error
                accuracy_function[mini_batch_number] = training_accuracy

                print("Training accuracy and prediction error in batch {}: {}, {}".format(
                        mini_batch_number, training_accuracy, prediction_error))

            # Save the weights at the end of each epoch
            saver.save(session, 'LSTM-weights', global_step=epoch)

            # Plots for network optimization at end of each epoch
            plt.subplot(311)
            plt.xlabel("Batch number")
            plt.ylabel("L2 error")
            plt.plot(error_function)
            plt.subplot(312)
            plt.xlabel("Batch number")
            plt.ylabel("Accuracy: mean difference between data point")
            plt.plot(accuracy_function)
            plt.subplot(313)
            plt.xlabel("Batch number")
            plt.ylabel("Learning rate")
            plt.plot(learning_rate_array)
            plt.show()

else: # if not training phase but generation

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, "{}LSTM-weights-0".format(weights_directory))

        desired_length = 20 # seconds
        sample_name = "musicdata_300.wav"
        generation_length = desired_length * (time_steps/10)

        seed_sequence = getSoundSample(store_directory, sample_name)
        working_sequence = seed_sequence
        generated_sequence = seed_sequence

        for steps in range(generation_length):
            network_prediction = session.run(
                predicted,
                feed_dict = {x: sound_sample})
            working_sequence = working_sequence[1:]
            working_sequence.append(network_prediction[0])
            generated_sequence.append(network_prediction[0])
            sound_sample = working_sequence

        generated_sequence = np.concat(generated_sequence)
        wavfile.write("MyTestSequence.wav", 22050, generated_sequence)
