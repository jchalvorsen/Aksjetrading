# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import csv
import pandas as pd
import numpy as np


FLAGS = None


def train():
	stock_type = "MHG"
	days_history = 4
	features = 4
	output_size = 2
	invis_layer_size = 32
	
	
	# Import data
	def get_dataset(name):
		df = pd.read_csv(name, sep=';', index_col=0, encoding="Latin1")
		df = df[::-1]
		df.columns = ['Siste', 'Kjoper', 'Selger', 'Hoy', 'Lav', 'Totalt_omsatt_NOK', 'Totalt_antall_aksjer_omsatt', 'Antall_off_handler', 'Antall_handler_totalt', 'VWAP']
		return df


	def get_currency_data(name):
		df = pd.read_csv(name, sep=';', index_col=0, encoding="Latin1")
		return df


	def append_currency_to_stock_price_df(df_stock, df_currency):
		df_stock["Dollarkurs"] = df_currency["1 USD"]


	stock = get_dataset("data\\" + stock_type + "_data.csv")
	nok = get_currency_data("data\\Valutakurser.csv")

	append_currency_to_stock_price_df(stock, nok)
	
	# Initialize labels
	Up = np.roll(stock.Siste.diff(1),-1) > 0
	n = len(Up)
	train = np.zeros((n, 2))
	train[:,0] = (Up == 0)
	train[:,1] = (Up == 1)
	train[:,1] = (Up == 1)
	cutoff = int(np.floor(n*0.8))
	Y_train = train[:cutoff,:]
	Y_test = train[cutoff:,:]
	Y_train = Y_train[days_history:]
	
	# Initialize training data
	input_dimension = features*days_history

	Siste_all = np.zeros((n, days_history))
	Siste = stock.Siste.values

	Omsatte_all = np.zeros((n, days_history))
	Omsatte = stock.Totalt_antall_aksjer_omsatt.values

	Antall_handler_all = np.zeros((n, days_history))
	Antall_handler = stock.Antall_handler_totalt.values

	Valutakurs_all = np.zeros((n, days_history))
	Valutakurs = stock.Dollarkurs.values


	def normalize(vector, ref):
		ref_min = min(ref)
		ref_max = max(ref)
		vector -= ref_min
		return vector / (ref_max-ref_min)
		

	for i in range(days_history):
		Siste_all[:,i] = np.roll(Siste, i)
		Omsatte_all[:,i] = np.roll(Omsatte, i)
		Antall_handler_all[:,i] = np.roll(Antall_handler, i)
		Valutakurs_all[:,i] = np.roll(Valutakurs, i)
	   
	Siste_all = normalize(Siste_all, Siste)    
	Omsatte_all = normalize(Omsatte_all, Omsatte)
	Antall_handler_all = normalize(Antall_handler_all, Antall_handler)
	Valutakurs_all = normalize(Valutakurs_all, Valutakurs)

	#plt.plot(Siste_all[cutoff:])
	#plt.show()

	X_data = Siste_all
	X_data = np.append(X_data,Omsatte_all, 1)
	X_data = np.append(X_data,Antall_handler_all, 1)
	X_data= np.append(X_data,Valutakurs_all, 1)


	#plt.plot(X_data[days_history:])
	#plt.show()
	#print(X_data.shape)

	X_train = X_data[:cutoff,:]
	X_test = X_data[cutoff:,:]
	X_train = X_train[days_history:]
	
	
	config = tf.ConfigProto(device_count = {'GPU': 0})
	sess = tf.InteractiveSession(config=config)
	# Create a multilayer model.

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder("float", [None, input_dimension], name='x-input')
		y_ = tf.placeholder("float", [None, output_size], name='y-input')


  # We can't initialize these variables to 0 - the network will get stuck.
	def weight_variable(shape):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=1)
		return tf.Variable(initial)

	def variable_summaries(var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Reusable code for making a simple neural net layer.
		It does a matrix multiply, bias add, and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
		# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, output_dim])
				variable_summaries(weights)
			with tf.name_scope('Wx'):
				preactivate = tf.matmul(input_tensor, weights)
				tf.summary.histogram('pre_activations', preactivate)
			activations = act(preactivate, name='activation')
			tf.summary.histogram('activations', activations)
		return activations

	hidden1 = nn_layer(x, input_dimension, invis_layer_size, 'layer1', act = tf.nn.tanh)


	# Do not apply softmax activation yet, see below.
	y = nn_layer(hidden1, invis_layer_size, output_size, 'layer2', act=tf.identity)

	with tf.name_scope('cross_entropy'):
		# The raw formulation of cross-entropy,
		#
		# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
		#                               reduction_indices=[1]))
		#
		# can be numerically unstable.
		#
		# So here we use tf.nn.softmax_cross_entropy_with_logits on the
		# raw outputs of the nn_layer above, and then average across
		# the batch.
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)
		tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
			cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	# Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '\train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '\test')
	tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

	def feed_dict(train):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		if train or FLAGS.fake_data:
			xs = X_train
			ys = Y_train
			#xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
		else:
			xs = X_test
			ys = Y_test
			#xs, ys = mnist.test.images, mnist.test.labels
		return {x: xs, y_: ys}

	for i in range(FLAGS.max_steps):
		if i % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		else:  # Record train set summaries, and train
			if i % 100 == 99:  # Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)
			else:  # Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary, i)
	train_writer.close()
	test_writer.close()


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
		tf.gfile.MakeDirs(FLAGS.log_dir)
	train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
					default=False,
					help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000,
					help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
					help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
					help='Keep probability for training dropout.')
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
					help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
					help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)