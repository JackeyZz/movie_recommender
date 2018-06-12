from Network import *

def get_batches(Xs, ys, batch_size):
	for start in range(0, len(Xs), batch_size):
		end = min(start + batch_size, len(Xs))
		yield Xs[start:end], ys[start:end]

## 训练网络
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import time
import datetime

losses = {'train':[], 'test':[]}

with tf.Session(graph=train_graph) as sess:
	
	#搜集数据给tensorBoard用
	# Keep track of gradient values and sparsity
	grad_summaries = []
	for g, v in gradients:
		if g is not None:
			grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
			sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
			grad_summaries.append(grad_hist_summary)
			grad_summaries.append(sparsity_summary)
	grad_summaries_merged = tf.summary.merge(grad_summaries)
	
	# Output directory for models and summaries
	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
	print("Writing to {}\n".format(out_dir))
	
	# Summaries for loss and accuracy
	loss_summary = tf.summary.scalar("loss", loss)

	# Train Summaries
	train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
	train_summary_dir = os.path.join(out_dir, "summaries", "train")
	train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	# Inference summaries
	inference_summary_op = tf.summary.merge([loss_summary])
	inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
	inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for epoch_i in range(num_epochs):
	
		#将数据集分成训练集和测试集，随机种子不固定
		train_X,test_X, train_y, test_y = train_test_split(features,  
															targets_values,  
															test_size = 0.2,  
															random_state = 0)  
	
		train_batches = get_batches(train_X, train_y, batch_size)
		test_batches = get_batches(test_X, test_y, batch_size)
	
		#训练的迭代，保存训练损失
		for batch_i in range(len(train_X) // batch_size):
			x, y = next(train_batches)

			categories = np.zeros([batch_size, 18])
			for i in range(batch_size):
				categories[i] = x.take(6,1)[i]

			titles = np.zeros([batch_size, sentences_size])
			for i in range(batch_size):
				titles[i] = x.take(5,1)[i]

			feed = {
				uid: np.reshape(x.take(0,1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
				user_age: np.reshape(x.take(3,1), [batch_size, 1]),
				user_job: np.reshape(x.take(4,1), [batch_size, 1]),
				movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
				movie_categories: categories,  #x.take(6,1)
				movie_titles: titles,  #x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: dropout_keep, #dropout_keep
				lr: learning_rate}

			step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  #cost
			losses['train'].append(train_loss)
			train_summary_writer.add_summary(summaries, step)  #
			
			# Show every <show_every_n_batches> batches
			if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
				time_str = datetime.datetime.now().isoformat()
				print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
					time_str,
					epoch_i,
					batch_i,
					(len(train_X) // batch_size),
					train_loss))
				
		#使用测试数据的迭代
		for batch_i  in range(len(test_X) // batch_size):
			x, y = next(test_batches)
			
			categories = np.zeros([batch_size, 18])
			for i in range(batch_size):
				categories[i] = x.take(6,1)[i]

			titles = np.zeros([batch_size, sentences_size])
			for i in range(batch_size):
				titles[i] = x.take(5,1)[i]

			feed = {
				uid: np.reshape(x.take(0,1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
				user_age: np.reshape(x.take(3,1), [batch_size, 1]),
				user_job: np.reshape(x.take(4,1), [batch_size, 1]),
				movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
				movie_categories: categories,  #x.take(6,1)
				movie_titles: titles,  #x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: 1,
				lr: learning_rate}
			
			step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  #cost

			#保存测试损失
			losses['test'].append(test_loss)
			inference_summary_writer.add_summary(summaries, step)  #

			time_str = datetime.datetime.now().isoformat()
			if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
				print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
					time_str,
					epoch_i,
					batch_i,
					(len(test_X) // batch_size),
					test_loss))

	# Save Model
	saver.save(sess, save_dir)  #, global_step=epoch_i
	print('Model Trained and Saved')