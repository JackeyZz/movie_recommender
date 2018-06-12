from model_Init import *

## 定义User的嵌入矩阵
def get_user_embedding(uid, user_gender, user_age, user_job):
	with tf.name_scope("user_embedding"):
		uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name = "uid_embed_matrix")
		uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name = "uid_embed_layer")
	
		gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1), name= "gender_embed_matrix")
		gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name = "gender_embed_layer")
	
		age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
		age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")
	
		job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name = "job_embed_matrix")
		job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name = "job_embed_layer")
	return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

## 将user嵌入矩阵全连接生成特征
def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
	with tf.name_scope("user_fc"):
		#第一层全连接
		uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name = "uid_fc_layer", activation=tf.nn.relu)
		gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name = "gender_fc_layer", activation=tf.nn.relu)
		age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name ="age_fc_layer", activation=tf.nn.relu)
		job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name = "job_fc_layer", activation=tf.nn.relu)
	
		#第二层全连接
		user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  #(?, 1, 128)
		user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  #(?, 1, 200)
	
		user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
	return user_combine_layer, user_combine_layer_flat

## 定义movie ID的嵌入矩阵
def get_movie_id_embed_layer(movie_id):
	with tf.name_scope("movie_embedding"):
		movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name = "movie_id_embed_matrix")
		movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name = "movie_id_embed_layer")
	return movie_id_embed_layer

## 对电影类型的多个嵌入向量做加和
def get_movie_categories_layers(movie_categories):
	with tf.name_scope("movie_categories_layers"):
		movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name = "movie_categories_embed_matrix")
		movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name = "movie_categories_embed_layer")
		if combiner == "sum":
			movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
	
	return movie_categories_embed_layer

## movies title的文本卷积网络实现
def get_movie_cnn_layer(movie_titles):
	#从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
	with tf.name_scope("movie_embedding"):
		movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name = "movie_title_embed_matrix")
		movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name = "movie_title_embed_layer")
		movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

	#对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
	pool_layer_lst = []
	for window_size in window_sizes:
		with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
			filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num],stddev=0.1),name = "filter_weights")
			filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")
	
			conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1,1,1,1], padding="VALID", name="conv_layer")
			relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias), name ="relu_layer")
	
			maxpool_layer = tf.nn.max_pool(relu_layer, [1,sentences_size - window_size + 1 ,1,1], [1,1,1,1], padding="VALID", name="maxpool_layer")
			pool_layer_lst.append(maxpool_layer)

	#Dropout层
	with tf.name_scope("pool_dropout"):
		pool_layer = tf.concat(pool_layer_lst, 3, name ="pool_layer")
		max_num = len(window_sizes) * filter_num
		pool_layer_flat = tf.reshape(pool_layer , [-1, 1, max_num], name = "pool_layer_flat")
	
		dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name = "dropout_layer")
	return pool_layer_flat, dropout_layer

## 将movie的各个层做全连接
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
	with tf.name_scope("movie_fc"):
		#第一层全连接
		movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name = "movie_id_fc_layer", activation=tf.nn.relu)
		movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name = "movie_categories_fc_layer", activation=tf.nn.relu)
	
		#第二层全连接
		movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  #(?, 1, 96)
		movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  #(?, 1, 200)
	
		movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
	return movie_combine_layer, movie_combine_layer_flat
	
## 构建计算图
#def calcGraph():
	

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
	#获取输入占位符
	uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
	#获取User的4个嵌入向量
	uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
	#得到用户特征
	user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
	#获取电影ID的嵌入向量
	movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
	#获取电影类型的嵌入向量
	movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
	#获取电影名的特征向量
	pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
	#得到电影特征
	movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer, 
																				movie_categories_embed_layer, 
																				dropout_layer)
	#计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
	with tf.name_scope("inference"):
		#将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
		#inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
#         	inference = tf.layers.dense(inference_layer, 1,
#                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
#                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
        #简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
#        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
		inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
		inference = tf.expand_dims(inference, axis=1)

	with tf.name_scope("loss"):
		# MSE损失，将计算值回归到评分
		cost = tf.losses.mean_squared_error(targets, inference )
		loss = tf.reduce_mean(cost)
	# 优化损失 
#train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(lr)
	gradients = optimizer.compute_gradients(loss)  #cost
	train_op = optimizer.apply_gradients(gradients, global_step=global_step)
print(inference)
























