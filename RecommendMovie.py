import tensorflow as tf
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from model_Init import *

load_dir = load_params()
## 获取tensor
def get_tensors(loaded_graph):

	uid = loaded_graph.get_tensor_by_name("uid:0")
	user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
	user_age = loaded_graph.get_tensor_by_name("user_age:0")
	user_job = loaded_graph.get_tensor_by_name("user_job:0")
	movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
	movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
	movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
	targets = loaded_graph.get_tensor_by_name("targets:0")
	dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
	lr = loaded_graph.get_tensor_by_name("LearningRate:0")
	#两种不同计算预测评分的方案使用不同的name获取tensor inference
	#inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
	inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0") # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
	movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
	user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
	return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat

## 指定用户和电影进行评分
def rating_movie(user_id_val, movie_id_val):
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
	
		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference,_, __ = get_tensors(loaded_graph)  #loaded_graph
	
		categories = np.zeros([1, 18])
		categories[0] = movies.values[movieid2idx[movie_id_val]][2]
	
		titles = np.zeros([1, sentences_size])
		titles[0] = movies.values[movieid2idx[movie_id_val]][1]
		
		feed = {
			uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
			user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
			user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
			user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
			movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
			movie_categories: categories,  #x.take(6,1)
			movie_titles: titles,  #x.take(5,1)
			dropout_keep_prob: 1}
	
		# Get Prediction
		inference_val = sess.run([inference], feed)  
	
		return (inference_val)

## 生成movie特征矩阵
def getMovieFec():
	loaded_graph = tf.Graph()  #
	movie_matrics = []
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(loaded_graph)  #loaded_graph

		for item in movies.values:
			categories = np.zeros([1, 18])
			categories[0] = item.take(2)

			titles = np.zeros([1, sentences_size])
			titles[0] = item.take(1)

			feed = {
				movie_id: np.reshape(item.take(0), [1, 1]),
				movie_categories: categories,  #x.take(6,1)
				movie_titles: titles,  #x.take(5,1)
				dropout_keep_prob: 1}

			movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)  
			movie_matrics.append(movie_combine_layer_flat_val)

	pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))
	movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
	
## 生成User特征矩阵
def getUserFec():
	loaded_graph = tf.Graph()  #
	users_matrics = []
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __,user_combine_layer_flat = get_tensors(loaded_graph)  #loaded_graph

		for item in users.values:

			feed = {
				uid: np.reshape(item.take(0), [1, 1]),
				user_gender: np.reshape(item.take(1), [1, 1]),
				user_age: np.reshape(item.take(2), [1, 1]),
				user_job: np.reshape(item.take(3), [1, 1]),
				dropout_keep_prob: 1}

			user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)  
			users_matrics.append(user_combine_layer_flat_val)

	pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
	users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

## 推荐电影
def recommend_same_type_movie(movie_id_val, top_k = 20):
    
    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        #推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        #results = (-sim[0]).argsort()[0:top_k]
        #print(results)
        
        #print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        #print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        temp=[]
        while len(results) != 6:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            #print(val)
            #print(movies_orig[val])
            temp.append(movies_orig[val])
        return np.array(movies_orig[movieid2idx[movie_id_val]]),np.array(temp)
		
### 
def recommend_your_favorite_movie(user_id_val,top_k=10):

    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

		#推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
	#     print(sim.shape)
	#     results = (-sim[0]).argsort()[0:top_k]
	#     print(results)
		
	#     sim_norm = probs_norm_similarity.eval()
	#     print((-sim_norm[0]).argsort()[0:top_k])
	
		#print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        temp=[]
        while len(results) != 6:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
			#print(val)
			#print(movies_orig[val])
            temp.append(movies_orig[val])

        return np.array(temp)

## 喜欢看这个电影的人还看了哪些电影
import random

def recommend_other_favorite_movie(movie_id_val, top_k = 20):

    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
        
        print(favorite_user_id)
	#     print(normalized_users_matrics.eval().shape)
	#     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
	#     print(favorite_user_id.shape)
	
        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
		
        print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id-1]))
        probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
	#     results = (-sim[0]).argsort()[0:top_k]
	#     print(results)
	
	#     print(sim.shape)
	#     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
		######print("喜欢看这个电影的人还喜欢看：")

        results = set()
        temp=[]
        while len(results) != 6:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
			######print(val)
			######print(movies_orig[val])
            temp.append(movies_orig[val])
		
        return np.array(temp)
if __name__=="__main__":
    temp=recommend_other_favorite_movie(1401,20)