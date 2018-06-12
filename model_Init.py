import tensorflow as tf
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def save_params(params):

	pickle.dump(params, open('params.p', 'wb'))


def load_params():

	return pickle.load(open('params.p', mode='rb'))

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))

#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040
#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1# 20 + 1 = 21

#电影ID个数
movie_id_max = max(features.take(1,1)) + 1 # 3952
#电影类型个数
movie_categories_max = max(genres2int.values()) + 1 # 18 + 1 = 19
#电影名单词个数
movie_title_max = len(title_set) # 5216

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

#电影名长度
sentences_size = title_count # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'

def get_inputs():
	uid = tf.placeholder(tf.int32, [None, 1], name="uid")
	user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
	user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
	user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
	
	movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
	movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
	movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
	targets = tf.placeholder(tf.int32, [None, 1], name="targets")
	LearningRate = tf.placeholder(tf.float32, name = "LearningRate")
	dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
	return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob
	
	
	
	
	
	
	
	
	
	
