import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import operator

#import tensorflow as tf

import os
import pickle
import re
#from tensorflow.python.ops import math_ops

def load_data():

	#读取User数据
	users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
	users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
	users = users.filter(regex='UserID|Gender|Age|JobID')
	users_orig = users.values
	#改变User数据中性别和年龄
	gender_map = {'F':0, 'M':1}
	users['Gender'] = users['Gender'].map(gender_map)

	age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
	users['Age'] = users['Age'].map(age_map)

	#读取Movie数据集
	movies_title = ['MovieID', 'Title', 'Genres']
	movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
	movies_orig = movies.values
	#将Title中的年份去掉
	pattern = re.compile(r'^(.*)\((\d+)\)$')

	title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
	movies['Title'] = movies['Title'].map(title_map)

	#电影类型转数字字典
	genres_set = set()
	for val in movies['Genres'].str.split('|'):
		genres_set.update(val)

	genres_set.add('<PAD>')
	genres2int = {val:ii for ii, val in enumerate(genres_set)}

	#将电影类型转成等长数字列表，长度是18
	genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

	for key in genres_map:
		for cnt in range(max(genres2int.values()) - len(genres_map[key])):
			genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])

	movies['Genres'] = movies['Genres'].map(genres_map)

	#电影Title转数字字典
	title_set = set()
	for val in movies['Title'].str.split():
		title_set.update(val)

	title_set.add('<PAD>')
	title2int = {val:ii for ii, val in enumerate(title_set)}

	#将电影Title转成等长数字列表，长度是15
	title_count = 15
	title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}

	for key in title_map:
		for cnt in range(title_count - len(title_map[key])):
			title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])

	movies['Title'] = movies['Title'].map(title_map)

	#读取评分数据集
	ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
	ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
	ratings = ratings.filter(regex='UserID|MovieID|ratings')

	#合并三个表
	data = pd.merge(pd.merge(ratings, users), movies)

	#将数据分成X和y两张表
	target_fields = ['ratings']
	features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

	features = features_pd.values
	targets_values = targets_pd.values

	return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig
def getUserMovie(user_id_val):
	#title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
	#pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig), open('preprocess.p', 'wb'))
	title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))
	temp=ratings.values[np.where(ratings.values[:,0]==user_id_val)]
	#print(temp)
	result=[]
	for val in temp:
		a=movies_orig[np.where(movies_orig[:,0]==val[1])]
		for x in a:
			result.append(x)
	return np.insert(np.array(result),3,values=temp[:,2],axis=1)
	#temp1=movies_orig[np.where(movies_orig[:,0]==temp[0,1])]
	#print(temp1)
if __name__=="__main__":
	print(getUserMovie(234))