import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
#import tensorflow as tf

import os
import pickle
import re
#from tensorflow.python.ops import math_ops

from urllib.request import urlretrieve
from os.path import isfile,isdir
from tqdm import tqdm
import zipfile
import hashlib

def _unzip(save_path, _, database_name, data_path):
	print('Extracting {}...'.format(database_name))
	with zipfile.ZipFile(save_path) as zf:
		zf.extractall(data_path)

def download_extract(database_name, data_path):
	DATASET_ML1M='ml-1m'
	if database_name==DATASET_ML1M:
		url='http://files.grouplens.org/datasets/movielens/ml-1m.zip'
		hashcode='c4d9eecfca2ab87c1945afe126590906'
		extract_path=os.path.join(data_path,'ml-1m')
		save_path=os.path.join(data_path,'ml-1m.zip')
		extract_fn=_unzip

	if os.path.exists(extract_path):
		print('Found {} Data'.format(database_name))
		return

	if not os.path.exists(data_path):
		os.makedirs(data_path)

	if not os.path.exists(save_path):
		with DLProgress(unit='B',unit_scale=True,miniters=1,desc='Downloading {}'.format(database_name)) as pbar:
			urlretrieve(
				url,
				save_path,
				pbar.hook)

	assert hashlib.md5(open(save_path,'rb').read()).hexdigest()==hashcode,\
		'{} file is corrupted. Remove the file and try again.'.format(save_path)

	os.makedirs(extract_path)
	try:
		extract_fn(save_path,extract_path,database_name,data_path)
	except Exception as err:
		shutil.rmtree(extract_path)
		raise err

	print('Done.')

class DLProgress(tqdm):
	last_block=0
	def hook(self,block_num=1,block_size=1,total_size=None):
		self.total=total_size
		self.update((block_num-self.last_block)*block_size)
		self.last_block=block_num
