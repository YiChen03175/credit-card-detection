'''
@File ShowSamplePath.py
@brief Show all the data sample in test folder
'''

from os import walk

# path of test folder
path = './images/test'

for dirpath, _, filenames in walk(path):
	for name in filenames:
		print(dirpath, name)