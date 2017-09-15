import os

def full_path(filename):
	return os.path.join(os.path.dirname(__file__), filename)

def read_file(filename):
	path = full_path(filename)
	with open(path, 'r') as file:
		return file.read()