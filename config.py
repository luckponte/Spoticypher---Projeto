# Project settings
import os

workDir = os.path.abspath(os.path.dirname(__file__))

DATA = {
	'data_path': workDir+'/data',
	'output_path': workDir+'/data/outputs'
}

KEYS = {
	'spotify': {
		'cid': "64bd2aecb64c40c287b7ccec188f31e0",
		'secret': "1344656d801d42d6b2cb8d3336c269cc",
		'redirect': "http://localhost",
	}
}