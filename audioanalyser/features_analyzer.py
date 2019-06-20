import config
import spotipy
import os
import math
import numpy as np
import pandas as pd 
import json
import glob
from spotipy.oauth2 import SpotifyClientCredentials 

sp = spotipy.Spotify() 

# expect a list of song ids
def getAudioFeatures(songs=[]):
	audioFeats = []
	# songs = list(filter(sanitizeItem, songs))
	for idx in range(0,len(songs),50):
		try:
			feats = pd.DataFrame(sp.audio_features(songs[idx:idx+50]))
			audioFeats.append(feats)
		except Exception as e:
			print("PROBLEM: \n {}".format(e))
			continue

	featList = pd.concat(audioFeats,axis=0)
	# print("FEATLIST: \n {}".format(featList))
	featList.index = np.arange(0, len(featList))
	return featList

def sanitizeItem(item):
	if item and item != None and item!='' and item!='nan' and not((type(item) is float) and math.isnan(item)):
		return True
	else:
		return False

# Auth stuff
client_credentials_manager = SpotifyClientCredentials(client_id=config.KEYS['spotify']['cid'], client_secret=config.KEYS['spotify']['secret']) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 

# Fetch the data
try:
	d = config.DATA['output_path']
	nationFolders = [os.path.join(d, o) for o in os.listdir(d) 
	                    if os.path.isdir(os.path.join(d,o))]
	if len(nationFolders) > 0:
		for folder in nationFolders:
			frames = []
			print(folder)
			parsedFiles = glob.glob(folder+'/parsed-*')
			if len(nationFolders) > 0:
				for file in parsedFiles:
					df = pd.read_csv(file)
					frames.append(df)
			if len(frames) > 0:
				songs = pd.concat(frames)
				songs = songs.drop_duplicates('id')
				songs = songs.dropna()
				songs.index = np.arange(0,len(songs))
				songsIds = songs['id'].values
				feats = getAudioFeatures(songsIds)
				audioAnalytics = ['id','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']
				feats = pd.concat([feats[audioAnalytics],songs['date']],axis=1,sort=False)
				feats.to_csv(folder+'/audio-features.csv',encoding='utf-8-sig')
except Exception as e:
	raise e

# songs = playlist["tracks"]["items"]
 
# ids = [] 
# for i in range(len(songs)): 
#     ids.append(songs[i]["track"]["id"]) 
# features = sp.audio_features(ids) 
# df = pd.DataFrame(features)