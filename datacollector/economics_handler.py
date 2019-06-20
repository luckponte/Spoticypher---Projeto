import config
from fycharts import SpotifyCharts
import pandas as pd
import numpy as np
import glob
from datetime import datetime as dt

def getMonthBegin(date):
	monthBegin = dt.strptime(date,'%Y-%m-%d')
	monthBegin = dt(monthBegin.year,monthBegin.month,1)
	return dt.strftime(monthBegin,'%Y-%m-%d')

def matchValueByDate(date1,date2,df,col):
	matchId = np.where(date1.values == date2)[0]
	if matchId.size == 0 and col != 'daily_rate':
		date2 = getMonthBegin(date2)
		matchId = np.where(date1.values == date2)[0]
		if matchId.size == 0:
			return None


	if col == 'daily_rate' and matchId.size == 0:
		return None
	else:
		val = df.loc[matchId[0],col]
	return val

def generateTableset(country):
	audioFeats = pd.read_csv(config.DATA['output_path']+'/'+country+'/audio-features.csv')
	econTables = fetchEconomicData(country)
	ecoData = []

	for df in econTables:
		cols = list(df)
		for col in cols:
			if(col == 'date'):
				continue
			audioFeats[col] = audioFeats.apply(lambda row: matchValueByDate(df['date'],row['date'],df,col),axis=1)

	print("NEW SET: \n {}".format(audioFeats));
	audioFeats.to_csv(config.DATA['output_path']+'/'+country+'/audio-economic-features.csv')

def fetchEconomicData(country):
	econFiles = glob.glob(config.DATA['data_path']+'/economics/'+country+'/*')
	econTables = []
	for file in econFiles:
		df = pd.read_csv(file,index_col=[0])
		econTables.append(df)
	return econTables