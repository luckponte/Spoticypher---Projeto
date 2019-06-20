import config
from fycharts import SpotifyCharts
import pandas as pd
import numpy as np

# Get top200 for everyday of a year
def fetchYearlyChartList(type,country="global",*args):
	chartObj = SpotifyCharts.SpotifyCharts()

	if type == 'daily':
		for year in args:
			nextYear = str(int(year)+1)
			fileName = config.DATA['output_path']+'/'+country+'/top-daily-200-'+country+'-'+year+'.csv'
			open(fileName, "w")
			chartObj.top200Daily(output_file=fileName,start=year+'-01-01',end=nextYear+'-01-01',region=country)
	elif type == 'weekly':
		for year in args:
			nextYear = str(int(year)+1)
			fileName = config.DATA['output_path']+'/top-weekly-200-'+country+'-'+year+'.csv'
			startDate = ''
			endDate  =''
			if year == '2017':
				startDate = '2017-01-06'
				endDate = '2017-12-29'
			elif year == '2018':
				startDate = '2018-01-05'
				endDate = '2018-12-28'
			elif year == '2019':
				startDate = '2019-01-04'
				endDate = '2019-05-10'
			else:
				print("Ano fora do intervalo")
				break

			open(fileName, "w")
			chartObj.top200Weekly(output_file=fileName,start=startDate,end=endDate,region=country)
	return fileName

# Parse the chart's csv files into Dataframes
def parseCharts(csv):
	charts = pd.read_csv(csv)
	charts = charts.drop_duplicates('id')

	parsed = charts[['id','Track Name','date']].copy()
	parsed.index = np.arange(0, len(parsed))
	return parsed

# Write new file
def exportCharts(chart,location=config.DATA['output_path']+"/foo.csv"):
	chart.to_csv(location,encoding='utf-8-sig')
	return True