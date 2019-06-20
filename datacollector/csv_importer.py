import config
import pandas as pd 
from datacollector.charts_handler import fetchYearlyChartList, parseCharts, exportCharts

print("Spotify Charts Importer")
year = input("year:")
country = input("country:")
dw = input("daily or weekly:")

try:
	fetchYearlyChartList(dw,country,year)
	filename = config.DATA['output_path']+'/'+country+'/top-'+dw+'-200-'+country+'-'+year+'.csv'
	chart = parseCharts(filename)
	filename = config.DATA['output_path']+'/'+country+'/parsed-top-'+dw+'-200-'+country+'-'+year+'.csv'
	exportCharts(chart,filename)
except Exception as e:
	raise e