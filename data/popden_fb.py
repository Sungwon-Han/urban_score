import pandas as pd
import convert
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--zl',help = 'zoom level')
parser.add_argument('--fbcsv', help='Facebook population density csv file path')
parser.add_argument('--ocsv', help='output .csv file')
args = parser.parse_args()

if args.zl ==None or args.fbcsv==None or args.ocsv==None:
	print("Error! please fill all facebook csv/ output csv file name")
	exit(0)

df = pd.read_csv(args.fbcsv)
aggregated_dict = OrderedDict()
#print(len(df))
for i in range(len(df)):
	x,y = convert.deg2num(df.iloc[i]['Lat'],df.iloc[i]['Lon'],args.zl)
	print(i, 'over', len(df), 'y',y,'x',x)
	try:
		aggregated_dict['{}_{}'.format(str(y),str(x))][0] = aggregated_dict['{}_{}'.format(str(y),str(x))][0]+df.iloc[i]['Population']
	except:
		aggregated_dict['{}_{}'.format(str(y),str(x))] = [df.iloc[i]['Population']]

aggregated_df = pd.DataFrame.from_dict(aggregated_dict,orient='index')

aggregated_df.to_csv(args.ocsv,header=False)
