import json
import csv
import os
import urllib.request
import random
from time import sleep
import argparse
import pandas as pd
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--zyx', help='input json zyx location file path')
parser.add_argument('--shufdat', help='input shuffled csv data file path')
parser.add_argument('--odir', help='output directory name')
parser.add_argument('--skip', help = 'skip first x areaIDs')

args = parser.parse_args()

if args.zyx==None or args.shufdat==None or args.token==None or args.odir==None or args.skip==None:
	print("Error! please fill all zyx json, shufdat csv, output directory, token, skip information")
	exit(0)

ftlist = None
token = args.token
zyx = args.zyx
dat = args.shufdat
odir = args.odir
skip = int(args.skip)
zfill_val = 5

base_url = "https://tiledbasemaps.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/"
#token = "lBRbigHSYEOvanHw_wB698Yr1izHkaahgcPFm1GOLE-Lj82yWtZEQ2Ks5OZ41U0RGcPT4j4voqrLU8XWaiNjrpdZ8TpSyzjldcGN9VV9iLLqxYO1hcaqx87APVshRKlgVct9CWtwG_PdDlArkK161A__mc3CkHE__46JSSxLlD6KSQnAqWal6mwdBX6kHoyHHhPyCjMr61x3GiuMI8CaD_HKuZtUJ_zL6Q_Vpq3DR-k."

csv_write_candidate = []
if not os.path.isdir('./'+odir):
	os.mkdir('./'+odir)

with open(zyx,'r') as json_file_zyx:
	data_zyx=json.load(json_file_zyx)
	data_zyx_keylist = list(data_zyx.keys())
	zfill_len = list(map(lambda x: len(x), data_zyx_keylist))
	if min(zfill_len)==max(zfill_len):
		zfill_val = min(zfill_len)

	dfx = pd.read_csv(dat)
	dfx = dfx.sort_values(by=['areaID'])
	dfx = dfx.tail(-skip)
	print(dfx.head())
	for index, row in dfx.iterrows():
		curr_zyxs = data_zyx[str(row['areaID']).zfill(zfill_val)]
		print(str(row['areaID']).zfill(zfill_val), row['Directory'], len(curr_zyxs))
		if not os.path.isdir('./'+odir+'/'+str(row['Directory'])):
			os.mkdir('./'+odir+'/'+str(row['Directory']))
		for curr_zyx in curr_zyxs:
			url = base_url+str(curr_zyx[0])+'/'+str(curr_zyx[1])+'/'+str(curr_zyx[2])+'?token='+token
			filename = './'+odir+'/'+str(row['Directory'])+'/'+str(curr_zyx[1])+'_'+str(curr_zyx[2])+'.png'
			try:
				print('try',filename)
				urllib.request.urlretrieve(url, filename)
			except:
				sleep(0.5)
				print('sleep and try',filename)
				urllib.request.urlretrieve(url, filename)

	'''
	data_zyx=json.load(json_file_zyx)
	data_zyx_keylist = list(data_zyx.keys()) 
	print(data_zyx_keylist)
	num_area = len(data_zyx_keylist)
	data_dat = pd.read_csv(dat)
	#remove unnamed
	#data_dat = data_dat.loc[:, ~data_dat.columns.str.contains('^Unnamed')]

	if len(data_dat)!= num_area:
		print(len(data_dat), num_area)
		print('Error on matching json and csv')
		exit(0)

	data_dat = data_dat[ftlist]

	dir_namelist = list(range(1,1+num_area))#note that elements are integer
	random.shuffle(dir_namelist)

	data_dat.insert(loc=0, column='Directory', value=dir_namelist)
	sorted_data_dat = data_dat.sort_values(by=['Directory'])
	sorted_data_dat.to_csv('./'+odir+'/shuffled.csv', index=False)

	for ind in list(range(len(dir_namelist))):
		curr_dir = str(dir_namelist[ind])
		#print(curr_dir)
		#print(data_dat.iloc[ind])
		curr_areaID = data_dat.iloc[ind]['areaID']
		if not os.path.isdir('./'+odir+'/'+str(curr_dir)):
			os.mkdir('./'+odir+'/'+str(curr_dir))
		#print(list(data_zyx.keys()))
		#for hungary, 1 to 001
		print(str(curr_areaID).zfill(3))
		curr_zyxs = data_zyx[str(curr_areaID).zfill(3)]
		for curr_zyx in curr_zyxs:
			url = base_url+str(curr_zyx[0])+'/'+str(curr_zyx[1])+'/'+str(curr_zyx[2])+'?token='+token
			filename = './'+odir+'/'+curr_dir+'/'+str(curr_zyx[1])+'_'+str(curr_zyx[2])+'.png'
			try:
				urllib.request.urlretrieve(url, filename)
			except:
				sleep(1)
				urllib.request.urlretrieve(url, filename)
	'''