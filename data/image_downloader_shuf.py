import json
import csv
import os
import urllib.request
import random
from time import sleep
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--zyx', help='input json zyx location file path')
parser.add_argument('--dat', help='input csv data file path')
parser.add_argument('--odir', help='output directory name')

args = parser.parse_args()

if args.zyx==None or args.dat==None or args.token==None or args.odir==None:
	print("Error! please fill all zyx_district json name, data_distric json name, temporal token information, output directory.")
	exit(0)

ftlist = None
token = args.token
zyx = args.zyx
dat = args.dat
odir = args.odir
zfill_val = 5

base_url = "https://tiledbasemaps.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/"
#token = "lBRbigHSYEOvanHw_wB698Yr1izHkaahgcPFm1GOLE-Lj82yWtZEQ2Ks5OZ41U0RGcPT4j4voqrLU8XWaiNjrpdZ8TpSyzjldcGN9VV9iLLqxYO1hcaqx87APVshRKlgVct9CWtwG_PdDlArkK161A__mc3CkHE__46JSSxLlD6KSQnAqWal6mwdBX6kHoyHHhPyCjMr61x3GiuMI8CaD_HKuZtUJ_zL6Q_Vpq3DR-k."

csv_write_candidate = []
if not os.path.isdir('./'+odir):
	os.mkdir('./'+odir)

with open(zyx,'r') as json_file_zyx:
	data_zyx=json.load(json_file_zyx)
	data_zyx_keylist = list(data_zyx.keys()) 
	print(data_zyx_keylist)
	num_area = len(data_zyx_keylist)
	data_dat = pd.read_csv(dat)
	zfill_len = list(map(lambda x: len(x), data_zyx_keylist))
	if min(zfill_len)==max(zfill_len):
		zfill_val = min(zfill_len)
	#remove unnamed
	#data_dat = data_dat.loc[:, ~data_dat.columns.str.contains('^Unnamed')]

	if len(data_dat)!= num_area:
		print(len(data_dat), num_area)
		print('Error on matching json and csv')
		exit(0)

	dir_namelist = list(range(1,1+num_area))#note that elements are integer
	random.shuffle(dir_namelist)
    
	print(dir_namelist)

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
		#for vietnam, 1001 to 01001
		print(str(curr_areaID).zfill(zfill_val))
		curr_zyxs = data_zyx[str(curr_areaID).zfill(zfill_val)]
		for curr_zyx in curr_zyxs:
			url = base_url+str(curr_zyx[0])+'/'+str(curr_zyx[1])+'/'+str(curr_zyx[2])+'?token='+token
			filename = './'+odir+'/'+curr_dir+'/'+str(curr_zyx[1])+'_'+str(curr_zyx[2])+'.png'
			try:
				urllib.request.urlretrieve(url, filename)
			except:
				sleep(1)
				urllib.request.urlretrieve(url, filename)

'''
with open('zyx_district_shuf.json','r') as json_file_zyx:
	with open('korea_demographic_data_shuf.json','r') as json_file_data:
		with open('temp_writer_shuf.csv','w') as csvfile:
			data_zyx = json.load(json_file_zyx)
			data_kd = json.load(json_file_data)
			csv_writer=csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

			csv_writer.writerow(['ID','Province','District','PAGE01_CY','PAGE02_CY','PAGE03_CY','PAGE04_CY','PAGE05_CY','n'])
			cnt = 1
			name_1_shuffle = list(data_zyx.keys())
			random.shuffle(name_1_shuffle)
			for name_1 in name_1_shuffle:
				name_2_shuffle = list(data_zyx[name_1].keys())
				random.shuffle(name_2_shuffle)
				for name_2 in name_2_shuffle:
					print(name_1,name_2)
					if data_zyx[name_1][name_2]=='':
						continue
					if len(data_zyx[name_1][name_2]["contains"])<5:
						continue
					temp_dir_name = str(cnt)
					if not os.path.isdir('./image_shuf/'+temp_dir_name):
						os.mkdir('./image_shuf/'+temp_dir_name)

					print([cnt, name_1, name_2,data_kd[name_1][name_2]['demographic']['PAGE01_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE02_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE03_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE04_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE05_CY']['value'].replace(",",""),len(data_zyx[name_1][name_2]["contains"])])

					for zyx in data_zyx[name_1][name_2]["contains"]:
						url = base_url+str(zyx[0])+'/'+str(zyx[1])+'/'+str(zyx[2])+'?token='+token
						filename = './image_shuf/'+temp_dir_name+'/'+str(zyx[1])+'_'+str(zyx[2])+'.png'
						try:
							urllib.request.urlretrieve(url, filename)
						except:
							sleep(1)
							urllib.request.urlretrieve(url, filename)

					csv_writer.writerow([cnt, name_1, name_2,data_kd[name_1][name_2]['demographic']['PAGE01_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE02_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE03_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE04_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE05_CY']['value'].replace(",",""),len(data_zyx[name_1][name_2]["contains"])])
					csv_write_candidate.append([cnt, name_1, name_2,data_kd[name_1][name_2]['demographic']['PAGE01_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE02_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE03_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE04_CY']['value'].replace(",",""),data_kd[name_1][name_2]['demographic']['PAGE05_CY']['value'].replace(",",""),len(data_zyx[name_1][name_2]["contains"])])
					cnt +=1

			random.shuffle(csv_write_candidate)

			for line in csv_write_candidate:
				csv_writer.writerow(line)
'''

