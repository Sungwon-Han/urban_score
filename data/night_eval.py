import glob
import os
from collections import OrderedDict
from PIL import Image
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tdir', help='target nightlight imagery directory')
parser.add_argument('--odir', help='output directory')
parser.add_argument('--ocsv', help='ouput csv file name')
args = parser.parse_args()

main_dir = args.tdir
out_dir =  args.odir
out_csv = args.ocsv

if os.path.isdir(out_dir):
	os.system('rm -r '+out_dir)
os.mkdir(out_dir)

all_districts = glob.glob(main_dir+'/*/*.png')
loc_list = []
sum_list = []
mean_list = []

for image_path in all_districts:
	print('Proceeding ',image_path,'...')
	im = Image.open(image_path,'r')
	pix_val = list(im.getdata())
	gray_pix_val = list(map(lambda x: 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2],pix_val))
	sum_val = sum(gray_pix_val)
	loc_list.append(image_path.split('/')[-1].split('.')[0])
	sum_list.append(sum_val)
	if len(gray_pix_val)==0:
		mean_list.append(0)
	else:
		mean_list.append(sum_val/len(gray_pix_val))

df_dict = OrderedDict()
df_dict['location'] = loc_list
df_dict['light_sum'] = sum_list
df_dict['light_mean'] = mean_list

df = pd.DataFrame.from_dict(df_dict)
print(df)
df.to_csv(out_dir+'/'+out_csv,index=False)