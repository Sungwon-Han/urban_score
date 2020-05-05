import json
import os
import glob
import argparse
from PIL import Image
import convert
import requests
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--ddir', help='daytime sat image directory')
parser.add_argument('--ondir', help='original nightlight directory name')
parser.add_argument('--cndir', help='cropped nightlight directory name')
parser.add_argument('--zl', help = 'nightlight zoom level')
parser.add_argument('--zdiff', help='zoom level difference')
parser.add_argument('--nation', help='nation code')

args = parser.parse_args()

if args.ddir==None or args.ondir==None:
	print("Error! please fill all night/daytime satellite image directory, output directory , zl difference information")
	exit(0)

ddir = args.ddir
ondir = args.ondir
cndir = args.cndir
zl = int(args.zl)
zdiff= int(args.zdiff)
nation = args.nation

if not os.path.isdir('./'+ondir):
	os.mkdir('./'+ondir)

#first, find target area that covers the given nation

url = 'https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/Countries/'+nation+'?f=pjson'
res = requests.get(url)
target_box = res.json()['countries'][0]['defaultExtent']
xmin, ymin = convert.deg2num(target_box['ymin'],target_box['xmin'],zl)
xmax, ymax = convert.deg2num(target_box['ymax'],target_box['xmax'],zl)
#ymax -> ymin
#xmin -> xmax

for y in range(ymax,ymin+1):
	for x in range(xmin,xmax+1):
		print('https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services/Earth_at_Night_2016/MapServer/tile/{}/{}/{}'.format(str(zl),str(y),str(x)))
		urllib.request.urlretrieve('https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services/Earth_at_Night_2016/MapServer/tile/{}/{}/{}'.format(str(zl),str(y),str(x)),'./'+ondir+'/{}_{}.png'.format(str(y),str(x)))


#second, match the daytime sat imagery to  
ndir_list = glob.glob(ondir+'/*.png')
ddir_list = glob.glob(ddir+'/*/*.png')

if not os.path.isdir(cndir):
	os.mkdir(cndir)

for ddir_image in ddir_list:
	sp = ddir_image.split('/')
	curr_img = sp[-1]
	curr_ddir = sp[-2]
	curr_out_path = cndir+'/'+curr_ddir
	if not os.path.isdir(curr_out_path):
		os.mkdir(curr_out_path)
	curr_y = int(curr_img.split('_')[0])
	curr_x = int(curr_img.split('_')[1].split('.')[0])
	night_y = curr_y//2**zdiff
	night_x = curr_x//2**zdiff
	pixel_y = curr_y%2**zdiff
	pixel_x = curr_x%2**zdiff

	try:
		curr_nightimg = Image.open('./'+ondir+'/'+str(night_y)+'_'+str(night_x)+'.png')
	except:
		urllib.request.urlretrieve('https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services/Earth_at_Night_2016/MapServer/tile/{}/{}/{}'.format(str(zl),str(night_y),str(night_x)),'./'+ondir+'/{}_{}.png'.format(str(night_y),str(night_x)))
		curr_nightimg = Image.open('./'+ondir+'/'+str(night_y)+'_'+str(night_x)+'.png')
	unit = curr_nightimg.size[0]//2**zdiff
	curr_area = (pixel_x*unit,pixel_y*unit,pixel_x*unit+unit,pixel_y*unit+unit)
	curr_cropped_nightimg = curr_nightimg.crop(curr_area)
	curr_cropped_nightimg.save(curr_out_path+'/'+curr_img)
	print(curr_ddir, curr_img, "converted to "+curr_out_path+'/'+curr_img)