#-*- coding:utf-8 -*-
import json
import convert as conv
import argparse
import urllib.request
import os
import requests
import argparse
import collections
import pandas as pd
from time import sleep

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#input : 
#zoom level, output zyx_district.json, data_district.json
parser = argparse.ArgumentParser()
parser.add_argument('--zl', help='zoom level')
parser.add_argument('--ozyx', help='output zyx_district json name')
parser.add_argument('--odat', help='output data_district csv name')
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--nation', help='enter target nation code. Default = KR')
parser.add_argument('--minor', help='minor county, Default = KR.Districts')
parser.add_argument('--dset', help='enter target dataset Default = KOR_MBR_2018')
parser.add_argument('--dcoll', help='Inner dataCollcetion name, Default = Gender')
parser.add_argument('--start', help='for separable credit use, input start if you want')
parser.add_argument('--end', help='for separable credit use, input end if you want')

args = parser.parse_args()

if args.zl== None or args.ozyx==None or args.odat==None or args.token==None:
	print("Error! please fill all zoom level, zyx_district json name, data_distric json name, temporal token information")
	exit(0)

zl = int(args.zl)
ozyx = args.ozyx
odat = args.odat
token = args.token
nation = 'KR'
minor = 'KR.Districts'
dset = 'KOR_MBR_2018'
dcoll = 'Gender'
geographyids = str(['01'])


if args.nation is not None:
	nation = args.nation
if args.minor is not None:
	minor = args.minor
if args.dset is not None:
	dset = args.dset
if args.dcoll is not None:
	dcoll = args.dcoll
dcoll = str([dcoll])

start = -1
end = -1
start_bool = False
end_bool =False

if args.start is not None:
	start = int(args.start)
	start_bool = True
	if start<0:
		print("You put wrong start; put non-negative number")
		exit(0)
if args.end is not None:
	end = int(args.end)
	end_bool = True
	if end<0:
		print("You put wrong end; put non-negative number that is larger than start")
		exit(0)

def extractZYX(zoomlevel,polygon):
	xlist = []
	ylist = []
	result = []
	for lnglat in polygon:
		#print(lnglat)
		lat = lnglat[1]
		lng = lnglat[0]
		xtile, ytile = conv.deg2num(lat,lng,zoomlevel)
		#print(xtile,ytile)
		xlist.append(xtile)
		ylist.append(ytile)
	
	xmin, xmax, ymin, ymax = min(xlist), max(xlist), min(ylist), max(ylist)
	new_xlist = list(range(xmin,xmax))
	new_ylist = list(range(ymin,ymax))

	print(xmin, xmax, ymin, ymax)
	tf_table = [tf[:] for tf in [[None] * (xmax-xmin+1)] * (ymax-ymin+1)]
	
	for y in list(range(ymin,ymax+1)):
		for x in list(range(xmin,xmax+1)):
			lat,lng = conv.num2deg(x,y,zoomlevel)
			tf_table[y-ymin][x-xmin] = 1 if Polygon(polygon).contains(Point(lng,lat)) else 0

	#return z,y,x only when more than 3 corners of image are inside the district boundary
	for y in new_ylist:
		for x in new_xlist:
			if tf_table[y-ymin][x-xmin] + tf_table[y-ymin][x-xmin+1] + tf_table[y-ymin+1][x-xmin] + tf_table[y-ymin+1][x-xmin+1] >=3:
				result.append([zoomlevel, y, x])
	#print(tf_table)
	return result



#print(kr_translation.eng2kor_district)
#Use 9-alpha on slack

url = 'https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/StandardGeographyQuery/'
params = { 'f':'pjson',
'sourceCountry':nation,
'token':token,
#'geographyLayers':major,
'geographyids':geographyids,
'returnSubGeographyLayer':'true',
'subGeographyLayer':minor,
#'DatasetID':dset,
'returnGeometry':'true',
'optionalCountryHierarchy':'census'
}
print(params)

res=requests.get(url, params=params)
res_json = res.json()
#print(res.text)
areaID_features = res_json['results'][0]['value']['features']

ozyx_dict = {}
area_dict = {}

#print(areaID_features[0])
for af in areaID_features:
	#if the area comes from different data collection, continue
	if af['attributes']['DatasetID']!=dset:
		continue
	#geometry
	zyx_list = extractZYX(zl,af['geometry']['rings'][0])
	print(af['attributes']['AreaName'], len(zyx_list))
	ozyx_dict[af['attributes']['AreaID']]=zyx_list
	area_dict[af['attributes']['AreaID']]={'minor':af['attributes']['AreaName'], 'major':af['attributes']['MajorSubdivisionName']}

ordered_ozyx_dict = collections.OrderedDict(sorted(ozyx_dict.items()))
ordered_areaID_list = list(ordered_ozyx_dict.keys())

print('number of minors')
print(len(ordered_areaID_list))

with open(ozyx+'.json','w') as json_file:
	json.dump(ordered_ozyx_dict,json_file)

if start_bool==False:
	start = 0
if end_bool==False:
	end = len(ordered_areaID_list)

selected_areaID_list = ordered_areaID_list[start:end]
print('selected_list')
print(selected_areaID_list)
print(len(selected_areaID_list))

sleep(5)

url = 'https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/Enrich?'
url += 'f=pjson&'
url += 'studyAreas=[{'
url += '"sourceCountry":"{}","layer":"{}","ids":{}'.format(nation,minor,str(selected_areaID_list))
url += '}]&'
url += 'dataCollections={}&'.format(dcoll)
url += 'inSR=4326&outSR=4326&'
url += 'token={}'.format(token)

#print(url)
res = requests.get(url)
#print(res.text)
res_json = res.json()

fields = res_json['results'][0]['value']['FeatureSet'][0]['fields']

code_list = []
for attribute_dict in fields:
		try:
			#only attributes have fullname
			fullname = attribute_dict["fullName"]
			code_list.append(attribute_dict["name"])
			desc_list.append(attribute_dict["alias"])
		except:
			continue

print(code_list)

pandas_baseline = {'areaID':[]}
for code in code_list:
	pandas_baseline[code]=[]

attributes_list = res_json['results'][0]['value']['FeatureSet'][0]['features']

for attr in attributes_list:
	attributes = attr['attributes']
	pandas_baseline['areaID'].append(str(attributes['StdGeographyID']))
	for code in code_list:
		pandas_baseline[code].append(attributes[code])

#print(pandas_baseline)

baseline_minor = []
baseline_major = []
for areaID in pandas_baseline['areaID']:
	baseline_minor.append(area_dict[areaID]['minor'])
	baseline_major.append(area_dict[areaID]['major'])
pandas_baseline['minor']=baseline_minor
pandas_baseline['major']=baseline_major

print(pandas_baseline)

#pandas work test
#pandas_baseline = {'areaID': ['11110', '11140', '11170', '11200', '11215', '11230', '11260', '11290', '11305', '11320'], 'PAGE01_CY': [14305, 11105, 22755, 33568, 36853, 34871, 40764, 51321, 30997, 36135], 'PAGE02_CY': [31628, 22852, 42268, 59850, 78542, 70187, 76726, 86865, 59514, 64560], 'PAGE03_CY': [32409, 29556, 55599, 76500, 87795, 78011, 91962, 98650, 68567, 70959], 'PAGE04_CY': [40019, 31677, 56517, 75562, 87776, 85361, 107604, 109820, 83551, 90596], 'PAGE05_CY': [37293, 30873, 52092, 61549, 67497, 81506, 91040, 96294, 81597, 81302], 'MAGE01_CY': [7309, 5568, 11667, 17061, 18900, 17932, 20849, 26406, 15870, 18593], 'MAGE02_CY': [15925, 11364, 20902, 29848, 38257, 35930, 38988, 43316, 29921, 33130], 'MAGE03_CY': [15857, 15055, 27590, 38378, 43738, 40436, 47831, 48990, 35692, 36018], 'MAGE04_CY': [20087, 16194, 27947, 38005, 42786, 43193, 53523, 54287, 41013, 43401], 'MAGE05_CY': [17001, 14127, 22813, 27823, 30868, 37216, 41868, 42689, 36134, 37144], 'FAGE01_CY': [6996, 5537, 11088, 16507, 17953, 16939, 19915, 24915, 15127, 17542], 'FAGE02_CY': [15703, 11488, 21366, 30002, 40285, 34257, 37738, 43549, 29593, 31430], 'FAGE03_CY': [16552, 14501, 28009, 38122, 44057, 37575, 44131, 49660, 32875, 34941], 'FAGE04_CY': [19932, 15483, 28570, 37557, 44990, 42168, 54081, 55533, 42538, 47195], 'FAGE05_CY': [20292, 16746, 29279, 33726, 36629, 44290, 49172, 53605, 45463, 44158]}

df = pd.DataFrame.from_dict(pandas_baseline)
s_name = ""
e_name = ""

if start_bool == True:
	s_name = "_start_"+str(start)
if end_bool == True:
	e_name = "_end_"+str(end)
df.to_csv(odat+s_name+e_name+'.csv', index=False)

