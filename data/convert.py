import requests
import math
import collections

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2quad(xtile, ytile, zoom):
	#note that return type is string!
	# To Binary
	x_bin = format(xtile,'b').zfill(zoom)
	y_bin = format(ytile,'b').zfill(zoom)
	# Insert digits alternately
	return "".join(list(map(lambda x: str(int(x[0])*2+int(x[1])),zip(y_bin,x_bin))))

def quad2num(quad):
	zoom = len(quad)
	t = list(map(lambda x: format(int(x),'b').zfill(2),list(quad)))
	t_y = int("".join([i[0] for i in t]),2)
	t_x = int("".join([i[1] for i in t]),2)
	return (t_x, t_y, zoom)


def bing_url_format(bingkey):

	url = 'https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Aerial?key={bingmapskey}'
	#key = 'AtNYVVZOIJlPZiEQN5YjAV1a7q-vbDv6FA9Z9w-Fe44Lo5wUc0IPTHqO6IC59rQg'
	res = requests.get(url.format(bingmapskey=bingkey))
	json_res = res.json()
	subdomains = ['t0','t1','t2','t3']
	default_subdomain = subdomains[0]
	current_bingmap_format = json_res['resourceSets'][0]['resources'][0]['imageUrl'].format(subdomain=default_subdomain, quadkey='{quadkey}')
	# return the request where only quadkey left
	return current_bingmap_format

def bing_label_format(lat,lng,z,bingkey):
	url = 'https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial/{latlng}?zl={zoomlevel}&key={bingmapskey}'
	res = requests.get(url.format(latlng = lat+','+lng, zoomlevel = z, bingmapskey=bingkey))
	json_res = res.json()
	vintage_start = json_res['resourceSets'][0]['resources'][0]['vintageStart']
	vintage_end = json_res['resourceSets'][0]['resources'][0]['vintageEnd']
	#make return value. It would be dictionary
	result = collections.OrderedDict()
	result['lat'] = lat
	result['lng'] = lng
	result['start'] = vintage_start
	result['end'] = vintage_end
	return result