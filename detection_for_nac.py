import sys, os, glob, shutil
import pandas as pd
import numpy as np
import time
from multiprocessing import Process, Pool

import download as dl
import register as rg
import const

R = 608
OVER = 51


from ctypes import *
import math
import random
import os, glob, sys
import cv2
import numpy as np
import time
import darknet
from PIL import Image
import PIL.ExifTags as ExifTags

SIZE = 512
OVER = 50

def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
	for detection in detections:
		x, y, w, h = detection[2][0],\
			detection[2][1],\
			detection[2][2],\
			detection[2][3]
		xmin, ymin, xmax, ymax = convertBack(
			float(x), float(y), float(w), float(h))
		pt1 = (xmin, ymin)
		pt2 = (xmax, ymax)
		cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
		cv2.putText(img,
					detection[0].decode() +
					" [" + str(round(detection[1] * 100, 2)) + "]",
					(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					[0, 255, 0], 2)
	return img


def get_latlon(path):
	# 画像ファイルを開く --- (*1)
	im = Image.open(path)
	# EXIF情報を辞書型で得る
	exif = {
		ExifTags.TAGS[k]: v
		for k, v in im._getexif().items()
		if k in ExifTags.TAGS
	}
	# GPS情報を得る --- (*2)
	gps_tags = exif["GPSInfo"]
	gps = {
		ExifTags.GPSTAGS.get(t, t): gps_tags[t]
		for t in gps_tags
	}
	# 緯度経度情報を得る --- (*3)
	def conv_deg(v):
		# 分数を度に変換
		d = float(v[0][0]) / float(v[0][1])
		m = float(v[1][0]) / float(v[1][1])
		s = float(v[2][0]) / float(v[2][1])
		return d + (m / 60.0) + (s / 3600.0)
	lat = conv_deg(gps["GPSLatitude"])
	lat_ref = gps["GPSLatitudeRef"]
	if lat_ref != "N": lat = 0 - lat
	lon = conv_deg(gps["GPSLongitude"])
	lon_ref = gps["GPSLongitudeRef"]
	if lon_ref != "E": lon = 0 - lon
	return lat, lon


netMain = None
metaMain = None
altNames = None


def YOLO(configPath, weightPath, metaPath, inputPath):

	global metaMain, netMain, altNames
	if not os.path.exists(configPath):
		raise ValueError("Invalid config path `" +
						 os.path.abspath(configPath)+"`")
	if not os.path.exists(weightPath):
		raise ValueError("Invalid weight path `" +
						 os.path.abspath(weightPath)+"`")
	if not os.path.exists(metaPath):
		raise ValueError("Invalid data file path `" +
						 os.path.abspath(metaPath)+"`")
	if netMain is None:
		netMain = darknet.load_net_custom(configPath.encode(
			"ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	if metaMain is None:
		metaMain = darknet.load_meta(metaPath.encode("ascii"))
	if altNames is None:
		try:
			with open(metaPath) as metaFH:
				metaContents = metaFH.read()
				import re
				match = re.search("names *= *(.*)$", metaContents,
								  re.IGNORECASE | re.MULTILINE)
				if match:
					result = match.group(1)
				else:
					result = None
				try:
					if os.path.exists(result):
						with open(result) as namesFH:
							namesList = namesFH.read().strip().split("\n")
							altNames = [x.strip() for x in namesList]
				except TypeError:
					pass
		except Exception:
			pass
	darknet_image = darknet.make_image(darknet.network_width(netMain),
									darknet.network_height(netMain),3)
	prev_time = time.time()
	
	nac = rg.read_nac(inputPath)
	height, width = nac.shape
	for y in range(0, height, R-OVER):
		for x in range(0, width, R-OVER):
			clp = rg.convert_to_uint8(nac[y:y+R, x:x+R])
			image_resized = cv2.resize(clp,
				(darknet.network_width(netMain),
					darknet.network_height(netMain)),
				interpolation=cv2.INTER_LINEAR)

			darknet.copy_image_from_bytes(darknet_image,image_resized.tobytes())

			detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.78)
			print(detections)
			if len(detections) > 0:
				image = cvDrawBoxes(detections, image_resized)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# cv2.imshow('Demo', image)
				name, ext = os.path.splitext(os.path.basename(inputPath))
				cv2.imwrite('result/{}_{}_{}result.jpg'.format(name, x, y), image)
				cv2.waitKey()
	print(time.time()-prev_time)


if __name__ == "__main__":
	configPath = "./task_crater/yolov4-custom.cfg"
	weightPath = "./task_crater/backup_v4/yolov4-custom_last.weights"
	metaPath = "./task_crater/crater.data"
	args = sys.argv
	inputPath = os.path.join(const.NAC_IMAGE_PATH, args[1])
	YOLO(configPath, weightPath, metaPath, inputPath)
