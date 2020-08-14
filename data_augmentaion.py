import glob, os, sys
import numpy as np
import random
import cv2

EXT = '.JPG'


def read_data(file_path):
	if not os.path.exists(file_path + '.txt'):
		return -1
	img = cv2.imread(file_path + EXT)
	ret = []
	with open(file_path + '.txt') as f:
		lines = [s.strip() for s in f.readlines()]
	for line in lines:
		line = line.split(' ')
		tmp_ret = {}
		tmp_ret['object_class'] = int(line[0])
		tmp_ret['x-center'] = float(line[1])
		tmp_ret['y-center'] = float(line[2])
		tmp_ret['width'] = float(line[3])
		tmp_ret['height'] = float(line[4])
		ret.append(tmp_ret)
	return img, ret


def ano_data_to_txt(data, path, kind):
	ret = ''
	for d in data:
		ret += '{} {} {} {} {}\n'.format(d['object_class'], d['x-center'], d['y-center'], d['width'], d['height']) 
	with open(path+'-'+kind+'.txt', mode='w') as f:
		f.writelines(ret)


def flip(img, data, path):
	flip_img = cv2.flip(img, 1)
	for d in data:
		d['x-center'] = 1 - d['x-center']
	cv2.imwrite(path+'-flip'+EXT, flip_img)
	ano_data_to_txt(data, path, 'flip')
	vflip_img = cv2.flip(flip_img, 0)
	for d in data:
		d['y-center'] = 1 - d['y-center']
	cv2.imwrite(path+'-vflip'+EXT, vflip_img)
	ano_data_to_txt(data, path, 'vflip')
	fvflip_img = cv2.flip(vflip_img, 1)
	for d in data:
		d['x-center'] = 1 - d['x-center']
	cv2.imwrite(path+'-fvflip'+EXT, fvflip_img)
	ano_data_to_txt(data, path, 'fvflip')
	for d in data:
		d['y-center'] = 1 - d['y-center']


# def resize(img, data, path):
# 	hight = img.shape[0]
# 	width = img.shape[1]
# 	half_img = cv2.resize(img, (hight//2, width//2))
# 	cv2.write(path+'-half'+EXT, half_img)
# 	ano_data_to_txt(data, path, 'half')
# 	twice_img = cv2.resize(img, (hight*2, width/2))
# 	cv2.write(path+'-twice'+EXT, twice_img)
# 	ano_data_to_txt(data, path, 'twice')


def contrast(img, data, path):
	min_table = 50
	max_table = 205
	diff_table = max_table - min_table
	LUT_HC = np.arange(256, dtype = 'uint8' )
	LUT_LC = np.arange(256, dtype = 'uint8' )

	for i in range(0, min_table):
		LUT_HC[i] = 0
	for i in range(min_table, max_table):
		LUT_HC[i] = 255 * (i - min_table) / diff_table
	for i in range(max_table, 255):
		LUT_HC[i] = 255

	for i in range(256):
		LUT_LC[i] = min_table + i * (diff_table) / 255

	if random.choice([True, False]):
		high_cont_img = cv2.LUT(img, LUT_HC)
		cv2.imwrite(path+'-contH'+EXT, high_cont_img)
		ano_data_to_txt(data, path, 'contH')
	else:
		low_cont_img = cv2.LUT(img, LUT_LC)
		cv2.imwrite(path+'-contL'+EXT, low_cont_img)
		ano_data_to_txt(data, path, 'contL')


def blur(img, data, path):
	average_square = (10,10)
	blur_img = cv2.blur(img, average_square)
	cv2.imwrite(path+'-blur'+EXT, blur_img)
	ano_data_to_txt(data, path, 'blur')


def gamma(img, data, path):
	gamma1 = 0.75
	gamma2 = 1.5
	LUT_G1 = np.arange(256, dtype = 'uint8' )
	LUT_G2 = np.arange(256, dtype = 'uint8' )
	for i in range(256):
		LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
		LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

	if random.choice([True, False]):
		high_cont_img = cv2.LUT(img, LUT_G1)
		cv2.imwrite(path+'-gammaH'+EXT, high_cont_img)
		ano_data_to_txt(data, path, 'gammaH')
	else:
		low_cont_img = cv2.LUT(img, LUT_G2)
		cv2.imwrite(path+'-gammaL'+EXT, low_cont_img)
		ano_data_to_txt(data, path, 'gammaL')


def main(path='./'):
	for jpg_path in glob.glob(path+'/*'+EXT):
		path, _ = 	os.path.splitext(jpg_path)
		img, ano_data = read_data(path)

		flip(img, ano_data, path)
		# if random.choice([True, False]):
		# 	resize(img, ano_data, path)
		if random.choice([True, False]):
			contrast(img, ano_data, path)
		elif random.choice([True, False]):
			blur(img, ano_data, path)
		else:
			gamma(img, ano_data, path)



if __name__ == '__main__':
	main(sys.argv[1:][0])