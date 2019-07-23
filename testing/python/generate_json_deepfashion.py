"""
{ Code for generating keypoints json with https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation.git .
Should be put into testing/python. }
"""
import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import pylab as plt
import pdb
import json

def generate_single(test_image):
	oriImg = cv.imread(test_image) # B,G,R order

	param, model = config_reader()
	multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
	caffe.set_mode_cpu()
	net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

	for m in range(len(multiplier)):
	    scale = multiplier[m]
	    imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
	    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
	    print (imageToTest_padded.shape)

	    net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
	    #net.forward() # dry run
	    net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
	    start_time = time.time()
	    output_blobs = net.forward()
	    print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))
	    
	#     pdb.set_trace()
	    # extract outputs, resize, and remove padding
	    heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
	    heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
	    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
	    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
	    
	    paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
	    paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
	    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
	    paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
	    
	    heatmap_avg = heatmap_avg + heatmap / len(multiplier)
	    paf_avg = paf_avg + paf / len(multiplier)

	import scipy
	print heatmap_avg.shape

	#plt.imshow(heatmap_avg[:,:,2])
	from scipy.ndimage.filters import gaussian_filter
	all_peaks = []
	peak_counter = 0

	for part in range(19-1):
	    x_list = []
	    y_list = []
	    map_ori = heatmap_avg[:,:,part]
	    map = gaussian_filter(map_ori, sigma=3)
	    
	    map_left = np.zeros(map.shape)
	    map_left[1:,:] = map[:-1,:]
	    map_right = np.zeros(map.shape)
	    map_right[:-1,:] = map[1:,:]
	    map_up = np.zeros(map.shape)
	    map_up[:,1:] = map[:,:-1]
	    map_down = np.zeros(map.shape)
	    map_down[:,:-1] = map[:,1:]
	    
	    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
	    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
	    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
	    id = range(peak_counter, peak_counter + len(peaks))
	    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	    all_peaks.append(peaks_with_score_and_id)
	    peak_counter += len(peaks)

	peaks_list = []
	peaks_dict = {"pose_keypoints":[], "face_keypoints":[],"hand_left_keypoints":[],"hand_right_keypoints":[]}
	for i in range(18):
		if len(all_peaks[i]) > 0:
			# print(i)
			add = list(all_peaks[i][0])[:3]
		else:
			add = [-1, -1, 0]
		peaks_list = peaks_list + add

	# pdb.set_trace()
	json_dict = {"version":1.1,"people": [peaks_dict]}

	json_name = test_image.rstrip('.jpg') + '_keypoints.json'
	with open(json_name, 'w') as f:
		json_string = json.dumps(json_dict, f)

if __name__ == '__main__':
	test_image = '../sample_image/09_1_front.jpg'
	generate_single(test_image)