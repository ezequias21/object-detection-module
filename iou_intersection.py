import numpy as np
import pandas as pd
import cv2
import os

def bb_intersection_over_union(ground_truth, predicted):
	if not predicted.any() or not ground_truth.any():
		return float(0)
	
	xA = max(ground_truth[0], predicted[0])
	yA = max(ground_truth[1], predicted[1])
	xB = min(ground_truth[2], predicted[2])
	yB = min(ground_truth[3], predicted[3])
	  
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	#area ground-truth rectangle
	ground_truth_area = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1)
	  
	#area predicted rectangle
	predicted_area = (predicted[2] - predicted[0] + 1) * (predicted[3] - predicted[1] + 1)

	iou = interArea / float(ground_truth_area + predicted_area - interArea)
	return iou

path_dir_base = '/home/estudante/Documentos/Estudo Jetson Nano/jetson-inference/build/aarch64/bin'
dir_ground_truth = 'ground_truth'
dir_bounding_box = 'bounding_box'
thresholds = ["0.5"]

# threshold = 0.5

def desnormalize(coords, width, height):
	coords[0] = int(coords[0]*width)
	coords[1] = int(coords[1]*height)
	coords[2] = int(coords[2]*width)
	coords[3] = int(coords[3]*height)
	return coords

def getBoxPoints(cx, cy, w, h):
	x1 = int(cx-w/2)
	y1 = int(cy-h/2)
	x2 = int(cx+w/2)
	y2 = int(cy+h/2)
	return np.array((x1, y1, x2, y2))

def adjustsBoxPoints(coords):
	coords = desnormalize(coords, 960, 720)
	return getBoxPoints(coords[0], coords[1], coords[2], coords[3])

def getIOUMetrics(threshold):
	metrics = pd.DataFrame(columns=['TP', 'FP', 'FN',  'IOU', 'threshold'])

	gt_files = os.listdir(os.path.join(path_dir_base, dir_ground_truth))
	bb_files = os.listdir(os.path.join(path_dir_base, dir_bounding_box, threshold))

	data = list(zip(gt_files, bb_files))

	for elem in data:
		metrics = calcIOU(elem[0], elem[1], metrics, threshold)

	return metrics

def calcIOU(gt_file, bb_file, metrics, threshold):
	ground_truth = pd.read_csv(os.path.join(path_dir_base, dir_ground_truth, gt_file), sep=' ', header=None)
	predicted = pd.read_csv(os.path.join(path_dir_base, dir_bounding_box, threshold, bb_file), sep=' ', header=None)

	for (_, gtrow) in ground_truth.iterrows():
		ioumax = 0
		bbclass = -1
		print('--------------------------------')
		for (_, prow) in predicted.iterrows():
			iou = bb_intersection_over_union(adjustsBoxPoints(np.array(gtrow)[1:]), adjustsBoxPoints(np.array(prow)[1:]))
			print(gtrow, iou, ioumax)
			if ioumax < iou: 
				ioumax = iou
				bbclass = np.array(prow)[0]

		
		TP = True if ioumax >= float(threshold) else False
		FP = True if ioumax < float(threshold) and ioumax > 0 else False
		FN = True if ioumax == 0 else False

		metrics = metrics.append({
			'TP': TP, 
			'FP': FP, 
			'FN': FN,  
			'IOU': ioumax, 
			'threshold': threshold
		}, ignore_index=True)
	print(metrics)
	return metrics

def addToMatrix(matrix, metrics, threshold): 
	matrix = matrix.append({
		'TP': metrics['TP'].sum(), 
		'FP': metrics['FP'].sum(),
		'FN': metrics['FN'].sum(),
		'threshold': threshold
	}, ignore_index=True)
	
	matrix =  matrix.apply(pd.to_numeric)
	return matrix
	 

def calMatrix():
	matrix = pd.DataFrame(columns=['TP', 'FP', 'FN', 'threshold'])

	for  threshold in thresholds:  
		metrics = getIOUMetrics(threshold)
		matrix = addToMatrix(matrix, metrics, threshold)

	return matrix



matrix = calMatrix()
print(matrix)
