#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:36:35 2023

@author: anna
"""
import tensorrt as trt
import sys
from PIL import Image
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt

from aws import *
from infer import TensorRTInfer
from image_batcher import ImageBatcher
import configparser


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_runtime = trt.Runtime(TRT_LOGGER)


import cv2 as cv

def readImagesFromVideo(max_frames, videoName, skip_frames):
    film = cv.VideoCapture(videoName)
    isTrue, frame = film.read()
    frame_count = 0
    
    frames = []
    while isTrue: 
        if frame_count < max_frames:
            isTrue, frame = film.read()
            if frame is None:
                break
            if (frame_count+1 % skip_frames != 0):
                frames.append(frame)
            
            frame_count += 1
        else: break    

    film.release()
    cv.destroyAllWindows()   
    return frames


def main():
    detected_bees = []

    film_base = '2-srednio' 
    film_path = film_base + '.avi' 
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    max_frames = config.get('DEFAULT', 'max_frames')
    xmin_oz = config.get('DEFAULT', 'xmin')
    xmax_oz = config.get('DEFAULT', 'xmax')
    ymin_oz = config.get('DEFAULT', 'ymin')
    ymax_oz = config.get('DEFAULT', 'ymax')
    skip_frames = config.get('DEFAULT', 'skip_frames')
    
    
    #max_frames = 600
    
    detection_engine_file = 'bee_detection_engine'

    classification_engine_file = 'varroa_detection_engine'

    detection_threshold = 0.5
    classification_threshold = 0.85
    
    preprocessor = 'fixed_shape_resizer'
        
    
    trt_infer_detection = TensorRTInfer(detection_engine_file, preprocessor, 'bbox', detection_threshold)
    trt_infer_classification = TensorRTInfer(classification_engine_file, preprocessor, 'bbox', classification_threshold)

    detection_frames = readImagesFromVideo(max_frames,film_path, None, skip_frames)   

    batcher_detection = ImageBatcher(None, *trt_infer_detection.input_spec(), preprocessor=preprocessor, image_array=detection_frames)
    idd=0
    
    for batch, images, scales in batcher_detection.get_batch():
        detections = trt_infer_detection.infer(batch, scales, detection_threshold) 
        idx = 0
        
        while (len(images)) > 0:
            for d in detections[0]:
                ymin,xmin,ymax,xmax = (d['ymin'], d['xmin'], d['ymax'], d['xmax'] )                
                
                if xmin < xmax and ymin < ymax and xmax < xmax_oz and ymax<ymax_oz and ymin > ymin_oz and xmin > xmin_oz:
                    cropped_image = images[0][ymin:ymax, xmin:xmax]
                    detected_bees.append(cropped_image)

            images.pop(0)
            idx = idx+1

        idd = idd+1                
    del detection_frames
    del images    

    batcher_classification = ImageBatcher(None, *trt_infer_classification.input_spec(), preprocessor=preprocessor, image_array = np.array(detected_bees))
    img_id = 0

    for c_batch, c_images, c_scales in batcher_classification.get_batch():
        detections_c = trt_infer_classification.infer(c_batch, c_scales, classification_threshold) 
        for cindex in range(len(c_images)):

            if len(detections_c[cindex]) > 0:
                connectAndSendToAWS(img_id, c_images[0])

        img_id = img_id +1         
   
if __name__ == "__main__":
    main()
