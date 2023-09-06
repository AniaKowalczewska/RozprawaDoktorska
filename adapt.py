#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Created on Sun Jul 30 16:50:11 2023

@author: anna
"""
import math
import logging
from infer import TensorRTInfer
from image_batcher import ImageBatcher

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import configparser

from PIL import Image
import numpy as np
import cv2 as cv
import time

logging.basicConfig(filename = "logs/adapt-{}.log".format(time.asctime()), format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_runtime = trt.Runtime(TRT_LOGGER)


h = 80
width = 1280
height = 720
cols = int(width/h)
rows = int(height/h)
bee_count = 0
avg_bees_per_frame = 0
frame_path = ""
detection_threshold = 0.5 
thres = 0.8
film_path = '2-srednio.avi' 
liczbaSiatek = 1

def set_config(max_frames, threshold,xmin, xmax,ymin, ymax):
    config = configparser.ConfigParser()
    config.set('DEFAULT', 'max_frames', str(max_frames))
    config.set('DEFAULT', 'threshold', str(threshold))
    config.set('DEFAULT', 'xmin', str(xmin))
    config.set('DEFAULT', 'xmax', str(xmax))
    config.set('DEFAULT', 'ymin', str(ymin))
    config.set('DEFAULT', 'ymax', str(ymax))

    with open('config.ini', 'w') as configfile:
          config.write(configfile)


def readImagesFromVideo(max_frames, videoName, outputPath):
    film = cv.VideoCapture(videoName)
    isTrue, frame = film.read()
    frame_count = 0
    frames = []
    while isTrue: 
        if frame_count < max_frames:
            isTrue, frame = film.read()
            if frame is None:
                break
            frames.append(frame)
            frame_count += 1
        else: break    

    film.release()
    cv.destroyAllWindows()   
    return frames

def update_grid(x,y,grid):
    x = math.floor(x/h)
    y = math.floor(y/h)
    if x==cols:
        x =x-1
    if y==rows:
        y =y-1
    try:
        grid[x][y]= grid[x][y] + 1
    except IndexError:
        print("Index error! x: {}, y: {}".format(x,y))
        print(grid)
    return (grid)  
        
def update_found_bees(min_x, min_y, max_y, grid):
    found = 0 
    for x in range(min_x, max_y+1):
        for y in range(min_y,max_y+1):
            found += grid[x][y]
    return found            
    
def prepare_list(siatki):
    bees_array = []
    unsorted_array = []
    
    for index in range(len(siatki)):
        bees = 0
        unsorted = []
        for x in range(len(siatki[index])):
            for y in range(len(siatki[index][0])):
                unsorted.append(((x, y), siatki[index][x][y]))
                bees = bees + siatki[index][x][y]
        bees_array.append(bees)
        unsorted_array.append(unsorted)
           
    
    return (bees_array, unsorted_array)
 
def wyznaczIPrzygotujSiatki(detekcje):
    if (len(detekcje) > 10):
        liczbaSiatek = 1
    else: liczbaSiatek = 4
    siatki = przygotujSiatki(liczbaSiatek) 
    return (liczbaSiatek, siatki)
    
        
def przygotujSiatki(liczbaSiatek):
    siatki = []
    for i in range(len(liczbaSiatek)):
        s = [ [0] * rows for _ in range(cols)]
        siatki.append(s) 
    return siatki           
    
    
def main():
    max_frames = 300

    detection_engine_file = 'detekcja_engine'
    preprocessor = 'fixed_shape_resizer'
    detections = []
    
    bees_array = []
    unsorted_array = []
    sredniaLiczbaPszczol = 0
    
    trt_infer_detection = TensorRTInfer(detection_engine_file, preprocessor, 'bbox', detection_threshold)
    
    detection_frames = readImagesFromVideo(max_frames,film_path, None)
    
    if len(detection_frames) < max_frames:
        max_frames = len(detection_frames)
    #logger.info('Batching frames')
   
    batcher_detection = ImageBatcher(None, *trt_infer_detection.input_spec(), preprocessor=preprocessor, image_array=detection_frames[0])
    #logger.info('Bee detection process started')
    for batch, images, scales in batcher_detection.get_batch():
        detekcjePierwszyObraz = trt_infer_detection.infer(batch, scales, detection_threshold) 
        liczbaSiatek, siatki = wyznaczIPrzygotujSiatki(detekcjePierwszyObraz)
   
   
    batcher_detection = ImageBatcher(None, *trt_infer_detection.input_spec(), preprocessor=preprocessor, image_array=detection_frames)
    #logger.info('Bee detection process started')
    

    for batch, images, scales in batcher_detection.get_batch():
        detections.append(trt_infer_detection.infer(batch, scales, detection_threshold)) 
        
    for i in range(len(detections)):
        for d in detections[i][0]:
            ymin = int(d['ymin'])
            ymax = int(d['ymax'])
            xmin = int(d['xmin'])
            xmax = int(d['xmax']) 
            if xmax < 0 or ymax<0 or ymin < 0 or xmin < 0:
                break
            if (liczbaSiatek == 1):
                xavg = (xmin + xmax) / 2
                yavg = (ymin + ymax) / 2
                update_grid(xavg, yavg, siatki[0])
            else:
                update_grid(xmin, ymin, siatki[0])
                update_grid(xmin, ymax, siatki[1])         
                update_grid(xmax, ymin, siatki[2])            
                update_grid(xmax, ymax, siatki[3])
    

    bees_array, unsorted_array = prepare_list(siatki)
    okna_z = [[],[],[],[]]   

    for index in range(liczbaSiatek):
        s = sorted(unsorted_array[index], key= lambda x:x[1])
        first = s.pop()
        min_x = first[0][0]
        min_y = first[0][1]
        max_x = min_x
        max_y = min_y
        found = first[1]
        sredniaLiczbaPszczol = math.floor(bees_array[index] / max_frames)
        while (found < sredniaLiczbaPszczol * max_frames * thres ) and len(s) > 0:
            current = s.pop()
            min_x = min(min_x, current[0][0])
            min_y = min(min_y, current[0][1])
            max_x = max(max_x, current[0][0])
            max_y = max(max_y, current[0][1])
            found = update_found_bees(min_x,min_y, max_y, siatki[index])
        okna_z[0].append(min_x)
        okna_z[1].append(min_y)
        okna_z[2].append(max_x)
        okna_z[3].append(max_y)

    
    
    min_x = min(okna_z[0])
    min_y = min(okna_z[1])
    max_x = max(okna_z[2])
    max_y = max(okna_z[3])

    min_x = min_x * h
    min_y = min_y * h
    max_x = max_x * h
    max_y = max_y * h
    
    skip_frames = 0
    if sredniaLiczbaPszczol > 5 and sredniaLiczbaPszczol < 10:
        skip_frames = 10
    elif sredniaLiczbaPszczol >= 10 and sredniaLiczbaPszczol < 15:
        skip_frames = 8
    elif sredniaLiczbaPszczol >= 15 and sredniaLiczbaPszczol < 20:
        skip_frames = 6
    elif sredniaLiczbaPszczol >=20 and sredniaLiczbaPszczol < 25:
        skip_frames = 4
    else: 
        skip_frames = 3        
    set_config(max_frames, thres, min_x, max_x, min_y, max_y, skip_frames)
    
   
if __name__ == "__main__":
    
    main()
 
    
    
