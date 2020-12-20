# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# based on work https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98





# import python3-numba python3-skimage python3-sklearn python3-netfilter

"""Example using Pytorch to detect objects in a given image."""

import argparse
import time
import sys
import os
from PIL import Image
from PIL import ImageDraw
from io import BytesIO, StringIO
import time
import base64
import json
import detect
#import tflite_runtime.interpreter as tflite
import platform
from datetime import datetime

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized



                
def printInfo(text):
    print(json.dumps({"type": "info", "data": text}))

def printError(text):
    print(json.dumps({"type": "error", "data": text}))

def printData(array, time):
    print(json.dumps({"type": "data", "data": array, "time": time}))


def detect(imgsz, model, names, colors, device, half):
    threshold = 0.4
    printInfo("ready")
    half = False
    conf_thres = 0.4
    iou_thres=0.45
    classes=None
    agnostic_nms=False

    
    rep = os.path.abspath(__file__)
    fn = rep + '.jpg'
    
    while True:
        line = sys.stdin.readline().rstrip("\n")
        try:
            output = []
            rawImage = BytesIO(base64.b64decode(line))
            image = Image.open(rawImage)
                       
            image.save(fn, "JPEG")
            
            dataset = LoadImages(fn, img_size=imgsz)
            
            start = time.perf_counter()

            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                pred = model(img, False)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
                
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    
                    output = []
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            labelID = names[int(cls)]
                            score = float(conf)
#                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            bbox = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2])-int(xyxy[0]),int(xyxy[3])-int(xyxy[1])]
                            output.append({"bbox": bbox, "class": labelID, "score": score})
          
                
            inference_time = time.perf_counter() - start
                   
            printData(output, (inference_time * 1000))
            
        except Exception as e:
            printError(str(e))




if __name__ == '__main__':
    source = '/home/Shinobi/plugins/pytorchyolo5/image2.jpg'
    weights =  ['yolov5s.pt']
    view_img = False
    save_img = True
    save_txt = False
    imgsz = 640
    conf_thres = 0.4
    iou_thres=0.45
    classes=None
    agnostic_nms=False
    save_conf=False
    
    with torch.no_grad():
        # Initialize
        set_logging()
        device = select_device('0') #cuda device, i.e. 0 or 0,1,2,3 or cpu'
        half = device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                
        detect(imgsz, model, names, colors, device, half)
