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


def detect(imgsz, model, names, colors, device):
    threshold = 0.4
    printInfo("ready")
    
    rep = os.path.abspath(__file__)
    fn = rep + '.jpg'
    
    while True:
        line = sys.stdin.readline().rstrip("\n")
        try:
            rawImage = BytesIO(base64.b64decode(line))
            image = Image.open(rawImage)
                       
#            now = datetime.now()
#            fn = rep + now.strftime("%H:%M:%S")
#            printInfo(fn)
#            
            image.save(fn, "JPEG")
            
            dataset = LoadImages(fn, img_size=imgsz)
            
            start = time.perf_counter()
#            interpreter.invoke()

            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            
            inference_time = time.perf_counter() - start
#            objs = detect.get_output(interpreter, threshold, scale)
            objs = []
            output = []
            for obj in objs:
                label = labels.get(obj.id, obj.id)
                labelID = obj[0]
                score = obj[1]
                bbox = obj[2]
                output.append({"bbox": bbox, "class": label, "score": score})
                

            output.append({"bbox": [10,20,100,150], "class": "ours", "score": 80})
            output.append({"bbox": [100,200,400,250], "class": "aldo", "score": 55})

            printData(output, (inference_time * 1000))
        except Exception as e:
            printError(str(e))




#    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
#    for path, img, im0s, vid_cap in dataset:
#        img = torch.from_numpy(img).to(device)
#        img = img.half() if half else img.float()  # uint8 to fp16/32
#        img /= 255.0  # 0 - 255 to 0.0 - 1.0
#        if img.ndimension() == 3:
#            img = img.unsqueeze(0)

#        # Inference
#        t1 = time_synchronized()
#        pred = model(img, False)[0]

#        # Apply NMS
#        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
#        t2 = time_synchronized()

#        # Apply Classifier
#        if classify:
#            pred = apply_classifier(pred, modelc, img, im0s)

#        # Process detections
#        for i, det in enumerate(pred):  # detections per image
#            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#            p = Path(p)  # to Path
#            save_path = str(save_dir / p.name)  # img.jpg
#            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#            s += '%gx%g ' % img.shape[2:]  # print string
#            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#            if len(det):
#                # Rescale boxes from img_size to im0 size
#                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                # Print results
#                for c in det[:, -1].unique():
#                    n = (det[:, -1] == c).sum()  # detections per class
#                    s += f'{n} {names[int(c)]}s, '  # add to string

#                # Write results
#                for *xyxy, conf, cls in reversed(det):
#                    if save_txt:  # Write to file
#                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                        with open(txt_path + '.txt', 'a') as f:
#                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                    if save_img or view_img:  # Add bbox to image
#                        label = f'{names[int(cls)]} {conf:.2f}'
#                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

#            # Print time (inference + NMS)
#            print(f'{s}Done. ({t2 - t1:.3f}s)')

#            # Stream results
##            if view_img:
##                cv2.imshow(str(p), im0)
##                if cv2.waitKey(1) == ord('q'):  # q to quit
##                    raise StopIteration

#            # Save results (image with detections)
#            if save_img:
#                if dataset.mode == 'image':
#                    cv2.imwrite(save_path, im0)
#                else:  # 'video'
#                    if vid_path != save_path:  # new video
#                        vid_path = save_path
#                        if isinstance(vid_writer, cv2.VideoWriter):
#                            vid_writer.release()  # release previous video writer

#                        fourcc = 'mp4v'  # output video codec
#                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
#                    vid_writer.write(im0)

##    if save_txt or save_img:
##        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
##        print(f"Results saved to {save_dir}{s}")

#    print(f'Done. ({time.time() - t0:.3f}s)')


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
                
        detect(imgsz, model, names, colors, device)
