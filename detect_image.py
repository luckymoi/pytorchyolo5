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
"""Example using Pytorch to detect objects in a given image."""

import argparse
import time
import sys
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


def main():
#    labels = load_labels("models/coco_labels.txt")
#    interpreter = make_interpreter("models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
#    interpreter.allocate_tensors()
    threshold = 0.4

    while True:
        line = sys.stdin.readline().rstrip("\n")
        try:
            rawImage = BytesIO(base64.b64decode(line))
            image = Image.open(rawImage)
            rep = os.path.abspath(__file__)
            
            now = datetime.now()
            fn = rep + now.strftime("%H:%M:%S")
            print(fn)
            
            image.save(fn, "JPEG")
            
#            scale = detect.set_input(interpreter, image.size,
#                                     lambda size: image.resize(size, Image.ANTIALIAS))

            start = time.perf_counter()
#            interpreter.invoke()
            inference_time = time.perf_counter() - start
#            objs = detect.get_output(interpreter, threshold, scale)
            output = []
            for obj in objs:
                label = labels.get(obj.id, obj.id)
                labelID = obj[0]
                score = obj[1]
                bbox = obj[2]
                output.append({"bbox": bbox, "class": label, "score": score})
            printData(output, (inference_time * 1000))
        except Exception as e:
            printError(str(e))


if __name__ == '__main__':
    main()
