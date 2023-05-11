from fastapi import FastAPI,Request, Form, UploadFile, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse,FileResponse
# from tortoise.contrib.fastapi import register_tortoise
# import rcq.quan as quan
import torch
from typing import List
import numpy as np
import sys
import os
root_path = os.getcwd()
sys.path.append("..")  
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolo_detect import detect
import os
from os.path import join
import time
import glob
import cv2
import base64
import requests
from io import BytesIO
from PIL import Image

# =========================== init ================================#
app = FastAPI()
template = Jinja2Templates("pages")

global_result_database = './data_result'
if not os.path.exists(global_result_database):
    os.makedirs(global_result_database)
# todaytime = time.strftime("%Y_%m_%d",time.localtime())
# print(todaytime.split('_')[-1]) # 方便后续改进代码
# ============================ inference ==========================#

@app.post("/inference")
async def inference(url: str = Query(default=...)):
    response_byte = requests.get(url)
    bytes_stream = BytesIO(response_byte.content)
    capture_img = Image.open(bytes_stream)
    imageby = cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)
    image_name = url.split('/')[-1]
    
    todaytime = time.strftime("%Y_%m_%d",time.localtime())
    files_path = global_result_database + todaytime
    if not os.path.exists(files_path):
        os.mkdir(files_path)
    
    if '.' in image_name:
        cv2.imwrite(os.path.join(os.path.dirname(__file__), files_path, image_name), imageby)
    else:
        image_name = image_name + '.jpg'
        cv2.imwrite(os.path.join(os.path.dirname(__file__), files_path, image_name), imageby)
    path = files_path + '/'+ image_name
    result = detect.run(source=path) 
    
    return result