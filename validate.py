from ultralytics import YOLO
import argparse
from PIL import Image
import torch
import os
def get_args():
    parser = argparse.ArgumentParser(description='validating_model')
    parser.add_argument('--data_yml_path',type=str,default='./yolo_data.yml')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train10/weights/best.pt')

    args = parser.parse_args()
    return args
def main(args):

    model = YOLO(args.model_path)
    results = model.val(data=args.data_yml_path, imgsz=640, batch=8, conf=0.3, iou=0.6, device="0")
    print(results)
if __name__ == '__main__':
    
    args =get_args()
    main(args)