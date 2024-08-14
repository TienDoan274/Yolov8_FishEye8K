from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import argparse
import torch
import cv2
from torchvision.transforms import ToTensor
from torchvision import transforms
import json
import os
def get_args():
    parser = argparse.ArgumentParser(description='train_fisheye_with_fasterRCNN')
    parser.add_argument('--image_path',type=str,default='./test/images/camera2_A_261.png')
    parser.add_argument('--label_path',type=str,default='./test/labels/camera2_A_261.txt')

    args = parser.parse_args()
    return args
def main(args):
    classes = ['Bus','Bike','Car','Pedestrian','Truck']
    with open(args.label_path) as f:
        lines = f.read().splitlines() 
    bboxes = []
    labels = []
    for line in lines:
        line = line.split(' ')
        labels.append(int(line[0]))
        bboxes.append(list(map(float,line[1:])))
    image = cv2.imread(args.image_path)
    img_shape = image.shape
    for box, label in zip(bboxes, labels):
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2)*img_shape[1])
        y_min = int((y_center - height / 2)*img_shape[0])
        x_max = int((x_center + width / 2)*img_shape[1])
        y_max = int((y_center + height / 2)*img_shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,0), 2)
        cv2.putText(image, str(label) , (x_min, y_min),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite("./runs/visualize.png", image)  
    cv2.waitKey(0)

if __name__ == '__main__':
    args =get_args()
    main(args)