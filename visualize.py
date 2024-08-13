from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import argparse
import torch
import cv2
from torchvision.transforms import ToTensor
from torchvision import transforms
import json
def get_args():
    parser = argparse.ArgumentParser(description='train_fisheye_with_fasterRCNN')
    parser.add_argument('--image_path',type=str,default='D:/TUHOC/VietNguyenAI/yolo/Fisheye8K/train/images/camera3_A_0.png')
    parser.add_argument('--label_path',type=str,default='D:/TUHOC/VietNguyenAI/yolo/Fisheye8K/train/labels/camera3_A_0.txt')
    parser.add_argument("--conf_thres", "-t", type=float, default=0.2)

    args = parser.parse_args()
    return args
def main(args):
    with open(args.label_path) as f:
        lines = f.read().splitlines() 
    print(lines[0])
    bboxes = []
    labels = []
    for line in lines:
        line = line.split(' ')
        labels.append(int(line[0]))
        bboxes.append(list(map(float,line[1:])))
    print(labels)
    print(bboxes)
    image = cv2.imread(args.image_path)
    img_shape = image.shape
    print('img_shape',img_shape)
    for box, label in zip(bboxes, labels):
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2)*img_shape[1])
        y_min = int((y_center - height / 2)*img_shape[0])
        x_max = int((x_center + width / 2)*img_shape[1])
        y_max = int((y_center + height / 2)*img_shape[0])
        print(x_min, y_min, x_max, y_max)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # cv2.putText(image, classes[label] + " {:.2f}".format(score), (xmin, ymin),
        #         cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 2)

    cv2.imwrite("test.jpg", image)  
    cv2.waitKey(0)

if __name__ == '__main__':
    args =get_args()
    main(args)