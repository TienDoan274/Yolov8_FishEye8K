from ultralytics import YOLO
from PIL import Image
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='yolov8_predict')
    parser.add_argument('--image_path',type=str,default='./train/images/camera15_A_40.png')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train10/weights/best.pt')

    args = parser.parse_args()
    return args
def main(args):
    model = YOLO(args.model_path)

    result = model.predict(args.image_path, imgsz=1024, conf=0.3,show_labels=True,iou = 0.6)
    img = result[0].plot(line_width=3, font_size=5)
    
    im = Image.fromarray(img)
    im.save('./runs/pred.png')

if __name__ == '__main__':
    args =get_args()
    main(args)