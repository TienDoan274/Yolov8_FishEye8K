from ultralytics import YOLO
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='train_yolov8')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train11/weights/best.pt')
    parser.add_argument('--yml_path',type=str,default='./yolo_data.yml')
    args = parser.parse_args()
    return args
def main(args):
    model = YOLO(model=args.model_path)
    model.train(data = args.yml_path,epochs=100,imgsz=640,batch=4,device=0,
                workers=8,patience=15,dropout= 0.1,label_smoothing= 0.1,lr0= 0.001,cos_lr= True)
if __name__ == "__main__":
    args = get_args()
    main(args)