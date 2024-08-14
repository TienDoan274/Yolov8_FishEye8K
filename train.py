from ultralytics import YOLO
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='train_yolov8')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train11/weights/best.pt')
    parser.add_argument('--yml_path',type=str,default='./yolo_data.yml')
    args = parser.parse_args()
    return args
def main(args):
    model = YOLO("D:/TUHOC/VietNguyenAI/yolo/yolov8n.pt")
    model.train(data = args.yml_path)
if __name__ == "__main__":
    args = get_args()
    main(args)