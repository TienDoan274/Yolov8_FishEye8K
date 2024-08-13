from ultralytics import YOLO
import argparse
from PIL import Image
def get_args():
    parser = argparse.ArgumentParser(description='train_fisheye_with_fasterRCNN')
    parser.add_argument('--image_path',type=str,default='D:/TUHOC/VietNguyenAI/yolo/Fisheye8K/train/images/camera3_A_0.png')
    parser.add_argument('--model_path',type=str,default='D:/TUHOC/VietNguyenAI/runs/detect/train11/weights/best.pt')
    parser.add_argument("--conf_thres", "-t", type=float, default=0.2)

    args = parser.parse_args()
    return args
def main(args):
    model = YOLO(args.model_path)

    # Run inference on 'bus.jpg' with arguments
    result = model.predict(args.image_path, save=False, imgsz=256, conf=0.3,show_labels=True)
    img = result[0].plot(line_width=3, font_size=15)
    
    im = Image.fromarray(img)
    im.save('result.png')

if __name__ == '__main__':
    args =get_args()
    main(args)