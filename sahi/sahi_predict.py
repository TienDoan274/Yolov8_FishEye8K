# arrange an instance segmentation model for test
from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='train_fisheye_with_fasterRCNN')
    parser.add_argument('--image_path',type=str,default='D:/TUHOC/VietNguyenAI/yolo/Fisheye8K/train/images/camera3_A_0.png')
    parser.add_argument('--model_path',type=str,default='D:/TUHOC/VietNguyenAI/runs/detect/train11/weights/best.pt')
    parser.add_argument("--conf_thres", "-t", type=float, default=0.3)
    parser.add_argument("--config_path", "-c", type=str, default="D:\TUHOC\VietNguyenAI\yolo\Fisheye8K\sahi\model_config.yml")

    args = parser.parse_args()
    return args
def main(args): 
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model_path,
        confidence_threshold=args.conf_thres,
        device="cuda:0",
        config_path=args.config_path
    )
    result = get_sliced_prediction(
        args.image_path,
        detection_model,
        slice_height = 256,
        slice_width = 256,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    result.export_visuals(export_dir="demo_data/")

if __name__ == '__main__':
    args =get_args()
    main(args)