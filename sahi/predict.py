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
    parser = argparse.ArgumentParser(description='predict_sahi')
    parser.add_argument('--image_path',type=str,default='./train/images/camera15_A_40.png')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train10/weights/best.pt')
    parser.add_argument("--config_path", "-c", type=str, default='./sahi/model_config.yml')

    args = parser.parse_args()
    return args
def main(args): 
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model_path,
        confidence_threshold=0.5,
        device="cuda:0",
        config_path=args.config_path
    )
    result = get_sliced_prediction(
        args.image_path,
        detection_model,
        slice_height = 512,
        slice_width = 512,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    result.export_visuals(export_dir="runs",file_name="sahi_pred")

if __name__ == '__main__':
    args =get_args()
    main(args)