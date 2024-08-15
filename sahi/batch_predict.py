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
    parser = argparse.ArgumentParser(description='batch_predict_sahi')
    parser.add_argument('--images_path',type=str,default='./test/images')
    parser.add_argument('--model_path',type=str,default='./runs/detect/train11/weights/best.pt')
    parser.add_argument("--config_path", "-c", type=str, default='./sahi/model_config.yml')

    args = parser.parse_args()
    return args
def main(args): 
    predict(
        model_type="yolov8",
        model_path=args.model_path,
        model_device='cuda:0',
        model_confidence_threshold=0.3,
        source=args.images_path,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

if __name__ == '__main__':
    args =get_args()
    main(args)