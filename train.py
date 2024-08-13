def main():
    from ultralytics import YOLO
    # Train a custom model
    model = YOLO("D:/TUHOC/VietNguyenAI/runs/detect/train8/weights/best.pt")
    model.train(data="D:\TUHOC\VietNguyenAI\yolo\Fisheye8K\yolo_data.yml", epochs=100, imgsz=256,device=0,batch=2,workers=4)
if __name__ == "__main__":
    main()