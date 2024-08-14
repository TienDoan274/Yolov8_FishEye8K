def main():
    from ultralytics import YOLO
    model = YOLO("./runs/detect/train8/weights/best.pt")
    model.train(data="./yolo_data.yml")
if __name__ == "__main__":
    main()