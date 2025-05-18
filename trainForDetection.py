from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(data="data.yaml", epochs = 100, batch=16, lr0=0.001, save=True)
