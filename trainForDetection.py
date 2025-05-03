from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(data="/kaggle/input/helmetriderplate/data.yaml", epochs = 50, patience=0, batch=16, lr0=0.0005, save=True)