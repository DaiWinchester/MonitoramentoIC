from ultralytics import YOLO, settings
model = YOLO('C:/Users/Filipe/Desktop/Projeto 1 - IC/results/runs/detect/train5/weights/best.pt')


if __name__ == '__main__':
    results = model.val( 
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,  # Non-Maximum Supression (NMS)
        split='test')  # train, val ou test

    # results = model.val()
    print(results)
