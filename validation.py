from ultralytics import YOLO, settings
model = YOLO('C:/Users/Filipe/Desktop/Projeto 1 - IC/results/runs/detect/train5/weights/best.pt')


if __name__ == '__main__':
    results = model.val(  # project='runs/detect',
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,  # Non-Maximum Supression (NMS)
        save_json=False,  # Save to JSON {image_id, cls, bbox, conf} of each image in dataset
        save_hybrid=False,
        # Bounding box labels + inference on the output image (Ultralytics 8.0.178 = results are wrong when this argmunent is True)
        split='test')  # train, val or test

    # results = model.val()
    print(results)