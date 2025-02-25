from ultralytics import YOLO, settings

settings.update({'runs_dir': 'C:/Users/Filipe/Desktop/Projeto 1 - IC/results/runs'})


#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
model = YOLO('yolo11m.pt')
model.model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model.train(data='velhinhos.yaml', epochs=50,
                #patience=15,
                batch=16,
                seed=42,
                freeze=20,
                #imgsz=540,
                #workers=8,
                #pretrained=True,
                #resume=False,  # resume training from last checkpoint
                #single_cls=False,  # Whether all classes will be the same (just one class)
                # project='runs/detect',  # Default = /home/{user}/Documents/ultralytics/runs
                #box=7.5,  # More recall, better IoU, less precission,
                #cls=0.5,  # Bbox class better
                #dfl=1.5,  # Distribution Focal Loss. Better bbox boundaries
                val=True)
                #Augmentations
                #degrees=0.3,
                #hsv_s=0.3,
                #hsv_v=0.3,
                #scale=0.5,
                #fliplr=0.5)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
