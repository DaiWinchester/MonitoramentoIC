from ultralytics import YOLO, settings

settings.update({'runs_dir': 'C:/Users/Filipe/Desktop/Projeto 1 - IC/results/runs'})


model = YOLO('yolo11m.pt')
model.model

if __name__ == '__main__':
    model.train(data='velhinhos.yaml', 
                epochs=50,
                batch=16,
                seed=42,
                freeze=20,
                val=True)



