import cv2
import time
from ultralytics import YOLO, settings

# Inserir o link para o arquivo best.pt na pasta weights
model = YOLO('caminho/best.pt')
#endereço para o vídeo
video_path = ""
webcam_id = 0
#substituir video_path ou webcam_id na função abaixo
cap = cv2.VideoCapture(webcam_id)

# Obter detalhes do vídeo original
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar saída do vídeo processado
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para salvar vídeo
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

count_s = 0
count_l = 0
count_d = 0
count_not_detected = 0
tempo = 0.0

def predict(isVideo, widthV, heightV):
    global count_s, count_l, count_d, tempo, count_not_detected

    #tempo inicial
    tempoInicial = time.time()

    while True:
        ret, frame = cap.read()
        image_resized = cv2.resize(frame, (widthV, heightV))
        #image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        tempo = time.time() - tempoInicial

        if not ret:
            break

        # Realizar a detecção de objetos
        results = model(image_resized,
                        conf=0.5,
                        iou=0.7,  # Non-Maximum Supression (NMS)
                        imgsz=540,
                        show=False,
                        save=False,
                        save_txt=False,  # Save bbox coordenation
                        save_conf=False,  # save_txt must be True
                        save_crop=False,
                        stream=False  # Do inference now (False) or after (True)
                        )

        # Desenhar caixas e rótulos na imagem
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bbox
                conf = float(box.conf[0])  # Confiança
                cls = int(box.cls[0])  # Classe detectada

                label = f"{model.names[cls]} {conf:.2f}"
                nome_classe = model.names[cls]

                print(nome_classe)
                #registra quantas vezes cada classe foi detectada
                if(nome_classe == 'sentada'):
                    count_s += 1
                elif(nome_classe == 'deitada'):
                    count_d += 1
                elif(nome_classe == 'levantada'):
                    count_l += 1
                else:
                    count_not_detected += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if(isVideo):
            out.write(frame)

        # Mostrar a saída
        cv2.imshow("YOLO Detection", frame)

        # Sair do loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    if(isVideo):
        out.release()
    cv2.destroyAllWindows()

def calcTempo():
    total = count_d + count_l + count_s + count_not_detected
    porcenLevantado = count_l / total
    porcenSentado = count_s / total
    porcenDeitado = count_d / total
    porcenNotDetected = count_not_detected / total

    tempoL = porcenLevantado*tempo/60
    tempoS = porcenSentado*tempo/60
    tempoD = porcenDeitado*tempo/60
    tempoOff = porcenNotDetected*tempo/60
    tempoT = tempo/60

    pctL = porcenLevantado * 100
    pctS = porcenSentado * 100
    pctD = porcenDeitado * 100
    pctOff = porcenNotDetected * 100

    print(f"Tempo em pé : {tempoL:.1f} minutos ({pctL:.1f}% do tempo total)")
    print(f"Tempo sentado: {tempoS:.1f} minutos ({pctS:.1f}% do tempo total)")
    print(f"Tempo deitado: {tempoD:.1f} minutos ({pctD:.1f}% do tempo total)")
    print(f"Tempo com usuário não detectado: {tempoOff:.1f} minutos ({pctOff:.1f}% do tempo total)")
    print(f"Duração total: {tempoT:.1f} minutos")


#se a entrada for através de webcam, isVideo = False
predict(isVideo=True, widthV=640, heightV=640)
calcTempo()
