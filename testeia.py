import cv2
import numpy as np
import face_recognition
from tracker import Tracker
from ultralytics import YOLO
import cvzone

reconhecimento = "Reconhecimento e contador"

# Abrindo a câmera
capturandovideo = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
tracker = Tracker()

# Tamanho do vídeo
larguravid = 1280
alturavid = 720

# Cor das caixas
vermelho = (0, 0, 255)
verde = (0, 255, 0)

listaframe = []
rostoconhecido = []
nomerosto = []

# Carregando as imagens e fazendo a função de reconhecimento
def carregarfaceconhecida():
    global rostoconhecido, nomerosto
    imagensarq = [
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian1.jpg",
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian2.jpg",
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian2.png",
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian.jpg",
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian3.png",
        r"C:\Users\ianca\Desktop\yolo_people_counter\Ian4.png"
    ]
    
    for imagemarq in imagensarq:
        try:
            image = face_recognition.load_image_file(imagemarq)
            reconhecrosto = face_recognition.face_encodings(image)[0]
            rostoconhecido.append(reconhecrosto)
            nomerosto.append("Ian")
        except Exception as e:
            print(f"Erro ao carregar imagem {imagemarq}: {e}")

carregarfaceconhecida()

while True:
    ret, frame = capturandovideo.read()
    if not ret:
        break

    # Diminuir tamanho do vídeo, tentando melhorar o FPS
    frame = cv2.resize(frame, (larguravid, alturavid))

    # Usando YOLO para detectar as pessoas
    results = model.predict(frame)
    contandopessoas = 0

    for r in results:
        caixaboxes = r.boxes
        for i in range(caixaboxes.shape[0]):
            cls = int(caixaboxes.cls[i].item())
            bb = caixaboxes.xyxy[i]

            if cls == 0:  
                x1, y1, x2, y2 = map(int, bb)  # Coordenadas das pessoas
                cv2.rectangle(frame, (x1, y1), (x2, y2), verde, 3)
                listaframe.append([x1, y1, x2, y2])
                contandopessoas += 1

    # Atualiza o tracker
    bbox_idx = tracker.update(listaframe)

    # Reconhecimento facial
    localizandorosto = face_recognition.face_locations(frame)
    facesreconhece2 = face_recognition.face_encodings(frame, localizandorosto)

    face_names = []
    for facereconhe in facesreconhece2:
        distancia = face_recognition.face_distance(rostoconhecido, facereconhe)

        
        if distancia.size > 0:
            best_match_index = np.argmin(distancia)
            name = "Desconhecido"
            
            if distancia[best_match_index] < 0.6:  
                name = nomerosto[best_match_index]

            face_names.append(name)
        else:
            face_names.append("Desconhecido")  


    cvzone.putTextRect(frame, f"Quantidade de pessoas: {contandopessoas}", (50, 60), 2, 2)

    for (top, right, bottom, left), name in zip(localizandorosto, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), vermelho, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    cv2 .imshow(reconhecimento, frame)

    # aperta Q para fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capturandovideo.release()
cv2.destroyAllWindows()