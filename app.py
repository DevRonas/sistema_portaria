import sys
sys.path.append(r"C:\ia_libs")

import cv2

# Carrega o detector de rostos nativo do OpenCV
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Liga a webcam (0 é a câmera padrão)
webcam = cv2.VideoCapture(0)

print("\n" + "="*50)
print("SISTEMA DE PORTARIA INICIADO!")
print("Olhe para a câmera. Aperte 'c' para tirar foto ou 'q' para sair.")
print("="*50 + "\n")

while True:
    sucesso, frame = webcam.read()
    if not sucesso:
        print("Erro ao acessar a câmera.")
        break

    # Converte para tons de cinza para processamento mais rápido
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostos na imagem
    rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha o quadrado verde em volta do rosto
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibe a janela de vídeo
    cv2.imshow("Camera da Portaria (Simulada)", frame)

    tecla = cv2.waitKey(1) & 0xFF

    # Salva a foto ao apertar 'c'
    if tecla == ord('c'):
        if len(rostos) > 0:
            cv2.imwrite("morador_cadastrado.jpg", frame)
            print("✨ Foto salva com sucesso como 'morador_cadastrado.jpg'!")
        else:
            print("❌ Nenhum rosto na tela para salvar.")

    # Fecha o programa ao apertar 'q'
    elif tecla == ord('q'):
        print("Fechando o sistema...")
        break

webcam.release()
cv2.destroyAllWindows()
