import cv2
import psycopg2
from ultralytics import YOLO
import easyocr
import re

# 1. Conecta ao Banco de Dados PostgreSQL do Condomínio
try:
    conexao = psycopg2.connect(
        host="localhost", database="postgres", user="postgres", password="portaria123", port="5432"
    )
    cursor = conexao.cursor()
    print("Conexão com o banco de dados realizada com sucesso!")
except Exception as e:
    print(f"Erro ao conectar no banco de dados: {e}")
    exit()

# 2. Inicializa o Leitor de Texto (OCR) configurado para Português/Inglês
print("[IA] Carregando motor de leitura de texto (EasyOCR)...")
leitor_texto = easyocr.Reader(['pt', 'en'], gpu=False)

# 3. Carrega o modelo YOLOv8 padrão
print("[IA] Carregando detector de objetos YOLOv8...")
modelo_yolo = YOLO('yolov8n.pt') 

# Liga a webcam (Simula a câmera da entrada de carros)
webcam = cv2.VideoCapture(0)

print("\n" + "="*50)
print("IA DE DETECÇÃO DE PLACAS INICIADA!")
print("Aperte 'q' para fechar.")
print("="*50 + "\n")

while True:
    sucesso, frame = webcam.read()
    if not sucesso:
        print("Erro ao acessar a câmera.")
        break

    # Roda o YOLOv8 no frame atual para procurar objetos (Carros, pessoas, etc)
    resultados = modelo_yolo(frame, verbose=False)[0]
    
    placa_detectada_texto = "Buscando Placa..."
    cor_box = (255, 0, 0) # Azul padrão

    # Opcional: Para testar na sua webcam sem um carro de verdade, vamos fazer
    # o EasyOCR ler qualquer texto nítido que você apontar para a câmera (como uma folha ou celular)
    try:
        # Pede para o OCR ler textos na tela
        leitura = leitor_texto.readtext(frame)
        
        for (bbox, texto, probabilidade) in leitura:
            # Limpa o texto tirando espaços e deixando em maiúsculo
            texto_limpo = texto.replace(" ", "").upper()
            
            # Validação simples: Se o texto parecer uma placa Mercosul (ABC1D23) ou Antiga (ABC1234)
            # Aqui aceitaremos qualquer texto com mais de 4 letras/números para você conseguir testar na webcam
            if len(texto_limpo) >= 5 and probabilidade > 0.40:
                placa_detectada_texto = texto_limpo
                cor_box = (0, 255, 0) # Muda para verde quando acha um texto forte
                
                print(f"🚗 TEXTO/PLACA DETECTADA: {placa_detectada_texto} (Confiança: {probabilidade:.2f})")
                
                # BUSCA NO BANCO: Verifica se essa placa pertence a algum morador cadastrado
                cursor.execute("SELECT nome, apartamento FROM moradores WHERE placa_carro = %s;", (placa_detectada_texto,))
                morador = cursor.fetchone()
                
                if morador:
                   print(f"🔓 VEÍCULO LIBERADO: Morador {morador[0]} do Ap {morador[1]}!")
                break
    except Exception as e:
        pass

    # Exibe o resultado na janela de vídeo
    cv2.putText(frame, f"Status: {placa_detectada_texto}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_box, 2)
    cv2.imshow("Camera de Transito - Leitura de Placas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cursor.close()
conexao.close()
webcam.release()
cv2.destroyAllWindows()
