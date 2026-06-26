import cv2
import psycopg2
import json
import numpy as np
from deepface import DeepFace
import easyocr

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

# 2. Inicializa os Motores de IA
detector_rosto = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("[IA] Carregando motor de leitura de texto (EasyOCR)...")
leitor_texto = easyocr.Reader(['pt', 'en'], gpu=False)

# Liga a Webcam
webcam = cv2.VideoCapture(0)

contador_frames = 0
ultima_placa_lida = "NENHUMA"
ultimo_id_gravado = None

print("\n" + "="*60)
print("SISTEMA DE PORTARIA UNIFICADO - MODO DIAGNÓSTICO")
print("="*60 + "\n")

while True:
    sucesso, frame = webcam.read()
    if not sucesso: break

    contador_frames += 1
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = detector_rosto.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    nome_tela = "Buscando..."
    cor_status = (255, 0, 0)

    # 🚗 [CÂMERA CARRO] - Leitura de Placas (A cada 7 frames)
    if contador_frames % 7 == 0:
        try:
            leitura = leitor_texto.readtext(frame)
            for (bbox, texto, probabilidade) in leitura:
                texto_limpo = texto.replace(" ", "").upper()
                if len(texto_limpo) >= 5 and probabilidade > 0.45:
                    ultima_placa_lida = texto_limpo
                    print(f"🚗 [PLACA] Detectada: {ultima_placa_lida}")
        except:
            pass

    # 👤 [CÂMERA PEDESTRE] - Reconhecimento Facial (A cada 10 frames)
    if len(rostos) > 0 and contador_frames % 10 == 0:
        print("[IA] Rosto detectado na tela! Analisando assinatura...")
        cv2.imwrite("temp_analise.jpg", frame)
        try:
            resultado_atual = DeepFace.represent(img_path="temp_analise.jpg", model_name="VGG-Face", enforce_detection=False)
            vetor_atual = np.array(resultado_atual[0]["embedding"])

            cursor.execute("SELECT id, nome, apartamento, placa_carro, vetor_rosto FROM moradores WHERE vetor_rosto IS NOT NULL;")
            moradores_banco = cursor.fetchall()

            melhor_distancia = 1.0
            morador_encontrado = None

            for id_morador, nome, ap, placa_banco, vetor_texto in moradores_banco:
                vetor_banco = np.array(json.loads(vetor_texto))
                distancia = 1 - (np.dot(vetor_atual, vetor_banco) / (np.linalg.norm(vetor_atual) * np.linalg.norm(vetor_banco)))

                if distancia < 0.58 and distancia < melhor_distancia:
                    melhor_distancia = distancia
                    morador_encontrado = (id_morador, nome, ap, placa_banco)

            if morador_encontrado:
                id_m, nome_m, ap_m, placa_m = morador_encontrado
                nome_tela = f"{nome_m} (Ap {ap_m})"
                cor_status = (0, 255, 0)
                placa_final = ultima_placa_lida if ultima_placa_lida != "NENHUMA" else placa_m

                print(f" 🔓 MORADOR ENCONTRADO: {nome_tela}")

                if id_m != ultimo_id_gravado:
                    cursor.execute("""
                        INSERT INTO historico_acessos (tipo_usuario, usuario_id, tipo_acao, placa_detectada, rosto_detectado_status)
                        VALUES ('Morador', %s, 'Entrada', %s, 'Reconhecido');
                    """, (id_m, placa_final))
                    conexao.commit()
                    ultimo_id_gravado = id_m
            else:
                nome_tela = "Nao Cadastrado"
                cor_status = (0, 0, 255)
                print(f"🚨 ALERTA: Não Cadastrado na câmera! Placa vinculada: {ultima_placa_lida}")
                
                if ultimo_id_gravado != 0:
                    cursor.execute("""
                        INSERT INTO historico_acessos (tipo_usuario, usuario_id, tipo_acao, placa_detectada, rosto_detectado_status)
                        VALUES ('Visitante', 0, 'Entrada', %s, 'Desconhecido');
                    """, (ultima_placa_lida,))
                    conexao.commit()
                    ultimo_id_gravado = 0

        except Exception as erro_detalhado:
            # 💡 AGORA O PYTHON VAI DIZER EXATAMENTE O QUE DEU ERRO!
            print(f"❌ Erro no bloco do rosto: {erro_detalhado}")

    # Desenha as caixas visuais na janela da portaria
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor_status, 2)
        cv2.putText(frame, nome_tela, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
    
    cv2.putText(frame, f"Ultima Placa: {ultima_placa_lida}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Portaria Inteligente - Monitoramento", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cursor.close()
conexao.close()
webcam.release()
cv2.destroyAllWindows()
