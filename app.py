import cv2
import psycopg2
import json
import numpy as np
from deepface import DeepFace

# 1. Conecta ao Banco de Dados PostgreSQL
try:
    conexao = psycopg2.connect(
        host="localhost", database="postgres", user="postgres", password="portaria123", port="5432"
    )
    cursor = conexao.cursor()
    print("Conexão com o banco de dados realizada com sucesso!")
except Exception as e:
    print(f"Erro ao conectar no banco de dados: {e}")
    exit()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

contador_frames = 0
ultimo_id_gravado = None 

print("\n" + "="*50)
print("SISTEMA DE PORTARIA INTELIGENTE UNIFICADO")
print("-> IA Automática rodando em segundo plano.")
print("-> Aperte 'c' a qualquer momento para CADASTRAR novo rosto.")
print("-> Aperte 'q' para fechar o programa.")
print("="*50 + "\n")

while True:
    sucesso, frame = webcam.read()
    if not sucesso: break

    contador_frames += 1
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    nome_tela = "Buscando..."
    cor_retangulo = (255, 0, 0) # Azul enquanto processa

    # 🚀 FLUXO 1: RECONHECIMENTO AUTOMÁTICO (Roda sozinho a cada 5 frames)
    if len(rostos) > 0 and contador_frames % 5 == 0:
        cv2.imwrite("rosto_verificacao.jpg", frame)
        try:
            resultado_atual = DeepFace.represent(img_path="rosto_verificacao.jpg", model_name="VGG-Face", enforce_detection=False)
            vetor_atual = np.array(resultado_atual[0]["embedding"])

            cursor.execute("SELECT id, nome, apartamento, vetor_rosto FROM moradores WHERE vetor_rosto IS NOT NULL;")
            moradores_banco = cursor.fetchall()

            melhor_distancia = 1.0 
            id_reconhecido = None
            morador_reconhecido = None

            for id_morador, nome, ap, vetor_texto in moradores_banco:
                vetor_banco = np.array(json.loads(vetor_texto))
                distancia = 1 - (np.dot(vetor_atual, vetor_banco) / (np.linalg.norm(vetor_atual) * np.linalg.norm(vetor_banco)))

                if distancia < 0.58 and distancia < melhor_distancia:
                    melhor_distancia = distancia
                    id_reconhecido = id_morador
                    morador_reconhecido = f"{nome} (Ap {ap})"

            if morador_reconhecido:
                nome_tela = morador_reconhecido
                cor_retangulo = (0, 255, 0) # Verde para Liberado
                print(f"🔓 ACESSO LIBERADO: {nome_tela}!")

                if id_reconhecido != ultimo_id_gravado:
                    sql_historico = """
                        INSERT INTO historico_acessos (tipo_usuario, usuario_id, tipo_acao, rosto_detectado_status)
                        VALUES ('Morador', %s, 'Entrada', 'Reconhecido');
                    """
                    cursor.execute(sql_historico, (id_reconhecido,))
                    conexao.commit()
                    print(f"🗄️ [BANCO] Entrada de {nome_tela} registrada no histórico!")
                    ultimo_id_gravado = id_reconhecido
            else:
                nome_tela = "Desconhecido"
                cor_retangulo = (0, 0, 255) # Vermelho para Alerta
                ultimo_id_gravado = None

        except Exception as e:
            pass

    # 🛠️ FLUXO 2: BOTÃO DE CADASTRO MANUAL (Ativa na hora se apertar 'c')
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('c'):
        if len(rostos) > 0:
            print("\n" + "[NOVO CADASTRO] " * 3)
            # No futuro, aqui pegaremos os dados digitados na tela (Dashboard)
            nome_novo = input("Digite o nome do novo morador: ")
            ap_novo = input("Digite o apartamento: ")
            placa_nova = input("Digite a placa do carro: ")
            
            print("[IA] Gerando assinatura facial para o novo cadastro...")
            cv2.imwrite("novo_morador_temp.jpg", frame)
            
            try:
                resultado_novo = DeepFace.represent(img_path="novo_morador_temp.jpg", model_name="VGG-Face", enforce_detection=True)
                vetor_real = resultado_novo[0]["embedding"]
                vetor_texto = json.dumps(vetor_real)

                comando_sql = """
                    INSERT INTO moradores (nome, apartmento, bloco, placa_carro, vetor_rosto)
                    VALUES (%s, %s, 'A', %s, %s);
                """
                # Ajustado 'apartmento' para 'apartamento' conforme sua tabela do banco
                cursor.execute("""
                    INSERT INTO moradores (nome, apartamento, bloco, placa_carro, vetor_rosto)
                    VALUES (%s, %s, 'A', %s, %s);
                """, (nome_novo, ap_novo, placa_nova, vetor_texto))
                conexao.commit()
                print(f"✨ SUCESSO: '{nome_novo}' foi salvo no banco e ja pode usar a IA!\n")
            except Exception as erro_cad:
                conexao.rollback()
                print(f"❌ Erro ao cadastrar: {erro_cad}\n")
        else:
            print("❌ Impossível cadastrar: Nenhum rosto visível na câmera.")

    elif tecla == ord('q'): break

    # Desenha as informações na tela
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor_retangulo, 2)
        cv2.putText(frame, nome_tela, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_retangulo, 2)

    cv2.imshow("Portaria Automatizada", frame)

cursor.close()
conexao.close()
webcam.release()
cv2.destroyAllWindows()
    