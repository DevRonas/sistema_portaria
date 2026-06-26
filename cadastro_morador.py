import cv2
import psycopg2
import json
from deepface import DeepFace

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

# Captura os dados do morador direto pelo terminal
print("\n" + "="*50)
print("TELA DE CADASTRO DE NOVO MORADOR")
print("="*50)
nome_novo = input("Digite o NOME COMPLETO do morador: ")
ap_novo = input("Digite o APARTAMENTO (Ex: 502): ")
bloco_novo = input("Digite o BLOCO (Ex: B): ").upper()
placa_nova = input("Digite a PLACA DO CARRO (Ex: ABC1D23): ").upper()
print("="*50 + "\n")

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

print("A câmera vai abrir. Fique bem de frente para a luz e aperte 'c' para salvar.")

while True:
    sucesso, frame = webcam.read()
    if not sucesso: break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in rostos:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("Camera de Cadastro - Olhe para ca e aperte 'c'", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('c'):
        if len(rostos) > 0:
            print("\n[IA] Extraindo assinatura facial real...")
            cv2.imwrite("rosto_novo_cadastro.jpg", frame)

            try:
                # O DeepFace gera a lista de números do rosto
                resultado = DeepFace.represent(img_path="rosto_novo_cadastro.jpg", model_name="VGG-Face", enforce_detection=True)
                
                # Correção da nova versão do DeepFace [0] para pegar o primeiro rosto da lista
                vetor_real = resultado[0]["embedding"]
                vetor_texto = json.dumps(vetor_real)

                # Salva o morador com todos os dados digitados e o rosto matemático
                comando_sql = """
                    INSERT INTO moradores (nome, apartamento, bloco, placa_carro, vetor_rosto)
                    VALUES (%s, %s, %s, %s, %s);
                """
                cursor.execute(comando_sql, (nome_novo, ap_novo, bloco_novo, placa_nova, vetor_texto))
                conexao.commit()
                
                print(f"\n✨ SUCESSO ABSOLUTO! {nome_novo} foi cadastrado com rosto real no banco!")
                break # Fecha a câmera automaticamente ao finalizar

            except Exception as erro:
                conexao.rollback()
                print(f"❌ Erro ao salvar no PostgreSQL: {erro}")
        else:
            print("❌ Erro: Nenhum rosto nítido detectado na imagem. Tente de novo.")

    elif tecla == ord('q'):
        print("Cadastro cancelado.")
        break

cursor.close()
conexao.close()
webcam.release()
cv2.destroyAllWindows()
