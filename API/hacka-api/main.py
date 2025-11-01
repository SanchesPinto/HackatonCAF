# --- Imports da API ---
import io
import sys
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests # <--- NOVO IMPORT

# --- Imports do Modelo (PyTorch & PIL) ---
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# ========================================================
# 1. CONSTANTES E CONFIGURAÇÕES DA API
# ========================================================
CLASS_NAMES = ["VOLANTE 8 / 9", "DISCO INOX CAF 8 FURO 5,0 MM (V.24)", "CRUZETA INOX CAF 8 (V.24)", "CARACOL MONTADO 8", "BOCAL CONJUNTO 8"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Caminho para o modelo treinado ---
MODELO_SALVO_PATH = "rna_etapa2_classificador_imagem.pth" # <-- MUDE ISSO

IMG_SIZE = 128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

transform_inferencia = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# ========================================================
# 2. Definição da Arquitetura
# ========================================================
class Rede(nn.Module):
    # ... (Nenhuma mudança aqui, o código da sua classe 'Rede' continua o mesmo) ...
    def __init__(self, cnn_output_size=512, classifier_dropout=0.5, num_classes=5):
        super(Rede, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        self.cnn_extractor = resnet
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size, 128), nn.ReLU(),
            nn.Dropout(p=classifier_dropout), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.cnn_extractor(x); output = self.classifier(features)
        return output

# ========================================================
# 3. Função de Carregamento
# ========================================================
def carregar_modelo_pronto(caminho_modelo, num_classes, device):
    # ... (Nenhuma mudança aqui) ...
    print(f"Carregando modelo para {num_classes} classes...")
    model = Rede(num_classes=num_classes)
    print(f"Carregando pesos de: {caminho_modelo}")
    try:
        checkpoint = torch.load(caminho_modelo, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Erro ao carregar o state_dict do modelo: {e}")
        return None
    model.to(device); model.eval()
    print("--- Modelo carregado e pronto para inferência ---")
    return model

# ========================================================
# 4. LÓGICA DE STARTUP DA API
# ========================================================
print("Iniciando a API e carregando o modelo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
model = carregar_modelo_pronto(MODELO_SALVO_PATH, NUM_CLASSES, device)
if model is None:
    print("Erro fatal: Não foi possível carregar o modelo.")
    sys.exit(1)

# ========================================================
# 5. INICIALIZAÇÃO E ENDPOINTS DA API
# ========================================================
app = FastAPI(title="API de Classificação")

# Modelos de Resposta e Requisição Pydantic
class PredictionResponse(BaseModel):
    filename: str # Usaremos a URL aqui
    prediction: str
    confidence_percent: float

class ImageRequest(BaseModel): # <--- NOVO MODELO
    """Define o JSON que o Manychat deve nos enviar"""
    image_url: str

# --- Endpoint de Upload (o que você já tinha) ---
@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    # ... (Nenhuma mudança aqui, este endpoint continua funcionando) ...
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform_inferencia(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            predicted_class_index = top_idx.item()
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = top_prob.item() * 100
        return PredictionResponse(
            filename=file.filename,
            prediction=predicted_class_name,
            confidence_percent=round(confidence, 2)
        )
    except Exception as e:
        return {"error": f"Erro ao processar a imagem: {str(e)}"}


# --- NOVO ENDPOINT PARA O MANYCHAT ---
@app.post("/predict-from-url/", response_model=PredictionResponse)
async def predict_from_url(request: ImageRequest):
    """
    Recebe uma URL (do Manychat), baixa a imagem,
    e retorna a predição.
    """
    try:
        # 1. Baixar a imagem da URL
        response = requests.get(request.image_url)
        response.raise_for_status() # Lança um erro se o download falhar
        
        # 2. Ler a imagem baixada (em memória)
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 3. Pré-processar a imagem (mesma lógica de antes)
        image_tensor = transform_inferencia(image).unsqueeze(0).to(device)
        
        # 4. Fazer a predição (mesma lógica de antes)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            predicted_class_index = top_idx.item()
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = top_prob.item() * 100
        
        # 5. Retornar o resultado
        return PredictionResponse(
            filename=request.image_url, # Retorna a URL como 'filename'
            prediction=predicted_class_name,
            confidence_percent=round(confidence, 2)
        )
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Falha ao baixar a imagem da URL: {str(e)}"}
    except Exception as e:
        return {"error": f"Erro ao processar a imagem: {str(e)}"}