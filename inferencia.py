import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder # Para carregar os nomes das classes
from PIL import Image
import os
import torch.nn.functional as F # Para usar o Softmax

# ============================
# 1. Definição da Arquitetura
# ============================
class Rede(nn.Module):
    def __init__(self, cnn_output_size=512, classifier_dropout=0.5, num_classes=5):
        super(Rede, self).__init__()
        resnet = models.resnet18(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False
            
        resnet.fc = nn.Identity()
        self.cnn_extractor = resnet
        
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.cnn_extractor(x)
        output = self.classifier(features)
        return output

# ============================
# 2. Definição das Transformações
# ============================
IMG_SIZE = 128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

#Transform necessário para inferência
transform_inferencia = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# ============================
# 3. Função para Carregar o Modelo e Classes
# ============================
def carregar_modelo(caminho_modelo, caminho_pasta_treino, device):
    """
    Carrega o modelo treinado e o mapa de classes.
    """
    print(f"Carregando mapa de classes de: {caminho_pasta_treino}")
    
    # Para pegar os nomes das classes na ordem correta
    # ImageFolder os ordena alfabeticamente, exatamente como no treino
    try:
        # Usamos um transform dummy só para instanciar o dataset
        temp_dataset = ImageFolder(root=caminho_pasta_treino, transform=transforms.ToTensor())
        class_names = temp_dataset.classes
        num_classes = len(class_names)
        print(f"Classes encontradas: {class_names}")
    except Exception as e:
        print(f"Erro ao ler pasta de treino para pegar nomes das classes: {e}")
        return None, None
        
    # Inicializa a arquitetura do modelo
    model = Rede(num_classes=num_classes)
    
    # Carrega os pesos salvos (o 'state_dict')
    print(f"Carregando pesos do modelo de: {caminho_modelo}")
    try:
        # map_location=device garante que funcione se você treinou na GPU e agora está no CPU
        checkpoint = torch.load(caminho_modelo, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Erro ao carregar o state_dict do modelo: {e}")
        return None, None

    # Move o modelo para o dispositivo (CPU ou GPU)
    model.to(device)
    
    # Define o modelo para o modo de avaliação
    model.eval()
    
    return model, class_names

# ============================
# 4. Função de Predição
# ============================
def prever_imagem(model, class_names, caminho_imagem, transform, device):
    """
    Carrega uma imagem, processa e retorna a predição.
    """
    print("\n--- Nova Predição ---")
    try:
        # 1. Carregar a imagem com PIL (Pillow)
        # .convert('RGB') garante que imagens com 4 canais (RGBA) funcionem
        image = Image.open(caminho_imagem).convert('RGB')
    except Exception as e:
        print(f"Erro ao abrir o arquivo de imagem: {caminho_imagem}")
        print(e)
        return

    # 2. Aplicar as transformações
    image_tensor = transform(image)
    
    # 3. Adicionar a "dimensão de batch"
    # O modelo espera (B, C, H, W), nosso tensor é (C, H, W)
    # .unsqueeze(0) adiciona uma dimensão no índice 0 -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    # 4. Mover o tensor para o dispositivo
    image_tensor = image_tensor.to(device)
    
    # 5. Fazer a predição (sem calcular gradientes)
    with torch.no_grad():
        outputs = model(image_tensor) # Saída são os logits (ex: tensor([[-1.2, 4.5, 0.1, ...]]))
        
        # 6. Converter logits para Probabilidades
        # Aplicamos Softmax para transformar os logits em probabilidades [0, 1]
        probabilities = F.softmax(outputs, dim=1)
        
        # 7. Pegar a classe com maior probabilidade
        # torch.max retorna (valor_max, indice_max)
        top_prob, top_idx = torch.max(probabilities, 1)
        
        # 8. Traddauzir o índice para o nome  classe
        predicted_class_index = top_idx.item() # .item() converte o tensor de 1 elemento em um número
        predicted_class_name = class_names[predicted_class_index]
        confidence = top_prob.item() * 100
        
    # 9. Mostrar os resultados de forma clara
    print(f"Imagem analisada: {os.path.basename(caminho_imagem)}")
    print(f"Resultado: A peça é da classe '{predicted_class_name}'")
    print(f"Confiança: {confidence:.2f}%")
    
    print("\nScore de todas as classes:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  - {class_names[i]}: {prob.item() * 100:.2f}%")

    #salva predicted_class_name em um arquivo txt
    with open("predicted_class.txt", "w") as f:
        f.write(predicted_class_name)   
        
# ============================
# 5. Execução Principal
# ============================
if __name__ == "__main__":
    
    # ----------------------
    MODELO_SALVO_PATH = "models/modelo2.pth"
    PASTA_DE_TREINO = "Datasets/treino" # Usado para pegar os nomes das classes
    IMAGEM_PARA_TESTAR = "imgs_teste/25.png" #caminho da imagem para teste
    # ----------------------
    
    # Define o dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Carrega o modelo
    model, class_names = carregar_modelo(MODELO_SALVO_PATH, PASTA_DE_TREINO, device)
    
    if model is not None and class_names is not None:
        # Testa uma imagem
        prever_imagem(model, class_names, IMAGEM_PARA_TESTAR, transform_inferencia, device)
        
# ============================