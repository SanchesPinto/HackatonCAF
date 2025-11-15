# Importações (quase as mesmas, adicionei ImageFolder)
from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder 
import os
import cv2
import numpy as np
import random 
import torchvision.models as models


# ============================
# 1. Definição dos Pipelines (Transforms)
# ============================

# Tamanho que a ResNet espera (pode ser 128x128, 224x224, etc.)
# Por enquanto em 128x128 para acelerar testes mas pode ser aumentado posteriormente
IMG_SIZE = 128

# Médias e desvios-padrão do ImageNet (correto para ResNet pré-treinada)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- Pipeline de TREINO (Com Data Augmentation) ---
transform_treino = transforms.Compose([
    # Recorta uma área aleatória da imagem e redimensiona para IMG_SIZE
    # Isso simula zoom e mudança de enquadramento.
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), 
    
    # Vira a imagem horizontalmente com 50% de chance
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Rotaciona a imagem aleatoriamente em até 15 graus
    # Simula a peça sendo fotografada em ângulos ligeiramente diferentes
    transforms.RandomRotation(degrees=15),
    
    # Altera aleatoriamente o brilho, contraste e saturação
    # Simula diferentes condições de iluminação
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    # Converte para Tensor
    transforms.ToTensor(),
    
    # Normaliza
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# --- Pipeline de VALIDAÇÃO e TESTE (Sem Augmentation) ---
# Aqui, não queremos aleatoriedade. Queremos apenas formatar a imagem.
transform_validacao_teste = transforms.Compose([
    # Apenas redimensiona para o tamanho exato
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    
    # Converte para Tensor
    transforms.ToTensor(),
    
    # Normaliza
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


# ============================
# 2. Modelo novo para classificação multi-classe
# ============================

class Rede(nn.Module):
    def __init__(self, cnn_output_size=512, classifier_dropout=0.5, num_classes=5):
        """
        
        Args:
            cnn_output_size (int): O tamanho do vetor de features da CNN 
                                   (512 para ResNet18).
            classifier_dropout (float): A probabilidade de dropout.
        """
        super(Rede, self).__init__()
        
        # --- 1. Encoder CNN ---
        # Lógica de carregar o ResNet18 pré-treinado e congelar 
        # os pesos para transfer learning.
        resnet = models.resnet18(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad = False
            
        # A camada 'fc' original da ResNet18 entra com 512 features.
        # Ao substituí-la por nn.Identity(), o modelo 'resnet' 
        # agora solta o vetor de features de 512 dimensões.
        resnet.fc = nn.Identity()
        self.cnn_extractor = resnet
        
        
        # --- 2. Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size, 128), # Recebe as 512 features
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes) # Saída alterada para a qtd de classes
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): O tensor de entrada com shape (B, C, H, W)
        """
        features = self.cnn_extractor(x)
        output = self.classifier(features)
        
        # A saída são os LOGITS. 
        return output

# ============================
# 3. Função para calcular acurácia e loss
# ============================


def avaliar_modelo(model, dataloader, criterion, device):
    model.eval() 
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            
            # CrossEntropyLoss espera rótulos como Long (inteiros) e shape (B,)
            y_batch = y_batch.to(device).long() 

            outputs = model(X_batch) # Saída são logits (B, num_classes)
            loss = criterion(outputs, y_batch) # CrossEntropyLoss compara (B, N) com (B,)
            total_loss += loss.item()
            
            # 1. 'outputs' tem shape (B, num_classes)
            # 2. torch.max(outputs.data, 1) retorna (valores_max, indices_max)
            # índices = predicted classes
            _, predicted = torch.max(outputs.data, 1)
            # (Alternativa: predicted = torch.argmax(outputs, dim=1))

            total += y_batch.size(0)
            # 3. Compara os índices previstos (predicted) com os rótulos reais (y_batch)
            correct += (predicted == y_batch).sum().item() 
    
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc
# ============================
# 4. Loop de treino
# ============================

def train_looping(model, train_loader, val_loader, criterion, writer, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    
    model.to(device) 

    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            # Rótulos como Long (inteiros) e shape (B,)
            y_batch = y_batch.to(device).long() 
            
            optimizer.zero_grad()
            outputs = model(X_batch) # Logits (B, num_classes)
            loss = criterion(outputs, y_batch) # CrossEntropyLoss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Cálculo de Acurácia Multi-Classe
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        train_loss_avg = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_loss, val_acc = avaliar_modelo(model, val_loader, criterion, device)
        print(f"Época {epoch+1:02d}, "
              f"Loss Treino: {train_loss_avg:.4f}, Acurácia Treino: {train_acc*100:.2f}%, "
              f"Loss Val: {val_loss:.4f}, Acurácia Val: {val_acc*100:.2f}%")

        writer.add_scalars("Losses", {"Train": train_loss_avg, "Validation": val_loss}, epoch)
        writer.add_scalars("Accuracies", {"Train": train_acc, "Validation": val_acc}, epoch)

    os.makedirs("models", exist_ok=True)
    model_path = "models/modelo2.pth" 

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, model_path)

    print(f"💾 Modelo salvo em: {model_path}")
    return model 

# ============================
# 5. Main 
# ============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- 1. Caminhos para os diretórios ---

    base_data_dir = "Datasets/" 
    train_dir = os.path.join(base_data_dir, "treino")
    val_dir = os.path.join(base_data_dir, "validacao")
    test_dir = os.path.join(base_data_dir, "teste")
    
    # --- 2. Carregamento dos Datasets ---
    
    try:
        train_dataset = ImageFolder(root=train_dir, transform=transform_treino)
        print(f"Dataset de TREINO carregado: {len(train_dataset)} imagens.")
        print(f"Classes de Treino: {train_dataset.classes}")
        
        val_dataset = ImageFolder(root=val_dir, transform=transform_validacao_teste)
        print(f"Dataset de VALIDAÇÃO carregado: {len(val_dataset)} imagens.")
        
        test_dataset = ImageFolder(root=test_dir, transform=transform_validacao_teste)
        print(f"Dataset de TESTE carregado: {len(test_dataset)} imagens.")
        
    except Exception as e:
        print(f"Erro ao carregar datasets. Verifique os caminhos e a estrutura de pastas.")
        print(f"Diretório de treino esperado: {train_dir}")
        print(f"Erro: {e}")
        return

    # Pega o número de classes do dataset de treino
    num_classes = len(train_dataset.classes)
    if num_classes <= 1:
        print("ERRO: São necessárias pelo menos 2 classes.")
        return
    print(f"Número de classes detectado: {num_classes}")
    

    BATCH_SIZE = 16 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    # --- 4. O resto do script (Modelo, Treino, Avaliação) ---

    writer = SummaryWriter(log_dir=f"runs/teste2{BATCH_SIZE}") 
    
    model = Rede(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss() 

    trained_model = train_looping(model, train_loader, val_loader, criterion, writer, device)

    print(f"\n=== Avaliação final no conjunto de teste ===")
    test_loss, test_acc = avaliar_modelo(trained_model, test_loader, criterion, device)
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc*100:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()