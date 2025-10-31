# Importações (quase as mesmas, adicionei ImageFolder)
from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder # <-- IMPORTANTE: Usaremos isso
import os
import cv2
import numpy as np
import random 
import torchvision.models as models
import pandas as pd




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
        
        # --- 1. O "Olho" (Encoder CNN) ---
        # A sua lógica de carregar o ResNet18 pré-treinado e congelar 
        # os pesos está perfeita para transfer learning.
        resnet = models.resnet18(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad = False
            
        # A camada 'fc' original da ResNet18 entra com 512 features.
        # Ao substituí-la por nn.Identity(), o modelo 'resnet' 
        # agora solta o vetor de features de 512 dimensões.
        resnet.fc = nn.Identity()
        self.cnn_extractor = resnet
        
        # --- 2. A "Memória" (LSTM) ---
        # REMOVIDA! Não precisamos mais dela.
        
        # --- 3. O "Juiz" (Classifier Head) ---
        # O classificador agora recebe a saída da CNN (cnn_output_size=512)
        # diretamente.
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
        
        # <-- MUDANÇA: A saída agora é (B, num_classes)
        # Esta saída são os LOGITS. 
        # NÃO aplique Softmax aqui, pois a CrossEntropyLoss fará isso.
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
            
            # <-- MUDANÇA: 
            # CrossEntropyLoss espera rótulos como Long (inteiros) e shape (B,)
            # Removemos .float() e .view(-1, 1)
            y_batch = y_batch.to(device).long() 

            outputs = model(X_batch) # Saída são logits (B, num_classes)
            loss = criterion(outputs, y_batch) # CrossEntropyLoss compara (B, N) com (B,)
            total_loss += loss.item()
            
            # <-- MUDANÇA: Cálculo de Acurácia para Multi-Classe
            # 1. 'outputs' tem shape (B, num_classes)
            # 2. torch.max(outputs.data, 1) retorna (valores_max, indices_max)
            #    Estamos interessados nos índices (a classe prevista).
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
    epochs = 11
    
    model.to(device) 

    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            # <-- MUDANÇA: Rótulos como Long (inteiros) e shape (B,)
            y_batch = y_batch.to(device).long() 
            
            optimizer.zero_grad()
            outputs = model(X_batch) # Logits (B, num_classes)
            loss = criterion(outputs, y_batch) # CrossEntropyLoss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # <-- MUDANÇA: Cálculo de Acurácia Multi-Classe
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
    model_path = "models/rna_etapa2_classificador_imagem.pth" # Mudei o nome do modelo

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

    # --- Carregar o Dataset (necessário aqui para pegar num_classes) ---
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_root_dir = "Datasets/DatasetFormatado"  
    try:
        full_dataset = ImageFolder(root=dataset_root_dir, transform=transform)
        print(f"Dataset carregado. Classes: {full_dataset.classes}")
        num_classes = len(full_dataset.classes) # <-- PEGA O NÚMERO DE CLASSES
        if num_classes <= 1:
            print("ERRO: São necessárias pelo menos 2 classes para CrossEntropyLoss.")
            return
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return

    # --- Divisão 80/10/10 ---
    total_len = len(full_dataset) 
    train_len = int(0.8 * total_len); val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len 
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    print(f"Total: {total_len}, Treino: {train_len}, Val: {val_len}, Teste: {test_len}")

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count() // 2 or 1) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    
    # --- Fim da configuração do Dataset ---

    writer = SummaryWriter(log_dir="runs/etapa2_multiclasse_experiment") 
    
    # <-- MUDANÇA: Passa 'num_classes' para o modelo
    model = Rede(num_classes=num_classes)
    
    # <-- MUDANÇA: Nova função de perda
    criterion = nn.CrossEntropyLoss() 

    trained_model = train_looping(model, train_loader, val_loader, criterion, writer, device)

    print(f"\n=== Avaliação final no conjunto de teste ===")
    test_loss, test_acc = avaliar_modelo(trained_model, test_loader, criterion, device)
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc*100:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()