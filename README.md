## 🏆 Vencedor (1º Lugar) | Classificador de Peças para Chatbot de Atendimento (Hackathon CAF Máquinas)

Este projeto foi o **vencedor do 1º lugar** no hackathon da **CAF Máquinas** em parceria com a **SECComp**. O desafio era criar soluções inovadoras para otimizar o atendimento ao cliente.

### O Problema

Identificamos um gargalo crítico no processo de venda de peças de reposição: a **dificuldade dos clientes em comunicar o nome técnico** das peças que necessitavam. Isso causava longos tempos de atendimento, dependência de atendentes experientes e potencial para erros na compra.

### A Solução

Desenvolvemos uma solução ponta-a-ponta que **automatiza a identificação de peças usando Visão Computacional integrada a um chatbot**.

1.  **Interface de Atendimento (Chatbot):** Criamos um fluxo de conversa no **ManyChat** (implantado no Instagram como PoC, mas adaptável ao WhatsApp). O bot guia o cliente e, quando ele não sabe o nome da peça, solicita uma foto.
2.  **Cérebro (Modelo de IA):** A foto é enviada para uma **API de Deep Learning (PyTorch)**. Este modelo classifica a imagem e retorna o nome técnico correto da peça.
3.  **Resultado:** O chatbot recebe o nome da peça (ex: "CARACOL MONTADO 8") e informa instantaneamente ao cliente, agilizando o pedido.

Como Prova de Conceito, o escopo foi focado em um único componente: o **"Conjunto de Moagem" do Moedor CAF 8**, composto por 5 peças principais.



### Desafio Técnico e Estratégia

O principal desafio foi a **ausência total de um dataset** de imagens das peças. Para contornar isso:

1.  **Geração Sintética de Dados:** Usamos IA para gerar imagens-base das 5 peças.
2.  **Data Augmentation Intensivo:** Aplicamos um pipeline robusto de transformações (`RandomRotation`, `ColorJitter`, `RandomResizedCrop`) para simular as diferentes condições de fotos que um cliente poderia enviar (ângulos, iluminação, zoom).
3.  **Arquitetura:** Usamos **Transfer Learning** com uma **ResNet18** pré-treinada, congelando o *backbone* e treinando apenas um classificador customizado.

### Stack de Tecnologias

* **Inteligência Artificial:** PyTorch, Torchvision (ResNet18), TensorBoard
* **Backend (API do Modelo):** FastAPI, Uvicorn
* **Chatbot (Frontend):** ManyChat
* **Deploy (Demo):** Instagram (via ManyChat), Ngrok (para expor a API ao chatbot)

### Resultados

* **🏆 Vencedor do Hackathon (1º Lugar):** A solução foi reconhecida pela sua viabilidade técnica, impacto direto no negócio e solução criativa para um problema real de atendimento.
* **🎯 Performance do Modelo:** O modelo alcançou **90% de acurácia** no conjunto de teste, validando a eficácia da abordagem, mesmo partindo de dados 100% sintéticos.
* **🚀 Demonstração:** O fluxo completo (Cliente -> Chatbot -> API -> Modelo -> Chatbot -> Cliente) foi demonstrado com sucesso.