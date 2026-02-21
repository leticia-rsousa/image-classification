## Image Classification
**Descrição Geral** 📄<br>
Este projeto implementa um **modelo de Deep Learning baseado em Redes Neurais Convolucionais (CNN)** para classificar imagens do dataset **CIFAR-10** utilizando **PyTorch**. O pipeline inclui **carregamento dos dados, pré-processamento, definição da arquitetura, treinamento, avaliação por classe e inferência em imagens externas**, demonstrando o fluxo completo de um modelo de visão computacional.

---
**Objetivo** 🎯 <br> 
Desenvolver um modelo de **classificação de imagens** capaz de reconhecer 10 categorias do CIFAR-10, validando o desempenho da rede e permitindo realizar predições em imagens reais após o treinamento.

---
**Tecnologias Utilizadas** 💻 <br>
* ***Python*** - linguagem principal.
* ***PyTorch*** - construção, treinamento e inferência do modelo.
* ***Torchvision*** - carregamento do CIFAR-10 e transformações.
* ***Torchsummary*** - resumo da arquitetura.
* ***Matplotlib*** - visualização de imagens.
* ***PIL (Pillow)*** - leitura de imagens externas.

---
**Arquitetura e Estrutura do Código** 🧱 <br><br>
***1. Seleção Automática do Dispositivo*** <br>
O código identifica automaticamente se há **GPU CUDA, MPS (Mac)** ou **CPU**, garantindo melhor desempenho no treinamento.

***2. Pré-processamento e Dataset*** <br>
O CIFAR-10 é carregado com normalização e convertido para tensores.
São definidos dois DataLoaders:
* ***loader_treino*** - dados embaralhados
* ***loader_teste*** - dados em ordem fixa para avaliação <br><br>
As classes são: <br>
plane, car, bird, cat, deer, dog, frog, horse, ship, truck.

***3. Modelo CNN – Arquitetura Inspirada na LeNet*** <br>
A rede inclui:
* ***Conv2D → ReLU → MaxPool*** 
* ***Conv2D → ReLU → MaxPool***
* ***Flatten***
* ***FC (120)*** 
* ***FC (84)***
* ***FC (10 classes)*** <br><br>
A arquitetura é exibida com **torchsummary**, detalhando número de parâmetros e camadas.

***4. Treinamento do Modelo*** <br>
O processo inclui:
* ***Forward pass*** 
* ***Cálculo da loss (CrossEntropy)***
* ***Backpropagation***
* ***Atualização dos pesos com Adam*** 
* ***Média de loss por época***
* ***Avaliação automática no conjunto de teste a cada época*** <br><br>
Ao fim de cada época, o script exibe: <br>
```Epoch [x/10], Erro em Treino: X.XXXX, Acurácia em Teste: XX.XX %```

***5. Avaliação Final por Classe*** <br>
Após o treinamento completo, o código calcula:
* ***Acurácia geral do modelo*** 
* ***Acurácia por classe individual*** <br><br>
Permitindo verificar quais categorias possuem maior ou menor desempenho.

***6. Salvamento e Carregamento do Modelo*** <br>
O modelo treinado é salvo em: <br>
```modelo_mp7.pth``` <br><br>
E posteriormente recarregado para uso em inferência.

***7. Inferência em Imagens Externas*** <br>
O projeto inclui uma função dedicada: <br>
```ia_classifica_imagem(image_path, model)``` <br><br>
Que:
* ***Carrega a imagem com PIL*** 
* ***Redimensiona para 32×32***
* ***Normaliza***
* ***Gera predição*** 
* ***Exibe a imagem com título contendo:*** classe prevista + confiança da predição (%)

---
**Conceitos e Funcionalidades Demonstradas** 🔍 <br><br>
✅ ***Carregamento e pré-processamento de datasets com Torchvision*** <br>
✅ ***Construção de CNNs com PyTorch*** <br>
✅ ***Treinamento completo com backpropagation*** <br>
✅ ***Avaliação quantitativa por classe*** <br>
✅ ***Salvamento e reutilização de modelos*** <br>
✅ ***Inferência real com imagens externas*** <br>

---
**Como Executar o Projeto** ▶️ <br><br>
***1. Instale as dependências (recomendado via requirements.txt):*** <br>
```pip install -r requirements.txt```

***2. Execute o script principal:*** <br>
```python image_classification.py```

***3. Adicione suas imagens para inferência e chame:*** <br>
```
ia_classifica_imagem("sua_imagem.jpg", model_carregado)
```

---
**Conclusão** 📌 <br>
O projeto demonstra um pipeline completo de **classificação de imagens com CNN**, passando por todas as etapas essenciais: preparação dos dados, definição da arquitetura, treinamento, avaliação e inferência. Ele consolida conceitos fundamentais de modelos convolucionais aplicados à visão computacional, utilizando PyTorch de forma prática e organizada.
