# Imports essenciais
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from torchsummary import summary

# Seleção automática do dispositivo de execução
if torch.cuda.is_available():
  device = torch.device("cuda")
  print("Dispositivo selecionado: GPU NVIDIA (CUDA)")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
  print("Dispositivo selecionado: GPU Apple (MPS)")
else:
  device = torch.device("cpu")
  print("Dispositivo selecionado: CPU")

print(f"Usando dispositivo: {device}")

# Hiperparâmetros
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Transformações padrão para CIFAR-10
transformacoes = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carregamento dos datasets
dataset_treino = torchvision.datasets.CIFAR10(
    root='./dados', train=True, download=True, transform=transformacoes)

dataset_teste = torchvision.datasets.CIFAR10(
    root='./dados', train=False, download=True, transform=transformacoes)

# DataLoaders
loader_treino = torch.utils.data.DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
loader_teste = torch.utils.data.DataLoader(dataset_teste, batch_size=batch_size, shuffle=False)

# Classes CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Número de imagens de treino: {len(dataset_treino)}")
print(f"Número de imagens de teste: {len(dataset_teste)}")
print(f"Número de lotes de treino: {len(loader_treino)}")
print(f"Número de lotes de teste: {len(loader_teste)}")

# Exibição de imagens
def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

dataiter = iter(loader_treino)
images, labels = next(dataiter)

print("Amostra de imagens de treino:")
imshow(torchvision.utils.make_grid(images[:4]))
print('Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Arquitetura CNN estilo LeNet para CIFAR-10
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # Camadas totalmente conectadas
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Exibição da arquitetura
modelo = ConvNet().to("cpu")
print("Arquitetura do Modelo:")
print(modelo)

summary(modelo, (3, 32, 32), device="cpu")

# Modelo no dispositivo final
modelo = ConvNet().to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

# Treinamento 
print("\nIniciando o treinamento...\n")
n_total_steps = (len(loader_treino))

for epoch in range(num_epochs):
  modelo.train()
  running_loss = 0.0

  # Itera sobre os batches do conjunto de treino
  for i, (images, labels) in enumerate(loader_treino):

    # Move os tensores (imagens e rótulos) para o dispositivo (CPU ou GPU), próximos do modelo
    images = images.to(device)
    labels = labels.to(device)

    # Passagem para frente (forward)
    # Aqui ocorre a previsão do modelo
    outputs = modelo(images)

    # Calcula o erro do modelo
    loss = criterion(outputs, labels)

    # Zera os gradientes acumulados de iterações anteriores
    optimizer.zero_grad()

    # Calcula os gradientes via backpropagation
    loss.backward()

    # Atualiza os pesos do modelo
    optimizer.step()

    # Soma o valor da perda para cálculo médio posterior
    running_loss += loss.item()

  # Após cada época, avalia o modelo no conjunto de teste (validação)
  # Coloca o modelo em modo de avaliação
  modelo.eval()

  # Desativa o cálculo de gradientes para economizar memória e tempo
  with torch.no_grad():

    n_correct = 0     # Contador de acertos
    n_samples = 0     # Contador de amostras

    # Loop sobre o conjunto de teste
    for val_images, val_labels in loader_teste:

      # Move imagens e rótulos para o dispositivo
      val_images = val_images.to(device)
      val_labels = val_labels.to(device)
      
      # Faz a interferência no conjunto de teste
      val_outputs = modelo(val_images)
      
      # torch.max retorna (valor, índice) -> pegamos o índice da classe prevista
      _, predicted = torch.max(val_outputs.data, 1)
      
      # Incrementa o total de amostras
      n_samples += val_labels.size(0)
      
      # Incrementa o contador de acertos
      n_correct += (predicted == val_labels).sum().item()

  # Calcula a acurácia e a perda média da época
  acc = 100.0 * n_correct / n_samples
  avg_loss = running_loss / n_total_steps

  # Exibe métricas de desempenho para a época atual
  print(f'Epoch [{epoch+1}/{num_epochs}], Erro em Treino: {avg_loss:.4f}, Acurácia em Teste: {acc:.2f} %')
print("\nTreinamento finalizado.\n")

# Avaliação no conjunto de teste
modelo.eval()
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  n_class_correct = [0 for _ in range(10)]
  n_class_samples = [0 for _ in range(10)]

  for images, labels in loader_teste:
    images = images.to(device)
    labels = labels.to(device)
    outputs = modelo(images)
    _, predicted = torch.max(outputs, 1)

    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

    for i in range(len(labels)):
      label = labels[i]
      pred = predicted[i]
      if label == pred:
        n_class_correct[label] += 1
      n_class_samples[label] += 1

  acc_geral = 100.0 * n_correct / n_samples
  print(f'Acurácia geral do modelo na base de teste: {acc_geral:.2f} %')
  print("-" * 30)

  for i in range(10):
    if n_class_samples[i] > 0:
      acc_classe = 100 * n_class_correct[i] / n_class_samples[i]
      print(f'Acurácia da classe {classes[i]}: {acc_classe:.2f} %')
    else:
      print(f'Acurácia da classe {classes[i]}: N/A')

# Salvamento do modelo
PATH = './modelo_mp7.pth'
torch.save(modelo.state_dict(), PATH)
print(f'Modelo salvo em: {PATH}')

# Carregamento para inferência
model_carregado = ConvNet().to(device)
model_carregado.load_state_dict(torch.load(PATH))
model_carregado.eval()

# Transformação para imagens externas
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Função de classificação para deploy
def ia_classifica_imagem(image_path, model):
  try:
    img_pil = Image.open(image_path).convert('RGB')
  except Exception as e:
    print(f"Erro ao carregar imagem {image_path}: {e}")
    return

  img_tensor = inference_transform(img_pil)
  img_tensor = img_tensor.unsqueeze(0).to(device)

  model.eval()
  with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted_idx = torch.max(outputs, 1)

  classe_predita = classes[predicted_idx.item()]
  confianca = torch.max(probabilities).item() * 100

  plt.imshow(img_pil)
  plt.title(rf'Classe Prevista: $\bf{{{classe_predita}}}$ (Confiança: {confianca:.2f}%)')
  plt.axis('off')
  plt.show()

# Execução das inferências
ia_classifica_imagem("imagem1.jpg", model_carregado)
ia_classifica_imagem("imagem2.jpg", model_carregado)
ia_classifica_imagem("imagem3.png", model_carregado)
ia_classifica_imagem("imagem4.jpg", model_carregado)
ia_classifica_imagem("imagem5.jpg", model_carregado)
ia_classifica_imagem("imagem6.jpg", model_carregado)
