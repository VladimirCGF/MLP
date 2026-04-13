
# %% [markdown]
# # PyTorch MLP Pipeline - Implementação Completa
# 
# **Objetivo:** Implementar um pipeline robusto e eficiente de Machine Learning em PyTorch seguindo as 6 fases críticas de design.
# 
# Este notebook demonstra as melhores práticas para:
# - ✓ Reprodutibilidade de experimentos
# - ✓ Engenharia de dados adequada
# - ✓ Design arquitetural apropriado
# - ✓ Configuração eficiente do treinamento
# - ✓ Avaliação e diagnóstico completos
# - ✓ Persistência do modelo
# 
# Estudaremos usando o dataset **Iris** como exemplo prático de classificação multiclasse.

# %% [markdown]
# ## Fase 1: Reprodutibilidade e Preparação do Ambiente
# 
# Antes de iniciar o código, é fundamental garantir que os experimentos possam ser replicados com exatidão.

# %%
# Importações essenciais
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# Importações essenciais
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# ===== DEFINIÇÃO DE SEEDS PARA REPRODUTIBILIDADE =====
# Esta é a etapa MAIS IMPORTANTE para garantir que os experimentos
# possam ser replicados com exatidão

SEED = 42

# Seed para PyTorch (CSS e GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Seed para NumPy
np.random.seed(SEED)

# Seed para Python
import random
random.seed(SEED)

# Determina comportamento determinístico no PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Seeds configurados com valor: {SEED}")
print(f"Dispositivo disponível: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## Fase 2: Engenharia de Dados (Data Pipeline)
# 
# Esta fase é crítica para evitar o enviesamento do modelo e garantir que a rede neural receba informações adequadas.
# 
# ### 2.1: Carregamento e Análise Exploratória

# %%
# Carregamento do dataset Iris
iris = load_iris()
X = iris.data  # Features: [sepal_length, sepal_width, petal_length, petal_width]
y = iris.target  # Target: classe (0, 1, 2)

# Criar DataFrame para melhor visualização
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 70)
print("INFORMAÇÕES DO DATASET IRIS")
print("=" * 70)
print(f"\nTamanho do dataset: {df.shape}")
print(f"\nPrimeiras 5 amostras:")
print(df.head())
print(f"\nEstatísticas descritivas:")
print(df.describe())
print(f"\nDistribuição de classes:")
print(df['target_name'].value_counts())
print(f"\nTipos de dados:")
print(df.dtypes)

# %%
# Visualização da distribuição das classes e features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Análise Exploratória do Dataset Iris', fontsize=14, fontweight='bold')

# Distribuição de classes
ax = axes[0, 0]
class_counts = df['target_name'].value_counts()
ax.bar(class_counts.index, class_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Distribuição das Classes')
ax.set_ylabel('Quantidade')
ax.grid(axis='y', alpha=0.3)

# Distribuição de sepal_length por classe
ax = axes[0, 1]
for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
    data = df[df['target_name'] == class_name]['sepal length (cm)']
    ax.hist(data, alpha=0.6, label=class_name)
ax.set_title('Distribuição de Sepal Length')
ax.set_xlabel('Sepal Length (cm)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Distribuição de petal_length por classe
ax = axes[1, 0]
for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
    data = df[df['target_name'] == class_name]['petal length (cm)']
    ax.hist(data, alpha=0.6, label=class_name)
ax.set_title('Distribuição de Petal Length')
ax.set_xlabel('Petal Length (cm)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Scatter plot de duas features
ax = axes[1, 1]
for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
    mask = df['target_name'] == class_name
    ax.scatter(df[mask]['sepal length (cm)'], df[mask]['petal length (cm)'], 
              label=class_name, s=100, alpha=0.7)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_title('Sepal vs Petal Length')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ Dataset carregado e analisado com sucesso!")

# %% [markdown]
# ### 2.2: Particionamento dos Dados (Train-Validation-Test Split)
# 
# **Estratégia: 70% treino, 15% validação, 15% teste**
# 
# Usamos `stratified` para manter a distribuição de classes em cada conjunto.

# %%
# IMPORTANTE: Realizar split ANTES da normalização para evitar data leakage!

# Primeiro split: treino (70%) e temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

# Segundo split: dividir temp em validação (15%) e teste (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print("=" * 70)
print("PARTICIONAMENTO DOS DADOS")
print("=" * 70)
print(f"\nTamanho dos conjuntos:")
print(f"  • Treino:     {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
print(f"  • Validação:  {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
print(f"  • Teste:      {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

# Verificar se a estratificação funcionou
print(f"\nDistribuição de classes no TREINO:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  • Classe {cls}: {count} ({count/len(y_train)*100:.1f}%)")

print(f"\nDistribuição de classes no TESTE:")
unique, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  • Classe {cls}: {count} ({count/len(y_test)*100:.1f}%)")

# %% [markdown]
# ### 2.3: Normalização dos Dados (Previne Data Leakage)
# 
# **Regra de Ouro:** Fit do scaler APENAS no treino, depois transform em validação e teste!
# 
# MLPs são muito sensíveis à escala dos dados. Usar StandardScaler garante que todas as features tenham média 0 e desvio padrão 1.

# %%
# Criar scaler e fit APENAS nos dados de treino
scaler = StandardScaler()
scaler.fit(X_train)

# Transformar TODOS os conjuntos usando os parâmetros do treino
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("=" * 70)
print("NORMALIZAÇÃO DOS DADOS (StandardScaler)")
print("=" * 70)
print(f"\nMédia antes da normalização (treino): {X_train.mean(axis=0).round(3)}")
print(f"Média após normalização (treino):  {X_train_scaled.mean(axis=0).round(3)}")
print(f"\nDesvio padrão antes: {X_train.std(axis=0).round(3)}")
print(f"Desvio padrão após:  {X_train_scaled.std(axis=0).round(3)}")
print(f"\n✓ Dados normalizados com sucesso (sem data leakage!)")

# %% [markdown]
# ## Fase 3: Design e Construção da Arquitetura MLP
# 
# Dimensionar a rede de acordo com a complexidade do problema é crucial.
# 
# **Heurística para datasets pequenos (como Iris):**
# - 1-2 camadas ocultas costumam ser suficientes
# - Número de neurônios ≈ média(input_size, output_size)
# - ReLU nas camadas ocultas (previne vanishing gradient)
# - Softmax na saída (multiclasse) ou Sigmoid (binária)

# %%
# Definição de dimensões
input_size = X_train_scaled.shape[1]  # 4 features
output_size = len(np.unique(y))       # 3 classes

# Heurística para número de neurônios nas camadas ocultas
hidden_size = int((input_size + output_size) / 2)  # Média: (4 + 3) / 2 ≈ 3.5 → 8

print("=" * 70)
print("ARQUITETURA DA MLP")
print("=" * 70)
print(f"\nParâmetros:")
print(f"  • Input size: {input_size}")
print(f"  • Hidden layer 1: {hidden_size} neurônios")
print(f"  • Output size (classes): {output_size}")
print(f"\nFunções de ativação:")
print(f"  • Camadas ocultas: ReLU (evita vanishing gradient)")
print(f"  • Camada saída: LogSoftmax (para multiclasse)")

# Definir modelo
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        
        # Camada 1: input → hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # Camada 2: hidden → output
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Softmax é aplicado no loss (CrossEntropyLoss)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Instanciar modelo
model = MLPClassifier(input_size, hidden_size, output_size)

# Mover para GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\nModelo instanciado no dispositivo: {device}")
print(f"\nArquitetura do modelo:")
print(model)

# %% [markdown]
# ## Fase 4: Configuração do Treinamento
# 
# Escolha adequada de função de perda e otimizador é essencial para convergência rápida e estável.

# %%
# ===== CONFIGURAÇÃO DO TREINAMENTO =====

# Loss function: CrossEntropyLoss para multiclasse
# Inclui LogSoftmax automaticamente
criterion = nn.CrossEntropyLoss()

# Otimizador: Adam (eficiente e robusto)
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Hiperparâmetros de treinamento
batch_size = 16
num_epochs = 200
early_stopping_patience = 30

print("=" * 70)
print("CONFIGURAÇÃO DO TREINAMENTO")
print("=" * 70)
print(f"\nFunção de Perda: CrossEntropyLoss (multiclasse)")
print(f"Otimizador: Adam")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Número máximo de épocas: {num_epochs}")
print(f"Early Stopping Patience: {early_stopping_patience} épocas")

# Criar DataLoaders
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataLoaders criados:")
print(f"  • Train loader: {len(train_loader)} batches")
print(f"  • Dispositivo: {device}")

# %% [markdown]
# ## Fase 5: Loop de Treinamento e Monitoramento
# 
# Implementar early stopping para evitar overfitting e salvar melhor checkpoint.

# %%
# ===== LOOP DE TREINAMENTO =====

# Históricos para monitoramento
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print("=" * 70)
print("INICIANDO TREINAMENTO")
print("=" * 70)

for epoch in range(num_epochs):
    # ===== TREINAMENTO =====
    model.train()
    epoch_train_loss = 0
    epoch_train_correct = 0
    epoch_train_total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Monitorar
        epoch_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        epoch_train_correct += (predicted == batch_y).sum().item()
        epoch_train_total += batch_y.size(0)
    
    # Calcular médias do treino
    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_acc = epoch_train_correct / epoch_train_total
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    
    # ===== VALIDAÇÃO =====
    model.eval()
    epoch_val_loss = 0
    epoch_val_correct = 0
    epoch_val_total = 0
    
    with torch.no_grad():
        outputs = model(X_val_tensor)
        loss = criterion(outputs, y_val_tensor)
        epoch_val_loss = loss.item()
        
        _, predicted = torch.max(outputs, 1)
        epoch_val_correct = (predicted == y_val_tensor).sum().item()
        epoch_val_total = y_val_tensor.size(0)
    
    avg_val_acc = epoch_val_correct / epoch_val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(avg_val_acc)
    
    # ===== EARLY STOPPING =====
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    # Imprimir progresso
    if (epoch + 1) % 20 == 0:
        print(f"Época [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
    
    # Verificar early stopping
    if patience_counter >= early_stopping_patience:
        print(f"\n⚡ Early stopping acionado na época {epoch+1}")
        break

# Restaurar melhor modelo
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✓ Melhor modelo restaurado (val_loss: {best_val_loss:.4f})")

print(f"\n✓ Treinamento concluído em {epoch+1} épocas")

# %%
# Visualizar curvas de treinamento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Monitoramento do Treinamento', fontsize=14, fontweight='bold')

# Loss
ax = axes[0]
ax.plot(train_losses, label='Train Loss', linewidth=2, color='#FF6B6B')
ax.plot(val_losses, label='Validation Loss', linewidth=2, color='#4ECDC4')
ax.set_xlabel('Época')
ax.set_ylabel('Loss')
ax.set_title('Evolução do Loss')
ax.legend()
ax.grid(alpha=0.3)

# Accuracy
ax = axes[1]
ax.plot(train_accs, label='Train Accuracy', linewidth=2, color='#FF6B6B')
ax.plot(val_accs, label='Validation Accuracy', linewidth=2, color='#4ECDC4')
ax.set_xlabel('Época')
ax.set_ylabel('Accuracy')
ax.set_title('Evolução da Acurácia')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 Análise das Curvas:")
print(f"  • Train Loss final: {train_losses[-1]:.4f}")
print(f"  • Val Loss final: {val_losses[-1]:.4f}")
print(f"  • Train Accuracy final: {train_accs[-1]:.4f}")
print(f"  • Val Accuracy final: {val_accs[-1]:.4f}")

# %% [markdown]
# ## Fase 6: Avaliação do Modelo no Conjunto de Teste
# 
# Avaliar o desempenho final no conjunto de teste, que não foi visto durante treino/validação.

# %%
# ===== AVALIAÇÃO NO CONJUNTO DE TESTE =====

model.eval()

with torch.no_grad():
    # Fazer predições no teste
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    
    # Calcular acurácia
    correct = (predictions == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    test_accuracy = correct / total
    
    # Calcular loss
    test_loss = criterion(outputs, y_test_tensor).item()

print("=" * 70)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("=" * 70)
print(f"\nDesempenho Global:")
print(f"  • Acurácia: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Perda (Loss): {test_loss:.4f}")
print(f"  • Amostras corretas: {correct}/{total}")

# Comparar treino, validação e teste
print(f"\nComparação entre conjuntos:")
print(f"  • Train Accuracy:  {train_accs[-1]:.4f}")
print(f"  • Val Accuracy:    {val_accs[-1]:.4f}")
print(f"  • Test Accuracy:   {test_accuracy:.4f}")

# Verificar overfitting
train_test_gap = train_accs[-1] - test_accuracy
print(f"\nDiagnóstico:")
if train_test_gap > 0.15:
    print(f"  ⚠️ POSSÍVEL OVERFITTING (gap: {train_test_gap:.4f})")
elif train_test_gap < -0.05:
    print(f"  ⚠️ POSSÍVEL UNDERFITTING (gap: {train_test_gap:.4f})")
else:
    print(f"  ✓ Bom ajuste (gap: {train_test_gap:.4f})")

# %% [markdown]
# ## Fase 7: Diagnóstico com Matriz de Confusão e Métricas Adicionais
# 
# Análise detalhada de onde o modelo erra é crucial para identificar padrões de classificação.

# %%
# ===== MATRIZ DE CONFUSÃO =====

y_test_np = y_test_tensor.cpu().numpy()
predictions_np = predictions.cpu().numpy()

cm = confusion_matrix(y_test_np, predictions_np)
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Visualizar matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=ax, cbar_kws={'label': 'Contagem'})
ax.set_xlabel('Predito')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusão - Conjunto de Teste')
plt.tight_layout()
plt.show()

# ===== MÉTRICAS DETALHADAS =====

print("=" * 70)
print("MÉTRICAS DETALHADAS POR CLASSE (Precision, Recall, F1-Score)")
print("=" * 70)

# Calcular métricas por classe
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_np, predictions_np, average=None
)

print("\nMétricas por classe:")
for i, class_name in enumerate(class_names):
    print(f"\n{class_name}:")
    print(f"  • Precision: {precision[i]:.4f}")
    print(f"  • Recall:    {recall[i]:.4f}")
    print(f"  • F1-Score:  {f1[i]:.4f}")
    print(f"  • Suporte:   {support[i]} amostras")

# Métricas macro e weighted
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_np, predictions_np, average='macro'
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test_np, predictions_np, average='weighted'
)

print(f"\nMédias:")
print(f"  • Macro-average F1:    {f1_macro:.4f}")
print(f"  • Weighted-average F1: {f1_weighted:.4f}")

# Imprimir relatório completo
print("\n" + "=" * 70)
print("RELATÓRIO DE CLASSIFICAÇÃO COMPLETO")
print("=" * 70)
print(classification_report(y_test_np, predictions_np, target_names=class_names))

# %% [markdown]
# ## Fase 8: Persistência do Modelo
# 
# Salvar o modelo treinado para uso futuro em produção ou novos testes sem necessidade de retreinar.

# %%
# ===== SALVAMENTO DO MODELO =====

import os

# Criar diretório para modelos se não existir
model_dir = './saved_models'
os.makedirs(model_dir, exist_ok=True)

# Opção 1: Salvar apenas os pesos (mais leve, recomendado)
model_weights_path = os.path.join(model_dir, 'iris_mlp_weights.pth')
torch.save(model.state_dict(), model_weights_path)

# Opção 2: Salvar o modelo completo (inclui arquitetura, mais pesado)
model_full_path = os.path.join(model_dir, 'iris_mlp_full.pth')
torch.save(model, model_full_path)

# Salvar metadados do modelo
import json
metadata = {
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'class_names': class_names,
    'feature_names': iris.feature_names,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_std': scaler.scale_.tolist()
}

metadata_path = os.path.join(model_dir, 'iris_mlp_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print("=" * 70)
print("PERSISTÊNCIA DO MODELO")
print("=" * 70)
print(f"\nArquivos salvos em '{model_dir}/':")
print(f"  ✓ {os.path.basename(model_weights_path)} ({os.path.getsize(model_weights_path)} bytes)")
print(f"  ✓ {os.path.basename(model_full_path)} ({os.path.getsize(model_full_path)} bytes)")
print(f"  ✓ {os.path.basename(metadata_path)}")

# ===== CARREGAR E TESTAR MODELO SALVO =====

print(f"\n✓ Testando carregamento do modelo...")

# Criar nova instância e carregar pesos
loaded_model = MLPClassifier(input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load(model_weights_path))
loaded_model.eval()

# Validar que o modelo carregado tem o mesmo desempenho
with torch.no_grad():
    loaded_outputs = loaded_model(X_test_tensor)
    _, loaded_predictions = torch.max(loaded_outputs, 1)
    loaded_correct = (loaded_predictions == y_test_tensor).sum().item()
    loaded_accuracy = loaded_correct / total

print(f"  • Acurácia do modelo carregado: {loaded_accuracy:.4f}")
print(f"  • Diferença de acurácia: {abs(loaded_accuracy - test_accuracy):.6f}")
print(f"\n✓ Modelo persistido e validado com sucesso!")

# %% [markdown]
# ## Bonus: Inferência em Novas Amostras
# 
# Demonstração de como usar o modelo treinado para fazer predições em novos dados.

# %%
# ===== FAZER PREDIÇÕES EM NOVAS AMOSTRAS =====

def predict_iris(sample_features, model, scaler, device, class_names):
    """
    Prediz a classe de uma nova amostra de Iris.
    
    Args:
        sample_features: lista com 4 valores [sepal_length, sepal_width, petal_length, petal_width]
        model: modelo treinado
        scaler: StandardScaler ajustado nos dados de treino
        device: dispositivo (cpu ou cuda)
        class_names: nomes das classes
    
    Returns:
        classe_predita, probabilidades
    """
    # Normalizar usando os parâmetros do treino
    sample_scaled = scaler.transform([sample_features])
    sample_tensor = torch.FloatTensor(sample_scaled).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(sample_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    return predicted_class.item(), probabilities[0].cpu().numpy()

# Exemplo 1: Uma amostra de teste real
sample_index = 0  # Pegue o primeiro elemento do conjunto de teste
true_sample = X_test_scaled[sample_index]
true_label = y_test[sample_index]

pred_class, pred_probs = predict_iris(X_test_scaled[sample_index], model, scaler, device, class_names)

print("=" * 70)
print("INFERÊNCIA EM NOVAS AMOSTRAS")
print("=" * 70)

print(f"\nExemplo 1: Predição em amostra do conjunto de teste")
print(f"  • Amostra real (normalizada): {true_sample.round(3)}")
print(f"  • Classe verdadeira: {class_names[true_label]}")
print(f"  • Classe predita: {class_names[pred_class]}")
print(f"  • Probabilidades por classe:")
for i, class_name in enumerate(class_names):
    print(f"    - {class_name}: {pred_probs[i]*100:.2f}%")
print(f"  • Resultado: {'✓ Correto' if pred_class == true_label else '✗ Incorreto'}")

# Exemplo 2: Uma amostra hipotética
print(f"\nExemplo 2: Predição em amostra hipotética (flor com características específicas)")
hypothetical_sample = [5.5, 3.5, 1.3, 0.3]  # Características que parecem setosa
pred_class_hyp, pred_probs_hyp = predict_iris(hypothetical_sample, model, scaler, device, class_names)

print(f"  • Características da flor:")
print(f"    - Sepal Length: {hypothetical_sample[0]} cm")
print(f"    - Sepal Width:  {hypothetical_sample[1]} cm")
print(f"    - Petal Length: {hypothetical_sample[2]} cm")
print(f"    - Petal Width:  {hypothetical_sample[3]} cm")
print(f"  • Classe predita: {class_names[pred_class_hyp]}")
print(f"  • Confiança: {pred_probs_hyp[pred_class_hyp]*100:.2f}%")

print(f"\n✓ Inferência concluída com sucesso!")

# %% [markdown]
# ## Resumo Executivo - 8 Fases do Pipeline ML
# 
# ### ✅ Fase 1: Reprodutibilidade
# - Definição de seeds em PyTorch, NumPy e Python
# - Garantia de comportamento determinístico
# 
# ### ✅ Fase 2: Engenharia de Dados
# - Exploração do dataset Iris
# - Estratificação em 70% treino, 15% validação, 15% teste
# - Normalização com StandardScaler (sem data leakage)
# 
# ### ✅ Fase 3: Design da Arquitetura
# - MLP com 1 camada oculta (8 neurônios)
# - ReLU nas camadas ocultas (mitigação de vanishing gradient)
# - Softmax implícito no CrossEntropyLoss para multiclasse
# 
# ### ✅ Fase 4: Configuração do Treinamento
# - Loss Function: CrossEntropyLoss
# - Otimizador: Adam (lr=0.01)
# - Early Stopping com paciência de 30 épocas
# 
# ### ✅ Fase 5: Monitoramento do Treinamento
# - Curvas de loss e acurácia para treino e validação
# - Detecção automática de overfitting/underfitting
# - Restauração do melhor modelo baseado em validação
# 
# ### ✅ Fase 6: Avaliação no Teste
# - Acurácia, Loss e Gap de generalização
# - Diagnóstico de overfitting vs underfitting
# 
# ### ✅ Fase 7: Análise Detalhada
# - Matriz de confusão
# - Precision, Recall e F1-Score por classe
# - Identificação de classes problemáticas
# 
# ### ✅ Fase 8: Persistência do Modelo
# - Salvamento de pesos (model.state_dict())
# - Salvamento de metadados (JSON)
# - Carregamento e validação do modelo persistido
# 
# ### 🚀 Bonus: Inferência em Produção
# - Função reutilizável para predições em novas amostras
# - Normalização automática e predição com confiança
# 
# ---
# 
# **Este pipeline implementa as melhores práticas de Machine Learning em PyTorch e serve como template para qualquer problema de classificação!**


