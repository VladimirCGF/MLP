#!/usr/bin/env python
# Script para verificar a função de ativação no arquivo mlp.py

import os

# Ler arquivo
with open('mlp.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Procurar pela definição da função de ativação
if 'self.relu1 = nn.Tanh()' in content:
    print("[OK] Ativação configurada como: Tanh")
elif 'self.relu1 = nn.ReLU()' in content:
    print("[ERROR] Ativação ainda é: ReLU")
else:
    print("[ERROR] Nenhuma ativação encontrada")

# Procurar no relatório - versão corrigida
if 'activation = "Tanh"' in content or "activation = 'Tanh'" in content:
    print("[OK] Relatório configurado para: Tanh")
elif 'activation = "ReLU"' in content or "activation = 'ReLU'" in content:
    print("[ERROR] Relatório ainda mostra: ReLU")

# Procurar por linhas específicas
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'activation = ' in line and ('Tanh' in line or 'ReLU' in line):
        print(f"\nLinha {i+1}: {line.strip()}")

# Contar ocorrências de variável relu1 vs ativações reais
relu1_count = content.count('self.relu1')
tanh_activation = content.count('nn.Tanh()')
relu_activation = content.count('nn.ReLU()')

print(f"\nEstatísticas de ativação:")
print(f"  • nn.Tanh(): {tanh_activation} vez(es)")
print(f"  • nn.ReLU(): {relu_activation} vez(es)")
print(f"  • self.relu1 (variável): {relu1_count} vez(es)")

