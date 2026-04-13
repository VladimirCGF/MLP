import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ler o valor de hidden_size do arquivo
with open('mlp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if 'hidden_size =' in line and '# 1 camada' in line:
            print(f"Linha encontrada: {line.strip()}")
            # Extrair o valor
            value = line.split('=')[1].split('#')[0].strip()
            print(f"Valor de hidden_size: {value}")
            break
