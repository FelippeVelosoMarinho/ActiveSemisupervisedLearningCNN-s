import random
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

from dataset_utils import filter_valid_images, safer_purge, showMultipleImages

from config import (
    RAW_DATA_DIR, 
    BASE_SPLIT_DIR, 
    CAT_DIR, 
    DOG_DIR
)

if not RAW_DATA_DIR.exists():
    raise FileNotFoundError(f"Diretório de dados brutos não encontrado: {RAW_DATA_DIR}")

print("Iniciando a busca de caminhos de arquivos...")
cat_paths = list(CAT_DIR.rglob("*.jpg"))
dog_paths = list(DOG_DIR.rglob("*.jpg"))

cat_files = sorted(cat_paths)
dog_files = sorted(dog_paths)

print(f"\nArquivos brutos: Gatos: {len(cat_files):,} | Cachorros: {len(dog_files):,}")

random_dog_files = random.sample(dog_files, min(9, len(dog_files))) 

try:
    random_dog_imgs = [cv2.imread(str(img_file)) for img_file in random_dog_files]
    random_dog_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in random_dog_imgs if img is not None]
    
    if random_dog_imgs:
        titles = ['Dog ' + str(i+1) for i in range(len(random_dog_imgs))]
        print("\n--- Exibindo Amostra de Imagens Brutas ---")
        showMultipleImages(random_dog_imgs, titles, (10, 15), 3, 3)
    else:
        print("Aviso: Nenhuma imagem de cachorro pôde ser lida pelo cv2 para visualização.")
except Exception as e:
    print(f"Erro ao tentar carregar imagens para visualização: {e}")

print("\n--- Iniciando filtragem de imagens corrompidas (PIL) ---")

cat_files_valid = filter_valid_images(cat_files)
dog_files_valid = filter_valid_images(dog_files)

print("\n" + "="*50)
print(f"Resultado Final (PIL): \n")
print(f"✅ Gatos Válidos: {len(cat_files_valid):,} (Pulados: {len(cat_files) - len(cat_files_valid):,})")
print(f"✅ Cachorros Válidos: {len(dog_files_valid):,} (Pulados: {len(dog_files) - len(dog_files_valid):,})")
total_valid = len(cat_files_valid) + len(dog_files_valid)
print(f"✅ TOTAL VÁLIDO: {total_valid:,}")
print("="*50)

print("\n--- Iniciando Separação do Dataset (Split) ---")

labels_cat = [0] * len(cat_files_valid)
labels_dog = [1] * len(dog_files_valid)

X = np.array(cat_files_valid + dog_files_valid)
y = np.array(labels_cat + labels_dog)

if total_valid < 10:
    print("ERRO: Número insuficiente de arquivos válidos para o split.")
    exit()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cálculo: 0.10 (val) / 0.80 (temp) = 0.125
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

print(f"Total de dados VÁLIDOS: {len(X):,}")
print(f"  - Treino (70%): {len(X_train):,}")
print(f"  - Validação (10%): {len(X_val):,}")
print(f"  - Teste (20%): {len(X_test):,}")
print("-" * 30)
print(f"Distribuição de classes (Teste): {y_test.mean():.4f} (Ideal: ~0.5)")


BASE_SPLIT_DIR.mkdir(exist_ok=True)

DEST_DIRS = {
    'train': BASE_SPLIT_DIR / 'train',
    'validation': BASE_SPLIT_DIR / 'validation',
    'test': BASE_SPLIT_DIR / 'test'
}

for key, dest_dir in DEST_DIRS.items():
    (dest_dir / 'Cat').mkdir(parents=True, exist_ok=True)
    (dest_dir / 'Dog').mkdir(parents=True, exist_ok=True)


def move_files(file_list, dest_key):
    """Copia os arquivos para as pastas de destino mantendo a estrutura de classes."""
    dest_path = DEST_DIRS[dest_key]
    for file_path_str in file_list:
        file_path = Path(file_path_str)
        
        # Determina a classe pelo nome da pasta pai no caminho original (Cat ou Dog)
        class_name = file_path.parent.name 

        final_dest = dest_path / class_name / file_path.name
        
        # Usamos copy, pois o original deve permanecer para re-execuções
        shutil.copy(file_path, final_dest)
        
    print(f"✅ Arquivos de {dest_key} COPIADOS: {len(file_list):,}")

print("\n--- Copiando arquivos para a estrutura de diretórios de destino ---")
move_files(X_train, 'train')
move_files(X_val, 'validation')
move_files(X_test, 'test')

print("\n--- Iniciando Limpeza Final (Verificação de Corrompimento pelo TF) ---")
for folder_name in ['train', 'validation', 'test']:
    folder_path = BASE_SPLIT_DIR / folder_name
    safer_purge(folder_path)

print("\n--- LIMPEZA FINALIZADA. Os datasets estão prontos para o Keras. ---")