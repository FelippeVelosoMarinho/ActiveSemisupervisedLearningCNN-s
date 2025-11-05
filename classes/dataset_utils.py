import os
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ==============================================================================
# Visualização de Imagens
# ==============================================================================

def showSingleImage(img, title, size):
    """Exibe uma única imagem."""
    fig, axis = plt.subplots(figsize=size)
    axis.imshow(img, 'gray')
    axis.set_title(title, fontdict={'fontsize': 20, 'fontweight': 'medium'})
    plt.show()
    
def showMultipleImages(imgsArray, titlesArray, size, x, y):
    """Exibe múltiplas imagens em uma grade (x colunas, y linhas)."""
    if x < 1 or y < 1:
        print("ERRO: X e Y não podem ser zero ou abaixo de zero!")
        return
    
    if x == 1 and y == 1:
        showSingleImage(imgsArray[0], titlesArray[0], size) # Ajuste para usar o primeiro elemento
        return

    # Lógica para mostrar imagens em uma grade
    fig, axis = plt.subplots(y, x, figsize=size)
    xId, yId, titleId = 0, 0, 0
    
    for i, img in enumerate(imgsArray):
        if i >= x * y:
            break # Evita iteração além da capacidade da grade

        # Lida com casos x=1 ou y=1 (eixos 1D)
        if x == 1 or y == 1:
            current_axis = axis[i]
        else:
            current_axis = axis[yId, xId]

        current_axis.set_title(titlesArray[titleId], fontdict={'fontsize': 18, 'fontweight': 'medium'}, pad=10)
        current_axis.set_anchor('NW')
        current_axis.imshow(img, 'gray')
        
        # Opcional: Desliga o eixo se o título for vazio
        if not titlesArray[titleId]:
            current_axis.axis('off')

        titleId += 1
        xId += 1
        if xId == x:
            xId = 0
            yId += 1
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Filtragem e Limpeza de Imagens Corrompidas
# ==============================================================================

def filter_valid_images(filepaths: list) -> list:
    """
    Verifica a validade de uma lista de caminhos de arquivos de imagem usando PIL.
    Filtra arquivos de tamanho zero e aqueles que não podem ser abertos/verificados.
    """
    valid_files = []
    corrupted_count = 0
    
    for i, fp in enumerate(filepaths):
        fp_str = str(fp) 
        
        # Otimização 1: Pula arquivos de tamanho zero
        if os.path.getsize(fp_str) == 0:
            corrupted_count += 1
            if corrupted_count % 500 == 1: # Log ocasional
                print(f"❌ Corrompido (tamanho zero): {fp_str}")
            continue
            
        try:
            with Image.open(fp_str) as im:
                im.verify() 
                # im.load() # Uma verificação ainda mais robusta é carregar os dados
            valid_files.append(fp)
        except Exception as e:
            # Otimização 2: Log mais claro
            corrupted_count += 1
            if corrupted_count % 500 == 1: # Log ocasional
                print(f"❌ Corrompido (PIL Error: {e.__class__.__name__}): {fp_str}") 
                
    print(f"\n✅ Concluído. {len(valid_files)} arquivos válidos | {corrupted_count} arquivos corrompidos.")
    return valid_files

def identify_corrupted_tf(file_path: Path) -> bool:
    """Tenta ler e decodificar a imagem usando rotinas do TensorFlow."""
    try:
        img_bytes = tf.io.read_file(str(file_path))
        _ = tf.image.decode_jpeg(img_bytes, channels=3)
        return False  # Não corrompido
    except Exception:
        return True   # Corrompido

def safer_purge(base_dir: Path):
    """Percorre o diretório e remove arquivos que o TensorFlow não consegue decodificar."""
    corrupted_files = []
    
    print(f"\n--- Verificando em: {base_dir.name} ---")
    
    for fp in base_dir.rglob("*.jpg"):
        if identify_corrupted_tf(fp):
            corrupted_files.append(fp)

    if corrupted_files:
        print(f"🗑️ Removendo {len(corrupted_files)} arquivos corrompidos...")
        for fp in corrupted_files:
            try:
                os.remove(fp)
            except Exception as e:
                print(f"   Erro ao remover {fp.name}: {e}")
                
    print(f"✅ Limpeza concluída em {base_dir.name}. Total removido: {len(corrupted_files):,}")