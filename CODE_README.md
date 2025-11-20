# Projeto ASSL — Guia de Código e Execução

Este documento descreve os principais módulos do repositório, explica os arquivos solicitados e mostra como executar as rotinas principais e baterias de experimentos localmente.

**Resumo rápido**

- **Propósito:** Implementação de um pipeline de Active Semi-Supervised Learning (ASSL) para CNNs, com scripts de treino/experimentos e notebooks de análise.
- **Ambiente:** Python 3.8+ (venv recomendado). Dependências em `requirements.txt`.

**Arquivos documentados**

- `models.py`: Implementações das arquiteturas/encoders e funções utilitárias para construir e carregar modelos. Use este módulo para instanciar o encoder ou classificador usado pelos scripts de treino. Normalmente expõe um construtor `build_model(...)` e helpers para carregar/salvar pesos.

- `assl_strategies.py`: Estratégias de seleção ativa e heurísticas (por-ex.: UCB, EL2N, pseudo-label ranking). Contém funções que recebem estatísticas de incerteza/EMA e retornam os `sample_ids` para aquisição.

- `assl_core.py`: Lógica principal do pipeline ASSL — contém laços de rounds, funções de aquisição, atualização das métricas EMA/UCB, e integração entre datasets L/U e os passos de treino (supervisionado / não-supervisionado). Este é o núcleo do fluxo experimental.

- `run_assl.py` (localizado em `notebooks/run_assl.py`): Entrypoint para executar um experimento ASSL (um único run ou round series) usando as funções em `assl_core.py` e `assl_strategies.py`. Parâmetros (dataset, batch, budget, seed) normalmente vêm de `experiments_config.json` ou argumentos de linha de comando.

- `plots.ipynb` (notebook `notebooks/plots.ipynb`, originalmente solicitado como `plots.py`): Notebook para gerar visualizações a partir do CSV de resultados (`experiments_results.csv` / `results/*.csv`). Contém a função `plot_assl_results(csv_path)` que plota test accuracy por round agregando por configuração.

- `ssl_pretext.py`: Scripts/utilitários para tarefas de pré-treinamento (pretext tasks) — usados para gerar representações iniciais antes do treino semi-supervisionado. Exemplos: RotNet, jigsaw, colorization, etc., dependendo da implementação.

- `experiments_config.json`: Arquivo JSON com a configuração de experimentos: grid de hiperparâmetros, seeds, datasets, tamanhos de imagem, valores de `lambda_u_max`, `pretext_task`, etc. `run_assl_grid.py` / `run_experiments.py` lê este arquivo para disparar múltiplos runs em lote.

- `results/` (pasta com resultados): contém CSVs de saída dos experimentos, por exemplo:
  - `assl_multitask_results.csv` — CSV principal com métricas por seed e round.
  - `assl_multitask_results_old.csv` — histórico/backup das execuções anteriores.

**Como preparar o ambiente**

- 1. Ative o ambiente virtual (exemplo com o venv presente em `tf_env`):

```powershell
& .\tf_env\Scripts\Activate.ps1
```

- 2. Instale dependências (se necessário):

```powershell
pip install -r requirements.txt
```

- 3. Verifique se o dataset requerido está disponível nas pastas `notebooks/data/` ou em `data/` e siga as instruções no `notebooks/README.md` para preparar MNIST/STL10/etc.

**Executando os principais scripts**

- `run_assl.py` (executa um experimento ASSL):

```powershell
python notebooks/run_assl.py --config experiments_config.json --experiment-name myrun
```

Opções comuns (ver o parser no início do script): `--config`, `--seed`, `--dataset`, `--batch`, `--rounds`, `--budget`.

- `run_assl_grid.py` (executa uma grade de experimentos definida em `experiments_config.json`):

```powershell
python notebooks/run_assl_grid.py --config notebooks/experiments_config.json
```

- `run_experiments.py` (helper para disparar/gerenciar jobs em série):

```powershell
python notebooks/run_experiments.py --config notebooks/experiments_config.json
```

- `ssl_pretext.py` (executa rotina de pretext task):

```powershell
python notebooks/ssl_pretext.py --task rot --dataset mnist --out checkpoints/pretext_rot.pt
```

As flags exatas dependem do parser implementado no script — cheque a seção de argumentos no topo de cada arquivo.

**Gerar plots e análises**

- Abrir o notebook `notebooks/plots.ipynb` em Jupyter / VS Code e executar a célula com a função `plot_assl_results(csv_path)`. Exemplo de uso em uma nova célula:

```python
from plots import plot_assl_results
plot_assl_results('notebooks/results/assl_multitask_results.csv')
```

Se preferir rodar sem Jupyter, converta o notebook para script ou crie um pequeno script Python que importe a função de plot (se houver uma versão `.py`).

**Interpretação dos arquivos em `results/`**

- Cada CSV contém colunas de configuração e métricas por round, por seed. Colunas típicas:
  - `dataset`, `img_size`, `pretext_task`, `lambda_u_max`, `seed`, `round`, `test_acc`, `val_acc`, `labeled_count`, etc.
- Use `notebooks/plots.ipynb` para agregar por `(pretext_task, lambda_u_max)` e traçar médias e desvios por round.

**Dicas para desenvolvimento e depuração**

- Se o treino “travar” ou demorar demasiado:

  - Rode com `DEBUG=True` (algumas implementações do notebook têm essa flag para desabilitar `@tf.function` e imprimir tempos).
  - Reduza `BATCH` e `MAX_U_STEPS` para um teste rápido.
  - Verifique uso de GPU com `nvidia-smi` e memória do sistema (no Windows, use `Task Manager` ou `nvidia-smi` via PowerShell).

- Mixed precision: se usar GPU AMP, ative a política pelo notebook e confirme a versão do TensorFlow; erros de LossScale podem ocorrer em versões antigas — use as guardas presentes nos scripts.

**Executando a bateria de testes mínima**

1. Ative o venv e instale dependências.
2. Faça um run rápido com uma configuração reduzida — por exemplo, editar `experiments_config.json` para usar apenas 1 seed, 1 round e dataset `mnist` com `img_size=32`.

```powershell
python notebooks/run_assl.py --config notebooks/experiments_config.json --seed 0 --rounds 1 --budget 10
```

3. Verifique a geração de um CSV novo em `notebooks/results/` e abra `notebooks/plots.ipynb` para visualizar.

**Onde procurar helpers e pontos de extensão**

- **Modelos e checkpoints:** `models.py` e `model2.keras` (exemplo de peso salvo). Modifique `build_model()` para trocar arquitetura.
- **Estratégias ativas:** Edite/adicione funções em `assl_strategies.py` para novas heurísticas.
- **Core pipeline:** `assl_core.py` para alterar o fluxo de rounds, warmup, ou integração com orquestradores.

**Notas finais / observações**

- Muitos notebooks contêm células experimentais; antes de rodar um pipeline completo, execute uma versão curta para confirmar compatibilidade de versões de TF/PyTorch/NumPy.
- Se preferir, posso:
  - Gerar um script de execução “minimal reproducible run” que executa um único round em CPU para validação rápida;
  - Adicionar um `tests/` com um teste unitário simples que verifica importação e construção do modelo.

---

Gerado automaticamente pelo assistente. Se quiser, eu atualizo o arquivo README principal (`README.md`) com um sumário curto e link para este `CODE_README.md`.
