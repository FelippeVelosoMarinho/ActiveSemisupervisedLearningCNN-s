# ActiveSemisupervisedLearningCNN-s

### Criando venv
```
python -m venv tf_env
source tf_env/bin/activate  # No Linux/Mac
tf_env\Scripts\activate   # No Windows (CMD)
```

### Instale as dependências
```
pip install -r requirements.txt
```

### Configurando venv no notebook
```
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name "TensorFlow Clean (venv)"
```
Depois selecione sua versão de kernel ao executar uma célula no jupyter ou vs code.