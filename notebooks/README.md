# Script
## Pretexto

* Rotação: classifica {0°, 90°, 180°, 270°}.

* Jigsaw 2×2: embaralha quadrantes e classifica uma das 24 permutações.

* Colorização: prediz RGB a partir de entrada em escala de cinza (perda MSE).

* Encoder: CNN pequena (≈200k parâmetros) para ser rápida em CPU.

* Linear Probe: congela o encoder e treina um classificador linear nos rótulos verdadeiros (10 classes) para medir qualidade das features.

* MNIST é convertido para 3 canais e redimensionado (compatível com as tarefas), STL-10 usa train (e opcional unlabeled indiretamente via wrappers).

#### MNIST, rotação, 3 épocas de pretexto e 3 de linear probe
```
python ssl_pretext.py --dataset mnist --task rotation --epochs-pretext 3 --epochs-linear 3 --img-size 96 --subset 5000
```

#### STL-10, jigsaw, 5 épocas pretexto e 5 de probe
```
python ssl_pretext.py --dataset stl10 --task jigsaw --epochs-pretext 5 --epochs-linear 5 --img-size 96 --batch-size 128
```

#### MNIST, colorização (MSE), depois probe
```
python ssl_pretext.py --dataset mnist --task colorization --epochs-pretext 5 --epochs-linear 5
```
