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

flowchart LR
    %% ------- Dados -------
    D[Dataset (MNIST / STL-10)]
    subgraph SPLITS[Data splits]
        L[Labeled pool L]
        U[Unlabeled pool U]
        T[Test set T]
    end

    D --> L
    D --> U
    D --> T

    %% ------- Modelo multitask -------
    subgraph MODEL[Multi-task ASSL + Pretext]
        direction TB
        ENC[Shared CNN Encoder]

        %% supervised branch
        subgraph SUP[Supervised branch]
            XL[x_L, y_L]
            CLS[Classifier Head]
            Lsup[L_sup (CE)]
        end

        %% SSL branch
        subgraph SSL[SSL (FixMatch-like)]
            XW[x_w (weak aug)]
            XS[x_s (strong aug)]
            PL[Pseudo-label + confidence mask]
            Luns[L_uns]
        end

        %% Pretext branch
        subgraph PRE[Pretext task branch]
            XPT[x_w → pretext transform<br/>(Rotation / Jigsaw / Colorization)]
            HEADPT[Pretext Head<br/>(Rot/Jig/Color)]
            Lpre[L_pre]
        end

        XL --> ENC
        ENC --> CLS --> Lsup

        XW --> ENC
        ENC --> PL
        XS --> ENC
        PL --> Luns

        XPT --> ENC --> HEADPT --> Lpre
    end

    %% ------ Loss combination ------
    Lsup --> LTOT[L_total = L_sup + λ_u * L_uns + λ_pre * L_pre]
    Luns --> LTOT
    Lpre --> LTOT

    %% ------ Active Loop ------
    subgraph ACTIVE[Active Learning Loop]
        direction TB
        SCORE[Entropy-based scoring on U]
        TOPK[Select top-k most uncertain]
        MOVE[Move selected from U to L]
    end

    U --> SCORE --> TOPK --> MOVE
    MOVE --> L
    LTOT --> SCORE
