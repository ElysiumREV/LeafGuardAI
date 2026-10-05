## Integrantes do Grupo
Jean Victor Yoshida Lima <br/>
João Pedro Cabrera Rodrigues Penna <br/>
Nícolas Justo Melão

# LeafGuardAI 🌿

O LeafGuardAI é um projeto de deep learning focado na identificação de doenças em plantas através da análise de imagens de folhas, combinando um pipeline avançado de pré-processamento de visão computacional com uma arquitetura EfficientNet-B0 ajustada.

## 🛠️ Pipeline de Processamento de Imagens

O pipeline de processamento de imagens foi projetado para maximizar a extração de características relevantes da folha, minimizando a interferência de ruídos e variações de iluminação. O fluxo consiste nas seguintes etapas sequenciais:

1.  **Segmentação de Folha (Extração de ROI):** 
    *   Conversão do espaço de cores BGR para **HSV** (*Hue, Saturation, Value*).
    *   Aplicação de uma máscara de cor para isolar tons de verde.
    *   Operações morfológicas (**Closing** e **Opening**) com kernel elíptico para refinar a máscara.
    *   Identificação do maior contorno e recorte para focar exclusivamente na folha, removendo fundos irrelevantes.

2.  **Correção Cromática e Balanceamento:** 
    *   Estimativa do viés de iluminação através do cálculo da média de cores dos pixels do fundo (fora da máscara da folha).
    *   Aplicação de um fator de correção para normalizar as cores da imagem, reduzindo a dependência de condições específicas de luz.

3.  **Denoising (Redução de Ruído):** 
    *   Utilização do algoritmo `fastNlMeansDenoisingColored` para remover ruídos granulares preservando bordas e detalhes de textura.

4.  **Otimização de Contraste (CLAHE):** 
    *   Conversão para o espaço de cores **Lab**.
    *   **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) aplicado ao canal de luminosidade ($L$) para realçar o contraste local e evidenciar padrões sutis de doenças.

5.  **Realce de Nitidez (Unsharp Masking):** 
    *   Aplicação de um desfoque Gaussiano combinado com a imagem original para criar um efeito de nitidez, enfatizando bordas e a estrutura vascular da folha.

6.  **Normalização Dimensional:** 
    *   Redimensionamento final para **256x256 pixels** utilizando interpolação `INTER_AREA` para compatibilidade com a entrada da rede neural.

## 🤖 Implementação e Integração de IA

O projeto integra conceitos avançados de Inteligência Artificial, focando em **Deep Learning** e **Visão Computacional** para a classificação de patologias vegetais.

### Pipeline Completo do Projeto:
**A. Gestão de Dados:**
*   **Split Estratificado:** Os dados são divididos em conjuntos de **Treino, Validação e Teste**, mantendo a distribuição original de classes.
*   **Balanceamento de Classes:** Implementação de **pesos por classe** (*class weights*) na função de perda para lidar com datasets desbalanceados, penalizando mais erros em classes minoritárias.

**B. Arquitetura do Modelo:**
*   **Base:** **EfficientNet-B0**, escolhida pelo equilíbrio ideal entre profundidade, largura e resolução.
*   **Transfer Learning:** Utilização de pesos pré-treinados no ImageNet para aproveitar a capacidade da rede de reconhecer formas e texturas genéricas.

**C. Estratégia de Treinamento (Multi-estágio):**
1.  **Estágio 1 (Warm-up):** O *backbone* da rede é congelado e apenas a camada de classificação (*head*) é treinada, evitando a destruição de pesos pré-treinados.
2.  **Estágio 2 (Fine-Tuning):** Descongelamento dos últimos blocos do *backbone* com uma taxa de aprendizado reduzida ($\text{lr} \approx 10^{-5}$) para especialização nos padrões botânicos.
*   **Otimização:** Otimizador **Adam** com escalonador **ReduceLROnPlateau**.
*   **Regularização:** **Label Smoothing** ($0.1$) para prevenir *overfitting* e melhorar a generalização.

**D. Aceleração e Estabilidade:**
*   **AMP (Automatic Mixed Precision):** Utilizado para reduzir o consumo de VRAM e acelerar o processamento em GPUs.
*   **Early Stopping:** Monitoramento da acurácia de validação para interromper o treino quando a melhora cessa, evitando o sobreajuste.

**E. Métricas de Avaliação:**
*   **Matriz de Confusão:** Para analisar confusões entre classes semelhantes.
*   **Relatório de Classificação:** Cálculo de **Precisão**, **Recall** e **Macro-F1 Score**.
*   **Acurácia Top-K:** Avalia se a classe correta está entre as $K$ previsões mais prováveis.

---

## 🚀 Desenvolvimento

Este projeto utiliza **uv** para gerenciar dependências e o ambiente virtual.

### Instalação

Após clonar o repositório, execute:

```bash
uv sync
```

Este comando instalará todas as dependências necessárias e configurará o ambiente virtual automaticamente.

### Adicionando Dependências

Para adicionar uma nova dependência ao projeto, use:

```bash
uv add <nome-do-pacote>
```

O `uv` atualizará automaticamente os arquivos `pyproject.toml` e `uv.lock`.

### Configurando a API do Gemini

Crie uma chave no [Google AI Studio](https://aistudio.google.com/api-keys) e cole-a no campo “Insira a API Key” na parte inferior da interface.

### Executando a GUI

```bash
uv run python -m leafguardai
```

ou:

```bash
uv run leafguardai
```

A interface gráfica classifica a imagem de uma folha selecionada utilizando o checkpoint treinado em `models/`. Treine o modelo primeiro.

## 📁 Dataset

Este projeto utiliza o dataset PlantVillage.

O dataset não está incluído neste repositório devido ao seu tamanho.

Você pode baixá-lo no [repositório original.](https://github.com/spMohanty/PlantVillage-Dataset)
Apenas o conjunto de imagens é necessário. Após baixar, coloque o diretório `color/` no seguinte caminho:
```
LeafGuardAI/
└── data/
    └── raw/
        └── PlantVillage/
            └── color/
```
A estrutura esperada é:
```
data/
└── raw/
    └── PlantVillage/
        └── color/
            ├── Apple___Apple_scab/
            ├── Tomato___healthy/
            └── ...
```

## 🏋️ Treinamento

```bash
uv run python scripts/train.py
```

Flags úteis:

```bash
uv run python scripts/train.py --epochs-stage1 10 --epochs-stage2 15 --patience 5
```

Os artefatos são gravados em `models/`:

- `best_model.pt` — melhor checkpoint por acurácia de validação
- `classes.json` — mapeamento de índice de classes
- `split.json` — caminhos estratificados de treino/val/teste
- `history.json` — métricas por época
- `metrics.json` — métricas finais de teste
- `classification_report.txt` — precisão/recall/F1 por classe
- `confusion_matrix.png`

### Treinamento com GPU AMD (ROCm + Docker)

No Linux com uma GPU AMD suportada pelo ROCm, o projeto pode ser treinado no container oficial ROCm/PyTorch. O host deve expor `/dev/kfd` e `/dev/dri` e usar o Docker Engine nativo.

Construa a imagem uma vez:

```bash
docker --context default compose build
```

Treine usando a GPU:

```bash
docker --context default compose run --rm train
```

O diretório atual é montado em `/workspace`; portanto, o dataset em `data/` e os artefatos gerados em `models/` permanecem no host. Argumentos extras de treino podem ser passados após o nome do serviço:

```bash
docker --context default compose run --rm train --batch-size 64 --epochs-stage1 10 --epochs-stage2 15
```

## 🔍 Inferência (CLI)

```bash
uv run python scripts/predict.py caminho/da/folha.jpg --top 3
```

## 📊 Exploração do Dataset

```bash
uv run python scripts/explore_dataset.py
uv run python scripts/test_dataset.py
```

## 🏗️ Estrutura Esperada do Projeto
```
LeafGuardAI/
├── src/
│   └── leafguardai/
├── scripts/
├── data/
├── models/
├── pyproject.toml
└── uv.lock
```
