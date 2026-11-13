## Integrantes do Grupo
Jean Victor Yoshida Lima <br/>
João Pedro Cabrera Rodrigues Penna <br/>
Nícolas Justo Melão

# LeafGuardAI 🌿

O LeafGuardAI é um projeto de deep learning focado na identificação de doenças em plantas através da análise de imagens de folhas, usando uma arquitetura EfficientNet-B0 ajustada com fotos de laboratório (PlantVillage) e fotos reais de campo. O projeto também inclui um pipeline opcional de pré-processamento de visão computacional (`ImageProcessor`).

## 🛠️ Pipeline de Processamento de Imagens

### Pipeline padrão (treino e inferência)

Por padrão, cada imagem passa pelas mesmas etapas no treino e no app (`src/leafguardai/vision/transforms.py` e `src/leafguardai/model/infer.py`):

1.  **Leitura:** a imagem é aberta em RGB, corrigindo a rotação de fotos de celular (orientação EXIF).
2.  **Redimensionamento:** o menor lado vai para **256 pixels**, mantendo a proporção.
3.  **Recorte central:** recorte de **224x224 pixels**, o tamanho de entrada da EfficientNet-B0.
4.  **Normalização:** os canais RGB são normalizados com a média e o desvio padrão do ImageNet.

No treino, as etapas 2 e 3 são substituídas por variações aleatórias (aumento de dados): recortes de tamanho e posição variados, espelhamento horizontal, rotação de até 15° e variação de brilho, contraste e saturação.

### `ImageProcessor` (opcional)

O `ImageProcessor` (`src/leafguardai/vision/processor.py`) é um pipeline de realce projetado para destacar características da folha e reduzir ruídos e variações de iluminação. Quando ativado com `--use-processor` no treino, ele é aplicado antes das etapas acima, no treino e no app. O fluxo consiste nas seguintes etapas sequenciais:

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
    *   Redimensionamento final para **256x256 pixels** utilizando interpolação `INTER_AREA`; em seguida, os transforms de avaliação recortam o centro em **224x224** e normalizam as cores para a entrada da rede neural.

> **Importante:** o `ImageProcessor` precisa ser usado do mesmo jeito no treino e na inferência. O modelo aprende com imagens no formato em que as viu; aplicar o processor só na inferência derrubou a acurácia de 99% para 64% em folhas do PlantVillage. Por isso, o treino grava em `preprocessing.json` se o processor foi usado (`--use-processor`), e o app repete automaticamente a mesma escolha. Nos nossos testes com fotos de campo, treinar sem o processor deu mais acerto (cerca de 60% contra 49%) e deixa a análise mais rápida, por isso ele vem desligado por padrão. Checkpoints antigos, sem `preprocessing.json`, continuam sendo usados com o processor.

## 🤖 Implementação e Integração de IA

O projeto integra conceitos avançados de Inteligência Artificial, focando em **Deep Learning** e **Visão Computacional** para a classificação de patologias vegetais.

### Pipeline Completo do Projeto:
**A. Gestão de Dados:**
*   **Split Estratificado por Folha:** As fotos do PlantVillage são divididas em **Treino, Validação e Teste** (70% / 15% / 15%), mantendo a distribuição de classes; cópias e versões giradas da mesma foto ficam sempre no mesmo conjunto.
*   **Fotos de Campo:** Além do PlantVillage (fotos de laboratório), o treino usa fotos reais de vários datasets de campo, divididas à parte (60% / 15% / 25%). A divisão é estável: cada foto cai sempre no mesmo conjunto, mesmo que outros datasets sejam adicionados. Fotos repetidas entre datasets são agrupadas, e cópias de imagens do PlantVillage, fotos iguais às de `data/external_test` e arquivos corrompidos são descartados.
*   **Mistura por Época:** Como há muito menos fotos de campo, elas são reamostradas para formar cerca de **40%** de cada época (`--field-share`).
*   **Balanceamento de Classes:** Implementação de **pesos por classe** (*class weights*) na função de perda para lidar com datasets desbalanceados, penalizando mais erros em classes minoritárias.

**B. Arquitetura do Modelo:**
*   **Base:** **EfficientNet-B0**, escolhida pelo equilíbrio ideal entre profundidade, largura e resolução.
*   **Transfer Learning:** Utilização de pesos pré-treinados no ImageNet para aproveitar a capacidade da rede de reconhecer formas e texturas genéricas.

**C. Estratégia de Treinamento (Multi-estágio):**
1.  **Estágio 1 (Warm-up):** O *backbone* da rede é congelado e apenas a camada de classificação (*head*) é treinada ($\text{lr} = 10^{-3}$, até 5 épocas), evitando a destruição de pesos pré-treinados.
2.  **Estágio 2 (Fine-Tuning):** Descongelamento da rede inteira (os 9 blocos do *backbone*) com taxa de aprendizado $\text{lr} = 10^{-4}$, por até 25 épocas, para adaptar todas as camadas às fotos de folhas, inclusive de campo. Use `--unfreeze-blocks` para descongelar menos blocos.
*   **Otimização:** Otimizador **Adam** com escalonador **ReduceLROnPlateau**.
*   **Regularização:** **Label Smoothing** ($0.1$) e aumento de dados (recortes, espelhamento, rotação e variação de cor) para prevenir *overfitting* e melhorar a generalização.

**D. Aceleração e Estabilidade:**
*   **AMP (Automatic Mixed Precision):** Utilizado para reduzir o consumo de VRAM e acelerar o processamento em GPUs.
*   **Early Stopping:** Monitoramento da acurácia de validação (nas fotos de campo, quando estão presentes) para interromper cada estágio após 5 épocas sem melhora, evitando o sobreajuste. O melhor checkpoint é o de maior acurácia nessa validação.

**E. Métricas de Avaliação:**
*   **Matriz de Confusão:** Para analisar confusões entre classes semelhantes.
*   **Relatório de Classificação:** Cálculo de **Precisão**, **Recall** e **Macro-F1 Score**.
*   **Acurácia Top-K:** Avalia se a classe correta está entre as $K$ previsões mais prováveis.
*   **Teste em Fotos de Campo:** Além do teste no PlantVillage, o modelo é avaliado separadamente nas fotos de campo de teste, que representam melhor o uso real do app.

---

## 🚀 Desenvolvimento

Este projeto utiliza **uv** para gerenciar dependências e o ambiente virtual.

### Instalação

Após clonar o repositório, execute:

```bash
uv sync
```

Este comando instalará todas as dependências necessárias e configurará o ambiente virtual automaticamente. No Windows e no Linux, o PyTorch instalado já tem suporte a GPU NVIDIA (CUDA); sem GPU, ele funciona normalmente na CPU.

### Adicionando Dependências

Para adicionar uma nova dependência ao projeto, use:

```bash
uv add <nome-do-pacote>
```

O `uv` atualizará automaticamente os arquivos `pyproject.toml` e `uv.lock`.

### Configurando a API do Gemini

Crie uma chave no [Google AI Studio](https://aistudio.google.com/api-keys) e cole-a no campo da chave na parte inferior da interface. Para não colar a chave toda vez, defina a variável de ambiente `GEMINI_API_KEY` ou crie um arquivo `.env` (veja `.env.example`) na pasta de onde você abre o app: o campo já vem preenchido (use o botão “Mostrar” para conferir o que colou).

### Executando a GUI

```bash
uv run python -m leafguardai
```

ou:

```bash
uv run leafguardai
```

A interface gráfica classifica a imagem de uma folha selecionada utilizando o modelo em `models/`. Não é preciso treinar: se não houver um modelo, o app baixa o modelo treinado do [release mais recente](https://github.com/ElysiumREV/LeafGuardAI/releases/latest) na primeira análise (cerca de 16 MB). A linha de comando (`scripts/predict.py`) faz o mesmo. O modelo fica em `models/` quando você roda do repositório, ou na pasta de dados do usuário (ex.: `~/.local/share/leafguardai/models`) quando o app é instalado como pacote; a variável `LEAFGUARDAI_MODELS_DIR` sobrescreve os dois.

O app mostra as 3 classes mais prováveis e avisa quando a confiança é baixa. Abaixo de 70%, o resultado aparece como "confiança moderada"; abaixo de 50%, como "resultado incerto", e nesse caso o app não pede orientações de tratamento ao Gemini e sugere outra foto. As faixas ficam em `CONFIDENCE_HIGH` e `CONFIDENCE_LOW` (`src/leafguardai/model/infer.py`), medidas em fotos de campo: acima de 70%, o modelo acerta cerca de 90% das vezes.

O campo opcional **Qual é a planta?** restringe a resposta às classes da planta escolhida, o que elimina a confusão entre plantas: nas fotos de campo de teste, o acerto sobe de 78,8% para 85,5% (83,8% sem contar as plantas com uma classe só). Nesse modo, a confiança só é considerada alta a partir de 90%. O campo lista apenas as plantas com duas ou mais classes no modelo; nas demais, use "Não sei / outra planta".

## 📁 Dataset

Este projeto utiliza o **PlantVillage** (fotos de folhas em laboratório) e um conjunto de **datasets de fotos de folhas em campo**.

Os datasets não estão incluídos neste repositório devido ao seu tamanho.

Você pode baixar o PlantVillage no [repositório original.](https://github.com/spMohanty/PlantVillage-Dataset)
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

Opcionalmente, o NewPlantDataset (uma versão do PlantVillage com cópias giradas e espelhadas) também é usado se estiver em `data/raw/NewPlantDataset/train/`.

Os datasets de fotos de campo podem ser baixados dos repositórios originais com o script abaixo, que já coloca as imagens no caminho correto (use `--only` para baixar só alguns, ex.: `--only plantdoc gvlid`):
```bash
uv run python scripts/download_field_datasets.py
```
```
LeafGuardAI/
└── data/
    └── raw/
        └── field/
```
A estrutura esperada é:
```
data/
└── raw/
    └── field/
        ├── PlantDoc/
        │   ├── train/
        │   │   ├── Apple Scab Leaf/
        │   │   └── ...
        │   └── test/
        ├── PlantWild/
        │   ├── apple black rot/
        │   └── ...
        ├── GVLiD/
        ├── ESCA/
        ├── PotatoEthiopia/
        ├── PlantPathology2021/
        ├── TomatoPakistan/
        └── TomatoDIU/
```

As pastas de cada dataset são associadas às classes do PlantVillage em `src/leafguardai/data/field.py`. Fontes:

- **PlantDoc** — Singh, D. et al. *PlantDoc: A Dataset for Visual Plant Disease Detection*. CODS-COMAD 2020. [Repositório](https://github.com/pratikkayal/PlantDoc-Dataset) (CC BY 4.0).
- **PlantWild** — Wei, T. et al. *Benchmarking In-the-wild Multimodal Plant Disease Recognition and A Versatile Baseline*. ACM MM 2024. [Hugging Face](https://huggingface.co/datasets/uqtwei2/PlantWild) (CC BY-NC-ND 4.0: uso não comercial, sem redistribuir imagens modificadas).
- **GVLiD** — *GrapeVine Leaf identification of the Diseases*. [Mendeley Data](https://data.mendeley.com/datasets/wkymf8bhcg) (CC BY 4.0).
- **ESCA-dataset** — Alessandrini, M. et al. [Mendeley Data](https://data.mendeley.com/datasets/89cnxc58kj/1) (CC BY 4.0).
- **Potato Leaf (Healthy and Late Blight)** — [Mendeley Data](https://data.mendeley.com/datasets/v4w72bsts5) (CC BY 4.0).
- **Plant Pathology 2021** — Thapa, R. et al., Cornell AgriTech. [Hugging Face](https://huggingface.co/datasets/timm/plant-pathology-2021) (Apache-2.0). O script baixa até 500 fotos por classe, reduzidas para 1024 px.
- **Tomato Leaf Disease Classification Dataset in Pakistan** — [Mendeley Data](https://data.mendeley.com/datasets/3mbnb82mxd) (CC BY 4.0). O script usa só as 830 fotos originais, sem as cópias aumentadas.
- **Tomato leaf diseases** — Daffodil International University. [Mendeley Data](https://data.mendeley.com/datasets/93h9p62kg4) (CC BY 4.0). Folhas destacadas fotografadas sobre fundo claro; o script baixa só as classes que existem no PlantVillage.

O treino ignora as fotos listadas em `src/leafguardai/data/field_exclude.txt` (slides, banners e fotos de fruto do PlantWild, escolhidas por triagem automática). As fotos do TomatoDIU não entram na validação nem no teste de campo, porque a mesma folha aparece em várias fotos: 80% vão para o treino e os 20% finais de cada classe, pela numeração dos arquivos, ficam reservados.

## 🏋️ Treinamento

```bash
uv run python scripts/train.py
```

Para gravar o modelo em outra pasta e compará-lo antes de substituir o atual:

```bash
uv run python scripts/train.py --output-dir models_novo
```

> **GPU:** no Windows e no Linux, o `uv sync` já instala o PyTorch com CUDA, e o treino usa a GPU NVIDIA automaticamente (na CPU, o treino leva muitas horas). No Linux com GPU AMD, use o Docker abaixo. Com GPU, o treino completo leva algumas horas; manter a pasta `data/` em um SSD deixa as épocas bem mais rápidas que em um HD.

Opções:

- `--output-dir pasta/` — grava os artefatos em outra pasta (o padrão `models/` substitui o modelo usado pelo app).
- `--epochs-stage1`, `--epochs-stage2`, `--patience` — épocas máximas de cada estágio (padrão 5 e 25) e épocas sem melhora antes de parar (padrão 5).
- `--unfreeze-blocks 9` e `--stage2-lr 1e-4` — quantos blocos do *backbone* treinar no estágio 2 e com que taxa de aprendizado (padrão: a rede inteira; com fotos de campo, isso levou o acerto em campo de 72,7% para 78,3% em relação a treinar só os 3 últimos blocos com $10^{-5}$).
- `--background-prob 0.5` — troca o fundo de parte das folhas do PlantVillage usando as máscaras da pasta `segmented/` do PlantVillage (coloque-a em `data/raw/PlantVillage/segmented/`). Desligada por padrão: deixou o modelo muito mais robusto a fundos artificiais, mas com fotos de campo no treino não melhorou o acerto em campo e deixa o treino ~3× mais lento.
- `--field-share 0.4` — fração de cada época vinda das fotos de campo; `--no-field` ignora as fotos de campo; `--field-datasets PlantDoc GVLiD ...` usa só os datasets de campo indicados.
- `--use-processor` — aplica o `ImageProcessor` também no treino (veja a nota na seção do pipeline).

Como o treino funciona:

- **Split por folha:** cópias, giros e espelhamentos da mesma foto ficam sempre na mesma parte (treino, validação ou teste), para que o teste não contenha imagens já vistas no treino.
- **Escolha do checkpoint:** com fotos de campo presentes, o melhor modelo é escolhido pela acurácia nas fotos de campo da validação.
- **Pré-processamento consistente:** o treino grava `preprocessing.json` junto do modelo, e o app aplica exatamente o mesmo pré-processamento na inferência. Usar na inferência um pré-processamento diferente do usado no treino derruba a acurácia (no checkpoint antigo, ligar o `ImageProcessor` só na inferência levou de 99% para 64% em folhas do PlantVillage). Checkpoints sem esse arquivo mantêm o comportamento original (com `ImageProcessor`).

Os artefatos são gravados em `models/`:

- `best_model.pt` — melhor checkpoint (por acurácia de validação, ou de validação de campo quando há fotos de campo)
- `classes.json` — mapeamento de índice de classes
- `preprocessing.json` — se o `ImageProcessor` foi usado no treino
- `split.json` — caminhos de treino/val/teste
- `history.json` — métricas por época
- `metrics.json` — métricas finais de teste (inclui `field_test_accuracy` quando há fotos de campo)
- `classification_report.txt` / `classification_report_campo.txt` — precisão/recall/F1 por classe
- `confusion_matrix.png` / `confusion_matrix_campo.png`

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
docker --context default compose run --rm train --batch-size 64 --output-dir models_novo
```

Baixe os datasets de campo no host (`uv run python scripts/download_field_datasets.py`) antes de treinar: dois deles vêm em `.rar`, e a extração precisa de `bsdtar`, `unrar` ou `7z`, que não estão na imagem Docker. No Linux, o `bsdtar` vem no pacote `libarchive-tools`.

## 🔍 Inferência (CLI)

```bash
uv run python scripts/predict.py caminho/da/folha.jpg --top 3 --planta Tomate
```

## 🧪 Avaliação em uma Pasta de Fotos

Para medir o modelo de `models/` em várias fotos de uma vez, organize-as em uma pasta com uma subpasta por classe (com o nome exato da classe, ex.: `Tomato___Late_blight/`) e rode:

```bash
uv run python scripts/evaluate_folder.py caminho/da/pasta
```

O script usa o mesmo caminho de inferência do app, salva uma planilha (`resultado.csv`) com a previsão de cada foto e mostra o acerto geral, o acerto por faixa de confiança, o acerto por classe e as confusões mais comuns.

## 📦 Publicando o Modelo (GitHub Release)

O modelo treinado não fica no repositório: ele é publicado como arquivo de um release, e o app o baixa sozinho. Depois de treinar o modelo final:

```bash
uv run python scripts/package_model.py
```

Isso gera `dist/leafguardai-model.zip` e `dist/leafguardai-model.zip.sha256` (o hash que o app confere antes de instalar). O zip tem `best_model.pt`, `classes.json`, `preprocessing.json` e `metrics.json` (use `--models-dir` para empacotar outra pasta). No GitHub, crie um release (ex.: `v1.0.0`) e anexe **os dois arquivos com exatamente esses nomes**. O app sempre baixa o modelo do release mais recente.

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

## 📄 Licença

O código-fonte está sob a licença [GNU GPL 3.0](LICENSE). Os pesos do modelo treinado, publicados nos releases, são um arquivo separado e não estão cobertos por ela: como foram treinados com datasets de terceiros, incluindo o PlantWild (CC BY-NC-ND 4.0), são disponibilizados **apenas para uso não comercial**. A licença de cada dataset está na seção [Dataset](#-dataset).
