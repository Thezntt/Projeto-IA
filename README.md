

## 🏥 Projeto 1: Classificação de Imagens de Raio-X com CNNs

### Objetivo

Desenvolver um modelo de *Deep Learning* utilizando Redes Neurais Convolucionais (CNNs) para classificar imagens de raio-X de tórax, distinguindo entre pacientes saudáveis e pacientes com pneumonia.

### Dataset

O modelo foi treinado com o dataset **"Chest X-ray (Pneumonia)"**, disponível publicamente no Kaggle. Este conjunto de dados contém milhares de imagens de raio-X de tórax pré-rotuladas.

### Metodologia

1.  **Pré-processamento:** As imagens foram redimensionadas e normalizadas para se adequarem à entrada da rede neural.
2.  **Data Augmentation:** Para melhorar a generalização do modelo e evitar *overfitting*, foram aplicadas técnicas de aumento de dados em tempo real, como rotações, *flips* horizontais e zoom.
3.  **Arquitetura do Modelo:** Foi construída uma Rede Neural Convolucional (CNN) customizada, composta por camadas de convolução (`Conv2D`), *pooling* (`MaxPooling2D`), *Batch Normalization* e camadas densas (`Dense`) para a classificação final.
4.  **Treinamento:** O modelo foi compilado com o otimizador Adam e a função de perda *binary crossentropy*, sendo treinado para distinguir as duas classes (Saudável vs. Pneumonia).

### Avaliação

A performance do modelo foi avaliada utilizando as seguintes métricas:

  * **Acurácia (Accuracy):** Percentual geral de acertos.
  * **Precisão (Precision) e Recall:** Para entender a capacidade do modelo em identificar casos positivos (Pneumonia) corretamente.
  * **Matriz de Confusão:** Para visualizar em detalhes os acertos e erros, especificamente os Falsos Positivos e Falsos Negativos.

*(Você pode inserir aqui um gráfico da sua Matriz de Confusão ou as métricas finais obtidas)*

-----

## 📈 Projeto 2: Previsão de Vendas Mensais no Varejo

### Objetivo

Aplicar e comparar diferentes modelos de regressão para prever as vendas mensais de uma rede de varejo, com base em variáveis como promoções, sazonalidade e características da loja.

### Dataset

O projeto utilizou o dataset **"Rossmann Store Sales"** do Kaggle, que contém dados históricos de vendas de mais de 1.000 lojas da rede Rossmann.

### Metodologia

1.  **Feature Engineering:**

      * **Variáveis Categóricas:** Codificação de variáveis como `StoreType` e `Assortment` (usando *One-Hot Encoding* ou *Label Encoding*).
      * **Variáveis Temporais:** Extração de features relevantes da data, como `Mês`, `DiaDaSemana` e `Ano`.
      * **Tratamento de Dados:** Limpeza de dados nulos e transformação de variáveis.

2.  **Modelagem:** Foram treinados e avaliados três modelos de regressão distintos:

      * Regressão Linear
      * Árvore de Decisão
      * XGBoost (Extreme Gradient Boosting)


## 🛠️ Tecnologias Utilizadas

  * **Python 3**
  * **Pandas** e **NumPy** (Para manipulação e análise de dados)
  * **Scikit-learn** (Para pré-processamento, modelos de regressão e métricas de avaliação)
  * **TensorFlow** e **Keras** (Para a construção e treinamento da CNN)
  * **XGBoost** (Para o modelo de boosting)
  * **Matplotlib** e **Seaborn** (Para visualização de dados)
  * **Jupyter Notebook** (Para desenvolvimento e prototipagem)


