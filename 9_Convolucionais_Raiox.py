"""
Script Python para treinar uma Rede Neural Convolucional (CNN)
para classificar imagens de Raio-X de tórax como 'NORMAL' ou 'PNEUMONIA'.

REQUISITO:
O arquivo 'kaggle.json' (baixado da sua conta Kaggle) deve estar
no mesmo diretório deste script antes da execução.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import subprocess
import shutil
import zipfile

# --- Constantes do Projeto ---
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10  # Você pode aumentar este valor para tentar obter melhor acurácia
BASE_DIR = 'chest_xray'
KAGGLE_JSON = 'kaggle.json'

def setup_dataset():
    """
    Instala a API do Kaggle, move a chave, baixa e descompacta o dataset.
    """
    print("--- Iniciando Configuração do Dataset ---")

    # 1. Verificar se o 'kaggle.json' existe
    if not os.path.exists(KAGGLE_JSON):
        print(f"Erro: Arquivo '{KAGGLE_JSON}' não encontrado.")
        print("Por favor, baixe-o da sua conta Kaggle e coloque-o no mesmo diretório.")
        sys.exit(1)

    # 2. Instalar a biblioteca do Kaggle
    print("Instalando a biblioteca Kaggle...")
    try:
        subprocess.run(["pip", "install", "kaggle", "-q"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Falha ao instalar o Kaggle: {e}")
        sys.exit(1)

    # 3. Configurar diretório e permissões da API Kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(KAGGLE_JSON, os.path.join(kaggle_dir, KAGGLE_JSON))
    os.chmod(os.path.join(kaggle_dir, KAGGLE_JSON), 0o600)

    # 4. Baixar o dataset
    print("Baixando o dataset 'chest-xray-pneumonia'...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "-q"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Falha ao baixar o dataset: {e}")
        sys.exit(1)

    # 5. Descompactar o dataset
    print("Descompactando arquivos...")
    with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    
    # 6. Limpar o arquivo .zip
    os.remove('chest-xray-pneumonia.zip')
    
    print("--- Configuração do Dataset Concluída ---")

def create_image_generators():
    """
    Cria os geradores de imagem, aplicando Data Augmentation no
    conjunto de treino.
    """
    print("Criando geradores de imagem...")
    
    train_dir = os.path.join(BASE_DIR, 'train')
    val_dir = os.path.join(BASE_DIR, 'val')
    test_dir = os.path.join(BASE_DIR, 'test')

    # Gerador de Treino (com Data Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Gerador de Validação e Teste (Apenas normalização)
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Criar os geradores a partir dos diretórios
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"Classes encontradas: {train_generator.class_indices}")
    return train_generator, val_generator, test_generator

def build_model():
    """
    Constrói a arquitetura do modelo CNN.
    """
    print("Construindo o modelo CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(512, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid') # Saída binária
    ])

    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def train_model(model, train_generator, val_generator):
    """
    Treina o modelo e retorna o histórico.
    """
    print(f"\n--- Iniciando Treinamento ({EPOCHS} épocas) ---")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )
    
    print("--- Treinamento Concluído ---")
    return history

def evaluate_model(model, history, test_generator):
    """
    Avalia o modelo no conjunto de teste, plota gráficos
    e imprime o relatório de classificação.
    """
    print("\n--- Iniciando Avaliação do Modelo ---")

    # 1. Avaliar no conjunto de teste
    loss, accuracy = model.evaluate(test_generator)
    print(f"\nAcurácia no Teste: {accuracy * 100:.2f}%")
    print(f"Perda no Teste: {loss:.4f}")

    # 2. Plotar gráficos de Acurácia e Perda
    plt.figure(figsize=(12, 4))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia ao longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Gráfico de Perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda ao longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.savefig('training_history.png')
    print("Gráfico do histórico salvo como 'training_history.png'")
    plt.show()

    # 3. Gerar previsões para Matriz de Confusão
    Y_pred = model.predict(test_generator)
    y_pred = (Y_pred > 0.5).astype(int).reshape(-1)
    y_true = test_generator.classes
    
    class_names = list(test_generator.class_indices.keys())

    # 4. Plotar Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro (True)')
    plt.xlabel('Previsto (Predicted)')
    
    plt.savefig('confusion_matrix.png')
    print("Matriz de Confusão salva como 'confusion_matrix.png'")
    plt.show()

    # 5. Imprimir Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("--- Avaliação Concluída ---")

def main():
    """
    Função principal para orquestrar o projeto.
    """
    # Etapa 1: Baixar e preparar os dados
    if not os.path.exists(BASE_DIR):
        setup_dataset()
    else:
        print(f"Diretório '{BASE_DIR}' já existe. Pulando o download.")

    # Etapa 2: Criar geradores de imagem
    train_gen, val_gen, test_gen = create_image_generators()

    # Etapa 3: Construir o modelo
    model = build_model()

    # Etapa 4: Treinar o modelo
    history = train_model(model, train_gen, val_gen)
    
    # Salvar o modelo treinado
    model.save('pneumonia_cnn_model.h5')
    print("Modelo salvo como 'pneumonia_cnn_model.h5'")

    # Etapa 5: Avaliar o modelo
    evaluate_model(model, history, test_gen)

if __name__ == "__main__":
    main()