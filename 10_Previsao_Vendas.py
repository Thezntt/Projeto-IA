import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# --- 1. Carregar os Dados ---

# URLs dos datasets (encontrados em um repositório público)
store_url = 'https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/store.csv'
train_url = 'https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv'

print("Carregando datasets...")
try:
    df_store = pd.read_csv(store_url, low_memory=False)
    df_train = pd.read_csv(train_url, low_memory=False)
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    print("Verifique a conexão com a internet ou as URLs dos arquivos.")

# --- 2. Preparação e Limpeza dos Dados ---

print("Preparando e limpando os dados...")
# Juntar os dados de treino com as informações das lojas
df = pd.merge(df_train, df_store, on='Store')

# Converter data para datetime
df['Date'] = pd.to_datetime(df['Date'])

# Vamos focar apenas em lojas abertas e que tiveram vendas
df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

# Tratamento de dados faltantes (ex: preencher 'CompetitionDistance' com a mediana)
median_dist = df['CompetitionDistance'].median()
df['CompetitionDistance'].fillna(median_dist, inplace=True)

# --- 3. Feature Engineering (Engenharia de Features) ---

print("Criando novas features...")
# Extrair features de sazonalidade (como solicitado)
df['Ano'] = df['Date'].dt.year
df['Mes'] = df['Date'].dt.month
df['Trimestre'] = df['Date'].dt.quarter
df['DiaDaSemana'] = df['Date'].dt.dayofweek

# Codificar 'Promo' (já é binária, 0 ou 1, o que é ótimo)
# 'Promo' é a nossa feature de promoção solicitada

# Codificar variáveis categóricas usando One-Hot Encoding
# StateHoliday, StoreType, e Assortment são categóricas
categorical_features = ['StateHoliday', 'StoreType', 'Assortment']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# --- 4. Preparação para Modelagem ---

print("Preparando dados para os modelos...")
# Definir as colunas que serão usadas como features (X) e o alvo (y)
target = 'Sales'

# Lista de features
features = [
    'Store',
    'DiaDaSemana',
    'Promo',
    'SchoolHoliday',
    'CompetitionDistance',
    'Ano',
    'Mes',
    'Trimestre'
]

# Adicionar as novas colunas criadas pelo get_dummies
# (ex: 'StateHoliday_a', 'StoreType_b', etc.)
for col in df.columns:
    if any(col.startswith(cat_feat) for cat_feat in categorical_features):
        features.append(col)

# Garantir que todas as features existem (em caso de algum filtro ter removido todas)
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]

# Dividir em dados de treino e teste
# Usaremos uma amostra menor para rodar mais rápido (sample_frac = 0.1)
# Se quiser usar todos os dados, remova as linhas .sample()
print("Usando uma amostra de 10% dos dados para agilidade...")
X_sample = X.sample(frac=0.1, random_state=42)
y_sample = y.loc[X_sample.index]

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# --- 5. Treinamento e Avaliação dos Modelos ---

# Dicionário para guardar os modelos
models = {
    'Regressão Linear': LinearRegression(),
    'Árvore de Decisão': DecisionTreeRegressor(max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror', n_jobs=-1)
}

# Lista para guardar os resultados
results = []

print("\n--- Iniciando Treinamento e Avaliação ---")

for name, model in models.items():
    print(f"Treinando {name}...")
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar o modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Guardar resultados
    results.append({
        'Modelo': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
    
    print(f"Resultados para {name}:")
    print(f"  RMSE (Erro Médio): {rmse:.2f}")
    print(f"  MAE (Erro Médio Absoluto): {mae:.2f}")
    print(f"  R² (Coef. de Determinação): {r2:.4f}\n")

# --- 6. Comparação Final ---

print("--- Comparação Final dos Modelos ---")
df_results = pd.DataFrame(results).set_index('Modelo')
print(df_results.sort_values(by='RMSE'))