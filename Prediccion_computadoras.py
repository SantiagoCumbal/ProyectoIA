import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Cargar el dataset
df = pd.read_csv("dataset_computadoras.csv")  # Asegúrate de tener el archivo en tu ruta de trabajo

# Crear categorías de precio (Baja, Media, Alta)
df['Price_Category'] = pd.qcut(df['Price_USD'], q=3, labels=['Baja', 'Media', 'Alta'])

# Eliminar columnas innecesarias
df_model = df.drop(columns=['ID', 'Price_USD'])

# Codificar variables categóricas
le = LabelEncoder()
for column in df_model.columns:
    if df_model[column].dtype == 'object':
        df_model[column] = le.fit_transform(df_model[column])

# Separar características y etiquetas
X = df_model.drop(columns=['Price_Category'])
y = df_model['Price_Category']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Baja', 'Media', 'Alta'],
            yticklabels=['Baja', 'Media', 'Alta'])
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.tight_layout()
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

# Importancia de características
importancias = pd.Series(model.feature_importances_, index=X.columns)
print("\nImportancia de características:\n")
print(importancias.sort_values(ascending=False))

conf_matrix_percent = confusion_matrix(y_test, y_pred, normalize='true') * 100

# Convertir a DataFrame con etiquetas
conf_matrix_df = pd.DataFrame(
    conf_matrix_percent,
    index=['Baja', 'Media', 'Alta'],
    columns=['Baja', 'Media', 'Alta']
).round(2)

# Imprimir la matriz en formato legible
print("\nMatriz de Confusión (en porcentaje):\n")
print(conf_matrix_df)


accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Exactitud del modelo: {accuracy * 100:.2f}%\n")

# Matriz de confusión normalizada (por filas)
conf_matrix_percent = confusion_matrix(y_test, y_pred, normalize='true') * 100

conf_matrix_df = pd.DataFrame(
    conf_matrix_percent,
    index=['Baja', 'Media', 'Alta'],
    columns=['Baja', 'Media', 'Alta']
).round(2)

print("Matriz de Confusión (en porcentaje):\n")
print(conf_matrix_df)

#primer resultado de entrenamiento es de 33,37%


from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\n✅ Exactitud MLP: {accuracy_mlp * 100:.2f}%")


params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor exactitud: {grid.best_score_}")


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\n✅ Exactitud mejorada MLP: {accuracy_mlp * 100:.2f}%")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Separar datos nuevamente después de codificar
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
