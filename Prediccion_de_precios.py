#Primer enfoque Sistema de predicción

#Creacion del modelo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("celulares.csv")
df = df.drop_duplicates()
print(df.isnull().sum())

X = df.drop("price_range", axis=1)
y = df["price_range"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("=== Matriz de Confusión ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))
print(f"\n🎯 Exactitud: {accuracy_score(y_test, y_pred)*100:.2f}%")