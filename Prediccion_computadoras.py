import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib
import numpy as np

# Configuración inicial
random_state = 42
plt.style.use('ggplot')

# 1. Cargar y preparar los datos
def load_and_prepare_data(filepath):
    """Carga y prepara el dataset para el modelado."""
    try:
        df = pd.read_csv(filepath)

        # Eliminar columnas de precio
        df = df.drop(columns=['ID', 'Price_USD'])

        # Extraer valores numéricos de RAM y Storage
        df['RAM_GB'] = df['RAM'].str.extract(r'(\d+)').astype(int)
        df['Storage_GB'] = df['Storage'].str.extract(r'(\d+)').astype(float)
        df = df.drop(columns=['RAM', 'Storage'])

        # Crear variable objetivo basada en calidad de componentes
        df['Quality_Score'] = calculate_quality_score(df)

        # Clasificar en categorías de calidad
        df['Quality_Category'] = pd.qcut(df['Quality_Score'], q=3, labels=['Baja', 'Media', 'Alta'])

        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def calculate_quality_score(df):
    """
    Calcula un score de calidad basado en los componentes.
    """
    score = 0

    score += df['CPU'].apply(lambda x: 3 if 'i7' in str(x) or 'Ryzen 7' in str(x) else 
                                  (2 if 'i5' in str(x) or 'Ryzen 5' in str(x) else 1))

    score += df['RAM_GB'] * 0.5
    score += df['Storage_GB'] * 0.1

    score += df['GPU'].apply(lambda x: 3 if 'RTX' in str(x) else 
                           (2 if 'GTX' in str(x) else 1))

    return (score - score.min()) / (score.max() - score.min()) * 100

# 2. Preprocesamiento de datos
def preprocess_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('Quality_Category', errors='ignore')
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(df['Quality_Category'])

    X = df.drop(columns=['Quality_Category', 'Quality_Score'])
    return X, y, label_encoders, le_target

# 3. Entrenamiento y evaluación de modelos
def train_and_evaluate(X, y, le_target):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), list(range(X.shape[1])))
    ])

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    rf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    }

    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    mlp_pipeline = make_imb_pipeline(
        preprocessor,
        SMOTE(random_state=random_state),
        MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            random_state=random_state
        )
    )

    mlp_pipeline.fit(X_train, y_train)
    y_pred_mlp = mlp_pipeline.predict(X_test)

    def evaluate_model(y_true, y_pred, model_name, le):
        print(f"\n{'='*50}")
        print(f"Evaluación del modelo {model_name}")
        print(f"{'='*50}")
        print("\nReporte de Clasificación:\n")
        print(classification_report(y_true, y_pred, target_names=le.classes_))

        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix_percent = confusion_matrix(y_true, y_pred, normalize='true') * 100

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Matriz de Confusión (Conteos)")
        plt.xlabel("Predicho")
        plt.ylabel("Real")

        plt.subplot(1, 2, 2)
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Matriz de Confusión (% por clase real)")
        plt.xlabel("Predicho")
        plt.ylabel("Real")

        plt.tight_layout()
        plt.show()

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✅ Exactitud del modelo: {accuracy * 100:.2f}%")

    evaluate_model(y_test, y_pred_rf, "Random Forest", le_target)
    evaluate_model(y_test, y_pred_mlp, "MLP Classifier", le_target)

    if hasattr(best_rf.named_steps['classifier'], 'feature_importances_'):
        importances = best_rf.named_steps['classifier'].feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de Características")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    return best_rf, mlp_pipeline

# 4. Guardar modelos y componentes
def save_models_and_components(rf_model, mlp_model, label_encoders, le_target):
    try:
        joblib.dump(rf_model, 'quality_random_forest_model.pkl')
        joblib.dump(mlp_model, 'quality_mlp_model.pkl')
        joblib.dump(label_encoders, 'quality_feature_encoders.pkl')
        joblib.dump(le_target, 'quality_label_encoder.pkl')

        print("\n✅ Modelos y componentes guardados correctamente:")
        print("- quality_random_forest_model.pkl")
        print("- quality_mlp_model.pkl")
        print("- quality_feature_encoders.pkl")
        print("- quality_label_encoder.pkl")
    except Exception as e:
        print(f"Error al guardar los modelos: {e}")

# 5. Funcion principal
def main():
    df = load_and_prepare_data("dataset_computadoras.csv")
    if df is None:
        return

    X, y, label_encoders, le_target = preprocess_data(df)
    rf_model, mlp_model = train_and_evaluate(X, y, le_target)
    save_models_and_components(rf_model, mlp_model, label_encoders, le_target)

if __name__ == "__main__":
    main()