import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import os

class ComputerPricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_file = "computer_model.pkl"
        self.preprocessor_file = "computer_preprocessor.pkl"
        self.dataset_path = "dataset_computadoras.csv"
        self.initialize_model()

    def initialize_model(self):
        """Carga o entrena el modelo"""
        if os.path.exists(self.model_file) and os.path.exists(self.preprocessor_file):
            try:
                self.model = joblib.load(self.model_file)
                self.preprocessor = joblib.load(self.preprocessor_file)
                print("Modelo cargado exitosamente")
                return
            except Exception as e:
                print(f"Error cargando modelo: {e}. Entrenando nuevo modelo...")
        
        self.train_model()

    def load_and_preprocess_data(self):
        """Carga y preprocesa el dataset"""
        df = pd.read_csv(self.dataset_path)
        
        # Limpieza básica de datos
        df = df.drop_duplicates().dropna()
        
        # Extraer valores numéricos de columnas combinadas
        df['RAM_GB'] = df['RAM'].str.extract(r'(\d+)').astype(int)
        df['Storage_GB'] = df['Storage'].str.extract(r'(\d+)').astype(float)
        df['Storage_Type'] = df['Storage'].apply(
            lambda x: 'SSD' if 'SSD' in str(x) else ('NVMe' if 'NVMe' in str(x) else 'HDD')
        )
        df['Screen_inch'] = df['Screen'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # Eliminar columnas originales
        df = df.drop(columns=['RAM', 'Storage', 'Screen'])
        
        return df

    def create_preprocessor(self):
        """Crea el pipeline de preprocesamiento"""
        numeric_features = ['RAM_GB', 'Storage_GB', 'Screen_inch']
        categorical_features = ['Brand', 'CPU', 'GPU', 'Storage_Type', 'Operating_System']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def train_model(self):
        """Entrena y guarda el modelo"""
        df = self.load_and_preprocess_data()
        self.preprocessor = self.create_preprocessor()
        
        X = df.drop('Price_USD', axis=1)
        y = df['Price_USD']
        
        X_processed = self.preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            early_stopping_rounds=10,
            eval_metric='mae'
        )
        
        self.model.fit(X_train, y_train, 
                      eval_set=[(X_test, y_test)],
                      verbose=True)
        
        # Evaluación
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n🔍 Evaluación del modelo:")
        print(f"- Error absoluto medio (MAE): ${mae:.2f}")
        print(f"- Coeficiente R²: {r2:.2f}")
        
        # Guardar modelo
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.preprocessor, self.preprocessor_file)
        print(f"\n💾 Modelo guardado en {self.model_file}")

    def predict_price(self, input_features):
        """Predice el precio para nuevas características"""
        try:
            input_df = pd.DataFrame([input_features])
            
            # Asegurar todas las columnas necesarias
            expected_cols = ['Brand', 'CPU', 'GPU', 'RAM_GB', 'Storage_GB', 'Storage_Type', 'Screen_inch', 'Operating_System']
            for col in expected_cols:
                if col not in input_df.columns:
                    # Valores por defecto
                    defaults = {
                        'RAM_GB': 8,
                        'Storage_GB': 256,
                        'Storage_Type': 'SSD',
                        'Screen_inch': 15.6,
                        'Operating_System': 'Windows 11'
                    }
                    input_df[col] = defaults.get(col, 'unknown')
            
            # Preprocesar y predecir
            processed_data = self.preprocessor.transform(input_df)
            price = self.model.predict(processed_data)[0]
            return max(300, round(float(price), 2))  # Precio mínimo $300
        
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            return None

# Función de interfaz para Flask
def predecir_precio_computadora(brand, cpu, gpu, ram, storage, storage_type, screen_size, os):
    """
    Args:
        brand: str (ej. 'Dell')
        cpu: str (ej. 'Intel i7')
        gpu: str (ej. 'NVIDIA RTX 3060')
        ram: int (ej. 16)
        storage: int (ej. 512)
        storage_type: str ('SSD', 'HDD' o 'NVMe')
        screen_size: float (ej. 15.6)
        os: str (ej. 'Windows 11')
    
    Returns:
        tuple: (precio_predicho, error_message)
    """
    try:
        predictor = ComputerPricePredictor()
        features = {
            'Brand': str(brand),
            'CPU': str(cpu),
            'GPU': str(gpu),
            'RAM_GB': int(ram),
            'Storage_GB': float(storage),
            'Storage_Type': str(storage_type),
            'Screen_inch': float(screen_size),
            'Operating_System': str(os)
        }
        
        price = predictor.predict_price(features)
        if price is None:
            return None, "No se pudo generar la predicción"
        
        return price, None
    
    except Exception as e:
        return None, f"Error en los datos de entrada: {str(e)}"

# Ejemplo de uso directo
if __name__ == "__main__":
    # Entrenar y probar el modelo
    predictor = ComputerPricePredictor()
    
    # Ejemplo de predicción
    precio = predictor.predict_price({
        'Brand': 'Dell',
        'CPU': 'Intel i7',
        'GPU': 'NVIDIA RTX 3060',
        'RAM_GB': 16,
        'Storage_GB': 512,
        'Storage_Type': 'SSD',
        'Screen_inch': 15.6,
        'Operating_System': 'Windows 11'
    })
    
    print(f"\n🖥️ Precio predicho para la configuración: ${precio:.2f}")