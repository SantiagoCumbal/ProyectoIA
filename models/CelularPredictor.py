import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

class PhonePricePredictor:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        """Carga o crea un nuevo modelo"""
        try:
            model = joblib.load('phone_model.pkl')
            print("Modelo cargado exitosamente")
            return model
        except:
            print("Entrenando nuevo modelo...")
            return self.train_model()

    def train_model(self):
        """Entrena y guarda un nuevo modelo"""
        # Datos de ejemplo (reemplaza con tu dataset real)
        brands = ['Samsung', 'Apple', 'Xiaomi', 'Huawei', 'Oppo']
        data = {
            'Brand': np.random.choice(brands, 300),
            'RAM_GB': np.random.choice([4, 6, 8, 12], 300),
            'Storage_GB': np.random.choice([64, 128, 256, 512], 300),
            'Screen_inch': np.round(np.random.uniform(5.0, 7.0, 300), 1),
            'Camera_MP': np.random.choice([12, 16, 32, 48, 64], 300),
            'Price_USD': np.zeros(300)
        }
        
        # Calcular precios realistas
        for i in range(300):
            base = 800 if data['Brand'][i] == 'Apple' else 400
            data['Price_USD'][i] = base + \
                data['RAM_GB'][i] * 30 + \
                data['Storage_GB'][i] * 0.8 + \
                data['Screen_inch'][i] * 50 + \
                data['Camera_MP'][i] * 2
        
        df = pd.DataFrame(data)
        
        # Preprocesamiento
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['RAM_GB', 'Storage_GB', 'Screen_inch', 'Camera_MP']),
            ('cat', OneHotEncoder(), ['Brand'])
        ])
        
        X = preprocessor.fit_transform(df.drop('Price_USD', axis=1))
        y = df['Price_USD']
        
        # Entrenamiento
        model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1)
        model.fit(X, y)
        
        # Guardar modelo
        joblib.dump(model, 'phone_model.pkl')
        return model

    def predict(self, features):
        """Realiza una predicción"""
        try:
            # Convertir a DataFrame
            input_df = pd.DataFrame([features])
            
            # Preprocesamiento (simplificado para el ejemplo)
            input_df['RAM_GB'] = input_df['RAM_GB'].astype(int)
            input_df['Storage_GB'] = input_df['Storage_GB'].astype(int)
            input_df['Screen_inch'] = input_df['Screen_inch'].astype(float)
            input_df['Camera_MP'] = input_df['Camera_MP'].astype(int)
            
            # Predicción simple (en un caso real usarías el preprocesador)
            price = 400 + \
                   (input_df['RAM_GB'][0] * 30) + \
                   (input_df['Storage_GB'][0] * 0.8) + \
                   (input_df['Screen_inch'][0] * 50) + \
                   (input_df['Camera_MP'][0] * 2)
            
            if input_df['Brand'][0] == 'Apple':
                price *= 1.5
            
            return {
                'price': max(100, round(float(price), 2)),
                'features': features
            }
        except Exception as e:
            return {'error': str(e)}

# Instancia global para el servidor (opcional)
predictor = PhonePricePredictor()