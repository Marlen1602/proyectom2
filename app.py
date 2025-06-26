from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

# Cargar modelo entrenado (asegúrate de tener estos archivos)
model = joblib.load('modelo.pkl')  # Guarda tu modelo entrenado
scaler = joblib.load('scaler_rfe.pkl')  # Guarda tu escalador

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convertir checkboxes a 0/1
        weather_rainy = 1 if request.form.get('weather_rainy') == '1' else 0
        weather_snowy = 1 if request.form.get('weather_snowy') == '1' else 0
        
        # Validar y convertir inputs numéricos
        def get_float_input(field_name, min_val=0):
            try:
                value = float(request.form[field_name])
                return max(value, min_val)
            except (ValueError, KeyError):
                raise ValueError(f"Valor inválido para {field_name.replace('_', ' ')}")

        input_data = np.array([
            get_float_input('synthetic_effort', 1),  # Synthetic_Effort (1-10)
            get_float_input('distance_km'),         # Distance_km
            get_float_input('prep_time'),           # Preparation_Time_min
            min(max(get_float_input('traffic_level'), 1), 5),  # Traffic_Level_Num (1-5)
            weather_rainy,                          # Weather_Rainy (0/1)
            weather_snowy                           # Weather_Snowy (0/1)
        ]).reshape(1, -1)

        
        # Escalar los datos
        scaled_data = scaler.transform(input_data)
        
        # Predecir con restricción física
        raw_pred = model.predict(scaled_data)[0]
        min_time = input_data[0][2] * 1.15  # Tiempo mínimo (prep_time * 1.15)
        prediction = max(raw_pred, min_time)
        
        return render_template('resultado.html',
                            prediction=round(prediction, 2),
                            min_time=round(min_time, 2),
                            input_data=request.form)
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)