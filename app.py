from flask import Flask, render_template, request
import joblib
import numpy as np
import logging
import os
import warnings

# Suprimir warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Inicializar Flask
app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Archivos del modelo Titanic
MODEL_PATH = 'titanic_model.pkl'
PCA_PATH = 'pca_model.pkl'

# Características utilizadas
feature_names = ['Age', 'Ticket', 'Fare', 'SibSp', 'Sex']

# Cargar modelo y PCA
def load_model_and_pca():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PCA_PATH):
        raise FileNotFoundError("Modelo o PCA no encontrado.")
    
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    
    return model, pca

# Intentar cargar al inicio
try:
    model, pca = load_model_and_pca()
    logging.info("✅ Modelo y PCA cargados correctamente.")
except Exception as e:
    logging.error(f"❌ Error al cargar el modelo o PCA: {str(e)}")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("🔍 Petición POST recibida")

        # Extraer datos del formulario con los nombres correctos
        edad = float(request.form['Age'])
        ticket = float(request.form['Ticket'])
        fare = float(request.form['Fare'])
        sibsp = float(request.form['SibSp'])
        sex = float(request.form['Sex'])

        # Construir input
        input_data = np.array([[edad, ticket, fare, sibsp, sex]])
        logging.info(f"🧾 Datos recibidos: {input_data}")

        # Aplicar PCA
        datos_transformados = pca.transform(input_data)
        logging.info(f"📉 Datos tras PCA: {datos_transformados}")

        # Predecir
        resultado = model.predict(datos_transformados)[0]
        pred = "🛟 Sobrevive" if resultado == 1 else "⚰️ No sobrevive"

        return render_template('formulario.html', prediction=pred)

    except Exception as e:
        logging.error(f"❌ Error en predicción: {str(e)}")
        return render_template('formulario.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    logging.info("🚢 Iniciando servidor Flask - Predicción Titanic")
    app.run(debug=True, host='0.0.0.0', port=5000)
