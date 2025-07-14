from flask import Flask, render_template, request
import numpy as np
import joblib

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el transformador PCA
modelo = joblib.load('titanic_model.pkl')
pca = joblib.load('pca_model.pkl')

# Ruta para mostrar el formulario
@app.route('/')
def formulario():
    return render_template('formulario.html')

# Ruta para procesar la predicción
@app.route('/predict', methods=['POST'])
def predecir():
    try:
        # Obtener los valores ingresados por el usuario
        sex = float(request.form['sex'])
        fare = float(request.form['fare'])
        pclass = float(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = float(request.form['sibsp'])

        # Crear arreglo y aplicar PCA
        datos = np.array([[sex, fare, pclass, age, sibsp]])
        datos_pca = pca.transform(datos)

        # Realizar la predicción
        prediccion = modelo.predict(datos_pca)[0]
        resultado = "🛟 Sobrevive" if prediccion == 1 else "⚰️ No sobrevive"

        return render_template('index.html', prediccion=resultado)

    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

# Ejecutar localmente (no se usa en Render)
if __name__ == '__main__':
    app.run(debug=True)
