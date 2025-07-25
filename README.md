# Proyecto IA
Desarrollar un sistema interactivo tipo chatbot que permita predecir el precio aproximado de celulares y computadoras a partir de las características técnicas proporcionadas por el usuario.

### Integrantes: 
> Edwin Sarango
> Santiago Cumbal
## Objetivos
1. Facilitar la estimación de precios para dispositivos tecnológicos de forma rápida, intuitiva y automatizada.
2. Guiar al usuario paso a paso mediante un flujo conversacional que recolecta las especificaciones técnicas clave del equipo.
3. Simular la lógica de un asesor tecnológico, utilizando modelos de predicción basados en reglas o inteligencia artificial (según el backend).
4. Ofrecer una experiencia web accesible, sin necesidad de conocimientos técnicos ni búsqueda manual de precios.
5. Integrar con modelos de Machine Learning o reglas heurísticas, mostrando cómo se puede conectar una interfaz amigable con motores de predicción.

## Desarrollo y resultados 
#  Chatbot Predictor de Precios

Este proyecto es un **chatbot interactivo** que permite predecir el precio estimado de **celulares** y **computadoras** según las especificaciones proporcionadas por el usuario. Utiliza un flujo conversacional paso a paso, procesamiento de texto y modelos de predicción (basados en reglas o machine learning).
---

## 📁 Estructura del Proyecto
📂 proyectoIA/
├── index.html # Interfaz principal del chatbot
├── styles.css # Estilos del chatbot
├── scripts/
│ ├── predict.js # Lógica de predicción para celulares y computadoras
│ └── conversational.js# Lógica del flujo conversacional
├── app/
│ └── (Flask API - opcional si usas ML backend)

---
### 1. `index.html`
- Contiene la estructura visual del chatbot.
- Incluye:
  - Encabezado con botón de cambio de modo (Celulares / Computadoras).
  - Contenedor de mensajes del chat.
  - Campo de entrada y botones de control.

---

### 2. `scripts/predict.js`
- Contiene los **modelos de predicción por reglas**.
- Estructura:
  - `phoneModel`: Modelo para celulares.
  - `computerModel`: Modelo para computadoras.

#### Cada modelo tiene:
- `extractFeatures(texto)`: Extrae RAM, almacenamiento, pantalla, procesador, etc.
- `predict(features)`: Estima el precio basado en ponderaciones de características.

---

### 3. `scripts/conversational.js`
- Controla el **flujo de conversación paso a paso**.
- Pregunta una característica a la vez.
- Guarda respuestas en `conversationState`.
- Al completar todas las preguntas, llama al predictor y muestra el resultado.

## 💬 Flujo Conversacional

El chatbot sigue un flujo guiado:

### Para celulares:
1. Marca / modelo
2. RAM
3. Almacenamiento
4. Tamaño de pantalla
5. Cámara

### Para computadoras:
1. Marca / modelo
2. Procesador
3. RAM
4. Tarjeta gráfica (GPU)
5. Almacenamiento
6. Tamaño de pantalla
7. Sistema operativo

Cuando se completan todas las preguntas, se muestra una predicción del precio estimado y un resumen visual.

---

## 🔘 Funcionalidades Adicionales

- `switchMode(mode)`: Cambia entre modo celulares y computadoras.
- `sendMessage()`: Envía el mensaje y procesa la lógica del paso siguiente.
- `newChat()`: Reinicia la conversación desde cero.
- `formatPredictionResult()`: Muestra el resultado con formato visual amigable.

---

## 🚀 ¿Cómo comenzar?

1. Abre `index.html` en tu navegador.
2. Selecciona el modo: **Celulares** o **Computadoras**.
3. Responde a cada pregunta que el chatbot te haga.
4. Recibe una predicción del precio y un resumen técnico.

---
## 📸 Vista previa

![demo] <img width="1917" height="979" alt="image" src="https://github.com/user-attachments/assets/aa9dd9c6-8871-4fc9-87ac-276a723e8142" />

---
Requerimentos básicos para AI TechAsistor:
```
flask
scikit-learn
pandas
numpy
joblib
flask-cors
```


## Conclusiones 
El desarrollo de este proyecto demuestra cómo un chatbot puede ser una herramienta eficiente y amigable para asistir a los usuarios en la recolección progresiva de datos, reemplazando formularios tradicionales por una experiencia conversacional más natural. A través de la integración de modelos de predicción basados en Machine Learning (para computadoras) y lógica basada en reglas (para celulares), se logró automatizar la estimación de precios tecnológicos en función de especificaciones técnicas clave como marca, memoria RAM, almacenamiento, pantalla y cámara.
El sistema fue diseñado con una arquitectura clara que separa el frontend (interfaz en HTML, CSS y JavaScript) del backend (CVS y PYs entrenados), lo que facilita la escalabilidad, mantenimiento y mejora continua del modelo predictivo. La implementación de un flujo conversacional paso a paso mejora la precisión en la recolección de datos, ya que guía al usuario de manera secuencial a través de las características necesarias para la predicción.
Además, el proyecto demostró la posibilidad de adaptar el mismo flujo a distintos tipos de dispositivos (como celulares y computadoras), haciendo el sistema flexible y extensible. Finalmente, se concluye que la combinación de tecnologías web con inteligencia artificial aplicada ofrece una solución robusta y práctica para resolver problemas del mundo real, como la estimación de precios, en tiempo real y desde una interfaz accesible para cualquier usuario.


## Licencia
EPN. Uso libre con fines educativos y de desarrollo.
