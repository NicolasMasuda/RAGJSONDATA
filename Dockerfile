# Usar la imagen oficial de Python 3.11
FROM python:3.11-slim-bookworm

# Permitir que las declaraciones y los mensajes de registro aparezcan inmediatamente en los registros
ENV PYTHONUNBUFFERED True

# Copiar el código local en la imagen del contenedor
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar el servicio web al iniciar el contenedor. Aquí usamos gunicorn
# con un proceso de trabajador y 8 hilos.
# Para entornos con múltiples núcleos de CPU, aumenta el número de trabajadores
# para que sea igual a los núcleos disponibles.
# El tiempo de espera se establece en 0 para deshabilitar los tiempos de espera de los trabajadores y permitir que Cloud Run maneje el escalado de instancias.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app