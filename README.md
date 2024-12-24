# Backend de Transcripción de Audio

Este proyecto es un backend desarrollado en FastAPI para la transcripción de archivos de audio y asociación de oradores con sus identidades basadas en el contenido del audio.

## Características

- Procesamiento de archivos de audio cargados por el cliente.
- Diarización para identificar segmentos de audio por orador.
- Asociación de nombres con los oradores basados en entidades nombradas extraídas del texto transcrito.
- Uso de modelos avanzados como Whisper y PyAnnote.

## Tecnologías utilizadas

- **FastAPI**: Framework para el desarrollo del backend.
- **Whisper**: Modelo para la transcripción de audio.
- **PyAnnote**: Herramienta para diarización de oradores.
- **spaCy**: Procesamiento de lenguaje natural para la detección de nombres propios.

## Configuración del proyecto

### Prerrequisitos

Asegúrate de tener instalados los siguientes programas:

- Python 3.9 o superior: [Instalación](https://www.python.org/downloads/)
- Pipenv (opcional para la gestión del entorno virtual):
  ```bash
  pip install pipenv
  ```

### Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/davidlealo/transcription-backend-fastapi.git
   cd transcription-backend-fastapi
   ```

2. Crea y activa un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno necesarias creando un archivo `.env`:

   ```env
   HUGGINGFACE_TOKEN=tu_token_de_huggingface
   OPENAI_API_KEY=tu_token_de_openai
   ```

5. Inicia el servidor:

   ```bash
   uvicorn main:app --reload
   ```

El servidor estará disponible en `http://localhost:8000`.

## Endpoints principales

### `/process_audio/` (POST)

- **Descripción**: Procesa un archivo de audio para transcribir su contenido y realizar diarización.
- **Parámetros**: Archivo de audio cargado.
- **Respuesta**: Transcripción del audio segmentada por orador.

### `/associate_person/` (POST)

- **Descripción**: Procesa un archivo de audio para identificar oradores y asociarlos con nombres basados en entidades nombradas.
- **Parámetros**: Archivo de audio cargado.
- **Respuesta**: Diccionario con oradores y nombres asociados.

## Pruebas

Para ejecutar pruebas, utiliza el comando:

```bash
pytest
```

Asegúrate de configurar las variables de entorno adecuadas antes de realizar las pruebas.

## Contacto

Si tienes preguntas, sugerencias o algún problema con el proyecto, no dudes en contactarme:

- **Email**: [davidlealo@gmail.com](mailto:davidlealo@gmail.com)
- **GitHub**: [https://github.com/davidlealo](https://github.com/davidlealo)
- **LinkedIn**: [https://www.linkedin.com/in/davidlealo/](https://www.linkedin.com/in/davidlealo/)

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

