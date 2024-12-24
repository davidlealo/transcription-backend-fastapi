from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from whisper import load_model
from pyannote.audio import Pipeline
import os

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por una lista específica de orígenes en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos
whisper_model = load_model("base")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")  # Obtén el token desde la variable de entorno

# Pipeline de diarización con autenticación
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=huggingface_token
)

# Carpeta temporal para guardar archivos de audio
os.makedirs("audio", exist_ok=True)

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # Guardar audio temporal
    file_path = f"audio/{file.filename}"
    with open(file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    
    # Diarización
    diarization = diarization_pipeline(file_path)
    speakers = []
    
    # Transcripción segmentada
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = int(turn.start * 1000), int(turn.end * 1000)
        text = whisper_model.transcribe(file_path, task="transcribe", segment=(start, end))['text']
        results.append({"speaker": speaker, "text": text})
        speakers.append(speaker)

    return {"transcription": results, "speakers": list(set(speakers))}
