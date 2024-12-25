from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from whisper import load_model
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:50958"],  # Cambia por el origen de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos
whisper_model = load_model("base")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Pipeline de diarización con autenticación
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=huggingface_token
)

# Carpeta temporal para guardar archivos de audio
os.makedirs("audio", exist_ok=True)

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    print(f"HUGGINGFACE_TOKEN: {huggingface_token}")
    print(f"OPENAI_API_KEY: {openai_api_key}")
    
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
