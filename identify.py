from fastapi import FastAPI, File, UploadFile
from pyannote.audio import Pipeline
from whisper import load_model
import spacy
import os

app = FastAPI()

# Modelos
nlp = spacy.load("en_core_web_sm")
whisper_model = load_model("base")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Pipeline de diarización
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=huggingface_token
)

@app.post("/associate_person/")
async def associate_person(file: UploadFile = File(...)):
    # Procesar audio
    file_path = f"audio/{file.filename}"
    with open(file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    
    diarization = diarization_pipeline(file_path)
    person_mapping = {}
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = int(turn.start * 1000), int(turn.end * 1000)
        text = whisper_model.transcribe(file_path, task="transcribe", segment=(start, end))['text']
        
        # Analizar texto para detectar nombres
        doc = nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        if names:
            person_mapping[speaker] = names[0]
    
    return {"person_mapping": person_mapping}
