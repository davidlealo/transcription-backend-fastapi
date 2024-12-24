import spacy

nlp = spacy.load("en_core_web_sm")  # Modelo de lenguaje en inglés

@app.post("/associate_person/")
async def associate_person(file: UploadFile = File(...)):
    # Procesar audio (como antes)
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
