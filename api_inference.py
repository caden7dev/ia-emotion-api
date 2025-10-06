# ID du modèle public sur Hugging Face Hub (pour tenter d'économiser de la mémoire)
# NOTE : Ce modèle fonctionne en ANGLAIS.
MODEL_ID = "dima806/distilbert-base-uncased-finetuned-emotion"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import logging

# Configuration du journal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_inference")

# Initialisation de l'API FastAPI
app = FastAPI()

# Le modèle sera chargé au démarrage du serveur (voir la fonction startup)
emotion_analyzer = None

# Définition de la structure de la requête (le corps JSON)
class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """Charge le modèle d'IA au démarrage du serveur."""
    global emotion_analyzer
    try:
        logger.info(f"Attempting to load model from Hugging Face Hub: {MODEL_ID}...")
        
        # Le pipeline va automatiquement télécharger et charger le tokenizer et le modèle
        # Utilisation du pipeline 'text-classification'
        emotion_analyzer = pipeline(
            "text-classification",
            model=MODEL_ID,
            return_all_scores=True  # Pour obtenir tous les scores d'émotion
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        # Si le chargement échoue, on laisse emotion_analyzer à None et l'endpoint /predict_emotion gérera l'erreur.

@app.get("/")
def read_root():
    """Endpoint de base pour vérifier que l'API est en ligne."""
    return {"status": "ok", "message": "Emotion detection API is running and waiting for model load."}

@app.post("/predict_emotion")
def predict_emotion(request: TextRequest):
    """
    Endpoint pour prédire les émotions à partir d'un texte donné.
    Le corps de la requête doit être un JSON: {"text": "votre texte ici"}
    """
    if emotion_analyzer is None:
        return {"error": "Model not loaded. Check server logs for memory or authorization errors."}, 500
    
    try:
        # Le pipeline retourne une liste de listes de dictionnaires
        results = emotion_analyzer(request.text)
        
        # Le format est [[{'label': 'joy', 'score': 0.9}, ...]]
        emotions = {item['label']: item['score'] for item in results[0]}
        
        return {"text": request.text, "emotions": emotions}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}, 500
