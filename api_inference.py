import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# ----------------------------------------------------------------------
# 1. Configuration du Logger
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 2. Chargement du Modèle (Correction pour Déploiement Cloud)
# ----------------------------------------------------------------------

# ID du modèle sur Hugging Face Hub (votre modèle)
# Lorsque vous chargez un modèle directement par son ID Hub,
# Transformers le télécharge automatiquement au premier appel (au démarrage).
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
# Initialisation de l'App FastAPI
app = FastAPI(
    title="Emotion Detection API",
    description="API to predict emotions (Happy, Sad, Angry, etc.) from text using a pre-trained Hugging Face model.",
    version="1.0.0"
)

# Variable pour stocker le pipeline (sera initialisé au premier appel pour économiser de la RAM)
# NOTE: Nous allons initialiser le pipeline directement pour capturer les erreurs de téléchargement
# le plus tôt possible lors du démarrage de Render.
emotion_pipeline = None

def load_model():
    """Charge le pipeline du modèle depuis Hugging Face Hub."""
    global emotion_pipeline
    try:
        logger.info(f"Attempting to load model from Hugging Face Hub: {MODEL_ID}...")
        
        # Le pipeline télécharge automatiquement le modèle s'il n'est pas en cache.
        emotion_pipeline = pipeline(
            "text-classification",
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            top_k=None # Permet d'obtenir tous les scores d'émotion
        )
        logger.info("Model loaded successfully from Hugging Face Hub.")
    except Exception as e:
        logger.error(f"FATAL ERROR: Could not load model from Hugging Face Hub. Reason: {e}")
        # Lancer une exception arrêtera le service si le chargement échoue
        raise RuntimeError(f"Model loading failed: {e}")

# Charger le modèle au démarrage de l'application
load_model()


# ----------------------------------------------------------------------
# 3. Schéma de Requête
# ----------------------------------------------------------------------
class EmotionRequest(BaseModel):
    """Définit le corps de la requête API."""
    text: str


# ----------------------------------------------------------------------
# 4. Route Principale
# ----------------------------------------------------------------------
@app.post("/predict_emotion")
async def get_emotion_prediction(request: EmotionRequest):
    """
    Accepte une chaîne de texte et retourne les probabilités d'émotion.
    """
    if not emotion_pipeline:
        # Mesure de sécurité si le chargement initial a échoué (ne devrait pas arriver avec load_model())
        raise HTTPException(status_code=503, detail="Model is not loaded or ready.")

    try:
        text = request.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        logger.info(f"Received text for prediction: '{text[:50]}...'")

        # Exécuter l'inférence
        results = emotion_pipeline(text)
        
        # L'API retourne une liste de listes, nous prenons le premier élément [0]
        # et le convertissons en un dictionnaire pour faciliter la lecture.
        formatted_results = {
            item['label']: item['score'] for item in results[0]
        }

        logger.info("Prediction successful.")
        return {"emotions": formatted_results}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

# ----------------------------------------------------------------------
# 5. Route de Santé (Health Check)
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    """Simple route pour vérifier que l'API est en cours d'exécution."""
    return {"status": "ok", "message": "Emotion Detection API is running and model is loaded."}
