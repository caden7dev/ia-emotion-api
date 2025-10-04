import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_PATH = "./final_emotion_model"
# Assurez-vous que le modèle est bien chargé
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le dossier du modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}")

# 1. Chargement du Modèle et du Tokenizer (effectué une seule fois au démarrage de l'API)
print("Démarrage de l'API : Chargement du modèle en mémoire...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

emotion_pipeline = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer
)
print("✅ Modèle d'analyse émotionnelle prêt.")

# 2. Définition de l'Application FastAPI
app = FastAPI(
    title="Emotion Inference API",
    description="API pour l'analyse émotionnelle des textes en temps réel (BERT Multilingue)."
)

# 3. Définition du Schéma de Données pour la Requête
class TextRequest(BaseModel):
    text: str

# 4. Définition du Endpoint de Prédiction
@app.post("/predict_emotion")
def predict_emotion(request: TextRequest):
    """
    Analyse le texte fourni et retourne l'émotion prédite et le score de confiance.
    """
    
    # 5. Prédiction : DEMANDE DES 5 MEILLEURS RÉSULTATS
    # Cela permet de vérifier l'hésitation du modèle
    result_top_k = emotion_pipeline(request.text, top_k=5)
    
    # Affiche les 5 meilleurs résultats pour le débogage dans le terminal Uvicorn
    print("\n--- Meilleurs résultats du Modèle ---")
    print(result_top_k)
    print("-----------------------------------\n")

    # 6. LOGIQUE DE CORRECTION POUR LE "NEUTRE" FAIBLE
    
    emotion_data = result_top_k[0] # Le résultat le plus confiant (souvent 'neutre')
    
    # Si le résultat principal est 'neutre', on regarde si une autre émotion est proche
    if emotion_data['label'] == 'neutre' and len(result_top_k) > 1:
        
        # Récupère l'émotion non-neutre la plus probable (le deuxième meilleur résultat)
        best_non_neutral = result_top_k[1]
        
        # Seuil d'hésitation : si l'émotion non-neutre est à moins de 5 points de pourcentage (0.05) du neutre
        NEUTRAL_TOLERANCE = 0.05 
        
        if (emotion_data['score'] - best_non_neutral['score']) < NEUTRAL_TOLERANCE:
            # Si le modèle hésite, on choisit l'émotion non-neutre pour l'action UX
            print(f"Correction: Le modèle hésitait (Neutre {emotion_data['score']:.2f} vs {best_non_neutral['label']} {best_non_neutral['score']:.2f}). Choix de {best_non_neutral['label']}.")
            
            emotion_label = best_non_neutral['label']
            score = best_non_neutral['score']
        else:
            # Si 'neutre' est beaucoup plus confiant (écart > 0.05), on le garde.
            emotion_label = emotion_data['label']
            score = emotion_data['score']
    
    else:
        # Si la meilleure prédiction n'est pas 'neutre', on la garde
        emotion_label = emotion_data['label']
        score = emotion_data['score']
        
    # --- LOGIQUE D'ACTION UX DÉPENDANT DE L'ÉMOTION FINALE ---
    ux_action = "Aucune action spécifique n'est suggérée pour cette émotion."
    
    # 🚨 RÈGLE 1 : GESTION DES ÉMOTIONS DE DÉTRESSE (Éthique et Sécurité)
    if emotion_label in ['tristesse', 'chagrin', 'remords', 'peur']: 
        ux_action = "Parle de ta situation à une personne de confiance pour te conseiller et t'aider."
    
    # RÈGLE 2 : GESTION DES ÉMOTIONS NÉGATIVES FORTES (Frustration / Colère)
    elif emotion_label in ['colère', 'ennui', 'déception', 'désapprobation']:
        ux_action = "Je vois que tu n'es pas de bonne humeur. Je te suggère de prendre une pause d'au moins 30 minutes pour te relaxer et revenir, et surtout n'oublie pas de sauvegarder ton travail en cours."
    
    # RÈGLE 3 : GESTION DES ÉMOTIONS POSITIVES (Encouragement)
    elif emotion_label in ['joie', 'excitation', 'amour', 'fierté', 'gratitude', 'optimisme', 'admiration']:
        ux_action = "Félicitation pour ta bonne humeur. Continue dans cet état d'esprit jusqu'à la fin."
    
    # FIN DE LA LOGIQUE D'ACTION UX
    
    return {
        "text": request.text,
        "emotion": emotion_label,
        "score": f"{score:.4f}",
        "ux_action_sugeree": ux_action
    }

# 7. Endpoint de Bienvenue
@app.get("/")
def home():
    return {"message": "Bienvenue à l'API d'Inférence Émotionnelle. Utilisez l'endpoint /predict_emotion."}
