import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
# --- CONFIGURATION ---
MODEL_PATH = "./final_emotion_model"
MODEL_NAME = "bert-base-multilingual-cased" # Gardez-le en commentaire ou pour référence
# ...
# Le tokenizer doit être le même que celui utilisé pour l'entraînement
# 💥 CORRECTION : Chargez le tokenizer depuis votre dossier local, pas depuis le Hub !
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 

# Le modèle sauvegardé contient la configuration des étiquettes (id2label)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
TEXT_TO_ANALYZE_FR = "Je suis vraiment frustré par la lenteur de ce code. Je n'arrive pas à avancer."
# L'utilisation du français est un test clé car nous avons entraîné le modèle sur de l'anglais
# avec un tokenizer multilingue.

def run_inference():
    print(f"Chargement du modèle entraîné depuis : {MODEL_PATH}")
    
    # Le tokenizer doit être le même que celui utilisé pour l'entraînement
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Le modèle sauvegardé contient la configuration des étiquettes (id2label)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Créer le 'pipeline' de classification (l'outil le plus simple pour l'inférence)
    emotion_pipeline = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer
    )

    print("\n--- TEST DE PRÉDICTION ---")
    print(f"Texte à analyser (FR) : '{TEXT_TO_ANALYZE_FR}'")
    
    # Lancer la prédiction
    result = emotion_pipeline(TEXT_TO_ANALYZE_FR)
    
    # Afficher le résultat
    if result:
        emotion_label = result[0]['label']
        score = result[0]['score']
        
        print("\n✅ Résultat de l'analyse émotionnelle :")
        print(f"   Émotion prédite : {emotion_label}")
        print(f"   Score de confiance : {score:.4f}")
        
        # --- L'innovation UX commence ici ---
        # Cette partie doit être intégrée à votre application React Native
        if emotion_label == 'frustration':
            print("\n💡 ACTION UX SUGGÉRÉE : Détectant de la frustration, l'application propose de sauvegarder et de prendre une pause de 10 minutes.")
        elif emotion_label == 'concentration':
            print("\n💡 ACTION UX SUGGÉRÉE : Détectant une forte concentration, l'application active un mode 'Ne pas déranger' automatique.")
        else:
             print("\n💡 ACTION UX SUGGÉRÉE : Aucune action spécifique n'est suggérée pour cette émotion.")


if __name__ == "__main__":
    # Assurez-vous d'abord que le dossier final_emotion_model existe (entraînement terminé)
    if os.path.exists(MODEL_PATH):
        run_inference()
    else:
        print(f"⚠️ Erreur : Le modèle n'a pas été trouvé à l'emplacement {MODEL_PATH}.")
        print("   Veuillez attendre que le script emotion_fine_tuning.py termine son entraînement.")