# --- IMPORTS et FONCTION DE PRÉPARATION ---
import os
import numpy as np
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load 

# --- CONFIGURATION GLOBALE ---
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
DATASET_NAME = "go_emotions"

# --- FONCTION DE PRÉPARATION DES DONNÉES ---
def load_and_prepare_data():
    print(f"1. Chargement du jeu de données '{DATASET_NAME}'...")
    raw_datasets = load_dataset(DATASET_NAME)

    id2label = raw_datasets["train"].features["labels"].feature.names
    print(f"   => {len(id2label)} catégories d'émotions brutes trouvées.")
    
    label2id = {name: i for i, name in enumerate(id2label)}
    
    def simplify_labels(example):
        if example["labels"]:
            # On prend uniquement la première étiquette pour le fine-tuning simple
            example["label"] = example["labels"][0] 
        else:
            # Assigner à "neutral" si la liste est vide (très rare)
            example["label"] = label2id['neutral']
        return example

    print("\n2. Simplification des étiquettes (mono-label)...")
    simplified_datasets = raw_datasets.map(simplify_labels)

    # Suppression des colonnes qui ne sont plus nécessaires ou qui interfèrent
    # On retire "author" par précaution, car il n'est pas toujours là
    simplified_datasets = simplified_datasets.remove_columns(["labels", "id"]) 

    print("\n3. Tokenisation des données...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    tokenized_datasets = simplified_datasets.map(tokenize_function, batched=True)
    print("   => Tokenisation terminée.")

    print("\n4. Mise au format PyTorch et nettoyage final...")
    # Renommer "label" en "labels" est crucial pour le Trainer
    final_datasets = tokenized_datasets.rename_column("label", "labels")
    final_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_dataset = final_datasets["train"]
    eval_dataset = final_datasets["test"]
    
    print("   => Préparation des données terminée.")
    return train_dataset, eval_dataset, id2label, label2id, MODEL_NAME

# --- FONCTION DE METRIQUES ET ENTRAÎNEMENT ---

# Chargement de la métrique F1 (dépend de scikit-learn)
metric = load("f1") 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Calcul du F1-score pondéré pour les 28 classes
    results = metric.compute(predictions=predictions, references=labels, average="weighted")
    # Ajout de l'accuracy pour la visibilité
    accuracy = np.mean(predictions == labels)
    results["accuracy"] = accuracy
    return results


if __name__ == "__main__":
    
    # 1. CHARGEMENT DES DONNÉES ET CONFIGURATION
    train_dataset, eval_dataset, id2label, label2id, model_name = load_and_prepare_data()

    print(f"Chargement du modèle pré-entraîné: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(id2label), 
        id2label=id2label, 
        label2id=label2id
    )
    # Le message "Some weights were not initialized..." est normal ici, car on ajoute le calque final.

   # 2. DÉFINITION DES ARGUMENTS D'ENTRAÎNEMENT (VERSION MINIMALE)
    print("\n2. Définition des arguments d'entraînement (minimal)...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
       eval_strategy="epoch",        # ANCIENNEMENT evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
         report_to="none",
    )

    # 3. CRÉATION ET LANCEMENT DU TRAINER
    print("\n3. Création du Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Démarrage du Fine-Tuning... Cela peut prendre beaucoup de temps sur CPU.")
    trainer.train()

    FINAL_MODEL_DIR = "./final_emotion_model"
    # Sauvegarde du modèle entraîné (le meilleur)
    trainer.save_model(FINAL_MODEL_DIR) 
    
    # Sauvegarde du tokenizer (très important pour l'inférence)
    trainer.tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print(f"\n✅ Modèle entraîné et sauvegardé sous : {FINAL_MODEL_DIR}")