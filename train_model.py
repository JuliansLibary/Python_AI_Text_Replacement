import spacy
from spacy.training.example import Example
import json

def load_train_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    train_data = []
    for entry in data:
        text = entry["text"]
        entities = [(ent["start"], ent["end"], ent["label"]) for ent in entry["entities"]]
        
        # Korrigiere die Entitäten, falls nötig
        corrected_entities = validate_and_correct_entities(text, entities)
        
        train_data.append((text, {"entities": corrected_entities}))
    return train_data

def validate_and_correct_entities(text, entities):
    """
    Überprüft und korrigiert die Position der Entitäten im Text.
    """
    valid_entities = []
    for start, end, label in entities:
        entity_text = text[start:end].strip()  # Entferne überflüssige Leerzeichen

        # Falls die Entität im Text gefunden wird und der Text exakt übereinstimmt
        if text[start:end] == entity_text:
            valid_entities.append((start, end, label))
        else:
            # Finde die tatsächliche Position der Entität im Text
            corrected_start = text.find(entity_text)
            if corrected_start != -1:
                corrected_end = corrected_start + len(entity_text)
                
                # Prüfen, ob die korrigierten Positionen passen
                if text[corrected_start:corrected_end] == entity_text:
                    valid_entities.append((corrected_start, corrected_end, label))
                else:
                    print(f"Warnung: Korrektur fehlgeschlagen für Entität '{entity_text}' in '{text}'")
            else:
                print(f"Warnung: Entität '{entity_text}' konnte in '{text}' nicht korrekt gefunden werden und wird ignoriert.")
    return valid_entities

def train_model():
    # Lade ein leeres Modell für Deutsch oder Englisch
    nlp = spacy.blank("de")  # Verwende "de" für Deutsch oder "en" für Englisch
    ner = nlp.add_pipe("ner", last=True)

    # Definiere das Label
    ner.add_label("VARIABLE")

    # Lade die Trainingsdaten aus JSON
    TRAIN_DATA = load_train_data("train_data.json")

    # Trainingsprozess starten
    optimizer = nlp.begin_training()
    n_epochs = 10  # Anzahl der Epochen, ggf. anpassbar
    for epoch in range(n_epochs):
        print(f"\nStarte Epoche {epoch + 1}/{n_epochs}")
        for i, (text, annotations) in enumerate(TRAIN_DATA, 1):
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.3, sgd=optimizer)
            if i % 10 == 0 or i == len(TRAIN_DATA):
                print(f"Verarbeitete Beispiele: {i}/{len(TRAIN_DATA)} in Epoche {epoch + 1}")

    # Speichere das trainierte Modell
    model_path = "app/models/fine_tuned_model"
    nlp.to_disk(model_path)
    print(f"Modell gespeichert unter: {model_path}")

if __name__ == "__main__":
    train_model()
