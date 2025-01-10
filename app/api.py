from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)

# Lade das feinabgestimmte Modell
nlp = spacy.load("app/models/fine_tuned_model")

# Einzelne Nachricht verarbeiten
@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.json
    message = data['message']
    
    # Analysiere den Text mit dem NER-Modell
    doc = nlp(message)
    
    # Ersetze alle erkannten Entitäten durch {VARIABLE}
    transformed_message = message
    for ent in doc.ents:
        transformed_message = transformed_message.replace(ent.text, "{VARIABLE}")
    
    return jsonify({"processed_message": transformed_message})

# Batch-Nachrichten verarbeiten
@app.route('/process_messages', methods=['POST'])
def process_messages():
    data = request.json
    messages = data['messages']
    
    transformed_messages = []
    for message in messages:
        doc = nlp(message)
        
        # Ersetze alle erkannten Entitäten durch {VARIABLE}
        transformed_message = message
        for ent in doc.ents:
            transformed_message = transformed_message.replace(ent.text, "{VARIABLE}")
        
        transformed_messages.append(transformed_message)
    
    return jsonify({"processed_messages": transformed_messages})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
