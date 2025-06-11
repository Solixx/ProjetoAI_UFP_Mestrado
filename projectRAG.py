import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine
import torch
from peft import PeftModel

url = "https://raw.githubusercontent.com/BrunoSilva077/dataset/main/ufp-courses-dataset.jsonl"
response = requests.get(url)
data = [json.loads(line) for line in response.text.splitlines()]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for dataset using 'response' key
documents = [entry['response'] for entry in data]
doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

# Initialize Gemma2 model and tokenizer
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LORA trained model
model2 = AutoModelForCausalLM.from_pretrained(model_name)
model2 = PeftModel.from_pretrained(model2, "./gemma2-finetuned-lora/final_model_adapter")


conversation_history = []

def retrieve_relevant_docs(query, top_k=3):
    # Encode query
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = [1 - cosine(query_embedding.cpu().numpy(), doc_emb.cpu().numpy()) for doc_emb in doc_embeddings]
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return top_k documents and their metadata
    return [(data[i], similarities[i]) for i in top_indices]

def generate_response(query, context, model):
    # Prepare prompt with retrieved context and history
    history_text = "\n".join([f"User: {h['query']}\nBot: {h['response']}" for h in conversation_history[-3:]])
    
    # Prepare prompt with history and context
    if context:
        prompt = f"Conversation History:\n{history_text}\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"
    else:
        prompt = f"Conversation History:\n{history_text}\n\nQuestion: {query}\n\nAnswer:"
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def main():
    print("Welcome to the UFP Chatbot! Ask about UFP courses or type 'quit' to exit.")

    print(f"\n 1. RAG\n 2. LORA\n 3. RAG + LORA")
    choice = input("Choose an option (1/2/3): ")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if choice == '1':
            # RAG only
            print("Using RAG to retrieve relevant documents...")
            relevant_docs = retrieve_relevant_docs(query)
            context = "\n".join([f"- {doc['instruction']}: {doc['response']}" for doc, _ in relevant_docs])
            
            if not context:
                context = "No relevant information found."
            
            response = generate_response(query, context, model)
        elif choice == '2':
            # LoRA only
            print("Using LoRA for response generation...")
            response = generate_response(query, None, model2)
        elif choice == '3':
            # RAG + LoRA
            print("Using RAG to retrieve relevant documents and LoRA for response generation...")
            relevant_docs = retrieve_relevant_docs(query)
            context = "\n".join([f"- {doc['instruction']}: {doc['response']}" for doc, _ in relevant_docs])
            
            if not context:
                context = "No relevant information found."
            
            response = generate_response(query, context, model2)
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            continue

        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()