#!/usr/bin/env python3
"""
Script to list all available Gemini models and their capabilities
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

def list_available_models():
    """List all available Gemini models using the API"""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Error: Please set up your Gemini API key in the .env file!")
        print("Create a .env file with: GEMINI_API_KEY=your_actual_api_key")
        return []
    
    try:
        genai.configure(api_key=api_key)
        
        print("🤖 Fetching available Gemini models...\n")
        
        # List all models
        models = genai.list_models()
        
        available_models = []
        
        for model in models:
            print(f"📋 Model: {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Description: {model.description}")
            print(f"   Version: {model.version}")
            print(f"   Input Token Limit: {model.input_token_limit:,}")
            print(f"   Output Token Limit: {model.output_token_limit:,}")
            print(f"   Supported Methods: {', '.join(model.supported_generation_methods)}")
            print(f"   Temperature: {model.temperature if hasattr(model, 'temperature') else 'N/A'}")
            print(f"   Top P: {model.top_p if hasattr(model, 'top_p') else 'N/A'}")
            print(f"   Top K: {model.top_k if hasattr(model, 'top_k') else 'N/A'}")
            print("-" * 80)
            
            available_models.append({
                'name': model.name,
                'display_name': model.display_name,
                'description': model.description,
                'version': model.version,
                'input_token_limit': model.input_token_limit,
                'output_token_limit': model.output_token_limit,
                'supported_methods': model.supported_generation_methods
            })
        
        print(f"\n✅ Found {len(available_models)} available models")
        
        # Save to JSON file for reference
        with open('available_models.json', 'w') as f:
            json.dump(available_models, f, indent=2)
        print("💾 Model information saved to 'available_models.json'")
        
        return available_models
        
    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")
        return []

def filter_generation_models(models):
    """Filter models that support generateContent method"""
    generation_models = []
    
    print("\n🎯 Models supporting 'generateContent' method:")
    print("=" * 60)
    
    for model in models:
        if 'generateContent' in model.get('supported_methods', []):
            generation_models.append(model)
            print(f"✅ {model['name']}")
            print(f"   {model['display_name']}")
            print(f"   Input limit: {model['input_token_limit']:,} tokens")
            print(f"   Output limit: {model['output_token_limit']:,} tokens")
            print()
    
    print(f"🎮 {len(generation_models)} models suitable for the guessing game")
    return generation_models

if __name__ == "__main__":
    models = list_available_models()
    if models:
        suitable_models = filter_generation_models(models)
        
        print("\n📝 Recommended models for the guessing game:")
        print("=" * 50)
        for model in suitable_models:
            model_id = model['name'].replace('models/', '')
            if any(keyword in model_id.lower() for keyword in ['flash', 'pro', 'gemini']):
                print(f"🌟 {model_id}")
                print(f"   {model['description']}")
                print()
