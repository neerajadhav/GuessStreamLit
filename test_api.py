#!/usr/bin/env python3
"""
Quick test to verify the Gemini AI integration works properly
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_connection():
    """Test basic Gemini API connection"""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Error: Please set up your Gemini API key in the .env file!")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test a simple prompt
        response = model.generate_content("Say 'Hello, AI Guessing Game!' if you're working correctly.")
        print("✅ Gemini API connection successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Gemini API: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini API Integration...")
    test_gemini_connection()
