import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("NO API KEY")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("Models:")
try:
    for m in client.models.list():
        if "pro" in m.name.lower():
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
