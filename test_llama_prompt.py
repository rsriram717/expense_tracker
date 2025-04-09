from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env file
load_dotenv()

# --- Inference.net Configuration ---
INFERENCE_API_KEY_ENV_VAR = "INFERENCE_API_KEY" 
INFERENCE_BASE_URL = "https://api.inference.net/v1/" 
LLAMA_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct/fp-8"

# Initialize OpenAI client
api_key = os.getenv(INFERENCE_API_KEY_ENV_VAR)
client = None

if api_key:
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=INFERENCE_BASE_URL,
        )
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
else:
    print(f"Warning: Environment variable {INFERENCE_API_KEY_ENV_VAR} not set.")
    exit(1)

# Test with specific prompt
test_prompt = "what color is the sun?"
print(f"\nTesting connection with prompt: '{test_prompt}'")

try:
    response = client.chat.completions.create(
        model=LLAMA_MODEL_NAME,
        messages=[{"role": "user", "content": test_prompt}],
        max_tokens=100,
        temperature=0.5
    )
    result = response.choices[0].message.content.strip()
    print("\nSUCCESS: Response received!")
    print(f"Response: '{result}'")
    print("\nConnection test successful. You can now run the batch categorizer.")
except Exception as e:
    print(f"\nFAILURE: Error calling API: {e}") 