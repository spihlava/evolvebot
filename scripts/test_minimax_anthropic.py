import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def test_minimax_anthropic():
    api_key = os.getenv("MINIMAX_API_KEY")
    api_base = os.getenv("MINIMAX_API_BASE", "https://api.minimax.io/anthropic")
    model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

    print(f"Testing Minimax Anthropic-compatible API...")
    print(f"Base URL: {api_base}")
    print(f"Model: {model}")

    client = Anthropic(
        api_key=api_key,
        base_url=api_base
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Hello! Respond with 'Anthropic Success' if this works."}
            ]
        )
        print("\nResponse Status: Success")
        print(f"Response Content: {response.content[0].text}")
    except Exception as e:
        print(f"\nResponse Status: Failed")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_minimax_anthropic()
