from openai import OpenAI
import os

'''client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )'''

endpoint_base_url = "http://localhost:8001/v1"

client = OpenAI(
    api_key="dummy",  # not used locally
    base_url=endpoint_base_url  # your proxy endpoint
    )

body = {
    "model": "gpt-4.1-mini",
    "messages": [
        {"role": "user", "content": "What is said about data management in the document? List the most important five keypoints."}
    ]
}

response = client.chat.completions.create(**body)

print(response)
