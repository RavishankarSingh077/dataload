
import os
from mistralai import Mistral

api_key = "auE256yKbGAZoaqvG00EqmrWYFgPcr61"
client = Mistral(api_key=api_key)

try:
    print("Sending request to Mistral...")
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("Success:", resp.choices[0].message.content)
except Exception as e:
    print("Error:", e)
