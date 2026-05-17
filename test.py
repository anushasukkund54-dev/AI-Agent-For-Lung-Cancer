from google import genai

client = genai.Client(api_key="AIzaSyAiFOIn9k5R1QKxE2S7WynJaN1iN5RiOaw")

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Hello"
)

print(response.text)