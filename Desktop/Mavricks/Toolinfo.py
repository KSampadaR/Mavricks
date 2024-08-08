import os
import google.generativeai as genai
import re

os.environ['GOOGLE_API_KEY'] = 'AIzaSyBx9G4qOMi2FPzG_bs4jEZ7_2Dnil93pGI'
api_key = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
files = [
    upload_to_gemini("cnc-milling-machine.jpg", mime_type="image/jpeg"),
]
chat_session = model.start_chat(history=[])
initial_msg = {
    "role": "user",
    "parts": [
        files[0],
        "Provide the only name of the machine present in the image in single line"
    ]
}
initial_response = chat_session.send_message(initial_msg)
print({initial_response.text})
