from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import os

# --------- Setup ---------
load_dotenv(override=True)
client = OpenAI()   # reads OPENAI_API_KEY from .env or environment

# --------- Load PDF content (Saia Design info) ---------
# Put SaiaDesignChatbot.pdf in the SAME folder as app.py
reader = PdfReader("SaiaDesignChatbot.pdf")
SaiaDesign = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        SaiaDesign += text + "\n"

# --------- Load summary text ---------
# Put summary.txt in the SAME folder as app.py
with open("summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# --------- System prompt ---------
name = "Saia Design"

system_prompt = f"""
You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s services, policies, contact info, and reputation. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and service offering which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client who came across the website. \
If you don't know the answer, say so.

## Summary:
{summary}

## Company Info / PDF:
{SaiaDesign}

With this context, please chat with the user, always staying in character as {name}.
""".strip()


# --------- Chat function for Gradio ---------
def chat(message, history):
    """
    `history` from gr.ChatInterface is a list of [user, assistant] pairs.
    We need to convert it into OpenAI's messages format.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# --------- Gradio UI ---------
demo = gr.ChatInterface(
    fn=chat,
    title="Saia Design Website Assistant",
    description="Ask anything about Saia Design: websites, pricing, SEO, services, and more.",
)

# For deployment (Render/Railway/etc.)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
