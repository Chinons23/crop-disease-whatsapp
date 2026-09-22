"""FastAPI webhook that receives WhatsApp images (via Twilio) and returns a
crop disease prediction.

Run locally:
    uvicorn src.bot.whatsapp_webhook:app --reload

Expose to Twilio during development with e.g. ngrok, and point your Twilio
WhatsApp sandbox "When a message comes in" webhook at:
    https://<your-ngrok-domain>/webhook
"""

import os

import requests
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from src.bot.handlers import handle_image_message

app = FastAPI(title="Crop Disease WhatsApp Bot")


@app.post("/webhook")
async def whatsapp_webhook(
    Body: str = Form(default=""),
    NumMedia: int = Form(default=0),
    MediaUrl0: str = Form(default=None),
):
    resp = MessagingResponse()

    if NumMedia and MediaUrl0:
        reply_text = handle_image_message(MediaUrl0)
    else:
        reply_text = (
            "👋 Send a photo of an affected crop leaf and I'll tell you what "
            "disease it might be."
        )

    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok"}
