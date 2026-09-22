# Deployment Notes

## Overview
The bot is a FastAPI app (`src/bot/whatsapp_webhook.py`) exposed as a webhook that
Twilio's WhatsApp API calls whenever a user sends a message.

## Local development
1. `uvicorn src.bot.whatsapp_webhook:app --reload`
2. Expose it publicly for Twilio to reach, e.g. with `ngrok http 8000`
3. In the Twilio Console, set the WhatsApp Sandbox "When a message comes in"
   webhook to `https://<ngrok-domain>/webhook`
4. Set environment variables:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `MODEL_CHECKPOINT` (path or URL to trained weights)
   - `CLASS_NAMES_PATH`

## Production options
- **Render / Railway / Fly.io** — simplest path for a small FastAPI service with a
  persistent webhook URL.
- **Hugging Face Spaces** (Gradio/FastAPI) — convenient if the model is already
  hosted on HF Hub.
- Keep the checkpoint small (MobileNetV3 / EfficientNet-Lite) to keep cold-start
  and inference latency low on low-cost hosting tiers.

## Model versioning
Push trained checkpoints to the Hugging Face Hub instead of committing them to
git:
```bash
huggingface-cli upload <your-username>/crop-disease-classifier models/checkpoints/best_model.pt
```
Then have `handlers.py` pull the weights from the Hub at startup instead of a
local path, if deploying to an ephemeral environment.

## Known constraints to design around
- Farmers may have low-bandwidth connections — keep the model small and image
  uploads should not require any extra preprocessing on their end.
- WhatsApp media messages expire; download and process them promptly in the
  webhook handler (see `handlers.py`).
