"""
Onion Farm Advisory Bot - WhatsApp Orchestrator
=================================================
Handles Twilio WhatsApp webhooks for Indian onion farmers.
Detects intent in Hindi/English, routes to farming tools.
"""

import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
import base64
from datetime import datetime

import boto3

from onion_tools import TOOLS_SCHEMA, execute_tool

# AWS Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

# Environment
CHAT_HISTORY_TABLE = os.environ.get("CHAT_HISTORY_TABLE", "onion-chat-history-prod")
MODEL_ID = "amazon.nova-pro-v1:0"
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

# ============================================================
# Intent Detection (Hindi + English)
# ============================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an AI assistant for Indian onion farmers.
You help with: crop management, weather, irrigation, pest control, and market prices.

The farmer may write in Hindi, Hinglish, Marathi, or English. Understand all.

Analyze the farmer's message and decide which tool to use:

1. crop_calendar — farmer asks about crop stage, what to do, growth progress
   Keywords: चरण, stage, क्या करें, today, अभी, nursery, रोपाई, कटाई
   
2. weather_advisory — farmer asks about weather, rain, temperature
   Keywords: मौसम, बारिश, rain, weather, तापमान, temperature, धूप

3. mandi_prices — farmer asks about market price, selling, rate
   Keywords: भाव, मंडी, rate, price, बेचें, sell, market, बाजार, कीमत

4. pest_management — farmer describes pests, diseases, spots, insects
   Keywords: कीड़ा, pest, रोग, disease, धब्बा, spot, पत्ती सूख, thrips, सड़न

5. irrigation_advisor — farmer asks about watering, irrigation
   Keywords: पानी, सिंचाई, irrigation, water, कब दें पानी

6. register_farmer — farmer wants to register, gives planting date
   Keywords: register, रजिस्टर, शुरू, मैंने लगाई, planted on, तारीख

Respond in JSON:
{
    "use_tool": true/false,
    "tool_name": "tool_name" | null,
    "parameters": { ... },
    "direct_response": "response if no tool needed (in Hindi)"
}

If farmer says hi/hello/namaste, respond with a welcome message listing what you can help with.
Always be respectful — use "आप" not "तू"."""


def detect_intent_and_route(message: str, phone: str) -> dict:
    """Detect farmer's intent and route to correct tool."""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": f"Farmer's message: {message}"}]}],
                "system": [{"text": ORCHESTRATOR_SYSTEM_PROMPT}],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.1},
            }),
        )
        result = json.loads(response["body"].read())
        response_text = result["output"]["message"]["content"][0]["text"]

        # Parse JSON response
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1]
            clean_text = clean_text.rsplit("```", 1)[0]

        intent = json.loads(clean_text)

        if intent.get("use_tool") and intent.get("tool_name"):
            tool_result = execute_tool(
                tool_name=intent["tool_name"],
                parameters=intent.get("parameters", {}),
                farmer_phone=phone,
            )
            return {"tool_used": intent["tool_name"], "result": tool_result}
        else:
            return {
                "tool_used": None,
                "result": {
                    "status": "success",
                    "advisory": intent.get("direct_response",
                        "🧅 नमस्ते! मैं आपका प्याज खेती सहायक हूं।\n\n"
                        "मुझसे पूछें:\n"
                        "• मौसम कैसा रहेगा?\n"
                        "• पानी कब दें?\n"
                        "• मंडी भाव क्या है?\n"
                        "• कीड़ा लगा है, क्या करें?\n"
                        "• मेरी फसल किस चरण में है?\n\n"
                        "शुरू करने के लिए बताएं — आपने प्याज कब लगाई?"
                    ),
                },
            }
    except json.JSONDecodeError:
        return {
            "tool_used": None,
            "result": {"status": "success", "advisory": response_text if "response_text" in dir() else "मैं समझ नहीं पाया। कृपया दोबारा बताएं।"},
        }
    except Exception as e:
        return {
            "tool_used": None,
            "result": {"status": "error", "message": f"Error: {str(e)}"},
        }


def format_response(orchestration_result: dict) -> str:
    """Format tool output for WhatsApp."""
    result = orchestration_result.get("result", {})

    if result.get("status") == "error":
        return f"⚠️ {result.get('message', 'कुछ गलत हुआ। दोबारा कोशिश करें।')}"

    if result.get("status") == "needs_input":
        return result.get("message", "कृपया और जानकारी दें।")

    # Get advisory from result
    advisory = result.get("advisory", result.get("message", ""))
    if not advisory:
        advisory = "✅ हो गया। और कुछ पूछना है?"

    # Truncate for WhatsApp
    if len(advisory) > 1500:
        advisory = advisory[:1450] + "\n\n... [और जानकारी के लिए पूछें]"

    return advisory


# ============================================================
# WhatsApp Send + Lambda Handler
# ============================================================


def send_whatsapp_reply(to: str, body: str):
    """Send WhatsApp reply via Twilio API."""
    if len(body) > 1500:
        body = body[:1450] + "\n\n... [और जानकारी के लिए पूछें]"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"

    encoded_to = urllib.parse.quote(to, safe='')
    encoded_from = urllib.parse.quote(
        f"whatsapp:{os.environ.get('TWILIO_WHATSAPP_NUMBER', '+14155238886')}",
        safe=''
    )
    encoded_body = urllib.parse.quote(body, safe='')
    data = f"To={encoded_to}&From={encoded_from}&Body={encoded_body}".encode()

    credentials = base64.b64encode(
        f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode()
    ).decode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    try:
        urllib.request.urlopen(req)
        print(f"WhatsApp reply sent to {to}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"Failed to send reply: {e} | {error_body[:300]}")
    except Exception as e:
        print(f"Failed to send reply: {e}")


def save_chat(phone: str, role: str, content: str):
    """Save message to chat history."""
    try:
        table = dynamodb.Table(CHAT_HISTORY_TABLE)
        table.put_item(Item={
            "phone": phone,
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content[:2000],
            "ttl": int(time.time()) + (30 * 86400),
        })
    except Exception:
        pass


def lambda_handler(event, context):
    """Main Lambda handler for Twilio WhatsApp webhook."""
    try:
        # Parse Twilio webhook
        if "body" in event:
            body = event["body"]
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode("utf-8")
            params = dict(urllib.parse.parse_qsl(body))
        else:
            params = event

        from_number = params.get("From", "").replace("whatsapp:", "")
        message_body = params.get("Body", "").strip()
        sender_name = params.get("ProfileName", "किसान")

        if not message_body:
            return {"statusCode": 200, "body": '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', "headers": {"Content-Type": "application/xml"}}

        print(f"[ONION-BOT] From: {from_number} | Name: {sender_name} | Msg: {message_body[:100]}")

        save_chat(from_number, "user", message_body)

        # Route and execute
        orchestration_result = detect_intent_and_route(message_body, from_number)

        # Format response
        reply_text = format_response(orchestration_result)

        save_chat(from_number, "assistant", reply_text)

        # Send reply
        send_whatsapp_reply(f"whatsapp:{from_number}", reply_text)

        return {"statusCode": 200, "body": '<?xml version="1.0" encoding="UTF-8"?><Response></Response>', "headers": {"Content-Type": "application/xml"}}

    except Exception as e:
        print(f"[ONION-BOT] ERROR: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)}), "headers": {"Content-Type": "application/json"}}
