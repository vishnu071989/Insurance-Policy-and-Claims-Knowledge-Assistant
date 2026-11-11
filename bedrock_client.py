import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from errors import BedrockError
from config import AWS_REGION, EMBED_MODEL_ID, LLM_MODEL_ID

_client = None
def _rt():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client

def embed(text: str) -> list:
    try:
        body = json.dumps({"inputText": text[:8000], "dimensions": 512, "normalize": True})
        r = _rt().invoke_model(modelId=EMBED_MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
        return json.loads(r["body"].read())["embedding"]
    except (ClientError, BotoCoreError) as e:
        raise BedrockError(f"Embedding call failed: {e}") from e

def converse(system: str, user: str, max_tokens: int = 900) -> str:
    if not LLM_MODEL_ID:
        raise BedrockError("LLM_MODEL_ID is not set (see .env.example).")
    try:
        r = _rt().converse(
            modelId=LLM_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        return r["output"]["message"]["content"][0]["text"].strip()
    except (ClientError, BotoCoreError, KeyError, IndexError) as e:
        raise BedrockError(f"LLM call failed: {e}") from e
