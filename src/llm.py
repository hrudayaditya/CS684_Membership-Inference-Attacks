"""Minimal DeepSeek chat client: reasoning off, JSON output, retries, token accounting."""
import json, os, threading, time
import requests

URL = "https://api.deepseek.com/chat/completions"
_lock = threading.Lock()
USAGE = {"prompt": 0, "cache_hit": 0, "completion": 0, "calls": 0}

def chat_json(model, system, user, max_tokens=1200, temperature=1.0, retries=6):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for k in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]}, json=body, timeout=180)
            if r.status_code == 200:
                j = r.json()
                u = j["usage"]
                with _lock:
                    USAGE["prompt"] += u["prompt_tokens"]; USAGE["completion"] += u["completion_tokens"]
                    USAGE["cache_hit"] += u.get("prompt_cache_hit_tokens", 0); USAGE["calls"] += 1
                return json.loads(j["choices"][0]["message"]["content"])
            if r.status_code not in (429, 500, 502, 503, 504): raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        except (requests.RequestException, json.JSONDecodeError):
            pass
        time.sleep(2 ** k)
    raise RuntimeError("LLM call failed after retries")

def usage_line():
    u = USAGE
    return f"calls={u['calls']} prompt={u['prompt']} (cached {u['cache_hit']}) completion={u['completion']}"
