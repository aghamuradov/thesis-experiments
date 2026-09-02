import httpx2

print("--- httpx2 GET ---")
try:
    r = httpx2.get("https://api.anthropic.com/v1/models", headers={"x-api-key": "test", "anthropic-version": "2023-06-01"})
    print(r.status_code, r.text[:200])
except Exception:
    import traceback; traceback.print_exc()

print("--- httpx2 POST ---")
try:
    r = httpx2.post("https://api.anthropic.com/v1/messages",
                     headers={"x-api-key": "test", "anthropic-version": "2023-06-01", "content-type": "application/json"},
                     json={"model": "claude-sonnet-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    print(r.status_code, r.text[:200])
except Exception:
    import traceback; traceback.print_exc()