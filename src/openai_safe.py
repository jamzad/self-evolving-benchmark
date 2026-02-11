from typing import Any, Dict, Optional

class ModelCaps:
    def __init__(self):
        self.temperature_supported: Dict[str, bool] = {}

def _temp_unsupported(e: Exception) -> bool:
    msg = str(e)
    return ("temperature" in msg) and ("Only the default (1) value is supported" in msg or "unsupported_value" in msg)

def chat_create_safe(
    client,
    caps: ModelCaps,
    *,
    model: str,
    messages,
    temperature: Optional[float] = None,
    **kwargs
) -> Any:
    # If we learned this model can't do temperature, omit it.
    if temperature is not None and model in caps.temperature_supported and not caps.temperature_supported[model]:
        return client.chat.completions.create(model=model, messages=messages, **kwargs)

    try:
        if temperature is None:
            resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
            caps.temperature_supported[model] = True
        return resp
    except Exception as e:
        if temperature is not None and _temp_unsupported(e):
            caps.temperature_supported[model] = False
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        raise
