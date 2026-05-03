import json
import os
from datetime import datetime, timezone


TRACE_FILE = os.getenv("TRACE_FILE", "/app/trace.json")


def _load_trace():
    if os.path.exists(TRACE_FILE):
        with open(TRACE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_trace(entries):
    trace_dir = os.path.dirname(TRACE_FILE)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    with open(TRACE_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_event(event: str, details: dict | None = None, status: str = "Success"):
    entries = _load_trace()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
    }
    if details:
        entry.update(details)
    entries.append(entry)
    _save_trace(entries)
    return entry
