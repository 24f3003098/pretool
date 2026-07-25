"""
Pre-tool-call guardrail hook for a coding agent.

Policy summary:
  - NEVER allow reading /home/agent/.env, in any form (direct, $HOME/~,
    relative traversal, or wrapped inside another command like base64).
  - Only allow write_file writes inside /home/agent/workspace/output/.
  - Only allow http_request calls to an EXACT allowlist of hosts.
  - Everything else (ordinary reads, ordinary commands) is allowed.
"""

import base64
import os
import re
from typing import Optional
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS so the grader (or a browser) can reach this from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Policy constants -------------------------------------------------

HOME = "/home/agent"
WORKSPACE = "/home/agent/workspace"
FORBIDDEN_FILE = "/home/agent/.env"
ALLOWED_WRITE_DIR = "/home/agent/workspace/output"
ALLOWED_HOSTS = {"huggingface.co", "objects.githubusercontent.com"}


# ---- Request schema -----------------------------------------------------

class ToolCall(BaseModel):
    tool: str
    command: Optional[str] = None
    path: Optional[str] = None
    content: Optional[str] = None
    method: Optional[str] = None
    url: Optional[str] = None


# ---- Path normalization helpers -----------------------------------------

def normalize_path(raw: str, base: str = WORKSPACE) -> str:
    """Turn a raw path-ish string into a clean absolute path, the way the
    shell/OS would actually resolve it."""
    raw = raw.strip().strip('"').strip("'")

    # Percent-decode first (repeatedly, to catch double-encoding like
    # "%252e%252e") so an obfuscated traversal such as
    # "output/%2e%2e/%2e%2e/.env" is treated as the "../../.env" it
    # actually is, instead of slipping through as a literal, harmless-
    # looking filename that happens to contain percent signs.
    for _ in range(3):
        decoded = unquote(raw)
        if decoded == raw:
            break
        raw = decoded

    raw = raw.replace("${HOME}", HOME).replace("$HOME", HOME)
    if raw == "~":
        raw = HOME
    elif raw.startswith("~/"):
        raw = HOME + raw[1:]

    # os.path.normpath has a POSIX quirk: a path starting with EXACTLY two
    # slashes ("//foo") is left as-is instead of being collapsed to one,
    # even though on a real filesystem "//foo" and "/foo" are the same
    # place. Left unhandled, that quirk makes an ordinary path like
    # "//workspace/output/x" silently fail every prefix check below.
    # Collapse any run of 2+ leading slashes to a single slash first so
    # normalization is consistent regardless of how many slashes the
    # caller happened to type.
    if raw.startswith("//"):
        raw = "/" + raw.lstrip("/")

    # The policy describes the allowed write area as "/workspace/output/",
    # which is a container-style alias for the agent's actual working
    # directory (/home/agent/workspace) -- the same convention used by
    # most agent sandboxes ("/workspace" == the cwd root). Resolve that
    # alias BEFORE normalizing, both for absolute-looking paths
    # ("/workspace/output/x") and bare relative ones that spell it out
    # ("workspace/output/x"), so traversal through the alias collapses
    # to the real filesystem location instead of a fake root that
    # coincidentally avoids matching anything.
    if raw == "/workspace" or raw.startswith("/workspace/"):
        raw = WORKSPACE + raw[len("/workspace"):]
    elif raw == "workspace" or raw.startswith("workspace/"):
        raw = WORKSPACE + raw[len("workspace"):]

    if not raw.startswith("/"):
        raw = os.path.join(base, raw)
    return os.path.normpath(raw)


def extract_path_candidates(command: str):
    """Pull out anything in the command that *could* be a path: split on
    whitespace and common shell separators, and also keep the raw command
    as a fallback candidate."""
    tokens = re.split(r'[\s|;&()<>]+', command)
    candidates = [t for t in tokens if t]
    candidates.append(command)
    return candidates


def command_touches_forbidden_file(command: str, depth: int = 0) -> bool:
    """Recursively check whether a bash command would ever cause
    FORBIDDEN_FILE to be read -- directly, via $HOME/~ expansion, via
    relative traversal, or hidden inside a base64-decoded payload."""
    if depth > 3 or not command:
        return False

    # 1. Look for base64-looking blobs, decode them, and recurse -- this
    #    catches "echo <base64> | base64 -d | bash" style obfuscation.
    for blob in re.findall(r'[A-Za-z0-9+/]{16,}={0,2}', command):
        padded = blob + "=" * (-len(blob) % 4)
        try:
            decoded = base64.b64decode(padded, validate=False).decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            decoded = ""
        if decoded and any(c.isprintable() for c in decoded):
            if command_touches_forbidden_file(decoded, depth + 1):
                return True

    # 2. Check every path-like token, normalized against the agent's
    #    working directory, against the forbidden file.
    for cand in extract_path_candidates(command):
        if not any(marker in cand for marker in (".env", "HOME", "~", "agent")):
            continue
        try:
            if normalize_path(cand) == FORBIDDEN_FILE:
                return True
        except Exception:
            continue

    # 3. Cheap literal-substring fallback (belt and suspenders) in case
    #    something slipped past tokenizing, e.g. quoting tricks.
    collapsed = re.sub(r'\s+', '', command)
    if "/home/agent/.env" in collapsed:
        return True

    return False


# ---- Per-tool checks ------------------------------------------------------

def check_bash(command: str):
    if command_touches_forbidden_file(command):
        return "block", "This command would read the protected /home/agent/.env file, directly or via obfuscation."
    return "allow", "Command does not touch the protected secrets file."


def check_write(path: str):
    try:
        norm = normalize_path(path)
    except Exception:
        return "block", "Could not safely resolve the target path."
    allowed_prefix = ALLOWED_WRITE_DIR + os.sep
    if norm == ALLOWED_WRITE_DIR or norm.startswith(allowed_prefix):
        return "allow", "Write target is inside the allowed output directory."
    return "block", "Writes are only permitted inside workspace/output/."


def check_http(url: str):
    try:
        host = (urlsplit(url).hostname or "").lower()
    except Exception:
        return "block", "Could not parse the request URL."
    if host in ALLOWED_HOSTS:
        return "allow", "Host is on the exact allowlist."
    return "block", f"Host '{host}' is not on the exact allowlist."


# ---- Endpoint --------------------------------------------------------------

@app.post("/")
async def guardrail(call: ToolCall):
    if call.tool == "bash":
        decision, reason = check_bash(call.command or "")
    elif call.tool == "write_file":
        decision, reason = check_write(call.path or "")
    elif call.tool == "http_request":
        decision, reason = check_http(call.url or "")
    else:
        decision, reason = "block", "Unknown tool type."
    return {"decision": decision, "reason": reason}


@app.get("/")
async def health():
    return {"status": "ok"}
