#!/usr/bin/env python3
"""
nanocode_vllm - minimal coder using vLLM OpenAI-compatible Chat Completions API
(using the OpenAI Python client; text-only; auto-detects served model)

Start vLLM server (OpenAI-compatible), e.g.:
  vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 30000 --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --structured-outputs-config.backend xgrammar

Env:
  VLLM_BASE_URL / OPENAI_BASE_URL  default http://localhost:8000/v1
  VLLM_MODEL                      optional; if unset or invalid, script auto-detects
  OPENAI_API_KEY                  optional; many vLLM servers accept "EMPTY"
"""

import glob as globlib, json, os, re, subprocess, urllib.request, urllib.parse, urllib.error
import json
import os
import re
import argparse
import subprocess
import datetime as _dt
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------- Config ----------
BASE_URL = os.environ.get(
    "VLLM_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
).rstrip("/")

REQUESTED_MODEL = os.environ.get("VLLM_MODEL")  # may be None/wrong
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "EMPTY"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)


# --- time helpers ---

def now_iso() -> str:
    # local time, ISO 8601 with seconds
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")

def ts_filename() -> str:
    # filesystem-safe timestamp
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%d-%H:%M:%S")


# ---------- Tools ----------
def read(args: Dict[str, Any]) -> str:
    lines = open(args["path"]).readlines()
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", len(lines)))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args: Dict[str, Any]) -> str:
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"


def edit(args: Dict[str, Any]) -> str:
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"


def glob(args: Dict[str, Any]) -> str:
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args: Dict[str, Any]) -> str:
    pattern = re.compile(args["pat"])
    hits: List[str] = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            if not os.path.isfile(filepath):
                continue
            with open(filepath, "r", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args: Dict[str, Any]) -> str:
    result = subprocess.run(
        args["cmd"], shell=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip() or "(empty)"



def web_search(args):
    """Search the web via DuckDuckGo HTML endpoint and return top results.
    Returns lines: '1. title - url'"""
    query = args.get("query", "").strip()
    max_results = int(args.get("max_results", 5))
    if not query:
        return "error: query is required"
    try:
        q = urllib.parse.quote(query)
        url = f"https://duckduckgo.com/html/?kl=us-en&q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        results = []
        for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I|re.S):
            href = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2))
            try:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    link = urllib.parse.unquote(qs["uddg"][0])
                else:
                    link = href
            except Exception:
                link = href
            title = re.sub(r"\s+", " ", title).strip()
            results.append((title, link))
            if len(results) >= max_results:
                break
        if not results:
            for m in re.finditer(r'href="([^"]*uddg=[^"]+)"[^>]*>(.*?)</a>', html, re.I|re.S):
                href = m.group(1)
                title = re.sub(r"<[^>]+>", "", m.group(2))
                try:
                    parsed = urllib.parse.urlparse(href)
                    qs = urllib.parse.parse_qs(parsed.query)
                    link = urllib.parse.unquote(qs.get("uddg", [href])[0])
                except Exception:
                    link = href
                title = re.sub(r"\s+", " ", title).strip()
                if title and link:
                    results.append((title, link))
                if len(results) >= max_results:
                    break
        if not results:
            return "none"
        return "\n".join(f"{i+1}. {t} - {u}" for i, (t, u) in enumerate(results))
    except urllib.error.URLError as e:
        return f"error: network - {e}"
    except Exception as e:
        return f"error: {e}"


def web_get(args):
    """Fetch a webpage and return plain text (stripped)."""
    url = args.get("url", "").strip()
    max_chars = int(args.get("max_chars", 6000))
    if not url:
        return "error: url is required"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text or "(empty)"
    except urllib.error.URLError as e:
        return f"error: network - {e}"
    except Exception as e:
        return f"error: {e}"

# --- Tool definitions: (description, schema, function) ---

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
    "web_search": (
        "Search the web and return top results as numbered list",
        {"query": "string", "max_results": "integer?"},
        web_search,
    ),
    "web_get": (
        "Fetch a webpage and return plain text (roughly extracted)",
        {"url": "string", "max_chars": "integer?"},
        web_get,
    ),
}

def run_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema() -> List[dict]:
    """Generate Chat Completions tool schema (OpenAI function calling format)."""
    result: List[dict] = []
    for name, (description, params, _fn) in TOOLS.items():
        properties: Dict[str, dict] = {}
        required: List[str] = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {"type": base_type}
            if not is_optional:
                required.append(param_name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return result


# ---------- Model auto-detection ----------
def detect_model_via_models_endpoint() -> Optional[str]:
    """Preferred: use OpenAI-compatible /v1/models."""
    models = client.models.list()
    ids = [m.id for m in (models.data or []) if getattr(m, "id", None)]
    return ids[0] if ids else None


def detect_model_via_probe() -> Optional[str]:
    """
    Fallback: send an invalid model name and try to parse a helpful error.
    This is less reliable, but can work when /v1/models is disabled.
    """
    try:
        client.chat.completions.create(
            model="__invalid_model__",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        msg = str(e)
        # Heuristic: extract text inside backticks: `...`
        m = re.search(r"`([^`]+)`", msg)
        if m:
            cand = m.group(1).strip()
            if cand and cand != "__invalid_model__":
                return cand
    return None


def pick_model(requested: Optional[str]) -> str:
    """
    Choose a model id that the server recognizes:
      1) If requested set and exists in /v1/models, use it.
      2) Else use first model from /v1/models.
      3) Else try probe fallback.
    """
    try:
        models = client.models.list()
        ids = [m.id for m in (models.data or []) if getattr(m, "id", None)]
        if ids:
            if requested and requested in ids:
                return requested
            return ids[0]
    except Exception:
        pass

    probed = detect_model_via_probe()
    if probed:
        # If /v1/models was blocked, we can still try using probed id.
        return requested or probed

    # Last resort: if user set something, try it; otherwise fail loud.
    if requested:
        return requested
    raise RuntimeError(
        "Could not auto-detect model. Set VLLM_MODEL to the served model id "
        "(example: export VLLM_MODEL='Qwen/Qwen3-8B')."
    )


MODEL = pick_model(REQUESTED_MODEL)


# ---------- API call ----------
def call_api(messages: List[dict]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=make_schema(),
        tool_choice="auto",
        temperature=0.0,
        max_tokens=8192,
    )
    return resp.model_dump()


# ---------- UI helpers ----------
def separator() -> str:
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def render_markdown(text: str) -> str:
    """
    Render Markdown nicely in a terminal using Rich.
    Falls back to plain text if Rich is unavailable.
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        md = Markdown(text, code_theme="monokai", hyperlinks=True)
        console.print(md)
        return "" # already printed
    except Exception:
        return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)

# --- history helpers ---

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_chat_history(session_timestamp: str, model: str, system_prompt: str, history_events: list):
    """
    Writes a single JSON file in chat_history/ named by timestamp.
    Each event includes its own timestamp (conversation timestamp per entry).
    """
    ensure_dir(f"chat_history/{model}")
    filename = os.path.join("chat_history", model, f"{ts_filename()}.json")
    payload = {
        "session_timestamp": session_timestamp,
        "model": model,
        "system_prompt": system_prompt,
        "events": history_events,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filename

def parse_args():
    p = argparse.ArgumentParser(description="nanocode_vllm - minimal coder using vLLM OpenAI-compatible Chat Completions API")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt/instructions string (overrides default).",
    )
    p.add_argument(
        "--system-file",
        default=None,
        help="Path to a file containing the system prompt (overrides --system).",
    )
    p.add_argument(
        "--not_save_history",
        action="store_true",
        help="Not save chat history JSON into chat_history/ at exit.",
    )

    p.add_argument(
        "--save_full_api_response",
        action="store_true",
        help="Save the whole API response object into chat history for transparency.",
    )
    return p.parse_args()

# ---------- Main ----------
def main() -> None:
    args = parse_args()
    print(f"{BOLD}nanocode_vllm{RESET} | {DIM}{MODEL}{RESET} | {BASE_URL}\n")

    messages: List[dict] = []
    system_prompt = args.system if args.system is not None else f"Concise coding assistant. cwd: {os.getcwd()}"

    session_timestamp = now_iso()
    history_events = []  # for saving (timestamps per event)

    def log_event(kind: str, **data):
        history_events.append({"timestamp": now_iso(), "type": kind, **data})

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break
            if user_input == "/c":
                messages = []
                log_event("control", command="/c")
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            # include system prompt each time as first message
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

            log_event("user", text=user_input)
            messages.append({"role": "user", "content": user_input})

            # agentic loop: keep calling API until no more tool calls
            while True:
                response = call_api(messages)

                # Keep the raw response if you want full reproducibility in history
                if args.save_full_api_response:
                    log_event("api_response", response=response)
                
                choice = (response.get("choices") or [{}])[0]
                message = (choice.get("message") or {})
                content = message.get("content")
                tool_calls = message.get("tool_calls") or []
                log_event("assistant", text=content, tool_calls=tool_calls)

                if content:
                    print(f"\n{CYAN}⏺{RESET}", end=" ")
                    out = render_markdown(content)
                    if out:
                        print(out)

                # record assistant message
                assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_message["tool_calls"] = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        assistant_message["tool_calls"].append(
                            {
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name": fn.get("name"),
                                    "arguments": fn.get("arguments", "{}"),
                                },
                            }
                        )
                messages.append(assistant_message)

                if not tool_calls:
                    break

                # process tool calls
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name")
                    try:
                        tool_args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        tool_args = {}

                    arg_preview = str(list(tool_args.values())[0])[:50] if tool_args else ""
                    print(f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})")

                    result = run_tool(tool_name, tool_args)
                    log_event("tool", name=tool_name, arguments=tool_args, output=result)

                    result_lines = result.split("\n")
                    preview = result_lines[0][:60] if result_lines else ""
                    if len(result_lines) > 1:
                        preview += f" ... +{len(result_lines) - 1} lines"
                    elif preview and len(preview) > 60:
                        preview += "..."
                    print(f"  {DIM}⎿  {preview}{RESET}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": result,
                        }
                    )

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            log_event("error", message=str(err))
            print(f"{RED}⏺ Error: {err}{RESET}")

    if not args.not_save_history:
        try:
            path = save_chat_history(
                session_timestamp=session_timestamp,
                model=MODEL,
                system_prompt=system_prompt,
                history_events=history_events,
            )
            print(f"{GREEN}⏺ Saved chat history:{RESET} {path}")
        except Exception as err:
            print(f"{RED}⏺ Failed to save history: {err}{RESET}")


if __name__ == "__main__":
    main()