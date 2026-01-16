#!/usr/bin/env python3
"""nanocode - minimal claude code alternative"""

import difflib, glob as globlib, json, os, re, subprocess, sys, termios, tty, urllib.request

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/messages" if OPENROUTER_KEY else "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-4.5" if OPENROUTER_KEY else "claude-opus-4-5")

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)

# --- Diff and confirmation ---

def confirm():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write(f"{DIM}[Enter] accept · [d] decline{RESET} ")
        sys.stdout.flush()
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                return True
            if ch in ("d", "D"):
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def confirm_change(path, old_content, new_content):
    old_lines = old_content.splitlines(keepends=True) if old_content else []
    new_lines = new_content.splitlines(keepends=True) if new_content else []
    diff = list(difflib.unified_diff(old_lines, new_lines, n=3))

    action = "New" if not old_content else "Delete" if not new_content else "Edit"
    width = min(os.get_terminal_size().columns, 80)
    bar = f"{DIM}{'─' * width}{RESET}"

    print(f"\n{bar}\n{YELLOW}● {action}:{RESET} {BOLD}{path}{RESET}\n{bar}")
    old_ln = new_ln = 0
    for line in diff[2:]:
        txt = line[1:].rstrip("\n")
        if line[0] == "@":
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            old_ln, new_ln = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
        elif line[0] == "-":
            print(f"{RED}{old_ln:>6} - {txt}{RESET}"); old_ln += 1
        elif line[0] == "+":
            print(f"{GREEN}{new_ln:>6} + {txt}{RESET}"); new_ln += 1
        else:
            print(f"{DIM}{old_ln:>6}   {txt}{RESET}"); old_ln += 1; new_ln += 1
    print(bar)

    if confirm():
        print(f"{GREEN}✓ Accepted{RESET}")
        return None
    reason = input(f"{DIM}Reason (optional):{RESET} ").strip() or "stop and wait for further instructions"
    status = {"New": "file NOT created", "Edit": "file NOT modified", "Delete": "file NOT deleted"}[action]
    msg = f"declined ({status}): {reason}"
    print(f"{RED}✗ {msg}{RESET}")
    return msg

# --- Tool implementations ---


def read(args):
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    path, new_content = args["path"], args["content"]
    try:
        old_content = open(path).read()
    except FileNotFoundError:
        old_content = ""
    result = confirm_change(path, old_content, new_content)
    if result:  # declined
        return result
    with open(path, "w") as f:
        f.write(new_content)
    return "ok"


def edit(args):
    path = args["path"]
    old_content = open(path).read()
    old, new = args["old"], args["new"]
    if old not in old_content:
        return "error: old_string not found"
    count = old_content.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    new_content = old_content.replace(old, new) if args.get("all") else old_content.replace(old, new, 1)
    result = confirm_change(path, old_content, new_content)
    if result:  # declined
        return result
    with open(path, "w") as f:
        f.write(new_content)
    return "ok"


def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args):
    proc = subprocess.Popen(
        args["cmd"], shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"


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
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        result.append(
            {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return result


def call_api(messages, system_prompt):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": messages,
                "tools": make_schema(),
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **({"Authorization": f"Bearer {OPENROUTER_KEY}"} if OPENROUTER_KEY else {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", "")}),
        },
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read())


def separator():
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def main():
    print(f"{BOLD}nanocode{RESET} | {DIM}{MODEL} ({'OpenRouter' if OPENROUTER_KEY else 'Anthropic'}) | {os.getcwd()}{RESET}\n")
    messages = []
    system_prompt = f"Concise coding assistant. cwd: {os.getcwd()}. When a file change is declined, respect the user's feedback and adjust accordingly. Do not retry the same change."

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
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            messages.append({"role": "user", "content": user_input})

            # agentic loop: keep calling API until no more tool calls
            while True:
                response = call_api(messages, system_prompt)
                content_blocks = response.get("content", [])
                tool_results = []
                declined = False

                for block in content_blocks:
                    if block["type"] == "text":
                        print(f"\n{CYAN}⏺{RESET} {render_markdown(block['text'])}")

                    if block["type"] == "tool_use":
                        tool_name = block["name"]
                        tool_args = block["input"]
                        arg_preview = str(list(tool_args.values())[0])[:50]
                        print(
                            f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
                        )

                        result = run_tool(tool_name, tool_args)
                        if "stop and wait" in result:
                            declined = True
                        result_lines = result.split("\n")
                        preview = result_lines[0][:60]
                        if len(result_lines) > 1:
                            preview += f" ... +{len(result_lines) - 1} lines"
                        elif len(result_lines[0]) > 60:
                            preview += "..."
                        print(f"  {DIM}⎿  {preview}{RESET}")

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": result,
                            }
                        )

                messages.append({"role": "assistant", "content": content_blocks})
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                if not tool_results or declined:
                    break

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()
