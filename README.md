# nanocode

Minimal Claude Code alternative. Single Python file, zero dependencies, ~340 lines.

Built using Claude Code, then used to build itself.

![screenshot](screenshot.png)

## Features

- Full agentic loop with tool use
- Tools: `read`, `write`, `edit`, `glob`, `grep`, `bash`
- Conversation history
- Colored terminal output
- Diff confirmation: file changes show colored diffs with line numbers; accept/decline via arrow keys, optional decline reason sent to model

## Usage

```bash
export ANTHROPIC_API_KEY="your-key"
python nanocode.py
```

### OpenRouter

Use [OpenRouter](https://openrouter.ai) to access any model:

```bash
export OPENROUTER_API_KEY="your-key"
python nanocode.py
```

To use a different model:

```bash
export OPENROUTER_API_KEY="your-key"
export MODEL="openai/gpt-5.2"
python nanocode.py
```

## Commands

- `/c` - Clear conversation
- `/q` or `exit` - Quit

## Tools

| Tool | Description |
|------|-------------|
| `read` | Read file with line numbers, offset/limit |
| `write` | Write content to file |
| `edit` | Replace string in file (must be unique) |
| `glob` | Find files by pattern, sorted by mtime |
| `grep` | Search files for regex |
| `bash` | Run shell command |

## Example

```
────────────────────────────────────────
❯ what files are here?
────────────────────────────────────────

⏺ Glob(**/*.py)
  ⎿  nanocode.py

⏺ There's one Python file: nanocode.py
```

## License

MIT
