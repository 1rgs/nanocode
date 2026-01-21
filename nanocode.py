#!/usr/bin/env python3
"""nanocode - minimal claude code alternative"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
import inspect
from pathlib import Path
import shutil
from typing import Callable, Literal, Optional, Protocol, TypeAlias
import glob as globlib, json, os, re, subprocess, urllib.request

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)

# --- Tool implementations ---
ToolDict: TypeAlias = dict[str, tuple[dict, Callable]]

TOOLS: ToolDict = {}

def tool(_func: Optional[Callable] = None, **param_descriptions: str):
    """
    A decorator that can automatically read function annotations and convert simple parameter types, 
    and convert them into the ToolDict type, where the tuple contains a dictionary in tool format and the function itself

    The usage is:
    @tool(parameter=description)
    """
    def decorator(func: Callable):
        func_name = func.__name__
        doc = inspect.getdoc(func) or "No description"
        sig = inspect.signature(func)
        
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            p_type = "string"
            if param.annotation == int: p_type = "integer"
            elif param.annotation == float: p_type = "number"
            elif param.annotation == bool: p_type = "boolean"
            
            properties[name] = {
                "type": p_type,
                "description": param_descriptions.get(name, "")
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        tool_schema = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        TOOLS[func_name] = (tool_schema, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)

@tool(path='Path uses forward slash style')
def read(path: str, offset: int = 0, limit: int = None) -> str:
    '''Read text file'''    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    end_index = offset + limit if limit is not None else len(lines)
    selected = lines[offset : end_index]
    
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))

@tool(path='Path uses forward slash style')
def write(path: str, content: str) -> str:
    '''Write file completely (overwrites)'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "ok"

@tool
def edit(path: str, old: str, new: str, all_matches: bool = False) -> str:
    '''Replace string in file'''    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    if old not in text:
        return "error: old_string not found"
    
    count = text.count(old)
    if not all_matches and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all_matches=True)"
    
    replacement = text.replace(old, new) if all_matches else text.replace(old, new, 1)
    
    with open(path, "w", encoding='utf-8') as f:
        f.write(replacement)
    return "ok"

@tool
def glob(pat: str, path: str = ".") -> str:
    '''Search for file list'''
    pattern = os.path.join(path, pat).replace("\\", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(files, key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0, reverse=True)
    return "\n".join(files) or "none"

@tool
def grep(pat: str, path: str = ".") -> str:
    '''Search for specific regex pattern in files'''
    pattern = re.compile(pat)
    hits = []
    for filepath in globlib.glob(os.path.join(path, "**"), recursive=True):
        if os.path.isdir(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass

    return "\n".join(hits[:50]) or "none" 

@tool
def bash(cmd: str) -> str:
    '''Shell is MSYS2'''    
    executable = shutil.which("bash")
    try:
        result = subprocess.run(
            [executable, "-c", cmd],
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Execution error: {str(e)}"
    
@tool(task='Specific task')
def explore(task: str, parallel: bool = True):
    """
    Execute sub-agent for exploration tasks, dispatch when you need to locate issues or summarize in large amounts of text.

    Please use the following format to pass tasks:

    # Task Topic
    Describe the problem you're encountering
    # Scope
    Most likely files
    # Task Steps
    Use ordered list to instruct sub-agent step by step, describe expected response
    """

    llm = DeepSeekLLM(
        model='deepseek-chat',
        api_key=os.environ['DEEP_SEEK_AGENT_KEY'],
        base_url='https://api.deepseek.com/chat/completions'
    )
    
    EXPLORE_SYSTEM_PROMPT = f"""
    You are skilled at locating information across numerous files. You are responsible for completing tasks according to requirements and providing concise summaries that meet task expectations.
    Use the done tool to report results.
    Current working directory: {Path.cwd().as_posix()}"""
    sub_tools = ['read', 'glob', 'grep', 'done'] 
    messages = [{'role': 'system','content': EXPLORE_SYSTEM_PROMPT}]
    
    for event in agent_loop(llm, messages, sub_tools):
        match event:
            case FinalResponseEvent(content=results):
                pass
            
    return results[-1]['content']

@tool
def done(report: str):
    '''At the end of the task, call this tool and input the required report.'''
    return report

# --- Events ---
@dataclass
class ToolCallEvent:
	tool: str
	args: str
	tool_call_id: str
     
@dataclass
class ToolResultEvent:
	tool: str
	result: str
	tool_call_id: str
     
@dataclass
class TextEvent:
    content: str

@dataclass
class FinalResponseEvent:
     content: list[dict]


# --- Define an LLM ---
class LLM(Protocol):
    medel: str
    api_key: str
    base_url: str

    def call(self, messages, tools) -> dict:
        ...
    
@dataclass
class DeepSeekLLM:
    model: str
    api_key: str
    base_url: str
    thinking: Literal['disabled', 'enabled'] = 'disabled'

    def call(self, messages: list[dict], tools: list[str]) -> dict:
        tool_schema = [TOOLS[key][0] for key in tools if key in TOOLS]

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": 8192,
                    "thinking": {"type": self.thinking},
                    "messages": messages,
                    "tools": tool_schema,
                },
                ensure_ascii=False
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            response = urllib.request.urlopen(request)
            return json.loads(response.read())
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode()
            raise Exception(f"HTTP Error: {e.code}, Detail: {error_detail}")

def agent_loop(llm: LLM, messages: list, tools: list[str]):
    while True:
        reponse = llm.call(messages, tools)

        message = reponse['choices'][0]['message']
        messages.append(message)

        tool_calls = message.get('tool_calls')
        content = message.get('content')

        if content:
            yield TextEvent(content=content)

        if not tool_calls:
            # TODO Maybe 
            yield FinalResponseEvent(content=messages)
            return

        if tool_calls:
            parallel_tools = []
            serial_tools = []
            for tool in tool_calls:
                tool_name = tool['function']['name']
                tool_args = json.loads(tool['function']['arguments'])
                tool_call_id = tool['id']

                if tool_name == 'done':
                    # Using the done tool to reply can reduce the issue of AI responding before completing the task.
                    yield FinalResponseEvent(content=messages)
                    return

                yield ToolCallEvent(tool=tool_name, args=str(tool_args), tool_call_id=tool_call_id)

                # Categorize the tools.
                if tool_args.get('parallel'):
                    parallel_tools.append(
                        {
                            'name': tool_name,
                            'arguments': tool_args,
                            'id': tool_call_id
                        }
                    )
                else:
                    serial_tools.append(
                        {
                            'name': tool_name,
                            'arguments': tool_args,
                            'id': tool_call_id
                        }
                    )

            # Execute serial tools
            for serial_tool in serial_tools:

                tool_call_id = serial_tool['id']
                name = serial_tool['name']
                args = serial_tool['arguments']

                func = TOOLS[name][1]

                try:
                    result = func(**args)
                except Exception as e:
                    result = f"error: {e}"

                messages.append({
                    'role': 'tool',
                    'content': json.dumps(result, ensure_ascii=False),
                    'tool_call_id': tool_call_id
                })        

                yield ToolResultEvent(tool=name, result=result, tool_call_id=tool_call_id)  

            # Execute parallel tools
            with ThreadPoolExecutor() as executor:
                futures_map = {}
                for parallel_tool in parallel_tools:

                    tool_call_id = parallel_tool['id']
                    name = parallel_tool['name']
                    args = parallel_tool['arguments']

                    func = TOOLS[name][1]
                    
                    future = executor.submit(func, **args)
                    futures_map[tool_call_id] = future

                for tool_call_id, future in futures_map.items():

                    try:
                        result = future.result()
                    except Exception as e:
                        result = f"error: {e}"

                    messages.append({
                        'role': 'tool',
                        'content': json.dumps(result, ensure_ascii=False),
                        'tool_call_id': tool_call_id
                    })

                    yield ToolResultEvent(tool=name, result=result, tool_call_id=tool_call_id)  




# --- cli decoration ---
def separator():
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def render_markdown(text: str) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    result = []

    for line in lines:

        if line.startswith("```"):
            result.append(line)
            continue

        line = re.sub(r"\*\*([^*]+?)\*\*", f"{BOLD}\\1{RESET}", line)
        line = re.sub(r"\*(.+?)\*", f"{DIM}\\1{RESET}", line)
        line = re.sub(r"`(.+?)`", f"{GREEN}\\1{RESET}", line)
        line = re.sub(r"\[(.+?)\]\(.+?\)", f"{BLUE}\\1{RESET}", line)

        if line.startswith("####"):
            line = f"{DIM}{line.strip('#').strip()}{RESET}"
        elif line.startswith("###"):
            line = f"{GREEN}{BOLD}{line.strip('#').strip()}{RESET}"
        elif line.startswith("##"):
            line = f"{BLUE}{BOLD}{line.strip('#').strip()}{RESET}"
        elif line.startswith("#"):
            line = f"{CYAN}{BOLD}{line.strip('#').strip()}{RESET}"
        elif line.startswith(">"):
            line = f"{DIM}  {line[1:].strip()}{RESET}"
        elif line.strip().startswith(("-", "*", "+ ")) and line.strip()[1].isspace():
            line = f"{CYAN}•{RESET} " + line.strip()[2:]

        result.append(line)

    return "\n".join(result)


# --- main ---
ONLY_READ = ['read', 'glob', 'grep', 'explore']
SYSTEM_PROMPT = f"""
You are a helpful assistant. Please assist the user in completing their tasks.
When the user makes a request, you should repeatedly ask questions to clarify their thoughts and obtain the necessary information to complete the task.
Then, present your plan to the user. If they agree, proceed with the action; otherwise, revise the plan.
Current working directory: {Path.cwd().as_posix()}"""

def main():

    deepseekllm = DeepSeekLLM(
        model='deepseek-reasoner',
        api_key=os.environ['DEEP_SEEK_AGENT_KEY'],
        base_url='https://api.deepseek.com/chat/completions',
        thinking='enabled'
    )

    messages = [{'role': 'system','content': SYSTEM_PROMPT}]
    while True:
        try:

            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())

            if user_input.startswith(('/q', '/exit')):
                break

            messages.append({'role': 'user', 'content': user_input})

            for e in agent_loop(llm=deepseekllm, messages=messages, tools=ONLY_READ):
                match e:
                    case TextEvent(content=content):
                        print(render_markdown(content))
                    case ToolCallEvent(tool=tool, args=args):
                        print(f'[{tool}] {args}'[:80])
                    case ToolResultEvent(tool=tool, result=result):
                        print(f'  - {tool}: {result}'[:80])
                    case FinalResponseEvent(content=content):
                        messages = content

        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()
