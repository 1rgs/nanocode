#!/usr/bin/env python3
"""nanocode - minimal claude code alternative"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
import inspect
from pathlib import Path
import shutil
from typing import Any, Callable, Literal, Optional, Protocol, TypeAlias
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
    
# --- Agent Tools ---
@tool(report='Your report')
def done(report: str) -> str:
    '''Call this when you finish the task'''
    return report

@tool(task='Task description', parallel='Whether to execute in parallel, usually True')
def explore(task: str, parallel: bool) -> str:
    """
    A sub-agent that executes exploration tasks. Dispatch when you need to locate issues or summarize in large amounts of text.

    Please use the following format to pass tasks:

    # Task Topic
    Explain the problem you encountered
    # Scope
    Most likely files
    # Task Steps
    Use ordered list to guide the sub-agent step by step, explain the expected response
    """

    llm = DeepSeekLLM(
        model='deepseek-chat',
        api_key=os.environ['DEEP_SEEK_AGENT_KEY'],
        base_url='https://api.deepseek.com/chat/completions'
    )

    # Define the subset of tools available to sub-tasks
    EXPLORE_SYSTEM_PROMPT = f"""
    You are good at locating information among numerous files. You are responsible for completing the task requirements and making a concise summary that meets the task requirements
    When you complete the task, use the done tool to report
    Current working directory: {Path.cwd().as_posix()}
    Files and directories under it: {"\n".join([f.name + ("/" if f.is_dir() else "") for f in Path.cwd().iterdir()])}"""
    sub_tools = ['read', 'glob', 'grep', 'done']
    messages = [{'role': 'system','content': EXPLORE_SYSTEM_PROMPT},
                {'role': 'user', 'content': task}]

    for event in agent_loop(llm, messages, sub_tools, max_step=20):
        match event:
            case FinalResponseEvent(messages):
                return messages[-1]['content']


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


# --- LLM API ---
class LLM(Protocol):
    model: str
    api_key: str
    base_url: str

    def call(self, messages, tools) -> dict:
        ...
    
@dataclass
class OpenAILike:
    model: str
    api_key: str
    base_url: str
    thinking: Literal['disabled', 'enabled'] = 'disabled'

    def call(self, messages: list[dict], tools: list[str], **extra_params) -> dict:
        tool_schema = [TOOLS[key][0] for key in tools if key in TOOLS]

        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "thinking": {"type": self.thinking},
            "messages": messages,
            "tools": tool_schema,
        }

        payload.update(extra_params)

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload,ensure_ascii=False).encode(),
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

@dataclass
class DeepSeekLLM(OpenAILike):
    pass

@dataclass
class ZAILLM(OpenAILike):
    pass

def agent_loop(llm: LLM, messages: list, tools: list[str], max_step: int = 200):

    def _execute_tool_func(name: str, args: dict):
        '''Call function'''
        func = TOOLS[name][1]
        try:
            return func(**args)
        except Exception as e:
            return f"error: {e}"

    def _handle_tool_result(name: str, result: Any, tool_call_id: str):
        '''Handle function execution result'''
        messages.append({
            'role': 'tool',
            'content': json.dumps(result, ensure_ascii=False),
            'tool_call_id': tool_call_id
        })
        return ToolResultEvent(tool=name, result=result, tool_call_id=tool_call_id)
    
    step = 0
    while step < max_step:
        step += 1
        is_final = False

        # TODO Context management

        # Handle reaching max steps
        if step >= max_step:
            messages.append({'role': 'user', 'content': 'You have reached the maximum steps. Report your current results immediately.\n1. What methods did you try\n2. What information did you get\n3. Suggest how to continue'})
            tools = []
            is_final = True

        # Call LLM api
        response = llm.call(messages, tools)
        message = response['choices'][0]['message']
        messages.append(message)

        # Parse returned content
        tool_calls = message.get('tool_calls')
        content = message.get('content')

        if content:
            yield TextEvent(content=content)
        if not tool_calls:
            if 'done' in tools:
                messages.append({'role': 'user', 'content': "Please use the done tool to finish the task"})
                continue
            is_final = True
        else:
            # --- Prepare tool call data ---
            parallel_tools = []
            serial_tools = []
            for tool in tool_calls:
                tool_name = tool['function']['name']
                tool_args = json.loads(tool['function']['arguments'])
                tool_call_id = tool['id']

                if tool_name == 'done':
                    is_final = True

                if tool_args.get('parallel'):
                    parallel_tools.append({'name': tool_name, 'args': tool_args, 'id': tool_call_id})
                else:
                    serial_tools.append({'name': tool_name, 'args': tool_args, 'id': tool_call_id})

            # --- Execute serial tools ---
            for tool in serial_tools:
                yield ToolCallEvent(tool=tool['name'], args=str(tool['args']), tool_call_id=tool['id'])
                result = _execute_tool_func(tool['name'], tool['args'])
                yield _handle_tool_result(tool['name'], result, tool['id'])

            # --- Execute parallel tools ---
            if parallel_tools:
                with ThreadPoolExecutor() as executor:
                    futures = {}
                    # Submit all parallel tasks
                    for tool in parallel_tools:
                        yield ToolCallEvent(tool=tool['name'], args=str(tool['args']), tool_call_id=tool['id'])
                        future = executor.submit(_execute_tool_func, tool['name'], tool['args'])
                        # Bind future with metadata so we know which tool it corresponds to when retrieving
                        futures[future] = tool
                    # Get results
                    for future in futures:
                        tool_info = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = f"error: {e}"
                        yield _handle_tool_result(tool_info['name'], result, tool_info['id'])

        if is_final:
            yield FinalResponseEvent(messages)
            return




class TerminalRenderer:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    ICON = "\uf111"

    @classmethod
    def print_start(cls, llm):
        print()
        print(f'{cls.BOLD}MyAgent | {cls.DIM}{llm.model}{cls.RESET}')
        print()

    @classmethod
    def separator(cls):
        """Generate an adaptive width separator line"""
        try:
            columns = os.get_terminal_size().columns
        except OSError:
            columns = 80
        print(f"{cls.DIM}{'─' * columns}{cls.RESET}")
    
    @classmethod
    def print_user_input(cls) -> str:
        """Render user input"""
        user_input = input(f"{cls.BOLD}{cls.BLUE}❯{cls.RESET} ").strip()
        return user_input

    @classmethod
    def print_text(cls, content):
        """Render regular text response"""
        print()
        print(f"{cls.CYAN}{cls.ICON}{cls.RESET} {cls.render_markdown(content.strip())}")
        print()

    @classmethod
    def print_tool_call(cls, tool_name, args):
        """Render tool call"""
        arg_preview = str(args)[:50]
        print(f"{cls.GREEN}{cls.ICON} {tool_name.capitalize()}{cls.RESET}({cls.DIM}{arg_preview}{cls.RESET})")

    @classmethod
    def print_tool_result(cls, result):
        """Render preview of tool return result"""
        lines = result.split("\n")
        preview = lines[0][:60]
        if len(lines) > 1:
            preview += f" ... +{len(lines) - 1} lines"
        elif len(lines[0]) > 60:
            preview += "..."
        print(f"  {cls.DIM}⎿  {preview}{cls.RESET}")

    @classmethod
    def print_error(cls, message):
        """Render error message"""
        print(f"{cls.RED}{cls.ICON} Error: {message}{cls.RESET}")

    @classmethod
    def print_info(cls, message):
        """Render info message"""
        print(f"{cls.GREEN}{cls.ICON} {message}{cls.RESET}")
        
    @classmethod
    def render_markdown(cls, text: str) -> str:
        """Parse simple Markdown syntax and convert to terminal escape sequences"""
        """Render AI output text (supports Markdown)"""
        if not text:
            return ""
        
        lines = text.split("\n")
        result = []
        is_code = False
        
        for line in lines:
            # Skip code block markers
            if line.startswith("```"):
                result.append(line)  
                is_code = not is_code
                continue
            if is_code:
                result.append(line)
                continue   

            # Bold
            line = re.sub(r"\*\*([^*]+?)\*\*", f"{cls.BOLD}\\1{cls.RESET}", line)
            # Italic/emphasis
            line = re.sub(r"\*(.+?)\*", f"{cls.DIM}\\1{cls.RESET}", line)
            # Inline code
            line = re.sub(r"`(.+?)`", f"{cls.GREEN}\\1{cls.RESET}", line)
            # Links (show only text)
            line = re.sub(r"$$(.+?)$$$.+?$", f"{cls.BLUE}\\1{cls.RESET}", line)

            # Headers
            if line.startswith("####"):
                line = f"{cls.DIM}{line.strip('#').strip()}{cls.RESET}"
            elif line.startswith("###"):
                line = f"{cls.GREEN}{cls.BOLD}{line.strip('#').strip()}{cls.RESET}"
            elif line.startswith("##"):
                line = f"{cls.BLUE}{cls.BOLD}{line.strip('#').strip()}{cls.RESET}"
            elif line.startswith("#"):
                line = f"{cls.CYAN}{cls.BOLD}{line.strip('#').strip()}{cls.RESET}"
            # Quote
            elif line.startswith(">"):
                line = f"{cls.DIM}  {line[1:].strip()}{cls.RESET}"
            # List items
            elif line.strip().startswith(("-", "*", "+ ")) and len(line.strip()) > 1 and line.strip()[1].isspace():
                line = f"{cls.CYAN}•{cls.RESET} " + line.strip()[2:]
            
            result.append(line)
        
        return f"\n  ".join(result)


# --- main ---
ONLY_READ = ['read', 'glob', 'grep', 'explore']
SYSTEM_PROMPT = f"""
You are a helpful assistant. Please assist the user in completing their tasks.
When the user makes a request, you should repeatedly ask questions to clarify their thoughts and obtain the necessary information to complete the task.
Then, present your plan to the user. If they agree, proceed with the action; otherwise, revise the plan.
Current working directory: {Path.cwd().as_posix()}"""
TR = TerminalRenderer

def main():

    # llm = DeepSeekLLM(
    #     model='deepseek-reasoner',
    #     api_key=os.environ['DEEP_SEEK_AGENT_KEY'],
    #     base_url='https://api.deepseek.com/chat/completions',
    # )

    llm = ZAILLM(
        model='GLM-4.7',
        api_key=os.environ['ZAI_KEY'],
        base_url='https://open.bigmodel.cn/api/paas/v4/chat/completions'
    )

    TR.print_start(llm)

    messages = [{'role': 'system','content': SYSTEM_PROMPT}]
    while True:
        try:

            user_input = TR.print_user_input()

            if not user_input:
                continue
            if user_input.startswith(('/q', '/exit')):
                break
            if user_input.startswith(('/c', '/clear')):
                messages = messages[:1]
                os.system('cls')
                continue

            messages.append({'role': 'user', 'content': user_input})

            for e in agent_loop(llm=llm, messages=messages, tools=ONLY_READ):
                match e:
                    case TextEvent(content=content):
                        TR.print_text(content)
                    case ToolCallEvent(tool=tool, args=args):
                        TR.print_tool_call(tool, args)
                    case ToolResultEvent(result=result):
                        TR.print_tool_result(result)
                    case FinalResponseEvent(content=content):
                        messages = content
                    
        except Exception as err:
            TR.print_error(err)



if __name__ == "__main__":
    main()
