from flask import Flask, render_template, request, redirect, url_for, jsonify, render_template, request, Response, send_file
import json
import os
import asyncio
import threading
from browser_use import Agent, BrowserConfig  
from browser_use.browser import BrowserProfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama

from weasyprint import HTML, CSS
from pydantic import SecretStr
from dotenv import load_dotenv
from report_generator import render_report 
import io
import uuid
from datetime import datetime

load_dotenv()

app = Flask(__name__)
DATA_FILE = 'tasks.json'
SETTINGS_FILE = 'settings.json'

browser_config = BrowserProfile(
    highlight_elements = False,
    user_data_dir=None
)
def load_tasks():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_tasks(tasks):
    with open(DATA_FILE, 'w') as f:
        json.dump(tasks, f, indent=4)

def get_next_id(tasks):
    return max([task["ID"] for task in tasks], default=0) + 1

@app.route('/')
def index():
    tasks = load_tasks()
    return render_template('index.html', tasks=tasks)

@app.route('/add', methods=['POST'])
def add_task():
    tasks = load_tasks()
    new_task = {
        "ID": get_next_id(tasks),
        "Task name": request.form['task_name'],
        "Task description": request.form['task_description'],
        "Tags": request.form['tags'].split(',')
    }
    tasks.append(new_task)
    save_tasks(tasks)
    return redirect(url_for('index'))

@app.route('/update/<int:task_id>', methods=['POST'])
def update_task(task_id):
    tasks = load_tasks()
    for task in tasks:
        if task["ID"] == task_id:
            task["Task name"] = request.form['task_name']
            task["Task description"] = request.form['task_description']
            task["Tags"] = request.form['tags'].split(',')
            break
    save_tasks(tasks)
    return redirect(url_for('index'))

@app.route('/delete/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    tasks = load_tasks()
    tasks = [task for task in tasks if task["ID"] != task_id]
    save_tasks(tasks)
    return redirect(url_for('index'))


def run_async_in_thread(coro):
    result = {}
    exception = {}

    def runner():
        try:
            result['value'] = asyncio.run(coro)
        except Exception as e:
            exception['error'] = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if 'error' in exception:
        raise exception['error']

    return result['value']

override_system_prompt = """
You are an expert in intelligent web automation. Your primary goal is to ensure reliable, human-like interactions with web interfaces by intelligently adapting to page structure and dynamics. Follow the directives below for robust and accurate automation:

1. Scrolling & Viewport Handling:
   - Detect and interact with scrollable containers (including nested ones).
   - Scroll to bring elements into view before interacting.
   - Support infinite scrolling: keep loading until the end is detected.
   - Use bounding box checks or visibility APIs to ensure elements are visible and unobstructed.

2. Element Interaction Strategy:
   - **Clickable Elements**: Wait until clickable (e.g., buttons, links, icons) and interact only when enabled.
   - **Hover Interactions**: Simulate hover when required to reveal hidden menus/tooltips.
   - **Sliders/Range Inputs**: Adjust to required value with appropriate drag or key events.
   - **Modals & Popups**: Detect presence and interact with visible modal content. Handle close/cancel/submit.
   - Handle dynamic content loading: wait for transitions, AJAX/XHR completions, or DOM mutations.

3. Forms, Inputs & Field Automation:
   - Fill text fields sequentially based on visible field order or label proximity.
   - Respect field dependencies (e.g., field B enabled only after filling A).
   - Handle autocomplete/dropdowns by simulating typing and selecting suggestions.

4. Option & Selection Handling:
   - **Dropdowns**: 
     - Expand the menu, retrieve all available (non-disabled) options.
     - Select the top available option unless a specific one is required.
     - Confirm selection by checking the visible label/value of the dropdown post-selection.
   - **Checkboxes**:
     - Verify current state before toggling to avoid unnecessary clicks.
     - Check the checkbox if it's not already checked, only when required.
     - Support bulk selection when checkboxes are in lists.
   - **Radio Buttons**:
     - Ensure the correct group is targeted.
     - Select the specified or default (first enabled) radio option.
     - Validate the selection state by checking the group afterward.

5. File Upload & Attachments:
   - Detect file input fields (visible or hidden).
   - Upload files using simulated file chooser input, waiting for processing (e.g., thumbnails or filenames).

6. Navigation & Process Flow:
   - Handle multi-step workflows by validating each stage before proceeding.
   - Detect and wait for loading states (spinners, skeleton loaders, etc.) to finish.
   - Always confirm actions by checking for success, confirmation, or error messages.
   
7. Robustness & Safety:
   - Never assume presence of elements — use timeouts, retries, and fallbacks.
   - Interact only when elements are enabled, visible, and stable (no animations in progress).
   - Use semantic cues (ARIA labels, roles, visible text) for locating and confirming actions when possible.
   - Log interactions and decisions for traceability and debugging.

Your automation must behave as a careful, context-aware user would, ensuring all interactions are validated and errors are gracefully handled.
"""

async def run_task_async(task):
    # Load settings from JSON
    settings = load_settings()

    # --- LLM Selection and Validation ---
    def get_llm(llm_name, llm_args):
        # Validate required fields for each LLM
        if llm_name == "gemini":
            api_key = llm_args.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API Key is required.")
            return ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp',
                api_key=SecretStr(api_key),
                temperature=0.2,
                seed=42
            )
        elif llm_name == "azure_openai":
            api_key = llm_args.get("azure_openai_api_key") or os.getenv("AZURE_OPENAI_API_KEY")
            api_version = llm_args.get("azure_openai_api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
            api_endpoint = llm_args.get("azure_openai_api_endpoint") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
            if not (api_key and api_version and api_endpoint):
                raise ValueError("Azure OpenAI API Key, Version, and Endpoint are required.")
            return AzureChatOpenAI(
                model="gpt-4o-mini",
                api_version=api_version,
                azure_endpoint=api_endpoint,
                api_key=SecretStr(api_key),
            )
        elif llm_name == "openai":
            api_key = llm_args.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API Key is required.")
            return ChatOpenAI(
                model="gpt-4o",
                api_key=SecretStr(api_key),
                temperature=0.2,
            )
        elif llm_name == "ollama":
            ollama_host = llm_args.get("ollama_host") or os.getenv("OLLAMA_HOST")
            if not ollama_host:
                raise ValueError("Ollama Host is required.")
            return ChatOllama(
                model="llama3",
                base_url=ollama_host,
                temperature=0.2,
            )
        else:
            raise ValueError(f"Unknown LLM: {llm_name}")

    # Get agent and planner LLMs from settings
    agent_llm = get_llm(settings.get("agent_llm", "gemini"), settings.get("agent_llm_args", {}))
    planner_llm = get_llm(settings.get("planner_llm", "gemini"), settings.get("planner_llm_args", {}))

    # --- Browser Profile from settings ---
    browser_profile = BrowserProfile(
        highlight_elements=settings.get("highlight_elements", False),
        headless=settings.get("headless_mode", False),
        user_data_dir=None
    )

    agent = Agent(
        task=task,
        llm=agent_llm,
        planner_llm=planner_llm,
        override_system_message=override_system_prompt,
        browser_profile=browser_profile
    )
    try:
        history = await agent.run()
        return history.model_dump()
    finally:
        if hasattr(agent, "close"):
            await agent.close()

async def run_tasks_concurrently(task_descriptions):
    coros = [run_task_async(description) for description in task_descriptions]
    return await asyncio.gather(*coros)

@app.route('/run', methods=['POST'])
def run():
    global history_data
    all_tasks = load_tasks()
    selected_task_names = request.form.getlist('tasks[]')
    selected_descriptions = [
        task["Task description"]
        for task in all_tasks
        if task["Task name"] in selected_task_names
    ]
    print("Selected task names:", selected_task_names)
    # Run the agent on descriptions in a fresh thread/event loop
    raw_results = run_async_in_thread(run_tasks_concurrently(selected_descriptions))
    history_data = {
        name: result
        for name, result in zip(selected_task_names, raw_results)
    }

    # Generate a unique test run ID and timestamps
    test_run_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f"test_run_{timestamp}_{test_run_id}.html"

    # Pass test_run_id to the report
    html_report = render_report(history_data, test_run_id=test_run_id)

    # Save the report to the History folder
    history_folder = os.path.join(os.path.dirname(__file__), "History")
    os.makedirs(history_folder, exist_ok=True)
    with open(os.path.join(history_folder, filename), "w", encoding="utf-8") as f:
        f.write(html_report)

    return Response(html_report, mimetype='text/html')

@app.route('/generate_tasks', methods=['POST'])
def generate_tasks():
    data = request.get_json()
    task_ids = data.get('task_ids', [])
    tasks = load_tasks()
    selected_descriptions = [task["Task description"] for task in tasks if task["ID"] in task_ids]
    results = run_async_in_thread(run_tasks_concurrently(selected_descriptions))
    return jsonify(results)

@app.route('/history')
def history():
    history_folder = os.path.join(os.path.dirname(__file__), "History")
    runs = []
    if os.path.exists(history_folder):
        for fname in sorted(os.listdir(history_folder), reverse=True):
            if fname.endswith(".html"):
                with open(os.path.join(history_folder, fname), encoding="utf-8") as f:
                    content = f.read()
                    def extract(field):
                        import re
                        m = re.search(rf'<strong>{field}:</strong>\s*([^<]+)', content)
                        return m.group(1).strip() if m else "N/A"
                    runs.append({
                        "id": extract("Test Run ID"),
                        "file": fname,
                        "status": extract("Status"),
                        "duration": extract("Total Duration"),
                        "tokens": extract("Total Tokens"),  
                        "start": extract("Start Time"),
                        "end": extract("End Time"),
                    })
    return render_template("history.html", runs=runs)

@app.route('/history/<filename>')
def history_report(filename):
    history_folder = os.path.join(os.path.dirname(__file__), "History")
    return send_file(os.path.join(history_folder, filename))

@app.route('/history/delete/<filename>', methods=['POST'])
def delete_history(filename):
    history_folder = os.path.join(os.path.dirname(__file__), "History")
    file_path = os.path.join(history_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "File not found"}), 404

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        # Default settings
        return {
            "agent_llm": "gemini",
            "agent_llm_args": {"gemini_api_key": ""},
            "planner_llm": "gemini",
            "planner_llm_args": {"gemini_api_key": ""},
            "headless_mode": False,
            "highlight_elements": False
        }
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        agent_llm = request.form.get('agent_llm', 'gemini')
        planner_llm = request.form.get('planner_llm', 'gemini')
        agent_llm_args = {}
        planner_llm_args = {}

        # Collect dynamic LLM args
        for key in request.form:
            if key.startswith('agent-llm_'):
                arg_name = key.replace('agent-llm_', '')
                agent_llm_args[arg_name] = request.form[key]
            if key.startswith('planner-llm_'):
                arg_name = key.replace('planner-llm_', '')
                planner_llm_args[arg_name] = request.form[key]

        headless_mode = 'headless_mode' in request.form
        highlight_elements = 'highlight_elements' in request.form

        settings_obj = {
            "agent_llm": agent_llm,
            "agent_llm_args": agent_llm_args,
            "planner_llm": planner_llm,
            "planner_llm_args": planner_llm_args,
            "headless_mode": headless_mode,
            "highlight_elements": highlight_elements
        }
        save_settings(settings_obj)
        return redirect(url_for('settings'))

    settings_obj = load_settings()
    # Pass LLM args as JS variables for dynamic rendering
    return render_template(
        'settings.html',
        settings=settings_obj,
        agentLlmSettings=settings_obj.get("agent_llm_args", {}),
        plannerLlmSettings=settings_obj.get("planner_llm_args", {})
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
