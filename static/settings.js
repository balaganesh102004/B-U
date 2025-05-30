const LLM_ARGS = {
    gemini: [
        {name: "gemini_api_key", label: "Gemini API Key", type: "password", required: true}
    ],
    azure_openai: [
        {name: "azure_openai_api_key", label: "Azure OpenAI API Key", type: "password", required: true},
        {name: "azure_openai_api_version", label: "API Version", type: "text", required: true},
        {name: "azure_openai_api_endpoint", label: "API Endpoint", type: "text", required: true}
    ],
    openai: [
        {name: "openai_api_key", label: "OpenAI API Key", type: "password", required: true}
    ],
    ollama: [
        {name: "ollama_host", label: "Ollama Host", type: "text", required: true}
    ],
    groq: [
        {name: "groq_api_key", label: "Groq API Key", type: "password", required: true}
    ]
};

function renderLlmArgs(selectId, argsDivId, currentSettings) {
    const select = document.getElementById(selectId);
    const argsDiv = document.getElementById(argsDivId);
    const llm = select.value;
    argsDiv.innerHTML = "";
    if (LLM_ARGS[llm]) {
        LLM_ARGS[llm].forEach(arg => {
            const value = currentSettings && currentSettings[arg.name] ? currentSettings[arg.name] : "";
            argsDiv.innerHTML += `
                <label for="${selectId}-${arg.name}">${arg.label}${arg.required ? ' *' : ''}</label>
                <input type="${arg.type}" id="${selectId}-${arg.name}" name="${selectId}_${arg.name}" value="${value}" ${arg.required ? 'required' : ''} autocomplete="off">
            `;
        });
    }
}

document.addEventListener("DOMContentLoaded", function() {
    // These objects should be rendered from Flask context if you want to prefill values
    const agentLlmSettings = window.agentLlmSettings || {};
    const plannerLlmSettings = window.plannerLlmSettings || {};

    renderLlmArgs("agent-llm", "agent-llm-args", agentLlmSettings);
    renderLlmArgs("planner-llm", "planner-llm-args", plannerLlmSettings);

    document.getElementById("agent-llm").addEventListener("change", function() {
        renderLlmArgs("agent-llm", "agent-llm-args");
    });
    document.getElementById("planner-llm").addEventListener("change", function() {
        renderLlmArgs("planner-llm", "planner-llm-args");
    });
});