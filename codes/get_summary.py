import subprocess

def call_summary_llm(text, model="mistral"):
    prompt = f"""
You are a clinical summarization assistant.
Based on the following clinical note, generate a short, human-readable paragraph summary that captures the overall medical context, conditions, interventions, and recommendations.
Text:
\"\"\"{text}\"\"\"
Summary:
"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("Summary LLM call failed:", result.stderr.decode())
        return "Error generating summary."
    return result.stdout.decode().strip()