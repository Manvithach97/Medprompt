import subprocess
import json
import re

def call_llm(prompt_text, model="mistral"):
    #calls llm 
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt_text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("LLM call failed:", result.stderr.decode())
        return ""
    return result.stdout.decode()


def clean_json_text(text: str) -> str:
    # removes js style comments , trailing commas , etc using regex
    text=re.sub(r'//.*?\n', '\n', text)

    text=re.sub(r',(\s*[}\]])', r'\1', text)
    text=re.sub(r'(?<=:\s)(\d+%)', r'"\1"', text)
    return text



def extract_combined(text, model="mistral"):
    #extracts both concise n detailed in one model call 
    prompt = f""" You are a clinical language assistant.
First, give a **concise structured summary** of the medical data in JSON format using only the following keys if present:
- Medication, Dose, Unit, Time, Date, Symptom, Diagnosis, Procedure, LabTest, VitalSign
Then, give a **detailed structured breakdown** of the information using custom categories based on what's actually mentioned.
Input:
\"\"\"{text}\"\"\"
Output:
{{
  "ConciseSummary": {{
    ...
  }},
  "DetailedSummary": {{
    ...
  }}
}}
    """

    response = call_llm(prompt, model=model)
    # 
    try:
        start=response.index("{")
        end=response.rindex("}") + 1
        json_str = response[start:end]

        #clean for formatting issues
        json_str = clean_json_text(json_str)

        #json parsing 
        structured = json.loads(json_str)
        return structured
    except Exception as e:
        print("Failed to extract JSON:", e)
        print("Raw model output:\n", response)
        #return raw response n error for debugging
        return {"RawResponse": response, "error": str(e)}
