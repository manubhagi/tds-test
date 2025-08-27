#app.py
import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
import io
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# LangChain / LLM imports (keep as you used)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 150))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        # Always resolve index.html relative to this file's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        keys_list: list of keys in order
        type_map: dict key -> casting function
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map




# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


import subprocess, sys, tempfile, os, pickle, base64, json
import pandas as pd

def write_and_run_temp_python(user_code: str, df: pd.DataFrame):
    """
    Writes a temp python script that:
      - injects df via pickle
      - defines plot_to_base64 helper
      - runs user_code
      - ensures JSON-safe stdout parsing
    """
    preamble = """import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
import pickle, base64, io, json
from PIL import Image

# Load dataframe from pickle
with open("df.pkl","rb") as f:
    df = pickle.load(f)

# Helper to encode plots to base64
def plot_to_base64(fig=None):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    from PIL import Image

    if fig is None:
        fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # resize + reencode to keep size <100KB if possible
    img = Image.open(buf)
    buf2 = BytesIO()
    img.save(buf2, format="PNG", optimize=True)
    data = buf2.getvalue()

    # fallback: webp (smaller)
    if len(data) > 100000:
        buf2 = BytesIO()
        img.save(buf2, format="WEBP", quality=80)
        data = buf2.getvalue()

    return base64.b64encode(data).decode("utf-8")
"""

    # write temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        df_path = os.path.join(tmpdir, "df.pkl")
        with open(df_path, "wb") as f:
            pickle.dump(df, f)

        with open(script_path, "w") as f:
            f.write(preamble + "\n" + user_code + "\n")

        # run subprocess
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # capture stdout
        out = completed.stdout.strip()

        # ✅ Only keep the last JSON object in stdout
        last_brace = out.rfind("{")
        if last_brace != -1:
            candidate = out[last_brace:]
        else:
            candidate = out

        try:
            parsed = json.loads(candidate)
            return parsed
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not parse JSON output: {str(e)}",
                "raw": candidate,
                "stderr": completed.stderr.strip(),
            }



# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
def run_agent_safely(llm_input: str) -> Dict:
    """
    1. Run the agent_executor.invoke to get LLM output
    2. Extract JSON, get 'code' and 'questions'
    3. Detect scrape_url_to_dataframe("...") calls in code, run them here, pickle df and inject before exec
    4. Execute the code in a temp file and return results mapping questions -> answers
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
        if not raw_out:
            return {"error": f"Agent returned no output. Full response: {response}"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if not isinstance(parsed, dict) or "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        questions: List[str] = parsed["questions"]

        # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
        urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
        pickle_path = None
        if urls:
            # For now support only the first URL (agent may code multiple scrapes; you can extend this)
            url = urls[0]
            tool_resp = scrape_url_to_dataframe(url)
            if tool_resp.get("status") != "success":
                return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
            # create df and pickle it
            df = pd.DataFrame(tool_resp["data"])
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name
            # Make sure agent's code can reference df/data: we will inject the pickle loader in the temp script

        # Execute code in temp python script
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message', exec_result)}", "raw": exec_result.get("raw")}

        # exec_result['result'] should be results dict
        results_dict = exec_result.get("result", {})
        # Map to original questions (they asked to use exact question strings)
        output = {}
        for q in questions:
            output[q] = results_dict.get(q, "Answer not found")
        return output

    except Exception as e:
        logger.exception("run_agent_safely failed")
        return {"error": str(e)}


from fastapi import Request

@app.post("/api")
async def analyze_data(request: Request):
    """Main API endpoint for file uploads and analysis"""
    try:
        # Check content type
        content_type = request.headers.get("content-type", "")
        
        if "multipart/form-data" in content_type:
            # Handle file uploads
            form = await request.form()
            questions_file = None
            data_file = None

            for key, val in form.items():
                if hasattr(val, "filename") and val.filename:  # it's a file
                    fname = val.filename.lower()
                    if fname.endswith(".txt") and questions_file is None:
                        questions_file = val
                    else:
                        data_file = val

            if not questions_file:
                raise HTTPException(400, "Missing questions file (.txt)")
                
            raw_questions = (await questions_file.read()).decode("utf-8")
            keys_list, type_map = parse_keys_and_types(raw_questions)
            
        elif "application/json" in content_type:
            # Handle JSON requests
            body = await request.json()
            questions = body.get("questions", "")
            data_url = body.get("data_url", None)
            
            if not questions:
                raise HTTPException(400, "Missing 'questions' field in JSON")
                
            raw_questions = questions
            keys_list, type_map = [], {}  # No key mapping for JSON requests
            
            # Handle data URL if provided
            if data_url:
                tool_resp = scrape_url_to_dataframe(data_url)
                if tool_resp.get("status") != "success":
                    raise HTTPException(400, f"Failed to fetch data from URL: {tool_resp.get('message')}")
                
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name
                
                df_preview = (
                    f"\n\nThe dataset from {data_url} has {len(df)} rows and {len(df.columns)} columns.\n"
                    f"Columns: {', '.join(df.columns.astype(str))}\n"
                    f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
                )
            else:
                pickle_path = None
                df_preview = ""
                
        else:
            raise HTTPException(400, "Unsupported content type. Use multipart/form-data for files or application/json for text")

        # Continue with the rest of the processing
        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        # Handle data file uploads (only for multipart requests)
        if "multipart/form-data" in content_type and data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                try:
                    if PIL_AVAILABLE:
                        image = Image.open(BytesIO(content))
                        image = image.convert("RGB")  # ensure RGB format
                        df = pd.DataFrame({"image": [image]})
                    else:
                        raise HTTPException(400, "PIL not available for image processing")
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")  
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            # Pickle for injection
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        # Build rules based on data presence
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            "Respond with the JSON object only."
        )

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process key mapping & type casting
        if keys_list and type_map:
            mapped = {}
            for idx, q in enumerate(result.keys()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        val = result[q]
                        if isinstance(val, str) and val.startswith("data:image/"):
                            # Remove data URI prefix
                            val = val.split(",", 1)[1] if "," in val else val
                        mapped[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        mapped[key] = result[q]
            result = mapped

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


    
from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)




@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST /api for actual analysis."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api for analysis.",
        "endpoints": {
            "file_upload": "POST /api - Upload files (multipart/form-data with questions.txt + data file)",
            "json_api": "POST /api - Send JSON with questions and optional data_url",
            "health_check": "GET /api - This endpoint"
        },
        "usage_examples": {
            "file_upload": "curl -X POST http://127.0.0.1:8000/api -F 'questions_file=@questions.txt' -F 'data_file=@dataset.csv'",
            "json_api": "curl -X POST http://127.0.0.1:8000/api -H 'Content-Type: application/json' -d '{\"questions\": \"What is the average age?\", \"data_url\": \"https://example.com/data.csv\"}'"
        },
        "supported_formats": {
            "questions": "Text file (.txt) or JSON string",
            "data": "CSV, Excel (.xlsx, .xls), Parquet, JSON, Images (.png, .jpg, .jpeg)"
        }
    })

