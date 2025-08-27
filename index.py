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
import concurrent.futures
import pickle
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
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

# LangChain / LLM imports
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

# Global variables for LLM and Agent
llm = None
agent_executor = None

def initialize_agent():
    """Initialize the LLM and agent with proper error handling"""
    global llm, agent_executor
    
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            return False
            
        logger.info("Initializing LLM with Gemini Pro...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis agent. Your job is to:
            1. Analyze questions and determine what data/analysis is needed
            2. Generate Python code to answer the questions
            3. Return results in the exact format requested
            
            Always respond with a JSON object containing:
            - "questions": list of original question strings
            - "code": Python code that creates a 'results' dictionary with question strings as keys
            
            Available tools: scrape_url_to_dataframe for web data
            Available libraries: pandas, numpy, matplotlib, seaborn, PIL
            Helper function: plot_to_base64() for encoding plots"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        tools = [scrape_url_to_dataframe]
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        logger.info("Agent initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM/Agent: {e}")
        llm = None
        agent_executor = None
        return False

# Try to initialize agent on startup
initialize_agent()

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


def write_and_run_temp_python(user_code: str, injected_pickle: str = None, timeout: int = 150):
    """
    Writes a temp python script that:
      - injects df via pickle if provided
      - defines plot_to_base64 helper
      - runs user_code
      - ensures JSON-safe stdout parsing
    """
    preamble = """import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
import pickle, base64, io, json
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Initialize results dictionary
results = {}

"""

    # Add DataFrame loading if pickle provided
    if injected_pickle:
        preamble += f"""
# Load dataframe from pickle
try:
    with open(r"{injected_pickle}", "rb") as f:
        df = pickle.load(f)
    print(f"Loaded DataFrame with shape: {{df.shape}}")
except Exception as e:
    print(f"Error loading DataFrame: {{e}}")
    df = pd.DataFrame()
"""

    preamble += """
# Helper to encode plots to base64
def plot_to_base64(fig=None):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    if fig is None:
        fig = plt.gcf()
    
    if not fig.get_axes():
        return ""
        
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        
        # Resize if PIL is available and image is too large
        if PIL_AVAILABLE and len(buf.getvalue()) > 100000:
            from PIL import Image
            img = Image.open(buf)
            # Resize to max 800px width while maintaining aspect ratio
            if img.width > 800:
                ratio = 800 / img.width
                new_size = (800, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buf2 = BytesIO()
            img.save(buf2, format="PNG", optimize=True)
            data = buf2.getvalue()
        else:
            data = buf.getvalue()
            
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        print(f"Error creating plot: {e}")
        return ""
    finally:
        plt.close(fig)

"""

    # write temp file and execute
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        
        with open(script_path, "w", encoding='utf-8') as f:
            f.write(preamble + "\n" + user_code + "\n")
            # Ensure results are printed as JSON
            f.write("""
# Print results as JSON
try:
    print(json.dumps({"status": "success", "result": results}))
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e)}))
""")

        # run subprocess
        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Code execution timeout"}

        # capture stdout
        out = completed.stdout.strip()
        stderr = completed.stderr.strip()

        # Try to find the JSON output
        lines = out.split('\n')
        json_line = None
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{"status":'):
                json_line = line
                break

        if json_line:
            try:
                return json.loads(json_line)
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the last line or entire output
        try:
            return json.loads(out.split('\n')[-1])
        except:
            return {
                "status": "error",
                "message": f"Could not parse execution output",
                "stdout": out,
                "stderr": stderr
            }


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    global agent_executor
    
    # Check if agent is initialized, try to initialize if not
    if agent_executor is None:
        logger.warning("Agent not initialized, attempting to initialize...")
        if not initialize_agent():
            return {"error": "Agent initialization failed. Please check GOOGLE_API_KEY environment variable and ensure all dependencies are installed."}
    
    # Double check agent_executor is available
    if agent_executor is None:
        return {"error": "Agent not available. Please check GOOGLE_API_KEY environment variable."}
        
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            try:
                response = agent_executor.invoke({"input": llm_input})
                raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
                if raw_out:
                    break
            except Exception as e:
                logger.error(f"Agent execution attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    return {"error": f"Agent execution failed: {str(e)}"}
                    
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format. Expected 'code' and 'questions' keys. Got: {list(parsed.keys())}"}

        code = parsed["code"]
        questions = parsed["questions"]

        # Handle scraping if no pickle path provided
        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    df.to_pickle(temp_pkl.name)
                    pickle_path = temp_pkl.name

        # Execute the code
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        
        if exec_result.get("status") != "success":
            return {
                "error": f"Code execution failed: {exec_result.get('message')}", 
                "details": exec_result
            }

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


@app.post("/api")
async def analyze_data(request: Request):
    """Main API endpoint for file uploads and analysis"""
    try:
        # Check if agent is available, try to initialize if not
        if agent_executor is None:
            logger.warning("Agent not initialized in API endpoint, attempting to initialize...")
            if not initialize_agent():
                raise HTTPException(500, "Agent initialization failed. Please check GOOGLE_API_KEY environment variable and ensure all dependencies are installed.")
        
        # Double check
        if agent_executor is None:
            raise HTTPException(500, "Agent not available. Please verify GOOGLE_API_KEY is set in environment variables.")
            
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
            data_file = None
            
        else:
            raise HTTPException(400, "Unsupported content type. Use multipart/form-data for files or application/json for text")

        # Initialize variables
        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        # Handle JSON data URL
        if "application/json" in content_type:
            body = await request.json()
            data_url = body.get("data_url", None)
            
            if data_url:
                tool_resp = scrape_url_to_dataframe(data_url)
                if tool_resp.get("status") != "success":
                    raise HTTPException(400, f"Failed to fetch data from URL: {tool_resp.get('message')}")
                
                df = pd.DataFrame(tool_resp["data"])
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    df.to_pickle(temp_pkl.name)
                    pickle_path = temp_pkl.name
                
                df_preview = (
                    f"\n\nThe dataset from {data_url} has {len(df)} rows and {len(df.columns)} columns.\n"
                    f"Columns: {', '.join(df.columns.astype(str))}\n"
                    f"First few rows:\n{df.head(3).to_string(index=False)}\n"
                )

        # Handle data file uploads (only for multipart requests)
        if "multipart/form-data" in content_type and data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()

            try:
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
                elif filename.endswith((".png", ".jpg", ".jpeg")):
                    if not PIL_AVAILABLE:
                        raise HTTPException(400, "PIL not available for image processing")
                    try:
                        image = Image.open(BytesIO(content))
                        image = image.convert("RGB")  # ensure RGB format
                        df = pd.DataFrame({"image": [image]})
                    except Exception as e:
                        raise HTTPException(400, f"Image processing failed: {str(e)}")
                else:
                    raise HTTPException(400, f"Unsupported data file type: {filename}")

                # Create pickle file
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    df.to_pickle(temp_pkl.name)
                    pickle_path = temp_pkl.name

                df_preview = (
                    f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                    f"Columns: {', '.join(df.columns.astype(str))}\n"
                    f"First few rows:\n{df.head(3).to_string(index=False)}\n"
                )
                
            except Exception as e:
                raise HTTPException(400, f"Failed to process data file: {str(e)}")

        # Build LLM input based on data presence
        if dataset_uploaded or pickle_path:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the provided dataset for answering questions.\n"
                "4) Create a 'results' dictionary with exact question strings as keys.\n"
                "5) For plots: use plot_to_base64() helper to return base64 image data.\n"
                "6) Return JSON with 'questions' (list) and 'code' (string) keys only.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, call scrape_url_to_dataframe(url) in your code.\n"
                "2) Create a 'results' dictionary with exact question strings as keys.\n"
                "3) For plots: use plot_to_base64() helper to return base64 image data.\n"
                "4) Return JSON with 'questions' (list) and 'code' (string) keys only.\n"
            )

        llm_input = (
            f"{llm_rules}\n\nQuestions to answer:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}\n"
            "Respond with only the JSON object containing 'questions' and 'code' keys."
        )

        # Run agent with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = future.result(timeout=LLM_TIMEOUT_SECONDS + 10)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process key mapping & type casting for file uploads
        if keys_list and type_map and len(result) > 0:
            mapped = {}
            result_items = list(result.items())
            for idx, (question, answer) in enumerate(result_items):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        # Handle base64 image data
                        if isinstance(answer, str) and answer.startswith("data:image/"):
                            val = answer.split(",", 1)[1] if "," in answer else answer
                        else:
                            val = answer
                        
                        if val not in (None, "", "Answer not found"):
                            mapped[key] = caster(val)
                        else:
                            mapped[key] = val
                    except (ValueError, TypeError):
                        mapped[key] = answer
                else:
                    # If more results than expected keys, use original question
                    mapped[question] = answer
            result = mapped

        # Clean up temporary pickle file
        if pickle_path and os.path.exists(pickle_path):
            try:
                os.unlink(pickle_path)
            except:
                pass  # Ignore cleanup errors

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=f"Internal server error: {str(e)}")


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
    global agent_executor
    
    # Try to initialize if not available
    if agent_executor is None:
        initialize_agent()
    
    agent_status = "initialized" if agent_executor else "failed to initialize"
    
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api for analysis.",
        "agent_status": agent_status,
        "google_api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
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

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for deployment platforms like Render
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
