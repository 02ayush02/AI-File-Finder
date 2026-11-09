Set-Location "$HOME\Downloads\ai-file-finder\ai-file-finder"
.\.venv\Scripts\Activate.ps1
$env:FILEFINDER_WATCH_DIRS = "C:\Users\ayush\Downloads"
$env:FILEFINDER_LLM_MODEL = "qwen2.5:3b"
$env:FILEFINDER_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
$env:FILEFINDER_RERANKER_MAX_LENGTH = "512"
streamlit run app.py