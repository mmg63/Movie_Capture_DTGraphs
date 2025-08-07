from flask import Flask
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "📽️ Movie DT Graph Builder API is running."

@app.route("/run")
def run_graph_builder():
    # Run your main.py as a subprocess
    result = subprocess.run(["python3", "main.py"], capture_output=True, text=True)
    return f"<pre>{result.stdout}\n\nErrors:\n{result.stderr}</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
