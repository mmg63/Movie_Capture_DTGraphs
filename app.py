# from flask import Flask
# import subprocess
# import os

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "✅ Movie Capture Graph Builder is live."

# @app.route("/run")
# def run_main_script():
#     result = subprocess.run(["python3", "main.py"], capture_output=True, text=True)
#     return f"<pre>{result.stdout or '[No output]'}\n\nErrors:\n{result.stderr or '[No errors]'}</pre>"

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     app.run(host="0.0.0.0", port=port)


from flask import Flask
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Movie Graph Processor is Live on Railway"

@app.route("/run")
def run_main_cloud():
    result = subprocess.run(["python3", "main_cloud.py"], capture_output=True, text=True)
    return f"<pre>{result.stdout or '[No output]'}\n\nErrors:\n{result.stderr or '[No errors]'}</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
