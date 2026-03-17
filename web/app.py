"""
web/app.py — Servidor Flask.
Ejecutar: python web/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template
from flask_session import Session
from web.routes.api import api_bp

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "selector_variables_v3"
app.config["SESSION_TYPE"]      = "filesystem"
app.config["SESSION_FILE_DIR"]  = ".flask_sessions"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(".flask_sessions", exist_ok=True)
Session(app)

app.register_blueprint(api_bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
