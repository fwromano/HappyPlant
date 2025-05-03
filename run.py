#!/usr/bin/env python3
"""
Launcher for 🌱 HappyPlant — creates/uses venv & installs only *missing* deps
"""

import os, sys, subprocess, platform, time, webbrowser, shutil, json
from pathlib import Path
from threading import Thread
from importlib.metadata import version, PackageNotFoundError

# ── colours ────────────────────────────────────────────────────────────────
class C: G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
say = lambda m,c=C.G: print(f"{c}{m}{C.N}")

# ── utility ────────────────────────────────────────────────────────────────
def py39():
    """return path to a python ≥3.9 interpreter"""
    if sys.version_info >= (3,9):
        return sys.executable
    for cmd in ('python3.12','python3.11','python3.10','python3.9','python3'):
        try:
            out = subprocess.check_output([cmd,'-c','import sys, json;print(json.dumps(sys.version_info[:2]))'],
                                          text=True)
            if tuple(json.loads(out)) >= (3,9):
                return cmd
        except Exception:
            pass
    say("Python ≥3.9 not found.", C.R); sys.exit(1)

def venv_py():
    return Path("venv")/("Scripts" if platform.system()=="Windows" else "bin")/"python"

def unsatisfied(requirements):
    """
    Return list of packages that are missing or wrong‑version vs requirements.txt
    """
    missing=[]
    for line in requirements:
        line=line.strip()
        if not line or line.startswith('#'): continue
        pkg, req_ver = (line.split('==')+[None])[:2]
        try:
            cur_ver = version(pkg)
            if req_ver and cur_ver != req_ver:
                missing.append(pkg)
        except PackageNotFoundError:
            missing.append(pkg)
    return missing

def ensure_venv():
    if not Path('venv').exists():
        say("Creating virtual env …", C.Y)
        subprocess.check_call([py39(), "-m", "venv", "venv"])
    if Path(sys.prefix).resolve()!=Path("venv").resolve():
        subprocess.check_call([str(venv_py()), __file__]); sys.exit(0)

def ensure_deps():
    req_path=Path('requirements.txt')
    if not req_path.exists(): say("requirements.txt missing", C.R); sys.exit(1)
    needs = unsatisfied(req_path.read_text().splitlines())
    if needs:
        say(f"Installing/upgrading: {', '.join(needs)}", C.Y)
        subprocess.check_call([str(venv_py()), "-m", "pip", "install",
                               "--upgrade", "--quiet",
                               "-r", "requirements.txt"])
    else:
        say("All dependencies satisfied ✔", C.G)

def open_browser(port):
    time.sleep(2); webbrowser.open(f"http://localhost:{port}")

# ── main ───────────────────────────────────────────────────────────────────
def main():
    say("🌱 HappyPlant Launcher", C.G)
    os.makedirs('templates', exist_ok=True); os.makedirs('uploads', exist_ok=True)
    if Path('index.html').exists() and not Path('templates/index.html').exists():
        shutil.move('index.html','templates/index.html')
    if not Path('.env').exists() and Path('.env.example').exists():
        shutil.copy('.env.example','.env'); say("👉  Edit .env with API keys", C.Y)

    ensure_venv()
    ensure_deps()

    port=5001
    Thread(target=open_browser, args=(port,), daemon=True).start()
    os.environ.update(FLASK_ENV="development", FLASK_DEBUG="True")
    say("Starting Flask on http://localhost:5001 …", C.G)
    import app
    app.app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)

if __name__=="__main__":
    main()
