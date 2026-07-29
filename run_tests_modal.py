"""Run the CPU smoke tests inside the Modal image (host has no torch).

    modal run run_tests_modal.py
"""

import modal

app = modal.App("pragma-tests")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .add_local_python_source("pragma")
    .add_local_file("preprocessed.sample.json", "/root/preprocessed.sample.json")
    .add_local_file("profiles.sample.json", "/root/profiles.sample.json")
    .add_local_file("tests/test_smoke.py", "/root/tests/test_smoke.py")
)


@app.function(image=image, timeout=1200)
def run():
    import subprocess, sys
    r = subprocess.run([sys.executable, "tests/test_smoke.py"], cwd="/root", capture_output=True, text=True)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    return {"returncode": r.returncode, "output": out}


@app.local_entrypoint()
def main():
    res = run.remote()
    print(res["output"])
    print("EXIT CODE:", res["returncode"])
