"""Generates the Colab notebooks. Edit the cells here, then run:
    python notebooks/build_notebook.py            # notebooks/gpuforge_colab.ipynb (clone or upload the repo)
    python notebooks/build_notebook.py --embed    # dist/gpuforge_colab_standalone.ipynb (repo embedded inside)
"""
import argparse
import base64
import subprocess
from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# GPUForge on an NVIDIA GPU

This notebook runs the whole GPUForge pipeline on a real NVIDIA GPU and produces **measured** numbers for your README and resume:

1. builds the custom CUDA kernels for this GPU and checks them
2. runs the full test suite, now including the GPU-only tests
3. numerically validates the custom kernels, cuBLAS and cuDNN against float64
4. trains the model with mixed precision
5. exports to ONNX and builds a TensorRT FP16 engine
6. benchmarks the kernels against cuBLAS and every inference backend
7. starts the inference server and load-tests its dynamic batching
8. writes everything into `RESULTS.md` and downloads it

**Before you start:** *Runtime → Change runtime type → T4 GPU → Save.* Then *Runtime → Run all*.
The whole notebook takes roughly 10–15 minutes on a free T4.

*Works on Kaggle too: turn on a GPU accelerator and set `REPO_URL` in step 2, since Kaggle has no upload dialog.*"""),

    md("## 1. Check the GPU"),
    code("""import os, subprocess, sys
import torch

CPU_TEST = os.environ.get("GPUFORGE_NOTEBOOK_CPU_TEST") == "1"  # only used to test this notebook without a GPU


def run(cmd: str, check: bool = True) -> int:
    \"\"\"Run a shell command and stream its output into the notebook.\"\"\"
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {cmd}")
    return rc


PY = sys.executable
run("nvidia-smi", check=False)
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"\\nGPU: {torch.cuda.get_device_name(0)} | sm_{major}{minor} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    print("Native bf16:", "yes, training uses bf16" if major >= 8 else "no (T4/V100), training uses fp16")
elif not CPU_TEST:
    raise RuntimeError("No NVIDIA GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then Run all again.")"""),

    md("""## 2. Get the code

* **Standalone notebook?** Nothing to do: the project is embedded in this notebook.
* **Pushed it to GitHub?** Paste the repo URL into `REPO_URL` (e.g. `https://github.com/<you>/gpuforge.git`).
* **Otherwise** leave it empty and you'll be asked to upload `gpuforge.zip`."""),
    code("""#@title Repository
REPO_URL = ""  #@param {type:"string"}

import base64, io, pathlib, zipfile

WORK = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
REPO = WORK / "gpuforge"

if os.environ.get("GPUFORGE_LOCAL_REPO"):  # testing hook
    REPO = pathlib.Path(os.environ["GPUFORGE_LOCAL_REPO"])
elif REPO_URL:
    if (REPO / ".git").exists():
        run(f"git -C {REPO} pull --ff-only")
    else:
        run(f"git clone --depth 1 {REPO_URL} {REPO}")
elif globals().get("EMBEDDED_REPO_ZIP"):
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_REPO_ZIP))) as z:
        z.extractall(WORK)
    print("Unpacked the embedded project")
elif not (REPO / "pyproject.toml").exists():
    from google.colab import files

    print("Choose gpuforge.zip from your computer...")
    uploaded = files.upload()
    with zipfile.ZipFile(next(iter(uploaded))) as z:
        z.extractall(WORK)

if not (REPO / "pyproject.toml").exists():
    raise RuntimeError(f"Could not find the project at {REPO}")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.makedirs("reports", exist_ok=True)
print("Working in", REPO)"""),

    md("""## 3. Install and compile the CUDA kernels

The kernels are compiled only for this GPU's architecture (`TORCH_CUDA_ARCH_LIST`), which keeps the build to a minute or two.
ONNX Runtime is swapped for its GPU build, and TensorRT is installed from pip to match the CUDA version PyTorch was built with."""),
    code("""os.environ["GPUFORGE_SKIP_EXT"] = "1"  # install the Python package first, compile the kernels next
run(f"{PY} -m pip install -q --no-build-isolation -e '.[dev]'")
os.environ.pop("GPUFORGE_SKIP_EXT")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    cuda_major = torch.version.cuda.split(".")[0]
    run(f"{PY} -m pip uninstall -q -y onnxruntime", check=False)
    run(f"{PY} -m pip install -q onnxruntime-gpu tensorrt-cu{cuda_major}")
    rc = run(f"{PY} setup.py build_ext --inplace > reports/build.log 2>&1", check=False)
    if rc != 0:
        run("tail -40 reports/build.log", check=False)
        raise RuntimeError("CUDA extension build failed - see the log above (full log: reports/build.log)")
    print("CUDA extension built for", os.environ["TORCH_CUDA_ARCH_LIST"])"""),

    md("""## 4. Check the custom kernels

Each GEMM variant is compared with a float64 reference. Errors should be tiny (around `1e-4` at this size)
and the same for every variant, because they do the same arithmetic in a different order."""),
    code("""import importlib
import gpuforge.kernels as K

importlib.reload(K)
print("CUDA extension loaded:", K.extension_available())
if torch.cuda.is_available():
    if not K.extension_available():
        raise RuntimeError("The extension did not load - rerun step 3 and check reports/build.log")
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    bias = torch.randn(1024, device="cuda")
    ref = torch.relu(a.double() @ b.double() + bias.double())
    for v in K.VARIANTS:
        out = K.gemm_bias_act(a, b, bias, "relu", v)
        print(f"{v:8s} max |error| vs float64 = {(out.double() - ref).abs().max().item():.2e}")
    x = torch.randn(64, 1000, device="cuda") * 20
    err = (K.softmax_rows(x).double() - torch.softmax(x.double(), -1)).abs().max().item()
    print(f"softmax  max |error| vs float64 = {err:.2e}")"""),

    md("## 5. Run the test suite (GPU tests included)"),
    code("""run(f"{PY} -m pytest -q -rs")"""),

    md("""## 6. Numerical validation

Custom kernels, cuBLAS and cuDNN are checked element by element against a theoretical error bound.
`bound_ratio` is how much of the allowed error was used: anything at or below 1.0 passes."""),
    code("""run(f"{PY} -m gpuforge.validation.suite --report reports/validation.json --db reports/analytics.db")"""),

    md("## 7. Train with mixed precision"),
    code("""workers = min(8, os.cpu_count() or 2)
extra = "model=resnet_tiny n_synthetic=2000 epochs=1 batch_size=64" if CPU_TEST else ""
run(f"{PY} -m gpuforge.train.train --config configs/train.yaml workers={workers} {extra}")"""),

    md("## 8. Export to ONNX and build the TensorRT engine"),
    code("""run(f"{PY} -m gpuforge.export.export --ckpt runs/latest/checkpoint.pt --out artifacts --trt --precision fp16")
run("ls -lh artifacts", check=False)"""),

    md("""## 9. Benchmarks

* **Kernels:** TFLOPS of the three custom GEMMs vs cuBLAS FP32 (and cuBLAS TF32 on Ampere+).
* **Inference:** latency and throughput for PyTorch FP32/FP16, ONNX Runtime and TensorRT at several batch sizes."""),
    code("""sizes = "256 512" if CPU_TEST else "512 1024 2048 4096"
run(f"{PY} -m gpuforge.bench.kernels --sizes {sizes} --db reports/analytics.db")

engine = "artifacts/model_fp16.plan"
engine_arg = f"--engine {engine}" if os.path.exists(engine) else ""
batches = "1 16" if CPU_TEST else "1 8 32 128"
run(f"{PY} -m gpuforge.bench.inference --ckpt runs/latest/checkpoint.pt --onnx artifacts/model.onnx "
    f"{engine_arg} --batches {batches} --db reports/analytics.db")"""),

    md("""## 10. Serve it and load-test dynamic batching

256 single-image requests arrive at once. Watch `gpuforge_mean_batch_size` in the metrics: the server merges them
into larger GPU batches instead of running 256 separate inferences."""),
    code("""import concurrent.futures as cf
import json, time, urllib.request
import numpy as np

backend, artifact = ("tensorrt", engine) if os.path.exists(engine) else ("onnx", "artifacts/model.onnx")
env = {**os.environ, "GPUFORGE_BACKEND": backend, "GPUFORGE_ARTIFACT": artifact,
       "GPUFORGE_MAX_BATCH": "64", "GPUFORGE_MAX_WAIT_MS": "5"}
log = open("reports/server.log", "w")
server = subprocess.Popen([PY, "-m", "uvicorn", "gpuforge.serve.server:app", "--port", "8000"],
                          env=env, stdout=log, stderr=subprocess.STDOUT)
BASE = "http://127.0.0.1:8000"


def get(path):
    return urllib.request.urlopen(BASE + path, timeout=5).read().decode()


def predict(x):
    req = urllib.request.Request(BASE + "/v1/predict", data=json.dumps({"inputs": x, "top_k": 3}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


try:
    for _ in range(240):
        try:
            print("server ready:", get("/readyz"))
            break
        except Exception:
            if server.poll() is not None:
                break
            time.sleep(0.5)
    else:
        raise RuntimeError("server did not become ready")
    if server.poll() is not None:
        raise RuntimeError("server exited:\\n" + open("reports/server.log").read()[-3000:])

    images = [np.random.randn(1, 3, 32, 32).astype(np.float32).tolist() for _ in range(256)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(64) as pool:
        responses = list(pool.map(predict, images))
    dt = time.perf_counter() - t0
    print(f"\\n{len(responses)} concurrent requests in {dt:.2f}s ({len(responses) / dt:.0f} requests/s, {backend} backend)")
    print("example response:", responses[0])
    print("\\n" + get("/metrics"))
finally:
    server.terminate()
    server.wait()
    log.close()"""),

    md("""## 11. Profile a few training steps (optional)

Prints the top GPU kernels by time - you'll see cuDNN convolution and cuBLAS kernel names, which is great interview material.
The trace (`reports/trace.json`) opens in [Perfetto](https://ui.perfetto.dev)."""),
    code("""batch = 32 if CPU_TEST else 256
model = "resnet_tiny" if CPU_TEST else "resnet_small"
amp = "bf16" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) or CPU_TEST else "fp16"
run(f"{PY} -m gpuforge.profiling.profile --model {model} --batch {batch} --steps 5 --amp {amp} --out reports/trace.json")"""),

    md("""## 12. Results tables

Generated only from what was measured above. Copy them into the README's *Results* section and use the numbers in your resume bullets."""),
    code("""from IPython.display import Markdown, display

run(f"{PY} -m gpuforge.analytics.report results --out reports/RESULTS.md > /dev/null")
display(Markdown(open("reports/RESULTS.md").read()))"""),

    md("""## 13. Performance-regression gate (optional demo)

Re-runs the kernel benchmark a few times to build baseline history, then runs the same check Jenkins uses to block slow merges.
On an unchanged build it should report no regressions. A free Colab GPU is shared and its clocks vary, so the demo
uses a 10% threshold; a dedicated CI GPU can use the stricter 5% default."""),
    code("""gate_sizes = "256 512" if CPU_TEST else "1024 2048 4096"  # small GEMMs are too noisy to gate on
for _ in range(4):
    run(f"{PY} -m gpuforge.bench.kernels --sizes {gate_sizes} --db reports/analytics.db > /dev/null")
run(f"{PY} -m gpuforge.analytics.report --db reports/analytics.db regress --threshold 10 --fail-on-regression",
    check=False)"""),

    md("## 14. Download the results"),
    code("""import shutil

shutil.copy("artifacts/export_report.json", "reports/export_report.json")
archive = shutil.make_archive(str(WORK / "gpuforge_results"), "zip", "reports")
print("Saved", archive)
try:
    from google.colab import files

    files.download(archive)
except ImportError:
    print("Not running in Colab - download the file from the file browser.")"""),
]

ap = argparse.ArgumentParser()
ap.add_argument("--embed", action="store_true", help="embed the committed repo (git HEAD) in the notebook")
args = ap.parse_args()
root = Path(__file__).resolve().parent.parent
out = Path(__file__).with_name("gpuforge_colab.ipynb")

if args.embed:
    # git archive = exactly the committed files, under a top-level gpuforge/ folder
    blob = subprocess.run(["git", "-C", str(root), "archive", "--format=zip", "--prefix=gpuforge/", "HEAD"],
                          check=True, capture_output=True).stdout
    b64 = base64.b64encode(blob).decode()
    embed = code(f"#@title Embedded project files (about {len(blob) // 1024} KB - no need to open this cell)\n"
                 f'EMBEDDED_REPO_ZIP = "{b64}"')
    embed.metadata["cellView"] = "form"
    idx = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## 2."))
    cells.insert(idx, embed)
    out = root / "dist" / "gpuforge_colab_standalone.ipynb"
    out.parent.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": [], "name": "gpuforge_colab.ipynb"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, out)
print("wrote", out)
