"""Run specwire on RunPod: a target pod and a draft pod, driven over SSH.

    uv run python -m specwire.cloud up                       # create both pods, start target server
    uv run python -m specwire.cloud rtt                      # gRPC round trip, draft pod -> target
    uv run python -m specwire.cloud bench target -- --modes target colocated --k 1 3 5 8
    uv run python -m specwire.cloud bench draft -- --modes split --k 1 3 5 8
    uv run python -m specwire.cloud status
    uv run python -m specwire.cloud down                     # terminate both pods

`bench target` runs on the target pod (target-only and co-located need the big GPU);
`bench draft` runs split mode from the draft pod against the target server. Results are
copied to results/. Datacenters and GPUs are options of `up`; --draft-datacenter puts the
draft elsewhere for the cross-region run.

Needs RUNPOD_API_KEY (env var, or the Windows user variable) and a RunPod secret holding
an HF token (--hf-secret). State (pod ids, SSH addresses, the gRPC token, prices) lives in
.cloud/, which git ignores. Each pod also terminates itself at --ttl-min, so a crashed
laptop can't leave one billing. Pods are always terminated, never stopped: a stopped pod
still bills for its disk.
"""

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from specwire.rpc import TOKEN_ENV

REST_URL = "https://rest.runpod.io/v1"
STATE_DIR = Path(".cloud")
STATE_FILE = STATE_DIR / "state.json"
SSH_KEY = STATE_DIR / "id_ed25519"
REMOTE_DIR = "/root/specwire"
TARGET_PORT = 50052
ROLES = ("target", "draft")

# The pod's start command. It carries no secrets: anyone with the account key can read it.
# Order matters: the TTL watchdog first, so a pod whose setup fails still dies on time.
# RunPod injects a pod-scoped RUNPOD_API_KEY and RUNPOD_POD_ID; that key may only use
# GraphQL to terminate, so the watchdog does that, retrying until the pod is gone.
START_SCRIPT = r"""
(
  while [ "$(date +%s)" -lt @TTL@ ]; do sleep 30; done
  while :; do
    printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl -sS -K - \
      -H 'Content-Type: application/json' \
      -d '{"query":"mutation { podTerminate(input: {podId: \"'"$RUNPOD_POD_ID"'\"}) }"}' \
      https://api.runpod.io/graphql
    sleep 60
  done
) >/var/log/specwire-ttl.log 2>&1 &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  curl ca-certificates openssh-server >/var/log/specwire-apt.log 2>&1
mkdir -p /root/.ssh /run/sshd /root/.cache/huggingface
printf '%s\n' "$SPECWIRE_SSH_PUBKEY" >/root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
printf '%s' "$HF_TOKEN" >/root/.cache/huggingface/token
ssh-keygen -A >/dev/null
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
sleep infinity
"""

# Run over SSH once the pod answers: install uv, unpack the code, sync the venv.
SETUP_SCRIPT = r"""
set -euo pipefail
if grep -q '{{' /root/.cache/huggingface/token; then
  echo "the RunPod secret for HF_TOKEN wasn't filled in: check --hf-secret" >&2; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
cd @DIR@
~/.local/bin/uv sync --frozen --no-dev --quiet
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'cuda ok')"
"""


def api_key():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and sys.platform == "win32":
        # A user variable set after this terminal opened isn't in os.environ yet.
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, "RUNPOD_API_KEY")[0]).strip()
        except OSError:
            pass
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def rest(method, path, body=None):
    """Call RunPod's REST API; return parsed JSON, or None for an empty or 404 reply."""
    req = urllib.request.Request(
        REST_URL + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + api_key(),
            "Content-Type": "application/json",
            "User-Agent": "specwire",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"RunPod {method} {path}: HTTP {e.code} {e.read().decode()[:500]}") from None
    return json.loads(raw) if raw else None


def start_command(ttl_epoch: int) -> str:
    return START_SCRIPT.replace("@TTL@", str(ttl_epoch))


def pod_body(role, args, ttl_epoch, pubkey):
    """The create-pod request for one role."""
    target = role == "target"
    return {
        "name": f"specwire-{role}-{ttl_epoch}",
        "imageName": args.image,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "interruptible": False,
        "gpuTypeIds": [args.target_gpu if target else args.draft_gpu],
        "gpuCount": 1,
        "dataCenterIds": [args.datacenter if target else args.draft_datacenter or args.datacenter],
        "containerDiskInGb": args.target_disk_gb if target else args.draft_disk_gb,
        # Explicit: the API's default is a 20 GB pod volume, billed even while stopped.
        "volumeInGb": 0,
        "ports": ["22/tcp"] + ([f"{TARGET_PORT}/tcp"] if target else []),
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [start_command(ttl_epoch)],
        # A reference RunPod fills in on the pod; the token itself never passes through here.
        "env": {
            "HF_TOKEN": "{{ RUNPOD_SECRET_" + args.hf_secret + " }}",
            "SPECWIRE_SSH_PUBKEY": pubkey,
        },
    } | ({"allowedCudaVersions": args.cuda} if args.cuda else {})


def create_pod(body, wait_s):
    """Create a pod, retrying while RunPod has none in stock (nothing bills meanwhile)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return rest("POST", "/pods", body)
        except RuntimeError as e:
            no_stock = "no instances" in str(e).lower() or "not enough free" in str(e).lower()
            if not no_stock or time.monotonic() > deadline:
                raise
            log(f"{body['gpuTypeIds'][0]} out of stock in {body['dataCenterIds'][0]}; retrying in 60 s")
            time.sleep(60)


def terminate(pod_id):
    rest("DELETE", f"/pods/{pod_id}")


def wait_for_ports(pod_id, ports, timeout_s=900):
    """Poll until the pod runs and RunPod has mapped every port; return (ip, {port: public})."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pod = rest("GET", f"/pods/{pod_id}")
        if pod is None:
            raise RuntimeError(f"pod {pod_id} disappeared")
        mappings = pod.get("portMappings") or {}
        if pod.get("desiredStatus") == "RUNNING" and pod.get("publicIp") and all(
            str(p) in mappings for p in ports
        ):
            return pod["publicIp"], {int(p): int(mappings[str(p)]) for p in ports}
        time.sleep(10)
    raise TimeoutError(f"pod {pod_id} had no public ports after {timeout_s} s")


def ssh_cmd(pod, *remote):
    return [
        "ssh", "-i", str(SSH_KEY), "-p", str(pod["ssh_port"]),
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={STATE_DIR / 'known_hosts'}",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15",
        f"root@{pod['ip']}", *remote,
    ]


def run_script(pod, script, check=True, capture=False):
    """Run a bash script on the pod, fed on stdin so nothing in it shows up in argv."""
    # Bytes, not text: on Windows text mode would turn \n into \r\n and break bash.
    return subprocess.run(
        ssh_cmd(pod, "bash", "-s"), input=script.encode(), check=check, capture_output=capture
    )


def wait_for_ssh(pod, timeout_s=900):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if run_script(pod, "true", check=False, capture=True).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError(f"no SSH to {pod['id']} after {timeout_s} s")


def push_code(pod):
    """Copy HEAD (git archive, so only committed files) to the pod."""
    # autocrlf off: on Windows, archive would otherwise write CRLF files for Linux.
    archive = subprocess.Popen(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
    )
    remote = f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR} && tar -x -C {REMOTE_DIR}"
    subprocess.run(ssh_cmd(pod, remote), stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise RuntimeError("git archive failed")


def set_up(pod):
    wait_for_ssh(pod)
    push_code(pod)
    run_script(pod, SETUP_SCRIPT.replace("@DIR@", REMOTE_DIR))
    log(f"{pod['role']} pod ready")


def start_target_server(state):
    pod = state["pods"]["target"]
    m = state["models"]
    script = f"""
cd {REMOTE_DIR}
export {TOKEN_ENV}={shlex.quote(state['token'])}
pkill -f 'specwire.rpc target' || true
nohup .venv/bin/python -m specwire.rpc target --host 0.0.0.0 --port {TARGET_PORT} \
  --model {shlex.quote(m['target'])} --device cuda --dtype {m['dtype']} {'--fast' if m['fast'] else ''} \
  >/root/target-server.log 2>&1 &
for i in $(seq 1 180); do
  grep -q listening /root/target-server.log && exit 0
  kill -0 $! 2>/dev/null || {{ tail -n 30 /root/target-server.log; exit 1; }}
  sleep 10
done
echo 'target server not listening after 30 min'; tail -n 30 /root/target-server.log; exit 1
"""
    run_script(pod, script)
    log("target server listening")


def git_dirty():
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if not line.startswith("??")]


def up(args):
    if STATE_FILE.exists():
        sys.exit(f"{STATE_FILE} exists: run `down` first (or delete it if those pods are gone)")
    if dirty := git_dirty():
        log(f"warning: {len(dirty)} uncommitted change(s) won't reach the pods (they get HEAD)")
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / ".gitignore").write_text("*\n")
    if not SSH_KEY.exists():
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "specwire", "-f", str(SSH_KEY)],
            check=True,
        )
    pubkey = SSH_KEY.with_suffix(".pub").read_text().strip()
    ttl_epoch = int(time.time()) + args.ttl_min * 60
    state = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "ttl_epoch": ttl_epoch,
        "token": secrets.token_urlsafe(32),
        "models": {"target": args.target_model, "draft": args.draft_model,
                   "dtype": args.dtype, "fast": not args.no_fast},
        "pods": {},
    }
    try:
        for role in ROLES:
            body = pod_body(role, args, ttl_epoch, pubkey)
            pod = create_pod(body, args.capacity_wait_min * 60)
            state["pods"][role] = {
                "role": role,
                "id": pod["id"],
                "gpu": body["gpuTypeIds"][0],
                "datacenter": body["dataCenterIds"][0],
                "cost_per_hr": pod.get("costPerHr"),
            }
            save(state)  # after every create, so `down` can find the pod if we crash
            log(f"created {role} pod {pod['id']} ({body['gpuTypeIds'][0]}, "
                f"{body['dataCenterIds'][0]}, ${pod.get('costPerHr')}/hr)")
        for role, pod in state["pods"].items():
            ports = [22] + ([TARGET_PORT] if role == "target" else [])
            ip, mapped = wait_for_ports(pod["id"], ports)
            pod |= {"ip": ip, "ssh_port": mapped[22]}
            if role == "target":
                pod["grpc"] = f"{ip}:{mapped[TARGET_PORT]}"
            if not pod["cost_per_hr"]:
                pod["cost_per_hr"] = (rest("GET", f"/pods/{pod['id']}") or {}).get("costPerHr")
        save(state)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(set_up, state["pods"].values()))
        start_target_server(state)
    except BaseException:
        log("setup failed: terminating the pods")
        down(args)
        raise
    log(f"up. pods terminate themselves at {datetime.fromtimestamp(ttl_epoch):%H:%M} unless `down` runs first")


def bench(args):
    state = load()
    pod = state["pods"][args.pod]
    m = state["models"]
    extra = list(args.bench_args)
    name = f"cloud-{args.pod}-{datetime.now():%Y%m%d-%H%M%S}.json"
    if "--out" in extra:  # keep the name, but always write under results/ to fetch it
        i = extra.index("--out")
        name = Path(extra[i + 1]).name
        del extra[i : i + 2]
    flags = {a for a in extra if a.startswith("--")}
    cmd = [".venv/bin/python", "-m", "specwire.bench", "--out", f"results/{name}"]
    defaults = {
        "--draft-model": m["draft"], "--target-model": m["target"],
        "--dtype": m["dtype"], "--device": "cuda",
    }
    # The draft pod's GPU is too small for the target-only and co-located baselines.
    if args.pod == "draft":
        defaults |= {"--modes": "split", "--target": state["pods"]["target"]["grpc"]}
    else:
        defaults["--modes"] = "target colocated"
    target_price = state["pods"]["target"]["cost_per_hr"]
    draft_price = state["pods"]["draft"]["cost_per_hr"]
    if target_price:
        defaults["--price-target"] = str(target_price)
    if draft_price:
        defaults["--price-draft"] = str(draft_price)
    for flag, value in defaults.items():
        if flag not in flags:
            cmd += [flag, *value.split()]
    if m["fast"] and "--fast" not in flags:
        cmd.append("--fast")
    cmd += extra
    script = f"""
cd {REMOTE_DIR}
export {TOKEN_ENV}={shlex.quote(state['token'])} PYTHONUNBUFFERED=1
{shlex.join(cmd)}
"""
    log(f"on {args.pod} pod: {shlex.join(cmd[2:])}")
    run_script(pod, script)
    fetch(pod, f"{REMOTE_DIR}/results/{name}", Path("results") / name)


def fetch(pod, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.run(ssh_cmd(pod, "cat", remote), check=True, capture_output=True).stdout
    local.write_bytes(data)
    log(f"saved {local}")


def rtt(args):
    """Median and spread of empty gRPC round trips from the draft pod to the target server."""
    state = load()
    script = f"""
cd {REMOTE_DIR}
export {TOKEN_ENV}={shlex.quote(state['token'])}
.venv/bin/python - <<'EOF'
import statistics, time
from specwire.rpc import TOKEN_ENV, TargetClient
import os
client = TargetClient({state['pods']['target']['grpc']!r}, os.environ[TOKEN_ENV])
client.memory()  # connect
ms = []
for _ in range({args.n}):
    start = time.perf_counter()
    client.memory()
    ms.append(1000 * (time.perf_counter() - start))
ms.sort()
print(f"gRPC round trip over {args.n} calls: median {{statistics.median(ms):.2f}} ms, "
      f"p10 {{ms[len(ms) // 10]:.2f}}, p90 {{ms[9 * len(ms) // 10]:.2f}}, max {{ms[-1]:.2f}}")
EOF
"""
    run_script(state["pods"]["draft"], script)


def status(args):
    state = load()
    left = (state["ttl_epoch"] - time.time()) / 60
    log(f"created {state['created']}; self-terminate in {left:.0f} min")
    total = 0.0
    for role, pod in state["pods"].items():
        live = rest("GET", f"/pods/{pod['id']}")
        st = live.get("desiredStatus") if live else "gone"
        total += float(pod.get("cost_per_hr") or 0) if live else 0
        log(f"{role:<6} {pod['id']}  {st:<10} {pod['gpu']}  {pod['datacenter']}  ${pod.get('cost_per_hr')}/hr")
    log(f"billing ${total:.2f}/hr while running")


def down(args):
    if not STATE_FILE.exists():
        log("no pods recorded")
        return
    state = load()
    failed = False
    for role, pod in state["pods"].items():
        try:
            terminate(pod["id"])
            log(f"terminated {role} pod {pod['id']}")
        except Exception as e:  # keep going so one failure doesn't leave the other running
            failed = True
            log(f"could not terminate {role} pod {pod['id']}: {e}")
    if failed:
        sys.exit("some pods may still be running: check the RunPod console")
    STATE_FILE.unlink()


def save(state):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def load():
    if not STATE_FILE.exists():
        sys.exit("no pods: run `up` first")
    return json.loads(STATE_FILE.read_text())


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("up", help="create both pods, set them up, start the target server")
    p.add_argument("--datacenter", default="EUR-IS-1")
    p.add_argument("--draft-datacenter", help="default: same as --datacenter")
    p.add_argument("--target-gpu", default="NVIDIA A100-SXM4-80GB")
    p.add_argument("--draft-gpu", default="NVIDIA L4")
    p.add_argument("--target-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-fast", action="store_true", help="skip static cache + CUDA graphs")
    p.add_argument("--ttl-min", type=int, default=120, help="pods terminate themselves after this")
    p.add_argument("--capacity-wait-min", type=int, default=10,
                   help="keep retrying an out-of-stock GPU this long")
    p.add_argument("--image", default="ubuntu:24.04", help="uv brings its own Python and CUDA torch")
    # Off by default: 13.0 is the newest version the API accepts, so the filter can
    # exclude hosts on newer drivers. Setup checks torch can use the GPU instead.
    p.add_argument("--cuda", nargs="+", default=[],
                   help="host CUDA versions to accept (e.g. 13.0); default: any")
    p.add_argument("--hf-secret", default="hf_token", help="RunPod secret holding the HF token")
    p.add_argument("--target-disk-gb", type=int, default=80)
    p.add_argument("--draft-disk-gb", type=int, default=40)
    p.set_defaults(func=up)

    p = sub.add_parser("bench", help="run specwire.bench on a pod; args after -- go to it")
    p.add_argument("pod", choices=ROLES)
    p.add_argument("bench_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=bench)

    p = sub.add_parser("rtt", help="measure gRPC round trips from the draft pod to the target")
    p.add_argument("-n", type=int, default=200)
    p.set_defaults(func=rtt)

    sub.add_parser("status", help="show the pods and what they cost").set_defaults(func=status)
    sub.add_parser("down", help="terminate both pods").set_defaults(func=down)

    args = parser.parse_args(argv)
    if args.cmd == "bench" and args.bench_args[:1] == ["--"]:
        args.bench_args = args.bench_args[1:]
    args.func(args)


if __name__ == "__main__":
    main()
