import argparse
import shlex

import pytest

from specwire import cloud


def up_args(**overrides):
    args = dict(
        image="ubuntu:24.04", target_gpu="NVIDIA A100-SXM4-80GB", draft_gpu="NVIDIA L4",
        cuda=["13.0"], datacenter="EUR-IS-1", draft_datacenter=None,
        target_disk_gb=80, draft_disk_gb=40, hf_secret="hf_token",
    )
    return argparse.Namespace(**(args | overrides))


def test_pod_bodies_share_a_datacenter_and_only_target_exposes_grpc():
    target = cloud.pod_body("target", up_args(), 1_900_000_000, "ssh-ed25519 AAAA")
    draft = cloud.pod_body("draft", up_args(), 1_900_000_000, "ssh-ed25519 AAAA")
    assert target["dataCenterIds"] == draft["dataCenterIds"] == ["EUR-IS-1"]
    assert target["gpuTypeIds"] == ["NVIDIA A100-SXM4-80GB"]
    assert draft["gpuTypeIds"] == ["NVIDIA L4"]
    assert target["ports"] == ["22/tcp", f"{cloud.TARGET_PORT}/tcp"]
    assert draft["ports"] == ["22/tcp"]
    assert target["volumeInGb"] == 0 and target["interruptible"] is False


def test_draft_datacenter_overrides_for_cross_region():
    args = up_args(draft_datacenter="US-KS-2")
    assert cloud.pod_body("target", args, 1, "k")["dataCenterIds"] == ["EUR-IS-1"]
    assert cloud.pod_body("draft", args, 1, "k")["dataCenterIds"] == ["US-KS-2"]


def test_pod_env_holds_only_a_secret_reference():
    env = cloud.pod_body("target", up_args(hf_secret="my_hf"), 1, "k")["env"]
    assert env["HF_TOKEN"] == "{{ RUNPOD_SECRET_my_hf }}"
    assert set(env) == {"HF_TOKEN", "SPECWIRE_SSH_PUBKEY"}


def test_start_command_arms_the_ttl_before_anything_else():
    script = cloud.start_command(1_900_000_123)
    assert '-lt 1900000123 ]' in script
    assert script.index("podTerminate") < script.index("apt-get")
    assert "@TTL@" not in script


@pytest.fixture
def pods(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cloud.STATE_DIR.mkdir()
    state = {
        "created": "x", "ttl_epoch": 0, "token": "tok en",
        "models": {"target": "T/8B", "draft": "D/1B", "dtype": "bfloat16", "fast": True},
        "pods": {
            "target": {"id": "t1", "ip": "1.2.3.4", "ssh_port": 22, "grpc": "1.2.3.4:40000",
                       "cost_per_hr": 1.79},
            "draft": {"id": "d1", "ip": "1.2.3.5", "ssh_port": 22, "cost_per_hr": 0.59},
        },
    }
    cloud.save(state)
    scripts = []
    monkeypatch.setattr(cloud, "run_script", lambda pod, script, **kw: scripts.append((pod, script)))
    monkeypatch.setattr(cloud, "fetch", lambda pod, remote, local: None)
    return scripts


def bench_argv(scripts):
    pod, script = scripts[-1]
    line = [ln for ln in script.splitlines() if "specwire.bench" in ln][0]
    return pod["id"], shlex.split(line), script


def test_bench_on_draft_pod_runs_split_against_the_target(pods):
    cloud.main(["bench", "draft", "--", "--k", "1", "3"])
    pod, argv, script = bench_argv(pods)
    assert pod == "d1"
    assert argv[argv.index("--modes") + 1] == "split"
    assert argv[argv.index("--target") + 1] == "1.2.3.4:40000"
    assert argv[argv.index("--price-target") + 1] == "1.79"
    assert argv[argv.index("--price-draft") + 1] == "0.59"
    assert "--fast" in argv and argv[-3:] == ["--k", "1", "3"]
    # The token goes in through the script's environment, not the bench's argv.
    assert "tok en" not in argv and "export SPECWIRE_TOKEN='tok en'" in script


def test_bench_keeps_user_flags_and_out_name(pods):
    cloud.main(["bench", "target", "--", "--modes", "target", "--out", "x/mine.json"])
    pod, argv, _ = bench_argv(pods)
    assert pod == "t1"
    assert argv.count("--modes") == 1 and argv[argv.index("--modes") + 1] == "target"
    assert argv[argv.index("--out") + 1] == "results/mine.json"
    assert "--target" not in argv
