#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
from datetime import datetime


def _safe_run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_run_metadata(out_dir, args=None):
    os.makedirs(out_dir, exist_ok=True)

    if args is not None:
        args_path = os.path.join(out_dir, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    git_commit = _safe_run(["git", "rev-parse", "HEAD"])
    with open(os.path.join(out_dir, "git_commit.txt"), "w", encoding="utf-8") as f:
        f.write(git_commit + "\n")

    env_txt = _safe_run(["python", "-m", "pip", "freeze"])
    with open(os.path.join(out_dir, "env.txt"), "w", encoding="utf-8") as f:
        f.write(env_txt + "\n")

    with open(os.path.join(out_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
