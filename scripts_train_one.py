r"""Train ONE LoRA adapter from a corpus file, in a clean process (no curate/field
/traces resident). Memory-instrumented. This is the per-adapter unit the day-chain
calls 4x (one fresh process each) to avoid accumulation. One-off (gitignored).

 python scripts_train_one.py <arm_name> [seed] e.g. B0_random 0
when seed is given, reads work/seed{N}/corpus_{arm}.json and writes the adapter
under adapters/seed{N}/ (mirrors run_ablation's per-seed isolation).
"""
import sys, json, time, threading
from config import Config
from train import train_lora

def avail_gb:
 for l in open("/proc/meminfo"):
 if l.startswith("MemAvailable:"): return int(l.split[1])/2**20
 return -1

def main:
 arm = sys.argv[1]
 seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
 low = [avail_gb]; stop = threading.Event
 def watch:
 while not stop.is_set:
 a = avail_gb
 if a < low[0]:
 low[0] = a; print(f" [mem] new low avail={a:.0f}G", flush=True)
 time.sleep(0.5)
 threading.Thread(target=watch, daemon=True).start

 cfg = Config
 if seed is not None:
 cfg.seed = seed
 cfg.work_dir = cfg.work_dir / f"seed{seed}"
 cfg.adapter_dir = cfg.adapter_dir / f"seed{seed}"
 corpus = json.load(open(cfg.work_dir / f"corpus_{arm}.json"))
 print(f"[train1] {arm} seed={seed}: {len(corpus)} examples, epochs={cfg.epochs}, "
 f"max_seq_len={cfg.max_seq_len}, avail={avail_gb:.0f}G", flush=True)
 t0 = time.time
 out = train_lora(corpus, cfg, run_name=arm)
 stop.set
 import os
 ok = os.path.isfile(os.path.join(out, "adapter_model.safetensors"))
 print(f"[train1] {arm} done in {time.time-t0:.0f}s, min_avail={low[0]:.0f}G, "
 f"adapter_saved={ok}", flush=True)
 print("TRAIN1_OK" if ok else "TRAIN1_FAIL", flush=True)

if __name__ == "__main__":
 sys.exit(main)
