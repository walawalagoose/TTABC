"""Run the TTABC ImageNet-A label-shift figure protocol, recording exact streams.

python scripts/run_label_shift.py /path/to/dataset --methods stata moon --output logs_label_shift
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttabc.utils import load_data


class RecordedBatches:
    def __init__(self, sampler, path):
        self.sampler, self.path = sampler, path

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        batches = []
        for batch in self.sampler:
            batches.append(list(batch))
            yield batch
        self.path.write_text(json.dumps(batches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--methods", nargs="+", choices=["stata", "moon"], default=["stata", "moon"])
    parser.add_argument("--output", type=Path, default=Path("logs_label_shift"))
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    original_loader = load_data.create_dataloader
    records = []
    for setting, gamma in (("no_shift", None), ("0.1", 0.1), ("0.01", 0.01),
                           ("0.001", 0.001), ("separate", -1)):
        for method in args.methods:
            output = args.output / method / setting
            output.mkdir(parents=True, exist_ok=False)
            stream_path = output / "batches.json"

            def create_recorded_loader(config, set_id):
                dataset, loader, names = original_loader(config, set_id)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_sampler=RecordedBatches(loader.batch_sampler, stream_path),
                    num_workers=config.workers, pin_memory=True,
                )
                return dataset, loader, names

            load_data.create_dataloader = create_recorded_loader
            command = [str(ROOT / "main.py"), args.data, "--test_sets", "A",
                       "-a", "ViT-B/16", "--gpu", str(args.gpu), "--run_type", method,
                       "--prompt_type", "no_prompt", "-b", "64", "--seed", str(args.seed),
                       "-j", str(args.workers), "-p", "50", "--log_dir", str(output)]
            if gamma is not None:
                command += ["--label_shift", "--label_shift_mode", "online",
                            "--label_shift_scope", "-1", "--label_shift_num_slots", "8",
                            "--label_shift_gamma", str(gamma)]
            (output / "command.json").write_text(json.dumps(command, indent=2))
            sys.argv = command
            print(f"Running {method}: {setting}", flush=True)
            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
            torch.cuda.empty_cache()
            log = next(output.glob("vit-b-16/A/*/*.log"))
            content = log.read_text()
            top1 = float(re.findall(r"Top-1 Acc.:\s+([\d.]+)", content)[-1])
            batches = json.loads(stream_path.read_text())
            indices = np.asarray([i for batch in batches for i in batch], dtype=np.int64)
            records.append(dict(method=method, setting=setting, top1=top1,
                                samples=len(indices), batches=len(batches),
                                stream_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                                log=str(log), seed=args.seed))
            (args.output / "results.json").write_text(json.dumps(records, indent=2))
    load_data.create_dataloader = original_loader
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
