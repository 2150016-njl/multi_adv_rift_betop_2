"""Package actual V6 run evidence; no inferred timestamp paths or deletion."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    if not run.is_dir():
        raise ValueError("run directory does not exist: " + str(run))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results, caches = run / ("v6_results_" + stamp + ".zip"), run / ("v6_cache_" + stamp + ".zip")
    files = [f for f in run.rglob("*") if f.is_file() and (f.suffix in {".json", ".jsonl", ".yaml", ".yml", ".log", ".txt", ".csv"} or (f.suffix == ".ckpt" and f.stat().st_size < 20 * 1024**2))]
    inventory = {}
    with zipfile.ZipFile(results, "x", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(files):
            name = file.relative_to(run).as_posix()
            archive.write(file, name)
            inventory[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        rift = Path(__file__).resolve().parents[1]
        sources = list((rift / "rift/cbv/planning/fine_tuner/rlft/joint_rift").glob("*.py"))
        sources += list((rift / "scripts").glob("*v6*"))
        sources += list((rift / "rift/cbv/recognition/rule_based").glob("*.py"))
        sources.append(rift / "scripts/run.py")
        for file in sources:
            name = "source/" + file.relative_to(rift).as_posix()
            archive.write(file, name)
            inventory[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps({"run_dir": str(run), "sha256": inventory}, indent=2))
    arrays = sorted(run.rglob("state_*.npz"))
    with zipfile.ZipFile(caches, "x", zipfile.ZIP_DEFLATED) as archive:
        for file in arrays:
            archive.write(file, file.relative_to(run).as_posix())
            metadata = file.with_suffix(".json")
            if metadata.is_file():
                archive.write(metadata, metadata.relative_to(run).as_posix())
    for file in (results, caches):
        print("DONE:", file, "MiB:", round(file.stat().st_size / 1024**2, 2), flush=True)
    print("Candidate snapshots:", len(arrays))


if __name__ == "__main__":
    main()
