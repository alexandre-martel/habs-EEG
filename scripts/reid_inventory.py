"""List available sessions without opening any BDF -> results/inventory.csv.

python scripts/reid_inventory.py --data ./sample          # local dataset root
python scripts/reid_inventory.py --hf                     # public HF file listing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid import inventory as inv_mod
from reid.config import RESULTS, default_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset root containing CURATED/ (default: auto)",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="use the Hugging Face file listing instead of local files",
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=None,
        help="participants_train.csv (default: <data>/participants_train.csv)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS / "inventory.csv")
    args = parser.parse_args()

    data_dir = args.data or default_data_dir()
    listing = inv_mod.list_hf() if args.hf else inv_mod.list_local(data_dir)
    files = inv_mod.parse_listing(listing)
    if files.empty:
        sys.exit(f"No CURATED files found ({'HF' if args.hf else data_dir}).")

    table_dir = args.participants.parent if args.participants else data_dir
    participants = inv_mod.load_participants(table_dir)
    inv = inv_mod.build_inventory(files, local=not args.hf)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(args.out, index=False)
    source = "Hugging Face listing" if args.hf else str(data_dir)
    print(
        f"{source}: {inv.participant_id.nunique()} participants, {len(inv)} sessions"
        f" -> {args.out}"
    )
    if participants is None:
        print("participants_train.csv not found: cohort unknown ('?').")
    print("\nPeople per cohort and available days:")
    print(
        inv_mod.day_patterns(inv_mod.attach_participants(inv, participants)).to_string()
    )
    print("\nPeople usable per scenario (task present on both days):")
    print(inv_mod.pair_counts(inv).to_string())
    print("\nRecording durations (s):")
    print(
        inv[["rest_duration_s", "artefact_duration_s"]].describe().round(1).to_string()
    )


if __name__ == "__main__":
    main()
