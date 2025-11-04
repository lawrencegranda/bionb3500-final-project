"""Build a WordNet sense map from gloss substrings defined in ``words.yaml``.

The generated JSON mirrors the structure required by ``utils.senses``:
``word -> sense label -> list of WordNet sensekeys``. Each sensekey targets the
specific lemma used in the configuration to keep downstream filtering
unambiguous.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from utils.senses import (  # pylint: disable=C0413,E0401
    GlossMap,
    build_sense_map,
    sense_map_to_json,
)


def load_gloss_map(words_path: Path) -> GlossMap:
    """Load the gloss substring configuration from ``words.yaml``."""

    with words_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    targets = payload.get("targets", [])
    gloss_map: GlossMap = {}

    for target in targets:
        word = target.get("word")
        senses = target.get("senses", {})
        if not word:
            continue
        gloss_map[word] = {label: set(glosses) for label, glosses in senses.items()}

    return gloss_map


def _write_json(
    data: Mapping[str, Mapping[str, Sequence[str]]], output_path: Path
) -> None:
    """Persist the sense map to ``output_path`` with pretty formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Build the command-line argument parser and parse ``argv``."""

    parser = argparse.ArgumentParser(
        description="Build a WordNet sense map from gloss substrings."
    )
    parser.add_argument(
        "--words",
        type=Path,
        default=REPO_ROOT / "config" / "words.yaml",
        help="Path to the YAML configuration containing gloss substrings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "senses" / "sense_map.json",
        help="Path to the JSON file that will store the generated sense map.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Program entrypoint."""

    args = _parse_args(argv)
    gloss_map = load_gloss_map(args.words)
    if not gloss_map:
        raise ValueError(f"No targets found in {args.words}")

    sense_map = build_sense_map(gloss_map)
    writable = sense_map_to_json(sense_map)
    _write_json(writable, args.output)

    print(f"Sense map written to {args.output}")


if __name__ == "__main__":
    main()
