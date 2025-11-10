"""Build a WordNet sense map from gloss substrings defined in ``words.yaml``.

The generated JSON mirrors the structure required by ``utils.senses``:
``word -> sense label -> list of WordNet sensekeys``. Each sensekey targets the
specific lemma used in the configuration to keep downstream filtering
unambiguous.
"""

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413,E0401
from src.types.senses import GlossMapType  # pylint: disable=C0413,E0401
from src.builders.sense_map import SenseMap  # pylint: disable=C0413,E0401


def load_gloss_map(words_path: Path) -> GlossMapType:
    """Load the gloss substring configuration from ``words.yaml``."""

    with words_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    targets = payload.get("targets", [])
    gloss_map: GlossMapType = {}

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


def main() -> None:
    """Program entrypoint."""

    args = get_args(__doc__)

    words_config_path = args.config.paths.target_words_path
    sense_map_path = args.config.paths.sense_map_path

    gloss_map = load_gloss_map(words_config_path)
    if not gloss_map:
        raise ValueError(f"No targets found in {words_config_path}")

    sense_map = SenseMap.from_gloss(gloss_map)
    writable = sense_map.to_json()
    _write_json(writable, sense_map_path)

    print("Sense map written to %s", sense_map_path)


if __name__ == "__main__":
    main()
