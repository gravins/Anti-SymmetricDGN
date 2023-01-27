from pathlib import Path
from typing import Any


def _dump_json(obj: Any, path: Path):
    import json

    json.dump(obj,
              path.open("w"),
              indent=2)


def _load_json(path: Path) -> Any:
    import json

    return json.load(path.open("r"))


def _dump_bin(obj: Any, path: Path):
    import pickle as pkl

    pkl.dump(obj, path.open("wb"))


def _load_bin(path: Path) -> Any:
    import pickle as pkl

    return pkl.load(path.open("rb"))


IO_HELPERS = {
    'json': (_dump_json, _load_json),
    'pkl': (_dump_bin, _load_bin),
    'pickle': (_dump_bin, _load_bin)
}
