import tempfile
from pathlib import Path

from contrasted.utils import load_labels, set_seed


def test_set_seed_runs():
    set_seed(42)


def test_load_labels():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# comment line\n")
        f.write("dom01 1.10.8.10\n")
        f.write("dom02 1.10.8.10\n")
        f.write("dom03 3.40.50.1620\n")
        label_path = Path(f.name)

    id_to_sf_idx, idx_to_sf = load_labels(str(label_path))

    assert id_to_sf_idx["dom01"] == 0
    assert id_to_sf_idx["dom02"] == 0
    assert id_to_sf_idx["dom03"] == 1

    assert idx_to_sf[0] == "1.10.8.10"
    assert idx_to_sf[1] == "3.40.50.1620"

    label_path.unlink()


def test_load_labels_empty_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("dom01 sf1\n")
        f.write("\n")
        f.write("dom02 sf2\n")
        label_path = Path(f.name)

    id_to_sf_idx, idx_to_sf = load_labels(str(label_path))

    assert len(id_to_sf_idx) == 2
    assert len(idx_to_sf) == 2

    label_path.unlink()
