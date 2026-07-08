"""TM-align structural validation wrapper."""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TMAlignResult:
    """Result from a TM-align structural alignment."""

    tm_score: float
    rmsd: float
    aligned_length: int
    query_length: int
    coverage: float


def find_tmalign_binary(path: str | None = None) -> Path:
    """Locate the TMalign binary at an explicit ``path`` or on ``PATH``.

    Raises ``FileNotFoundError`` if it cannot be found.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"TMalign binary not found at specified path: {path}")

    resolved = shutil.which("TMalign")
    if resolved is not None:
        return Path(resolved)

    raise FileNotFoundError(
        "TMalign binary not found on PATH. Install TMalign or specify "
        "tmalign_binary=<path> in the config."
    )


_RE_ALIGNED = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)")
_RE_TM_SCORE_CHAIN1 = re.compile(
    r"TM-score=\s*([\d.]+)\s+\(if normalized by length of Chain_1"
)
_RE_LENGTH_CHAIN1 = re.compile(r"Length of Chain_1:\s*(\d+)\s+residues")


def _required_match(
    pattern: re.Pattern[str], stdout: str, description: str
) -> re.Match[str]:
    match = pattern.search(stdout)
    if match is None:
        raise ValueError(
            f"Could not parse {description} from TMalign output:\n{stdout}"
        )
    return match


def _parse_tmalign_output(stdout: str) -> TMAlignResult:
    """Parse TMalign stdout into a ``TMAlignResult``.

    Raises ``ValueError`` if a required field is missing from the output.
    """
    m_aligned = _required_match(_RE_ALIGNED, stdout, "aligned length/RMSD")
    aligned_length = int(m_aligned.group(1))
    rmsd = float(m_aligned.group(2))

    m_tm = _required_match(_RE_TM_SCORE_CHAIN1, stdout, "TM-score (Chain_1)")
    tm_score = float(m_tm.group(1))

    m_len = _required_match(_RE_LENGTH_CHAIN1, stdout, "Chain_1 length")
    query_length = int(m_len.group(1))

    coverage = aligned_length / query_length if query_length > 0 else 0.0

    return TMAlignResult(
        tm_score=tm_score,
        rmsd=rmsd,
        aligned_length=aligned_length,
        query_length=query_length,
        coverage=coverage,
    )


def run_tmalign(
    query_structure: Path,
    target_structure: Path,
    binary: str = "TMalign",
) -> TMAlignResult:
    """Align ``query_structure`` against ``target_structure`` and parse the result.

    Both paths point to PDB/mmCIF files. Raises ``RuntimeError`` if TMalign
    exits non-zero, or ``ValueError`` if its output cannot be parsed.
    """
    result = subprocess.run(
        [binary, str(query_structure), str(target_structure)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TMalign failed (exit {result.returncode}):\n{result.stderr}"
        )
    return _parse_tmalign_output(result.stdout)


_STRUCTURE_EXTENSIONS = [".pdb", ".cif", ".pdb.gz", ".cif.gz", ".ent", ".ent.gz"]


def resolve_structure_path(domain_id: str, structure_dir: Path) -> Path | None:
    """Find a structure file for ``domain_id`` in ``structure_dir``, or ``None``.

    Tries extensions .pdb, .cif, .pdb.gz, .cif.gz, .ent, .ent.gz in order.
    """
    for ext in _STRUCTURE_EXTENSIONS:
        candidate = structure_dir / f"{domain_id}{ext}"
        if candidate.exists():
            return candidate
    return None
