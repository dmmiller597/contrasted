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
    """Locate the TMalign binary.

    Args:
        path: Explicit path to the binary, or None to search PATH.

    Returns:
        Path to the TMalign binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
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


def _parse_tmalign_output(stdout: str) -> TMAlignResult:
    """Parse TMalign stdout into a TMAlignResult.

    Raises:
        ValueError: If required fields cannot be parsed from the output.
    """
    m_aligned = _RE_ALIGNED.search(stdout)
    if not m_aligned:
        msg = "Could not parse aligned length/RMSD from TMalign output"
        raise ValueError(f"{msg}:\n{stdout}")

    aligned_length = int(m_aligned.group(1))
    rmsd = float(m_aligned.group(2))

    m_tm = _RE_TM_SCORE_CHAIN1.search(stdout)
    if not m_tm:
        msg = "Could not parse TM-score (Chain_1) from TMalign output"
        raise ValueError(f"{msg}:\n{stdout}")
    tm_score = float(m_tm.group(1))

    m_len = _RE_LENGTH_CHAIN1.search(stdout)
    if not m_len:
        msg = "Could not parse Chain_1 length from TMalign output"
        raise ValueError(f"{msg}:\n{stdout}")
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
    """Run TMalign on two structures and parse the result.

    Args:
        query_structure: Path to the query PDB/mmCIF file.
        target_structure: Path to the target PDB/mmCIF file.
        binary: Path or name of the TMalign binary.

    Returns:
        Parsed TMAlignResult.

    Raises:
        RuntimeError: If TMalign exits with a non-zero return code.
        ValueError: If the output cannot be parsed.
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
    """Find a structure file for a domain ID.

    Tries common extensions: .pdb, .cif, .pdb.gz, .cif.gz, .ent, .ent.gz

    Args:
        domain_id: The domain identifier.
        structure_dir: Directory containing structure files.

    Returns:
        Path to the structure file, or None if not found.
    """
    for ext in _STRUCTURE_EXTENSIONS:
        candidate = structure_dir / f"{domain_id}{ext}"
        if candidate.exists():
            return candidate
    return None
