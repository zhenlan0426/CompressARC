from pathlib import Path
import shutil


def save_top_level_py_files(dest_dir: str | Path) -> None:
    """Copy every *.py file in the repository root into a code_snapshot folder
    inside dest_dir.

    Parameters
    ----------
    dest_dir : str | Path
        Destination directory where the snapshot should be placed. Typical use:
        a timestamped sub-folder inside run_results/.
    """

    dest_path = Path(dest_dir)
    snapshot_dir = dest_path / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent  # repo root because this file sits at top level

    for py_file in project_root.iterdir():
        if py_file.is_file() and py_file.suffix == ".py":
            shutil.copy2(py_file, snapshot_dir / py_file.name)
