from pathlib import Path

def prepare_save_path(path):
    """
    Ensure parent directory exists for a file path.
    Accepts str or Path. Returns Path or None.
    """
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path