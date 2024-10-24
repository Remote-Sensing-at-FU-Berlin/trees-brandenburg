from shutil import copytree
from pathlib import Path
from typing import List
from warnings import warn
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_subset(indir: Path, outdir: Path, classes: List[str]) -> None:
    """Subset training data according to class labels

    .. note:: The function expects a particular structure where subdirectories in
      ``indir`` correspond to your target classes.

    .. note:: Data is copied, not moved.

    :param indir: Input directory
    :type indir: Path
    :param outdir: Path where data should be copied to
    :type outdir: Path
    :param classes: List of classes to keep
    :type classes: List[str]
    """
    assert indir.is_dir(), "'indir` is not a directory"

    outdir.mkdir(exist_ok=True)

    for child in indir.iterdir():
        if child.is_dir() and (stem := child.stem) in classes:
            try:
                copytree(child, outdir / stem)
            except FileExistsError:
                warn(f"Directory {outdir}/{stem} or some part of it already exists. No data was copied for this class")


def generate_data_overview(src: Path, pattern: str = "*.tif", encode_labels: bool = True, resolve_paths: bool = True) -> pd.DataFrame:
    """Generate a DataFrame with paths, labels and optionally encoded target labels

    .. note:: The function expects a particular structure where subdirectories in
      ``indir`` correspond to your target classes.

    :param src: Directory with training data (see note)
    :type src: Path
    :param pattern: Pattern used to select files, defaults to "*.tif"
    :type pattern: str, optional
    :param encode_labels: Choose if a numeric label encoding on the target class be done, defaults to True
    :type encode_labels: bool, optional
    :param resolve_paths: Make sure returned file paths are absolute, defaults to True
    :type resolve_paths: bool, optional
    :return: Dataframe with paths, labels and optionally encoded target labels
    :rtype: pd.DataFrame
    """    
    target_classes: List[str] = []
    image_paths: List[Path] = []
    for file in src.rglob(pattern):
        target_classes.append(file.parent.stem)
        image_paths.append(file.resolve() if resolve_paths else file)
    
    out: pd.DataFrame = pd.DataFrame({"images": image_paths, "labels": target_classes})

    if encode_labels:
        out["encoded_labels"] = LabelEncoder().fit_transform(out["labels"])

    return out
