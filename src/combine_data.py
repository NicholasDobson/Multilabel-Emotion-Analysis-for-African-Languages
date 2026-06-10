import argparse
import logging
from pathlib import Path
from datasets import concatenate_datasets, load_from_disk, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # merging two different types of augmentation
    parser = argparse.ArgumentParser(description="Combine BT and Gemini augmented datasets")
    parser.add_argument("--lang", required=True, help="Language code")
    args = parser.parse_args()

    data_dir = Path("data")
    bt_path = data_dir / f"{args.lang}_augmented"
    gemini_path = data_dir / f"{args.lang}_gemini_augmented"
    out_path = data_dir / f"{args.lang}_combined_augmented"

    if not bt_path.exists():
        logger.warning(f"BT augmented data not found at {bt_path}. Skipping.")
        return
    if not gemini_path.exists():
        logger.warning(f"Gemini augmented data not found at {gemini_path}. Skipping.")
        return

    logger.info(f"Loading BT data from {bt_path}")
    bt_ds = load_from_disk(str(bt_path))

    logger.info(f"Loading Gemini data from {gemini_path}")
    gemini_ds = load_from_disk(str(gemini_path))

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_language_data

    splits = load_language_data(args.lang)
    baseline_len = len(splits["train"])

    bt_train = bt_ds["train"]
    gemini_train = gemini_ds["train"]

    logger.info(f"Baseline train length: {baseline_len}")
    logger.info(f"BT train length: {len(bt_train)} (+{len(bt_train) - baseline_len} new)")
    logger.info(f"Gemini train length: {len(gemini_train)} (+{len(gemini_train) - baseline_len} new)")

    gemini_new_only = gemini_train.select(range(baseline_len, len(gemini_train)))

    combined_train = concatenate_datasets([bt_train, gemini_new_only])

    logger.info(f"Combined train length: {len(combined_train)}")

    combined_dict = DatasetDict({
        "train": combined_train,
        "validation": bt_ds["validation"],
        "test": bt_ds["test"],
    })

    combined_dict.save_to_disk(str(out_path))
    logger.info(f"Saved combined dataset to {out_path}")

if __name__ == "__main__":
    main()
