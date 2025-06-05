import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run model analysis.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for processing."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="Maximum length of the tokenized input.",
    )
    parser.add_argument("--layers_to_skip", type=int, help="Number of layers to skip.")
    parser.add_argument(
        "--dataset_column",
        type=str,
        help="The specific column of the dataset to use.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Optional argument to specify the size of the dataset.",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="eval",
        help="Subset of the dataset to use (e.g., 'train', 'eval').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--device", type=str, help="Device to run the model on ('cpu', 'cuda')."
    )
    return parser.parse_args()
