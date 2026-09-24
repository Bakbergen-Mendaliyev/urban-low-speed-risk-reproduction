"""One public entry point. Default verification performs no training or inference."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "verify",
            "preprocess",
            "features",
            "train",
            "predict",
            "evaluate",
            "external",
            "figures",
        ],
        default="verify",
    )
    parser.add_argument(
        "--dataset", choices=["Chicago", "City-M", "Chicago-external"], default="Chicago"
    )
    parser.add_argument("--fold", type=int, choices=range(1, 5), default=1)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="Explicitly authorize new local computation; never needed for frozen verification or display export.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.dataset == "Chicago-external" and args.stage not in ["preprocess", "external"]:
        parser.error(
            "Chicago-external is only valid with preprocess or external; no external fitting is permitted."
        )
    if args.stage not in ["verify", "figures"] and not args.allow_expensive:
        parser.error(
            "This stage creates new local computation. Supply --allow-expensive after reading README; it never overwrites verified results."
        )
    if args.stage == "verify":
        from src.evaluation.frozen import verify

        verify(config)
    elif args.stage == "figures":
        from src.visualization.displays import render

        render(config)
    elif args.stage == "preprocess":
        from src.data.preprocess import run

        run(config, args.dataset)
    elif args.stage == "features":
        from src.features.prepare import prepare_fold

        prepare_fold(config, args.dataset, args.fold)
    elif args.stage == "train":
        from src.models.train import train_fold

        train_fold(config, args.dataset, args.fold)
    elif args.stage == "predict":
        from src.models.predict import predict_fold

        predict_fold(config, args.dataset, args.fold)
    elif args.stage == "external":
        from src.evaluation.external import replay

        replay(config)
    else:
        from src.evaluation.historical import evaluate

        evaluate(config, args.dataset)


if __name__ == "__main__":
    main()
