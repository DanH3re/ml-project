import argparse
from src.pipeline.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="POS Tagging Pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Run only these steps (e.g. step_01_data step_02_embedding)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="STEP",
        help="Skip these steps",
    )
    args = parser.parse_args()

    run(steps_filter=args.steps, skip_filter=args.skip)


if __name__ == "__main__":
    main()
