"""Check validity of formalizations."""
import argparse
import json
import os

import tqdm

from LeanEuclid.E3.checker import Checker
from LeanEuclid.E3.utils import ROOT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Book", "UniGeo"],
        required=True,
        help="Testing dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        choices=[
            "",
            "Parallel",
            "Triangle",
            "Quadrilateral",
            "Congruent",
            "Similarity",
        ],
        required=True,
        help="Testing category",
    )
    parser.add_argument(
        "--mode",
        choices=["bvars", "skipApprox", "onlyApprox", "full"],
        default="skipApprox",
        help="E3 checker mode",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["text-only", "multi-modal"],
        required=True,
        help="Reasoning Type",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for result directory",
    )
    args = parser.parse_args()

    cnt = 0
    tot = 0

    for c in args.category:
        print("Category: ", c)
        pred_dir = os.path.join(
            ROOT_DIR,
            "result",
            "statement",
            args.dataset,
            args.reasoning,
            str(args.num_examples) + "shot",
            args.prefix,
            c,
        )
        result_dir = os.path.join(
            ROOT_DIR,
            "result",
            "equivalence",
            args.dataset,
            args.reasoning,
            str(args.num_examples) + "shot",
            args.prefix,
            c,
        )

        if args.dataset == "UniGeo":
            testing_idx = range(1, 21)
        else:
            testing_idx = [i for i in range(1, 49) if i not in [2, 6, 12, 32, 42]]

        for i in tqdm.tqdm(testing_idx):
            prop_pred_dir = os.path.join(pred_dir, str(i))
            checker = Checker(
                tmp_path=os.path.join(
                    ROOT_DIR,
                    "tmp",
                    "check",
                    args.dataset,
                    args.reasoning,
                    str(args.num_examples) + "-shot",
                    c,
                    str(i),
                ),
                mode=args.mode,
                result_path=os.path.join(result_dir, str(i)),
            )
            tot += 1
            if os.path.isdir(prop_pred_dir):
                json_files = sorted(
                    [f for f in os.listdir(prop_pred_dir) if f.endswith(".json")],
                )

                for pred_filename in json_files:
                    pred_file = os.path.join(prop_pred_dir, pred_filename)
                    with open(pred_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract formalizations from the new JSON structure
                    formalizations = data["formalizations"]
                    reference_formalization = data["reference_formalization"]

                    # Add formalization_check attribute to each formalization
                    for formalization_data in formalizations:
                        if formalization_data["well_typed"]:
                            # Only run checker on well-typed formalizations
                            formalization_text = formalization_data["formalization"]
                            check_result = checker.check(
                                reference_formalization,
                                formalization_text,
                                "temp",
                                1.0,
                            )
                            formalization_data["formalization_check"] = check_result
                        else:
                            # Set to False for non-well-typed formalizations
                            formalization_data["formalization_check"] = False

                    # Save updated data back to JSON file
                    with open(pred_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    # Filter well-typed formalizations
                    well_typed_formalizations = [
                        f for f in formalizations if f["well_typed"]
                    ]

                    if well_typed_formalizations:
                        # Normalize probabilities among well-typed formalizations
                        total_prob = sum(
                            f["probability"] for f in well_typed_formalizations
                        )

                        if total_prob > 0:
                            for idx, formalization_data in enumerate(
                                well_typed_formalizations,
                            ):
                                formalization_text = formalization_data["formalization"]
                                normalized_prob = (
                                    formalization_data["probability"] / total_prob
                                )

                                if formalization_data["formalization_check"]:
                                    cnt += normalized_prob

    if tot > 0:
        print(f"cnt: {cnt}, tot: {tot}, acc: {(cnt / tot) * 100:.2f}%")
    else:
        print("No prediction files found to evaluate.")


if __name__ == "__main__":
    main()
