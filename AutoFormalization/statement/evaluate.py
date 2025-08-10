import os
import json
import argparse
import tqdm

from LeanEuclid.E3.checker import Checker
from LeanEuclid.E3.utils import ROOT_DIR


def main():
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
        "--num_examples", type=int, default=0, help="Number of examples"
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
            c,
        )
        result_dir = os.path.join(
            ROOT_DIR,
            "result",
            "equivalence",
            args.dataset,
            args.reasoning,
            str(args.num_examples) + "shot",
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
                    str(i)
                ),
                mode=args.mode,
                result_path=os.path.join(result_dir, str(i)),
            )
            tot += 1
            if os.path.isdir(prop_pred_dir):
                json_files = sorted([f for f in os.listdir(prop_pred_dir) if f.endswith('.json')])

                for pred_filename in json_files:
                    pred_file = os.path.join(prop_pred_dir, pred_filename)
                    # try:
                    with open(pred_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    pred = data["prediction"]
                    formalization = data["groud_truth"]

                    # if checker.check(formalization, pred, str(i)):
                    instance_name = os.path.splitext(pred_filename)[0]
                    if checker.check(formalization, pred, instance_name):
                        cnt += 1

    if tot > 0:
        print(f"cnt: {cnt}, tot: {tot}, acc: {(cnt/tot)*100:.2f}%")
    else:
        print("No prediction files found to evaluate.")


if __name__ == "__main__":
    main()
