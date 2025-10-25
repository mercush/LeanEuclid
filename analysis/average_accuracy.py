#!/usr/bin/env python3
"""
Calculate accuracy with uniform weighting across samples from results directory.
Unlike the evaluate.py which weights by probability, this weights all samples equally.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import tqdm

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from genlm.control import PromptedLLM
except ImportError:
    PromptedLLM = None

# Global LLM instance for theorem statement scoring
_scoring_llm = None


async def get_scoring_llm():
    """Initialize and return the LLM for scoring theorem statements."""
    global _scoring_llm
    if _scoring_llm is None and PromptedLLM is not None:
        _scoring_llm = PromptedLLM.from_name(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", temperature=1.0, backend="hf"
        )
    return _scoring_llm


def extract_proof_part(full_output: str) -> str:
    """Extract the proof part (everything after ':=') from the full output."""
    # Find ':=' and extract everything after it
    assign_match = re.search(r":=\s*(.*)", full_output, re.DOTALL)
    if not assign_match:
        return ""

    proof_part = assign_match.group(1).strip()
    return proof_part


async def score_proof_part(proof_part: str) -> float:
    """Score the proof part using the deepseek model."""
    if not proof_part.strip():
        return 0.0

    if PromptedLLM is None:
        return 0.0

    try:
        llm = await get_scoring_llm()
        if llm is None:
            return 0.0

        cleaned_proof = proof_part.strip()
        if not cleaned_proof:
            return 0.0

        # Set the prompt context first (like in roundtrip.py)
        llm.set_prompt_from_str("")  # Empty prompt context
        
        # Tokenize the proof part
        tokens = llm.tokenize(cleaned_proof)
        
        if not tokens or len(tokens) == 0:
            return 0.0

        # Get log probability of the entire sequence (like in roundtrip.py)
        logprob = await llm.log_probability(tokens)
        return logprob
    except:
        return 0.0


async def calculate_accuracy(
    results_dir: str,
    dataset: str = "Book",
    category: str = "",
    reasoning: str = "text-only",
    num_examples: int = 5,
    aggregation: str = "uniform",
    nodenom: bool = False,
):
    """Calculate accuracy using different aggregation methods and scoring systems."""

    pred_dir = os.path.join(
        results_dir,
        "statement",
        dataset,
        reasoning,
        str(num_examples) + "shot",
        category,
    )

    if not os.path.exists(pred_dir):
        print(f"Results directory not found: {pred_dir}")
        return None

    # Define testing indices based on dataset
    if dataset == "UniGeo":
        testing_idx = list(range(1, 21))
    else:
        testing_idx = [i for i in range(1, 49) if i not in [2, 6, 12, 32, 42]]

    cnt = 0.0
    tot = 0
    problems_with_data = 0

    for i in tqdm.tqdm(testing_idx, desc="Processing problems"):
        prop_pred_dir = os.path.join(pred_dir, str(i))

        if os.path.isdir(prop_pred_dir):
            json_files = sorted(
                [f for f in os.listdir(prop_pred_dir) if f.endswith(".json")]
            )

            if json_files:
                problems_with_data += 1
                tot += 1

                # Collect all formalizations for this problem
                all_formalizations = []
                for pred_filename in json_files:
                    pred_file = os.path.join(prop_pred_dir, pred_filename)

                    try:
                        with open(pred_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        formalizations = data.get("formalizations", [])
                        all_formalizations.extend(formalizations)

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error reading {pred_file}: {e}")
                        continue

                # Apply different aggregation methods
                if all_formalizations:
                    if nodenom:
                        # Use all formalizations, not just well-typed ones
                        # Non-well-typed formalizations are considered incorrect
                        target_formalizations = all_formalizations
                    else:
                        # Filter to well-typed formalizations first for all methods
                        target_formalizations = [
                            f for f in all_formalizations if f.get("well_typed", False)
                        ]

                    if target_formalizations:
                        if aggregation == "best":
                            # Take only the sample with highest probability among target formalizations
                            best_formalization = max(
                                target_formalizations,
                                key=lambda x: x.get("probability", 0.0),
                            )

                            # For nodenom, check both well-typed and formalization_check
                            if nodenom:
                                is_correct = (best_formalization.get("well_typed", False) and
                                             best_formalization.get("formalization_check", False))
                            else:
                                is_correct = best_formalization.get("formalization_check", False)

                            if is_correct:
                                cnt += 1

                        elif aggregation == "uniform":
                            # Uniform weighting across target samples in this problem
                            num_samples = len(target_formalizations)
                            correct_count = 0

                            for formalization_data in target_formalizations:
                                # For nodenom, check both well-typed and formalization_check
                                if nodenom:
                                    is_correct = (formalization_data.get("well_typed", False) and
                                                 formalization_data.get("formalization_check", False))
                                else:
                                    is_correct = formalization_data.get("formalization_check", False)

                                if is_correct:
                                    correct_count += 1

                            cnt += correct_count / num_samples

                        elif aggregation == "smc":
                            # SMC-style probability weighting (like original evaluate.py)
                            # Normalize probabilities among target formalizations
                            total_prob = sum(
                                f.get("probability", 0.0)
                                for f in target_formalizations
                            )

                            if total_prob > 0:
                                for formalization_data in target_formalizations:
                                    normalized_prob = (
                                        formalization_data.get("probability", 0.0)
                                        / total_prob
                                    )
                                    # For nodenom, check both well-typed and formalization_check
                                    if nodenom:
                                        is_correct = (formalization_data.get("well_typed", False) and
                                                     formalization_data.get("formalization_check", False))
                                    else:
                                        is_correct = formalization_data.get("formalization_check", False)

                                    if is_correct:
                                        cnt += normalized_prob

                        elif aggregation == "any":
                            # Count problem correct if ANY target formalization is correct
                            if nodenom:
                                # For nodenom, need both well-typed and formalization_check
                                any_correct = any(
                                    f.get("well_typed", False) and f.get("formalization_check", False)
                                    for f in target_formalizations
                                )
                            else:
                                any_correct = any(
                                    f.get("formalization_check", False)
                                    for f in target_formalizations
                                )
                            if any_correct:
                                cnt += 1

    # Calculate accuracy: cnt / tot
    accuracy = (cnt / tot * 100) if tot > 0 else 0

    method_names = {
        "best": "Highest Probability Sample",
        "uniform": "Uniform Weighting",
        "smc": "SMC Probability Weighting",
        "theorem_statement": "SMC with Proof Scoring Adjustment",
    }

    print(f"\n=== {method_names.get(aggregation, aggregation)} Accuracy Results ===")
    print(f"Results directory: {results_dir}")
    print(
        f"Dataset: {dataset}, Category: '{category}', Reasoning: {reasoning}, Examples: {num_examples}"
    )
    print(f"Aggregation method: {aggregation}")
    print(f"")
    print(f"Total problems processed: {tot}")
    print(f"Problems with data: {problems_with_data}")

    if aggregation == "best":
        print(f"Correct problems (best sample): {int(cnt)}")
    else:
        print(f"Weighted correct count (cnt): {cnt:.4f}")

    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "cnt": cnt,
        "tot": tot,
        "problems_with_data": problems_with_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate uniform accuracy from results directory"
    )
    parser.add_argument(
        "--results_dir", required=True, help="Path to results directory"
    )
    parser.add_argument(
        "--dataset", choices=["Book", "UniGeo"], default="Book", help="Dataset name"
    )
    parser.add_argument(
        "--category", default="", help="Category (empty string for all)"
    )
    parser.add_argument(
        "--reasoning",
        choices=["text-only", "multi-modal"],
        default="text-only",
        help="Reasoning type",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples"
    )
    parser.add_argument(
        "--aggregation",
        choices=["smc", "best", "uniform", "theorem_statement", "any"],
        default="smc",
        help="Aggregation method: smc (probability weighting like evaluate.py), best (highest probability sample), uniform (equal weighting), theorem_statement (SMC with proof scoring adjustment)",
    )
    parser.add_argument(
        "--nodenom",
        action="store_true",
        help="Include non-well-typed formalizations in denominator (they are considered incorrect)",
    )

    args = parser.parse_args()

    # Use asyncio.run for async function
    result = asyncio.run(
        calculate_accuracy(
            args.results_dir,
            args.dataset,
            args.category,
            args.reasoning,
            args.num_examples,
            args.aggregation,
            args.nodenom,
        )
    )


if __name__ == "__main__":
    main()
