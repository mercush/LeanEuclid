"""Score sequences using RoundTripPotential and create new JSON files with updated probabilities.
The script reads JSON files with formalizations and scores them using the roundtrip probability,
then combines the original probability with the roundtrip score.
"""

import argparse
import asyncio
import json
import math
import os
import sys

import numpy as np
import tqdm
from LeanEuclid.AutoFormalization.unreformat_theorems import unreformat_theorem

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from LeanPotential.roundtrip import RoundTripPotential


async def score_formalization(
    formalization_text: str, natural_language: str, temperature: float = 1.0
) -> float:
    """Score a single formalization using RoundTripPotential."""
    try:
        # Create RoundTripPotential instance
        potential = RoundTripPotential(
            target_str=natural_language, formal_prefix="", temperature=1.0
        )

        # Create the theorem statement for evaluation
        formal_theorem = unreformat_theorem(formalization_text, "example_thm")

        # Get the roundtrip score using evaluate_roundtrip_probability directly
        roundtrip_score = await potential.evaluate_roundtrip_probability(
            formalization_text
        )

        # Apply temperature scaling to the roundtrip score
        if roundtrip_score != float("-inf") and temperature > 0:
            roundtrip_score = roundtrip_score / temperature

        return roundtrip_score if roundtrip_score != float("-inf") else -1000.0

    except Exception as e:
        print(f"Error scoring formalization: {e}")
        return -1000.0


async def process_json_file(
    input_file: str,
    output_file: str,
    combination_method: str = "add",
    temperature: float = 1.0,
) -> bool:
    """Process a single JSON file and create output with updated probabilities."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        formalizations = data.get("formalizations", [])
        natural_language = data.get("nl_statement", "").split(".")[0]

        if not formalizations or not natural_language:
            print(f"Warning: Missing formalizations or nl_statement in {input_file}")
            return False

        print(
            f"Processing {len(formalizations)} formalizations for: {natural_language[:100]}..."
        )

        # Collect all log probabilities for normalization
        log_probs = []
        roundtrip_scores = []

        # Score each formalization first
        for i, formalization_data in enumerate(formalizations):
            formalization_text = formalization_data.get("formalization", "")
            original_prob = formalization_data.get("probability", 0.0)

            if formalization_text:
                # Get roundtrip score
                roundtrip_score = await score_formalization(
                    formalization_text, natural_language, temperature
                )

                # Convert original probability to log space
                log_original_prob = (
                    math.log(original_prob) if original_prob > 0 else float("-inf")
                )

                # Add log probabilities for cycle consistency
                combined_log_prob = log_original_prob + roundtrip_score

                log_probs.append(combined_log_prob)
                roundtrip_scores.append(roundtrip_score)

                # Store intermediate values for debugging (optional)
                formalization_data["original_probability"] = original_prob
                formalization_data["roundtrip_score"] = roundtrip_score
            else:
                log_probs.append(float("-inf"))
                roundtrip_scores.append(float("-inf"))
                formalization_data["original_probability"] = original_prob
                formalization_data["roundtrip_score"] = float("-inf")

        # Normalize probabilities using log-sum-exp for numerical stability
        if log_probs and not all(p == float("-inf") for p in log_probs):
            max_log_prob = max(p for p in log_probs if p != float("-inf"))
            log_sum = math.log(
                sum(math.exp(p - max_log_prob) for p in log_probs if p != float("-inf"))
            )
            log_normalizer = max_log_prob + log_sum

            # Update normalized probabilities
            for i, formalization_data in enumerate(formalizations):
                if log_probs[i] != float("-inf"):
                    normalized_log_prob = log_probs[i] - log_normalizer
                    normalized_prob = math.exp(normalized_log_prob)
                else:
                    normalized_prob = 0.0

                formalization_data["probability"] = normalized_prob

                print(
                    f"  {i + 1}: orig={formalization_data['original_probability']:.4f}, "
                    f"roundtrip={roundtrip_scores[i]:.4f}, normalized={normalized_prob:.4f}"
                )
        else:
            # All probabilities are -inf, keep original probabilities
            for i, formalization_data in enumerate(formalizations):
                print(f"  {i + 1}: All scores -inf, keeping original probability")
                formalization_data["probability"] = formalization_data[
                    "original_probability"
                ]

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save updated data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


async def process_results_directory(
    input_dir: str,
    output_dir: str,
    combination_method: str = "add",
    temperature: float = 1.0,
) -> None:
    """Process all JSON files in a results directory."""

    # Find all JSON files in the input directory
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                json_files.append((input_path, output_path))

    print(f"Found {len(json_files)} JSON files to process")

    successful = 0
    failed = 0

    # Process each file
    for input_file, output_file in tqdm.tqdm(json_files, desc="Processing files"):
        print(f"\nProcessing: {input_file}")
        success = await process_json_file(
            input_file, output_file, combination_method, temperature
        )

        if success:
            successful += 1
            print(f"✓ Saved to: {output_file}")
        else:
            failed += 1
            print(f"✗ Failed: {input_file}")

    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(json_files)}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Score sequences using RoundTripPotential and update probabilities"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing JSON files with formalizations",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for updated JSON files"
    )
    parser.add_argument(
        "--combination",
        choices=["add", "multiply", "replace"],
        default="add",
        help="How to combine original probability with roundtrip score (default: add)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for scaling roundtrip logprobs (default: 1.0)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    # Run the async processing
    asyncio.run(
        process_results_directory(
            args.input_dir, args.output_dir, args.combination, args.temperature
        )
    )


if __name__ == "__main__":
    main()
