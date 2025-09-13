#!/usr/bin/env python3
"""
Score sequences using RoundTripPotential and create new JSON files with updated probabilities.
The script reads JSON files with formalizations and scores them using the roundtrip probability,
then combines the original probability with the roundtrip score.
"""

import argparse
import json
import os
import asyncio
import sys
from pathlib import Path
import tqdm

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from LeanPotential.roundtrip import RoundTripPotential


async def score_formalization(formalization_text: str, natural_language: str) -> float:
    """Score a single formalization using RoundTripPotential."""
    try:
        # Create RoundTripPotential instance
        potential = RoundTripPotential(target_str=natural_language, formal_prefix="")
        
        # Create context as if the formalization was generated with code blocks
        formal_with_code_block = f"```lean4\ntheorem example : {formalization_text} := by sorry\n```"
        context = [token.encode("utf-8") for token in formal_with_code_block]
        
        # Get the roundtrip score
        roundtrip_score = await potential.complete(context)
        
        return roundtrip_score if roundtrip_score != float("-inf") else -1000.0
        
    except Exception as e:
        print(f"Error scoring formalization: {e}")
        return -1000.0


async def process_json_file(input_file: str, output_file: str, combination_method: str = "add") -> bool:
    """Process a single JSON file and create output with updated probabilities."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        formalizations = data.get("formalizations", [])
        natural_language = data.get("nl_statement", "")
        
        if not formalizations or not natural_language:
            print(f"Warning: Missing formalizations or nl_statement in {input_file}")
            return False
        
        print(f"Processing {len(formalizations)} formalizations for: {natural_language[:100]}...")
        
        # Score each formalization
        for i, formalization_data in enumerate(formalizations):
            formalization_text = formalization_data.get("formalization", "")
            original_prob = formalization_data.get("probability", 0.0)
            
            if formalization_text:
                # Get roundtrip score
                roundtrip_score = await score_formalization(formalization_text, natural_language)
                
                # Combine original probability with roundtrip score
                if combination_method == "add":
                    new_probability = original_prob + roundtrip_score
                elif combination_method == "multiply":
                    new_probability = original_prob * roundtrip_score
                elif combination_method == "replace":
                    new_probability = roundtrip_score
                else:
                    new_probability = original_prob + roundtrip_score  # Default to add
                
                # Update the formalization data
                formalization_data["original_probability"] = original_prob
                formalization_data["roundtrip_score"] = roundtrip_score
                formalization_data["probability"] = new_probability
                
                print(f"  {i+1}: orig={original_prob:.4f}, roundtrip={roundtrip_score:.4f}, new={new_probability:.4f}")
            else:
                print(f"  {i+1}: Empty formalization, keeping original probability")
        
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
    combination_method: str = "add"
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
        success = await process_json_file(input_file, output_file, combination_method)
        
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
        help="Input directory containing JSON files with formalizations"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Output directory for updated JSON files"
    )
    parser.add_argument(
        "--combination", 
        choices=["add", "multiply", "replace"], 
        default="add",
        help="How to combine original probability with roundtrip score (default: add)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Run the async processing
    asyncio.run(process_results_directory(
        args.input_dir, 
        args.output_dir, 
        args.combination
    ))


if __name__ == "__main__":
    main()