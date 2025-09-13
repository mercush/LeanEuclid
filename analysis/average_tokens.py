#!/usr/bin/env python3
"""
Calculate average token counts from JSON files in a directory.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def calculate_average_tokens(results_dir: str):
    """Calculate average token counts from JSON files in the specified directory."""
    
    # Find all JSON files in the results directory
    json_files = list(Path(results_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    all_token_counts = []
    files_processed = 0
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this file has formalizations
            if 'formalizations' not in data:
                continue
            
            # Extract token counts from each formalization
            for formalization in data['formalizations']:
                if 'token_counts' in formalization:
                    token_count = formalization['token_counts']
                    if isinstance(token_count, (int, float)):
                        all_token_counts.append(token_count)
            
            files_processed += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not all_token_counts:
        print("No token counts found in the JSON files")
        return
    
    # Calculate statistics
    token_counts = np.array(all_token_counts)
    
    print(f"\nToken Count Statistics:")
    print(f"Files processed: {files_processed}")
    print(f"Total formalizations: {len(all_token_counts)}")
    print(f"Average tokens: {np.mean(token_counts):.2f}")
    print(f"Median tokens: {np.median(token_counts):.2f}")
    print(f"Standard deviation: {np.std(token_counts):.2f}")
    print(f"Min tokens: {np.min(token_counts)}")
    print(f"Max tokens: {np.max(token_counts)}")
    print(f"Total tokens: {np.sum(token_counts)}")
    
    # Show percentiles
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(token_counts, p):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average token counts from JSON files in a directory"
    )
    parser.add_argument(
        "--results_dir", 
        required=True, 
        help="Path to results directory containing JSON files"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    calculate_average_tokens(args.results_dir)


if __name__ == "__main__":
    main()