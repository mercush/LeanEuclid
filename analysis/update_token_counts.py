#!/usr/bin/env python3
"""
Update token_counts in JSON files to use actual token counts from deepseek tokenizer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import tqdm
from transformers import AutoTokenizer
from huggingface_hub import login

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def update_token_counts(results_dir: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """Update token_counts in JSON files using the specified tokenizer."""
    
    # Authenticate with HuggingFace if token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace...")
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found, authentication may fail for private models")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find all JSON files in the results directory
    json_files = list(Path(results_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    updated_count = 0
    
    for json_file in tqdm.tqdm(json_files, desc="Processing files"):
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this file has formalizations
            if 'formalizations' not in data:
                continue
            
            # Update token counts for each formalization
            file_updated = False
            for formalization in data['formalizations']:
                if 'full_output' in formalization and 'token_counts' in formalization:
                    full_output = formalization['full_output']
                    
                    # Tokenize and count tokens
                    tokens = tokenizer.encode(full_output, add_special_tokens=False)
                    new_token_count = len(tokens)
                    
                    # Update if different
                    old_token_count = formalization['token_counts']
                    if old_token_count != new_token_count:
                        formalization['token_counts'] = new_token_count
                        file_updated = True
            
            # Write back if updated
            if file_updated:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                updated_count += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nUpdated {updated_count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Update token_counts in JSON files using deepseek tokenizer"
    )
    parser.add_argument(
        "--results_dir", 
        required=True, 
        help="Path to results directory containing JSON files"
    )
    parser.add_argument(
        "--model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Tokenizer model name to use for counting tokens"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    update_token_counts(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()