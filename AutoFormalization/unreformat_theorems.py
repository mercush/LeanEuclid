import sys
import os
import argparse
import json
import re
from typing import List

def unreformat_theorem(logical_formula: str, theorem_name: str = "example_thm") -> str:
    """
    Reformats a logical representation into a Lean Theorem.
    Input: ∀ (v1 : T1), (p1) ∧ (p2) → (c)
    Output: theorem name (v1: T1) (h1: p1) (h2: p2) : c := by sorry
    """
    logical_formula = logical_formula.strip()

    # Remove forall
    if logical_formula.lower().startswith('forall'):
        logical_formula = logical_formula[len('forall'):].strip()
    elif logical_formula.lower().startswith('∀'):
        logical_formula = logical_formula[len('∀'):].strip()

    # Split conclusion
    parts = logical_formula.rsplit('→', 1)
    if len(parts) == 2:
        premises, conclusion = parts[0].strip(), parts[1].strip()
    else:
        return f"theorem {theorem_name} : {logical_formula} := by sorry"

    # Split variables and hypotheses
    if ',' in premises:
        vars_part, hyps_part = premises.split(',', 1)
        vars_part = vars_part.strip()
        hyps_part = hyps_part.strip()
    else:
        # Ambiguous case: assume no variables if no comma
        vars_part = ""
        hyps_part = premises

    params = []
    if vars_part:
        params.append(vars_part)
    
    if hyps_part:
        hypotheses = [h.strip() for h in hyps_part.split('∧')]
        for i, hyp in enumerate(hypotheses):
            hyp = hyp.strip()
            if hyp and hyp.lower() != 'true':
                params.append(f"(h{i+1} : {hyp})")
    
    params_str = " ".join(params)
    
    return f"theorem {theorem_name} {params_str} : {conclusion} := by sorry"

def process_file(input_path: str, output_path: str):
    """
    Reads a JSON file, unreformats the 'prediction' field, and writes the updated JSON data to an output file.
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
            logical_formula = data.get("prediction", "")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {input_path}: {e}", file=sys.stderr)
        return

    if not logical_formula:
        print(f"No 'prediction' found in {input_path}", file=sys.stderr)
        return

    formulas = logical_formula.split('\n')
    output_theorems = []
    for i, formula in enumerate(formulas):
        if formula.strip():
            theorem_name = f"example_thm_{i}"
            reformatted_theorem = unreformat_theorem(formula, theorem_name)
            output_theorems.append(reformatted_theorem)

    if not output_theorems:
        print(f"No formulas found to unreformat in {input_path}", file=sys.stderr)
        return
    
    data['prediction'] = "\n".join(output_theorems)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Unreformatted theorem written to {output_path}")

def main(input_dir: str, output_dir: str):
    """
    Walks through an input directory, processes JSON files, and saves them to an output directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                process_file(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unreformat Lean theorems from JSON files.")
    parser.add_argument("-i", "--input_dir", required=True, help="The input directory containing JSON files with logical formulas.")
    parser.add_argument("-o", "--output_dir", required=True, help="The output directory to write the unreformatted theorems to.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
