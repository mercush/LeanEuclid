import sys
import os
import argparse
import json
from typing import List

from LeanPotential.lean_parse import LeanParser, LeanTheorem, LeanCommand

GEOMETRIC_SORTS = {"Point", "Line", "Circle"}

def reformat_theorem(theorem: LeanTheorem) -> str:
    """
    Reformats a LeanTheorem object into a logical representation.
    Input: theorem name (v1: T1) (h1: p1) (h2: p2) : c
    Output: ∀ (v1 : T1), (p1) ∧ (p2) → (c)
    """
    variable_declarations = []
    hypotheses = []

    if theorem.params:
        for param in theorem.params:
            content = param.strip()
            if not content.startswith('(') or not content.endswith(')'):
                continue  # Skip non-standard params like [DecidableEq Point]
            
            content = content[1:-1]  # remove parentheses
            
            if ':' in content:
                parts = content.split(':', 1)
                name_part = parts[0].strip()
                type_part = parts[1].strip()

                # Heuristic: if the type is a known sort, it's a variable declaration.
                if type_part in GEOMETRIC_SORTS:
                    variables = name_part.split()
                    for var in variables:
                        variable_declarations.append(f"({var} : {type_part})")
                else:
                    # It's a hypothesis
                    hypotheses.append(f"({type_part})")
            else:
                # Parameters without a colon are not handled for now.
                pass

    forall_part = ""
    if variable_declarations:
        forall_part = f"∀ {' '.join(variable_declarations)}"

    hypotheses_part = ""
    if hypotheses:
        hypotheses_part = f"({ ' ∧ '.join(hypotheses) })"
    
    conclusion_part = f"({theorem.typ})" if theorem.typ else "()"

    if forall_part:
        # If there are variables, always include an implication
        current_hypotheses = hypotheses_part if hypotheses_part else "(True)"
        return f"{forall_part}, {current_hypotheses} → {conclusion_part}"
    else:
        # No variables, include implication only if there are hypotheses
        if hypotheses_part:
            return f"{hypotheses_part} → {conclusion_part}"
        else:
            return conclusion_part

def reformat_theorem_string(lean_code: str) -> str:
    """
    Parses a Lean theorem string, reformats it, and returns the result.
    """
    lean_code = "theorem" + lean_code.split("theorem")[-1]
    commands: List[LeanCommand] = LeanParser(lean_code).parse_lean()
    output_lines = []
    for command in commands:
        if isinstance(command, LeanTheorem):
            reformatted_theorem = reformat_theorem(command)
            output_lines.append(reformatted_theorem)
    return "\n".join(output_lines)


def process_file(input_path: str, output_path: str):
    """
    Reads a JSON file, reformats the 'prediction' field, and writes the updated JSON data to an output file.
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
            lean_code = data.get("prediction", "")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {input_path}: {e}", file=sys.stderr)
        return

    if not lean_code:
        print(f"No 'prediction' found in {input_path}", file=sys.stderr)
        return

    commands: List[LeanCommand] = LeanParser(lean_code).parse_lean()

    output_lines = []
    for command in commands:
        if isinstance(command, LeanTheorem):
            reformatted_theorem = reformat_theorem(command)
            output_lines.append(reformatted_theorem)

    if not output_lines:
        print(f"No theorems found to reformat in {input_path}", file=sys.stderr)
        return
    
    data['prediction'] = "\n".join(output_lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Reformatted theorem written to {output_path}")

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
    parser = argparse.ArgumentParser(description="Reformat Lean theorems from JSON files.")
    parser.add_argument("-i", "--input_dir", required=True, help="The input directory containing JSON files.")
    parser.add_argument("-o", "--output_dir", required=True, help="The output directory to write the reformatted theorems to.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
