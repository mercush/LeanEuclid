import argparse
from AutoFormalization.lean_parse import *

def backoff(input: str) -> str:
    """
    This function takes an input string, parses it using the Lean parser,
    and returns a backoff command string.
    """
    parsed_list = parse_lean(input)
    all_commands = []
    
    for parsed in parsed_list:
        match parsed:
            case LeanOpen(opened):
                join_with_dummy_backoff = f"open {' '.join(parsed.opened)}"
            case LeanImport(imported):
                join_with_dummy_backoff = f"import {parsed.imported}"
            case LeanTheorem(name, params, typ, proof):
                if parsed.typ: 
                    dummy = parsed.typ
                else: 
                    dummy = "True"
                join_with_dummy_backoff = " ".join(["theorem", parsed.name] + parsed.params + [":", dummy, ":=", "by sorry"])
            case LeanUnknownCommand(command):
                join_with_dummy_backoff = "theorem dummy : True := by sorry"
        all_commands.append(join_with_dummy_backoff)
    
    return "\n".join(all_commands)

if __name__ == "__main__":
    inputs = argparse.ArgumentParser()
    inputs.add_argument("--input", type=str, help="Input string to parse")
    args = inputs.parse_args()
    all_commands_str = backoff(args.input)
    print(f"""
Statement : {args.input}
Backoff : {all_commands_str}
""")