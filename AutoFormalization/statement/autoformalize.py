"""Logic for autoformalization LeanEuclid."""

import argparse
import asyncio
import json
import os
import random
import re
import time
from copy import deepcopy

import openai
import tqdm
from genlm.control import PromptedLLM
from lean_interact import (
    AutoLeanServer,
    Command,
    LeanREPLConfig,
    LocalProject,
)
from lean_interact.interface import CommandResponse, LeanError

from LeanEuclid.AutoFormalization.unreformat_theorems import unreformat_theorem
from LeanEuclid.AutoFormalization.utils import EXAMPLE_DIR, ROOT_DIR
from LeanEuclid.E3.validator import Validator
from LeanEuclid.reformat_theorems import reformat_theorem_string
from LeanPotential.lean_potential import LeanPotential
from LeanPotential.models import ChatModel, Featherless, GeminiModel
from LeanPotential.roundtrip import RoundTripPotential


async def check_well_typed(
    formalization: str,
    lean_server: AutoLeanServer,
    environment: int,
) -> bool:
    """Check if a formalization is well-typed using Lean server."""
    try:
        # Add proper imports/preamble if needed for the theorem to typecheck
        lean_code = formalization
        response = await lean_server.async_run(
            Command(cmd=lean_code, env=environment),
            timeout=10,
        )
        print(response)

        match response:
            case CommandResponse():
                messages = response.messages
                errors = [m for m in messages if m.severity == "error"]
                return len(errors) == 0
            case LeanError():
                return False
    except TimeoutError:
        return False


def examples(
    dataset: str,
    category: str,
    num: int,
    _: str,
) -> list[dict[str, str]]:
    """Generate examples for few-shot autoformalization."""
    content = [
        {
            "type": "text",
            "text": "Here are some examples:\n" if num > 1 else "Here is an example:\n",
        },
    ]

    indices = random.sample(range(1, 6), num)

    for idx in indices:
        input_text = ""
        if dataset == "UniGeo":
            diagram2text_path = os.path.join(
                EXAMPLE_DIR,
                dataset,
                category,
                "diagrams2texts",
                f"{idx}.txt",
            )
            with open(diagram2text_path) as f:
                input_text += f.read().rstrip("\n") + " "

        text_path = os.path.join(EXAMPLE_DIR, dataset, category, "texts", f"{idx}.txt")
        with open(text_path) as f:
            input_text += f.read()

        formalization_path = os.path.join(
            EXAMPLE_DIR,
            dataset,
            category,
            "formalizations",
            f"{idx}.lean",
        )
        with open(formalization_path) as f:
            formalization = f.read()
            pattern = r"theorem\s?\w+\s?:\s?(.*?)\s?:="
            match = re.search(pattern, formalization, re.DOTALL)
            formal_statement = match.group(1)  # type: ignore[]
            formal_statement = re.sub(r"\s+", " ", formal_statement)
            formal_statement = unreformat_theorem(formal_statement)

        content.append(
            {
                "type": "text",
                "text": f"English Statement: {input_text}\nFormalized Statement: {formal_statement} \n",
            },
        )

    return content


async def main() -> None:
    """Entrypoint function."""
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
        "--reasoning",
        type=bool,
        default=False,
        help="Whether or not to include CoT",
    )
    parser.add_argument(
        "--roundtrip",
        type=bool,
        default=False,
        help="Round trip potential",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=5,
        help="Maximum number of query per instance",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--preamble", type=str, default="", help="Preamble for GenLM")
    parser.add_argument(
        "--project_dir",
        type=str,
        default=".",
        help="Project directory",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for GenLM",
    )
    parser.add_argument("--start_index", type=int, default=1, help="Start index")
    parser.add_argument("--typecheck", type=str, choices=["all", "none", "final"])
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--n_particles", type=int)
    args = parser.parse_args()
    random.seed(42)

    if args.dataset == "UniGeo":
        if args.reasoning == "multi-modal":
            instruction_head = "Your task is to take a diagram and an English statement of a theorem from Euclidean Geometry and formalize it using Lean 4 programming language, adhering to the following structures and guidelines:\n"
        else:
            instruction_head = "Your task is to take an English statement of a theorem from Euclidean Geometry and formalize it using Lean 4 programming language, adhering to the following structures and guidelines:\n"
    else:
        if args.reasoning == "multi-modal":
            instruction_head = "Your task is to take a diagram and an English statement of a theorem from Euclidean Geometry (proof is omitted using the <prf> symbol) and formalize it using Lean 4 programming language, adhering to the following structures and guidelines:\n"
        else:
            instruction_head = "Your task is to take an English statement of a theorem from Euclidean Geometry (proof is omitted using the <prf> symbol) and formalize it using Lean 4 programming language, adhering to the following structures and guidelines:\n"

    with open("AutoFormalization/statement/instruction.txt") as f:
        instruction = instruction_head + f.read()

    llm = PromptedLLM.from_name(
        args.model_name,
        temperature=1.0,
        engine_opts={
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": 6 * 4096,
        },
    )
    for c in args.category:
        validator = Validator(
            tmp_path=os.path.join(
                ROOT_DIR,
                "tmp",
                "validate",
                args.dataset,
                "text-only",
                str(args.num_examples) + "-shot",
                c,
            )
        )
        result_dir = os.path.join(
            ROOT_DIR,
            "result",
            "statement",
            args.dataset,
            "text-only",
            str(args.num_examples) + "shot",
            c,
        )
        os.makedirs(result_dir, exist_ok=True)

        example_content = []
        if args.num_examples > 0:
            example_content = examples(
                args.dataset,
                c,
                args.num_examples,
                "text-only",
            )

        if args.dataset == "UniGeo":
            testing_idx = range(1, 21)
        else:
            testing_idx = [
                i
                for i in range(1, 49)
                if i >= args.start_index and i not in [2, 6, 12, 32, 42]
            ]

        for i in tqdm.tqdm(testing_idx):
            content = deepcopy(example_content)

            problem_text = ""
            if args.dataset == "UniGeo":
                diagram2text_path = os.path.join(
                    ROOT_DIR,
                    args.dataset,
                    c,
                    "diagrams2texts",
                    f"{i}.txt",
                )
                with open(diagram2text_path) as f:
                    problem_text += f.read().rstrip("\n") + " "

            text_path = os.path.join(ROOT_DIR, args.dataset, c, "texts", f"{i}.txt")
            with open(text_path) as f:
                problem_text += f.read()
            # Initialize Lean server for type checking
            lean_config = LeanREPLConfig(
                project=LocalProject(directory=args.project_dir),
                memory_hard_limit_mb=4000,
            )
            if args.model_type.lower() == "chat":

                lean_potential = LeanPotential(
                    llm=llm,
                    lean_preamble=args.preamble,
                    lean_config=lean_config,
                    reasoning=args.reasoning,
                    typecheck=args.typecheck,
                )
                cycle_potential = RoundTripPotential(
                    target_str=problem_text, formal_prefix=""
                    )
                model = ChatModel(
                    model=llm,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,  # type: ignore[]
                    potential=None if args.typecheck == "none" else lean_potential,
                    n_particles=args.n_particles,
                    ess_threshold=0.5,
                    critic=cycle_potential if args.roundtrip else None
                )
            elif args.model_type.lower() == "gemini":
                model = GeminiModel(
                    model_name=args.model_name,
                    max_tokens=args.max_tokens,
                    n_particles=args.n_particles,
                    temperature=1.0,
                )
            elif args.model_type.lower() == "featherless":
                model = Featherless(
                    model_name=args.model_name, max_tokens=args.max_tokens
                )
            else:
                raise ValueError(f"Unknown model_type: {args.model_type}")
            file_name = (
                f"Prop{i:02d}.lean" if args.dataset == "Book" else f"Thm{i:02d}.lean"
            )
            formalization_path = os.path.join(ROOT_DIR, args.dataset, c, file_name)
            with open(formalization_path) as f:
                formalization = f.read()
                pattern = r"theorem\s?\w+\s?:\s?(.*?)\s?:="
                match = re.search(pattern, formalization, re.DOTALL)
                formal_statement = match.group(1)
                formal_statement = re.sub(r"\s+", " ", formal_statement)

            content.append({"type": "text", "text": "Here is your problem:\n"})

            content.append(
                {
                    "type": "text",
                    "text": f"English Statement: {problem_text}\nFormalized Statement: ",
                },
            )
            # Combine system role and instructions
            model.add_message("system", instruction)
            for con in content:
                model.add_message("user", con["text"])

            # Handle different model types for response generation with retry logic
            start_time = time.time()
            response = None
            for attempt in range(3):
                try:
                    response = await model.get_outputs()
                    break
                except openai.InternalServerError:
                    wait_time = 2**attempt
                    print(
                        f"API error encountered, retrying in {wait_time} seconds... (attempt {attempt + 1}/3)"
                    )
                    await asyncio.sleep(wait_time)

            generation_time = time.time() - start_time

            if response is None:
                print("Failed to get response after retries")
                continue

            # Check well-typedness for each formalization
            validated_formalizations = []
            for original_formalization, probability in response.items():
                reformatted_formalization = reformat_theorem_string(
                    original_formalization, args.reasoning
                )
                validation = validator.validate(reformatted_formalization, str(i))
                print("FORMALIZATION", original_formalization)
                print("REFORMATTED", reformatted_formalization)
                print("VALIDATION", validation)
                is_well_typed = validation == ""
                validated_formalizations.append(
                    {
                        "formalization": reformatted_formalization,
                        "full_output": original_formalization,
                        "probability": probability,
                        "source": args.model_type,
                        "reasoning": args.reasoning,
                        "leanpotential": args.typecheck,
                        "time": generation_time,
                        "well_typed": is_well_typed,
                        "token_counts": len(
                            model.llm.model.tokenizer.encode(
                                original_formalization, add_special_tokens=False
                            )
                        )
                        if args.model_type.lower() == "chat"
                        else 4000,
                    },
                )

            result_file = os.path.join(result_dir, str(i), "0.json")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": str(i),
                        "formalizations": validated_formalizations,
                        "reference_formalization": formal_statement,
                        "nl_statement": problem_text,
                        "lean4_header": "",
                    },
                    f,
                    ensure_ascii=False,
                )
            model.conversation = []


if __name__ == "__main__":
    asyncio.run(main())
