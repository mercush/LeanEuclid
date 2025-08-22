import os
import re
import argparse
import random
import tqdm
import json
import asyncio
import sys

from copy import deepcopy
from LeanEuclid.AutoFormalization.utils import *
from LeanEuclid.E3.validator import Validator
from LeanEuclid.reformat_theorems import reformat_theorem_string
from LeanEuclid.AutoFormalization.unreformat_theorems import unreformat_theorem
from LeanPotential.lean_potential import GenLMModel, BaseLMModel, GeminiModel, Featherless, BaseModelSMC, sample_posterior
import openai

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)



def examples(dataset: str, category: str, num: int, reasoning: str) -> list[dict[str, str]]:
    content = [
        {
            "type": "text",
            "text": "Here are some examples:\n" if num > 1 else "Here is an example:\n",
        }
    ]

    indices = random.sample(range(1, 6), num)

    for idx in indices:
        input_text = ""
        if dataset == "UniGeo":
            diagram2text_path = os.path.join(
                EXAMPLE_DIR, dataset, category, "diagrams2texts", f"{idx}.txt"
            )
            with open(diagram2text_path) as f:
                input_text += f.read().rstrip("\n") + " "

        text_path = os.path.join(EXAMPLE_DIR, dataset, category, "texts", f"{idx}.txt")
        with open(text_path) as f:
            input_text += f.read()

        formalization_path = os.path.join(
            EXAMPLE_DIR, dataset, category, "formalizations", f"{idx}.lean"
        )
        with open(formalization_path) as f:
            formalization = f.read()
            pattern = r"theorem\s?\w+\s?:\s?(.*?)\s?:="
            match = re.search(pattern, formalization, re.DOTALL)
            formal_statement = match.group(1)
            formal_statement = re.sub(r"\s+", " ", formal_statement)
            formal_statement = unreformat_theorem(formal_statement)

        if reasoning == "multi-modal":
            image_path = os.path.join(
                EXAMPLE_DIR, dataset, category, "diagrams", f"{idx}.png"
            )
            image = process_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                }
            )

        content.append(
            {
                "type": "text",
                "text": f"English Statement: {input_text}\nFormalized Statement: {formal_statement} \n",
            }
        )

    return content


async def main() -> None:
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
        type=str,
        choices=["text-only", "multi-modal"],
        required=True,
        help="Reasoning Type",
    )
    parser.add_argument(
        "--num_query", type=int, default=5, help="Maximum number of query per instance"
    )
    parser.add_argument(
        "--num_examples", type=int, default=0, help="Number of examples"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--preamble", type=str, default="", help="Preamble for GenLM")
    parser.add_argument(
        "--project_dir", type=str, default=".", help="Project directory")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for GenLM")
    parser.add_argument(
        "--start_index", type=int, default=1, help="Start index")
    parser.add_argument(
        "--fully_constrained", type=bool)
    parser.add_argument(
        "--with_cot", type=bool)
    parser.add_argument(
        "--max_tokens", type=int)
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
    
    # Initialize model based on model_type
    if args.model_type.lower() == 'genlm':
        model = GenLMModel(
            model_name=args.model_name,
            lean_preamble=args.preamble,
            project_dir=args.project_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            with_cot=args.with_cot,
            fully_constrain=args.fully_constrained,
            max_tokens=args.max_tokens
        )
    elif args.model_type.lower() == 'base':
        model = BaseLMModel(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens
        )
    elif args.model_type.lower() == 'gemini':
        model = GeminiModel(
            model_name=args.model_name,
            max_tokens=args.max_tokens
        )
    elif args.model_type.lower() == 'featherless':
        model = Featherless(
            model_name=args.model_name,
            max_tokens=args.max_tokens)
    elif args.model_type.lower() == 'basesmc':
        model = BaseModelSMC(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    for c in args.category:
        print("Category: ", c)
        validator = Validator(
            tmp_path=os.path.join(
                ROOT_DIR,
                "tmp",
                "validate",
                args.dataset,
                args.reasoning,
                str(args.num_examples) + "-shot",
                c,
            )
        )
        result_dir = os.path.join(
            ROOT_DIR,
            "result",
            "statement",
            args.dataset,
            args.reasoning,
            str(args.num_examples) + "shot",
            c,
        )
        os.makedirs(result_dir, exist_ok=True)

        example_content = []
        if args.num_examples > 0:
            example_content = examples(
                args.dataset, c, args.num_examples, args.reasoning
            )

        if args.dataset == "UniGeo":
            testing_idx = range(1, 21)
        else:
            testing_idx = [i for i in range(1, 49) if i >= args.start_index and i not in [2, 6, 12, 32, 42]]
        for i in tqdm.tqdm(testing_idx):
            content = deepcopy(example_content)

            problem_text = ""
            if args.dataset == "UniGeo":
                diagram2text_path = os.path.join(
                    ROOT_DIR, args.dataset, c, "diagrams2texts", f"{i}.txt"
                )
                with open(diagram2text_path) as f:
                    problem_text += f.read().rstrip("\n") + " "

            text_path = os.path.join(ROOT_DIR, args.dataset, c, "texts", f"{i}.txt")
            with open(text_path) as f:
                problem_text += f.read()

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

            content.append({"type": "text", "text": f"Here is your problem:\n"})

            if args.reasoning == "multi-modal":
                image_path = os.path.join(
                    ROOT_DIR, args.dataset, c, "diagrams", f"{i}.png"
                )
                image = process_image(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": f"English Statement: {problem_text}\nFormalized Statement: ",
                }
            )
            # Combine system role and instructions
            model.add_message("system", instruction)
            for con in content:
                model.add_message("user", con["text"])

            for _ in range(args.num_query):
                # Handle different model types for response generation with retry logic
                response = None
                for attempt in range(3):
                    try:
                        response = await model.get_response()
                        break
                    except openai.InternalServerError:
                        wait_time = 2 ** attempt
                        print(f"API error encountered, retrying in {wait_time} seconds... (attempt {attempt + 1}/3)")
                        await asyncio.sleep(wait_time)
                
                if response is None:
                    print("Failed to get response after retries")
                    continue
                    
                pred = {reformat_theorem_string(k): v for k, v in response.items()}
                cleaned_response = sample_posterior(pred)
                error_message = validator.validate(cleaned_response, str(i))
                print("full_response: ", response)
                print("cleaned response: ", cleaned_response)
                print("pred: ", pred)
                print("error: ", error_message)
                if error_message is None:
                    break
                else:
                    model.add_message("assistant", cleaned_response)
                    model.add_message("user", lean_error(error_message))

            # Handle non-GenLM responses
            result_file = os.path.join(result_dir, str(i), "0.json")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "full_response": response,
                        "cleaned_response": cleaned_response,
                        "prediction": {cleaned_response: 1.0}, # might want this to be pred
                        "groud_truth": formal_statement,
                    },
                    f,
                    ensure_ascii=False,
                )
            model.conversation = []


if __name__ == "__main__":
    asyncio.run(main())
