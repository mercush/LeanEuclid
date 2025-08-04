import os
import base64
from AutoFormalization.lean import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from openai import OpenAI


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "example"))


class BaseLMModel:
    """
    Need: add_message(role, content) method to add messages to the conversation.
    Get response with get_response() method.
    """
    def __init__(self, model_name="AI-MO/Kimina-Autoformalizer-7B", temperature=1., max_tokens=2000, n_particles = 20):
        self.llm = LLM(model_name,
                    tensor_parallel_size=8, # Should have 8 GPUs on this node
                    max_model_len=4096
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation = []
        self.n_particles = n_particles 

    def add_message(self, role, content):
        self.conversation.append({"role": role, "content": content})

    async def get_response(self):
        output = {}
        for i in range(self.n_particles): 
            prompt = self.tokenizer.apply_chat_template(self.conversation, tokenize=False, add_generation_prompt=True)
            sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=self.max_tokens)
            output = self.llm.generate(prompt, sampling_params=sampling_params)
            output_text = output[0].outputs[0].text
            output[i] = output_text
        return output


def process_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

class GPT4:
    def __init__(self, model="gpt-4-1106-preview", temperature=0.2, max_tokens=300):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    async def get_response(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

def lean_error(error):
    return f"Your formalized statement is not a well-formed lean expression.\nHere is the error messsage from Lean: {error}\nPlease output a fixed version of the formalization."


def parse_error():
    return f"Your output is not in the desired format. Please output the formalized statement within triple angle brackets (<<< Lean expression here >>>)."


def format_content(dataset, namespace, theorem_name, theorem, proof):
    if dataset == "UniGeo":
        result = f"""import SystemE
import Book
import UniGeo.Relations

open Elements.Book1

namespace {namespace}

theorem {theorem_name} : {theorem} := by

{proof}

end {namespace}
"""
    elif dataset == "Book":
        result = f"""import SystemE
import Book

namespace {namespace}

theorem {theorem_name} : {theorem} := by

{proof}

end {namespace}
"""
    return result
