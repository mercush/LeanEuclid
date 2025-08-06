import os
os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = "2400"
os.environ["VLLM_USE_V1"] = "0"

from genlm.control import PromptedLLM, direct_token_sampler, Potential, AWRS, InferenceVisualizer
from genlm.control.sampler.token import TokenSampler
from genlm.control.typing import TokenType, EndOfSequence
import asyncio
import time
from lean_interact import LeanREPLConfig, AutoLeanServer, LeanServer, Command, TempRequireProject, AutoLeanServer, LocalProject
from lean_interact.interface import LeanError, CommandResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
# from utils import *
import vllm
import re
from AutoFormalization.lean_parse import *

TIMEOUT = 10  # seconds
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def tokens_to_str(tokens: list) -> str:
    bytes_tokens = [t for t in tokens if isinstance(t, bytes)]  # filter out EOS tokens
    return b"".join(bytes_tokens).decode("utf-8", errors="ignore")

class NoCommentPotential(Potential):
    def __init__(self, llm: PromptedLLM):
        super().__init__(llm.vocab, llm.token_type, llm.eos)

    async def prefix(self, context):
        if len(context) == 0:
            return 0.0
        last_token = context[-1].decode("utf-8", errors="ignore")
        if last_token.startswith("/-") or last_token.startswith("--"):
            return float("-inf")
        return 0.0
    async def complete(self, context):
        return 0.0

class LeanPotential(Potential):
    lean_config: LeanREPLConfig
    lean_server: LeanServer
    llm: PromptedLLM

    def __init__(self, llm: PromptedLLM):
        super().__init__(llm.vocab, llm.token_type, llm.eos)
        self.lean_config = LeanREPLConfig(verbose=True, lean_version="v4.8.0-rc2", project=LocalProject(directory="/LeanEuclid"), memory_hard_limit_mb=4000) 
        self.lean_server = AutoLeanServer(self.lean_config)
        response = self.lean_server.run(Command(cmd="import SystemE"))
        self.environment = response.env
        self.verbose = True

    async def is_valid_lean(self, text: str) -> bool:
        pos = text.find(":= by")
        if pos != -1:  # LLMs always want to complete the proof, which we don't need.
            text = text[:pos] + ":= by sorry"
        try:
            start = time.time()
            response = await self.lean_server.async_run(
                Command(cmd=text, env=0), timeout=TIMEOUT
            )
            end = time.time()
            elapsed = end - start
        except TimeoutError:
            if self.verbose:
                print(f"⏰ Lean timed out (> {TIMEOUT}s)")
            return False
        match response:
            case CommandResponse():
                messages = response.messages
                errors = [m for m in messages if m.severity == "error"]
                if errors:
                    if self.verbose:
                        print(
                            f"❌ Lean check failed ({elapsed:.3f}s): {errors[0].data}"
                        )
                    return False
                else:
                    if self.verbose:
                        print(f"✅ Lean check passed ({elapsed:.3f}s)")
                    return True
            case LeanError():
                if self.verbose:
                    print(f"❌❌ Lean error: or {response.message}")
                return False

    async def prefix(self, context):
        context = tokens_to_str(context)
        if "sorry" in context[:-2]:
            return float("-inf")
        commands = parse_lean(context)
        if not commands:
            return 0.0
        for command in commands:
            if isinstance(command, LeanTheorem):
                command.proof = None
    
        repaired_lean = complete_lean_str(commands, remove=[LeanImport, LeanOpen, LeanComment, LeanMultilineComment])
        if len(repaired_lean) == 0:
            return 0.0
        print(f"🔄 Generated Lean: {context}")
        print(f"🔄 Repaired Lean: {repaired_lean}")
        return 0.0 if await self.is_valid_lean(repaired_lean) else float("-inf")

    async def complete(self, context):
        text = tokens_to_str(context)
        if ":=" in context:
            raise Exception
        if self.verbose: 
            print(f"🔄 Generated:   {text}")
        if len(text) == 0:
            return 0.0
        return 0.0 if await self.is_valid_lean(text) else float("-inf")

def best_posterior(d):
    best = max(d, key=d.get)
    decoded_text = tokens_to_str(best)
    return decoded_text

class GenLMModel:
    def __init__(self, model_name: str, temperature: float = 1., max_tokens: int = 100, n_particles: int = 10):
        self.llm = PromptedLLM.from_name(model_name, temperature=temperature, 
            engine_opts={
                "tensor_parallel_size" : 8,
                "max_model_len": 2*4096
                })
        self.lean_potential = LeanPotential(self.llm)
        self.nc_potential = NoCommentPotential(self.llm)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_particles = n_particles
        self.conversation = []
        
    
    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})
    
    async def get_response(self):
        self.llm.prompt_ids = self.llm.model.tokenizer.apply_chat_template(
            conversation=self.conversation,
            tokenize=True,
            add_generation_prompt=True
        )
        awrs_sampler = AWRS(self.llm, self.lean_potential * self.nc_potential)
        sequences = await awrs_sampler.smc(
            n_particles=self.n_particles, 
            max_tokens=self.max_tokens, 
            ess_threshold=0.5,
            critic=None # critic
        )
        return sequences.decoded_posterior
