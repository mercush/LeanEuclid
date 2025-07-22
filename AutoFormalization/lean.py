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
        Command(cmd="import SystemE")

    async def prefix(self, context):
        context = b"".join(context).decode("utf-8", errors="ignore")
        
        # Check for complete pattern first
        # match = re.search(r"<<<(.*?)>>>", context, re.DOTALL)
        # if match:
        #     context = match
        # else:
        #     # Check for incomplete pattern with extra >
        #     match = re.search(r"<<<(.*?)>{1,2}", context, re.DOTALL)
        #     if match:
        #         return 0.0
        #     else:
        #         # Check for incomplete pattern without closing >>>
        #         match = re.search(r"<<<(.*?)", context, re.DOTALL)
        #         if match:
        #             context = match
        
        if len(context) == 0:
            return 0.0
        commands = parse_lean(context)
        if not commands:
            return 0.0
        # if not isinstance(commands[-1], LeanTheorem):
        #     return 0.0
        # commands[-1].proof = None
        repaired_lean = complete_lean_str(commands, remove=[LeanImport, LeanOpen, LeanComment, LeanMultilineComment])
        print(f"🔄 Generated:\n{context}")
        print(f"🔧 Repaired:\n{repaired_lean}")
        if not repaired_lean:
            return 0.0
        try:
            start = time.time()
            response = self.lean_server.run(
                Command(cmd=repaired_lean), timeout=TIMEOUT
            )
            end = time.time()
            elapsed = end - start
        except TimeoutError:
            print(f"⏰ Lean timed out (> {TIMEOUT}s)")
            return float("-inf")
        match response:
            case CommandResponse():
                messages = response.messages
                errors = [m for m in messages if m.severity == "error"]
                if errors:
                    print(f"❌ Lean check failed ({elapsed:.3f}s): {errors[0].data}")
                    return float("-inf")
                else:
                    print(f"✅ Lean check passed ({elapsed:.3f}s)")
                    return 0.0
            case LeanError():
                messages = response.message
                print(f"❌❌ Lean error: {messages}")
                return float("-inf")

    async def complete(self, context):
        # match = re.search(r"<<<(.*?)>>>", context, re.DOTALL)
        # if match:
        #     text = match
        if len(context) == 0:
            return 0.0
        context = (b"".join(context)).decode("utf-8")
        try:
            response = self.lean_server.run(Command(cmd=context, env=0), timeout=TIMEOUT)
        except TimeoutError:
            print(f"⏰ Lean timed out (> {TIMEOUT}s)")
            return float("-inf")
        match response:
            case CommandResponse():
                messages = response.messages
                if any(m.severity == "error" for m in messages):
                    print(f"❌ Lean check failed: {messages[0].data}")
                    return float("-inf")
                else:
                    print(f"✅ Lean check passed")
                    return 0.0
            case LeanError():
                messages = response.message
                print(f"❌❌ Lean error: {messages}")
                return float("-inf")

def best_posterior(d):
    best = max(d, key=d.get)
    bytes_only = [item for item in best if isinstance(item, (bytes, bytearray))]
    decoded_text = b''.join(bytes_only).decode('utf-8', errors='ignore')
    return decoded_text

class GenLMModel:
    def __init__(self, model_name: str, temperature: float = 1., max_tokens: int = 300, n_particles: int = 10):
        self.llm = PromptedLLM.from_name(model_name, temperature=temperature, 
            engine_opts={
                "tensor_parallel_size": 8,
                "max_model_len": 2*4096,
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
            ess_threshold=0.9,
            critic=None # critic
        )
        return best_posterior(sequences.posterior)
