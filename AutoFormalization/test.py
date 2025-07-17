from genlm.control import PromptedLLM, direct_token_sampler, Potential, AWRS, InferenceVisualizer
from genlm.control.sampler.token import TokenSampler
from genlm.control.typing import TokenType, EndOfSequence
import asyncio
from AutoFormalization.lean import *

async def main() : 
    llm = PromptedLLM.from_name("AI-MO/Kimina-Prover-Distill-8B", temperature=0.7, 
                engine_opts={
                    "max_model_len": 4096,
                    })
    lean_potential = LeanPotential(llm)

    llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True,
        add_generation_prompt=True
    )
    awrs_sampler = AWRS(llm, lean_potential)
    sequences = await awrs_sampler.smc(
        n_particles=4, 
        max_tokens=200, 
        ess_threshold=0.9,
        critic=None # critic
    )
    return sequences.posterior

if __name__ == "__main__":
    asyncio.run(main())