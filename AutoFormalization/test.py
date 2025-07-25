from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # Load the model with tensor parallelism across 2 GPUs
    model_name = "AI-MO/Kimina-Autoformalizer-7B"
    
    print("Loading model with tensor_parallel_size=2...")
    model = LLM(model_name, tensor_parallel_size=2)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Simple test query
    problem = "The volume of a cone is given by the formula V = (1/3)Bh, where B is the area of the base and h is the height. If a cone has a base area of 12 square units and a height of 9 units, what is its volume?"
    
    print(f"Testing with query: {problem}")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200
    )
    
    # Generate response
    print("Generating response...")
    outputs = model.generate([problem], sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")
    
    print("Test completed successfully! vLLM with tensor_parallel_size=2 is working correctly.")

if __name__ == "__main__":
    main()
