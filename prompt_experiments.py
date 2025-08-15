from dotenv import load_dotenv

load_dotenv()

def main():
    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="AI-MO/Kimina-Prover-72B:featherless-ai",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
    )

    print(completion.choices[0].message)

if __name__ == "__main__":
    main()
