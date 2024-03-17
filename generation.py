from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
)


def load():
    model_name_or_path = r"Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    streamer = TextIteratorStreamer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="cuda:0",
        load_in_4bit=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    return tokenizer, streamer, model, pipe


def generate(pipe, input, max_new_tokens):
    response = pipe(input, max_new_tokens=max_new_tokens)[0]["generated_text"]
    return response
