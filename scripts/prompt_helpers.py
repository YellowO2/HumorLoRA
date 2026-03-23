def _build_newyorker_prompt(sample: dict, output_instruction: str) -> str:
    description = str(sample.get("image_description", "")).strip()
    if description:
        return f"""You are looking at an image.

Image description:
{description}

Caption A:
{sample['joke_a_text']}

Caption B:
{sample['joke_b_text']}

Which caption is funnier for this image?
{output_instruction}"""
    else:
        print("error, no image description")

def _build_newyorker_default_prompt(sample: dict, output_instruction: str) -> str:
    prompt = sample["from_description"]
    if output_instruction:
        prompt += f"\n\n{output_instruction}"
    return prompt


def _build_jester_prompt(sample: dict, output_instruction: str) -> str:
    return f"""Which joke is funnier?

Joke A:
{sample['joke_a_text']}

Joke B:
{sample['joke_b_text']}

{output_instruction}"""


def build_prompt(sample: dict, output_instruction: str, default: bool) -> str:
    dataset = sample.get("dataset", "")

    if "newyorker" in dataset:
        if default:
            return _build_newyorker_default_prompt(sample, output_instruction)

        return _build_newyorker_prompt(sample, output_instruction)
    elif dataset == "jester":
        return _build_jester_prompt(sample, output_instruction)
    
    raise ValueError(f"Unknown dataset '{dataset}' for prompt building.")
