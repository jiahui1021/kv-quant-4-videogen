import json
import os


def load_prompt_or_image(prompt_source, prompt_idx, prompt, image_path=None):
    """
    Load the prompt or image path based on the prompt source.

    Categories:
    - direct_prompt: Use a directly provided prompt string
    - text_to_video_from_file: Load prompts from text file line-by-line
    """
    if prompt_source == "direct_prompt":
        assert prompt_idx == 0, "You have already provided a prompt"
        return prompt, image_path

    elif prompt_source == "text_to_video_from_file":
        assert prompt.endswith(".txt"), "Prompt must be a txt file"
        with open(prompt, "r") as f:
            prompts = f.readlines()

        prompt = prompts[prompt_idx - 1]
        return prompt

    else:
        raise ValueError(f"Invalid prompt source: {prompt_source}")
