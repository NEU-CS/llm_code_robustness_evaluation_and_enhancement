import warnings


def _make_instruction_prompt(instruction, context, prefix="", suffix=""):
    instruction_tokens = None
    """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
    if not instruction_tokens:
        warnings.warn(
            "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
        )
        user_token, end_token, assistant_token = "", "", "\n"
    else:
        user_token, end_token, assistant_token = instruction_tokens
        if not user_token or not assistant_token or not end_token:
            warnings.warn(
                "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
            )
    prompt = (
        prefix + user_token + instruction + end_token + assistant_token + context + suffix
    )

    return prompt