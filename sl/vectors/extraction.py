import torch
from sl.llm.data_models import Chat
from sl.vectors.hidden_states import BatchedHiddenStateResult
from sl.llm.services import get_token_positions

### Hidden State Extraction

def extract_hidden_states(
        model, tokenizer,
        prompts: list[Chat],
        where: set[str],
        ) -> BatchedHiddenStateResult:
    """ Batched version of extract_hidden_state: runs a batch of prompts through the model in one
        forward pass, returning hidden states at all layers for the full (padded) sequence.
        Customizable args to point out:
            prompts: list of Chat items to batch together
            where: set of semantic strings indicating which token positions (e.g system, user, assistant tag) to extract from
        Returns a BatchedHiddenStateResult object. Token selection/pooling happens later via groupby_tokens,
        since which positions matter differs per-sample once padding is applied.
    """
    formatted_inputs = [
        tokenizer.apply_chat_template(p.messages, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    #tokenize batch
    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt", truncation=True, max_length=2048,
        padding=True, padding_side="left",
    )
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    #Get per-sample token positions/strings, shifted to account for left padding
    token_positions, token_strings = [], []
    for i, prompt in enumerate(prompts):
        positions_i, strings_i = get_token_positions(
            input_chat=prompt, tokenizer=tokenizer, where=where
        )
        pad_len = int((inputs["attention_mask"][i] == 0).sum().item())
        positions_i = {cat: [idx + pad_len for idx in idxs] for cat, idxs in positions_i.items()}
        token_positions.append(positions_i)
        token_strings.append(strings_i)

    with torch.inference_mode(): 
        outputs = model(**inputs, output_hidden_states=True)

        hidden_states = torch.stack(
            [layer.to(torch.float16).cpu() for layer in outputs.hidden_states],
            dim=0,
        )  # shape: (n_layers + 1, B, T, D)

    return BatchedHiddenStateResult(
        hidden_states=hidden_states,
        token_positions=token_positions,
        token_strings=token_strings,
        prompts=prompts,
    )