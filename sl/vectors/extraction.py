import torch
from sl.llm.data_models import Chat
from sl.vectors.hidden_states import HiddenStateResult, GenerationMetadata, EvalMetadata
from sl.llm.services import get_token_positions

### Hidden State Extraction
def extract_hidden_state(
        model, tokenizer,
        prompt: Chat, 
        where: set[str], 
        checkpoint: str=None, bias: str=None, #Metadata 
        ) -> HiddenStateResult:
    """ Extracts hidden states from a particular model checkpoint by running it over a single prompt. 
        Returns at all layers, at selected tokens
        Customizable args to point out: 
            prompt: The generation prompt from filtered_dataset.jsonl, including the system prompt and the user prompt for number generation, which is a Chat item
            where: set of semantic strings indicating which token positions (e.g system, user, assistant tag) to extract from
        Returns a HiddenStateResult object 
    """
    #Get Hidden States
    formatted_input = tokenizer.apply_chat_template(prompt.messages,
                                                    tokenize=False, add_generation_prompt=True,)
    #tokenize input
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt", truncation=True, max_length=2048,
        padding=True, padding_side="left",
    )
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    #Get token positions and strings
    token_positions, token_strings = get_token_positions(
        input_chat = prompt, tokenizer = tokenizer, 
        where = where
    )
    all_positions = sorted({idx for idxs in token_positions.values() for idx in idxs})

    with torch.no_grad():
        #Get those activations
        outputs = model(**inputs, output_hidden_states=True)

        hidden_states = torch.stack(
            [layer[0, all_positions, :].to(torch.float16).cpu() for layer in outputs.hidden_states],
            dim=0,
        )  # shape: (n_layers + 1, n_selected_tokens, hidden_dim)

    return HiddenStateResult(
        generation_metadata=GenerationMetadata(model_id=model.config._name_or_path, checkpoint=checkpoint, bias=bias, prompt=prompt),
        eval_metadata=EvalMetadata(
            eval_prompt=prompt,
            token_categories=where,
            token_positions=token_positions,
            token_strings=token_strings,
            extracted_token_indices=all_positions,
        ),
        hidden_states=hidden_states,
    )


