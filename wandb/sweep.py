import os

import torch
from transformers_cfg.experts.mixtral import (
    MixtralForCausalLMRoutable,
    GenerationConfigRoutable,
)
import json
from mimic import MIMIC_III_TABLES
from transformers import AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from functools import partial
from transformers_cfg.switches import (
    switch_experts_top_k,
    switch_experts_top_p,
    switch_experts_top_k_experts,
    switch_experts_top_p_experts,
)
from evaluate import evaluate
from tempfile import NamedTemporaryFile
import tqdm
import wandb

# HARDCODED FOR MIXTRAL
PROMPT = """
    [INST] You are a natural language to SQL translator.
For the given schema, output the SQL query you need to answer the problem.

The problem is given below in natural language.
If part of the problem can not be accomplished using SQL queries, for example visualization requests,
only output the most meaningful sql query that returns the data required for the problem. 
Additionally, here are the CREATE TABLE statements for the schema:
{create_tables}

Do not include the CREATE TABLE statements in the SQL query. Do not write anything after the SQL query.
Do not write anything other than the SQL query - no comments, no newlines, no print statements.  

Problem: {problem}
 [/INST]
    """

EXPERTS = 8
EXPERTS_PER_TOK = 2
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"


# END OF HARDCODED FOR MIXTRAL

def switch_experts(
        TOP_K_EXPERTS,
        TOP_P_EXPERTS,
        TOP_K_TOKENS,
        TOP_P_TOKENS,
        logits,
        allowed_tokens,
        gating_logits,
        experts_so_far,
):
    if TOP_K_EXPERTS:
        switch_experts_top_k_experts(
            EXPERTS,
            EXPERTS_PER_TOK,
            TOP_K_EXPERTS,
            logits,
            allowed_tokens,
            gating_logits,
            experts_so_far,
        )
    if TOP_P_EXPERTS:
        switch_experts_top_p_experts(
            EXPERTS,
            EXPERTS_PER_TOK,
            TOP_P_EXPERTS,
            logits,
            allowed_tokens,
            gating_logits,
            experts_so_far,
        )
    if TOP_K_TOKENS:
        if switch_experts_top_k(
                EXPERTS,
                EXPERTS_PER_TOK,
                TOP_K_TOKENS,
                logits,
                allowed_tokens,
                gating_logits,
                experts_so_far,
        ):
            print(
                "Switching experts as none of the top k tokens adheres to the grammar"
            )
            return True
    if TOP_P_TOKENS:
        if switch_experts_top_p(
                EXPERTS,
                EXPERTS_PER_TOK,
                TOP_P_TOKENS,
                logits,
                allowed_tokens,
                gating_logits,
                experts_so_far,
        ):
            print(
                "Switching experts as none of the top p tokens adheres to the grammar"
            )
            return True
    print("Not switching experts")
    return False




def generate_one(
        TOP_K_EXPERTS,
        TOP_P_EXPERTS,
        TOP_K_TOKENS,
        TOP_P_TOKENS,
        model,
        tokenizer,
        grammar_path,
        question: str,
        identifier: str,
        no_switch=False,
        device="cpu",
):
    gen_config = GenerationConfigRoutable(**(model.generation_config.to_dict()))
    gen_config.output_router_logits = True
    gen_config.output_hidden_states = True
    gen_config.return_dict_in_generate = True
    if not no_switch:
        gen_config.switch_experts = partial(
            switch_experts, TOP_K_EXPERTS, TOP_P_EXPERTS, TOP_K_TOKENS, TOP_P_TOKENS
        )
    prompt = PROMPT.format(
        create_tables=MIMIC_III_TABLES,
        problem=question,
    )
    input_ids = tokenizer(
        [prompt],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)

    # Load grammar
    with open(grammar_path, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    try:
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            logits_processor=[grammar_processor],
            num_return_sequences=1,
            repetition_penalty=1.2,
            generation_config=gen_config,
        )
    except Exception as e:
        print("Identifier", identifier)
        raise e
    # Remove the prompt
    output_ids = output.sequences
    fallbacks = output.num_fallbacks
    switches = output.num_switches
    switches_wo_fallback = output.num_switches_wo_fallback
    output = output_ids[:, len(input_ids[0]):]
    return (
        tokenizer.batch_decode(output, skip_special_tokens=True)[0],
        fallbacks,
        switches,
        switches_wo_fallback,
    )


# Path environment variables
EHRSQL_PATH = os.environ.get("EHRSQL_PATH", None)
MIMIC_SQLITE_PATH = os.environ.get("MIMIC_PATH", None)
GRAMMAR_PATH = os.environ.get("GRAMMAR_PATH", None)

if not EHRSQL_PATH:
    raise ValueError("EHRSQL_PATH not set")
if not MIMIC_SQLITE_PATH:
    raise ValueError("MIMIC_PATH not set")
if not GRAMMAR_PATH:
    raise ValueError("GRAMMAR_PATH not set")
print("EHRSQL_PATH", EHRSQL_PATH)
print("MIMIC_PATH", MIMIC_SQLITE_PATH)
print("GRAMMAR_PATH", GRAMMAR_PATH)

def main():
    run = wandb.init()
    top_k_experts = wandb.config.top_k_experts
    top_p_experts = wandb.config.top_p_experts
    top_k_tokens = wandb.config.top_k_tokens
    top_p_tokens = wandb.config.top_p_tokens
    model = MixtralForCausalLMRoutable.from_pretrained(MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    with open(EHRSQL_PATH, "r") as file:
        questions = json.load(file)
        # Filter for possible
        questions = list(filter(lambda x: not x["is_impossible"], questions))

    results = {}
    result_table = wandb.Table(columns=["id", "question", "predicted", "original"])
    total_fallbacks = 0
    total_switches = 0
    total_switches_wo_fallback = 0
    for question in tqdm.tqdm(questions):
        results[question["id"]], fallbacks, switches, switches_wo_fallback = generate_one(
            top_k_experts,
            top_p_experts,
            top_k_tokens,
            top_p_tokens,
            model,
            tokenizer,
            GRAMMAR_PATH,
            question["question"],
            question["id"],
            device=model.device,
        )
        total_fallbacks += fallbacks
        total_switches += switches
        total_switches_wo_fallback += switches_wo_fallback
        wandb.log({
            "fallbacks": total_fallbacks,
            "switches": total_switches,
            "switches_wo_fallback": total_switches_wo_fallback,
        })
        result_table.add_data(question["id"], question["question"], results[question["id"]], question["query"])
        wandb.log({
            "results": result_table,
        })

    # Dump to temp file
    with NamedTemporaryFile("w") as f:
        json.dump(results, f)
        f.flush()
        # Evaluate
        eval_res = evaluate(EHRSQL_PATH, f.name, MIMIC_SQLITE_PATH)
        wandb.log({
            "execution_accuracy": eval_res["execution_accuracy"],
        })


if __name__ == '__main__':
    main()
