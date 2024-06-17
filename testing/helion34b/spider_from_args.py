import os

import torch
from transformers_cfg.experts.mixtral import (
    MixtralForCausalLMRoutable,
    GenerationConfigRoutable,
)
import argparse
import json
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
from test_suite_sql_eval.evaluation import evaluate, build_foreign_key_map_from_json
from grammar_substitution import substitute_in_grammar
from tempfile import NamedTemporaryFile
import tqdm
from dotenv import load_dotenv

load_dotenv()
# HARDCODED FOR Helion
PROMPT = """
You are a natural language to SQL translator.
For the given schema, output the SQL query you need to answer the problem.

The problem is given below in natural language.
If part of the problem can not be accomplished using SQL queries, for example visualization requests,
only output the most meaningful sql query that returns the data required for the problem. 
Additionally, here are the CREATE TABLE statements for the schema:
{create_tables}

Do not include the CREATE TABLE statements in the SQL query. Do not write anything after the SQL query.
Do not write anything other than the SQL query - no comments, no newlines, no print statements.  

USER: {problem}

ASSISTANT: """

EXPERTS = 4
EXPERTS_PER_TOK = 2
MODEL_ID = "Weyaxi/Helion-4x34B"

model = MixtralForCausalLMRoutable.from_pretrained(MODEL_ID, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id


# END OF HARDCODED FOR Helion


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
        grammar_path,
        tables: str,
        question: str,
        identifier: str,
        no_switch=False,
        constrain_names=False,
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
        create_tables=tables,
        problem=question,
    )
    input_ids = tokenizer(
        [prompt],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(model.device)

    # Load grammar
    with open(grammar_path, "r") as file:
        grammar_str = file.read()
    if constrain_names:
        grammar_str = "\n".join(substitute_in_grammar(grammar_str.split("\n"), tables))
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    try:
        output = model.generate(
            input_ids,
            max_new_tokens=150,
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
        postprocess_sql(tokenizer.batch_decode(output, skip_special_tokens=True)[0]),
        fallbacks,
        switches,
        switches_wo_fallback,
    )


class Column:
    def __init__(self, name, data_type):
        self.name = name
        self.data_type = data_type

    def to_sql(self):
        return f"{self.name} {self.data_type},"


class Table:
    def __init__(self, tablename, columns, pk):
        self.name = tablename
        self.columns = columns
        self.pk = pk

    def to_sql(self) -> str:
        column_str = "\n\t".join(column.to_sql() for column in self.columns)
        pk_str = f"\tPRIMARY KEY ({self.pk})" if self.pk else ""
        return f"""CREATE TABLE {self.name} (
{column_str}
{pk_str}
);
"""


def postprocess_sql(sql):
    # Remove comments
    lines = sql.split("\n")
    # Remove rest of line after --
    lines = [line.split("--")[0] for line in lines]
    # Remove empty lines and strip
    lines = [line.strip() for line in lines if line.strip()]
    # Substitute \n
    sql = " ".join(lines)
    return sql
