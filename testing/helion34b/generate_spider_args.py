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


# Path environment variables
SPIDER_QUERY_PATH = os.environ.get("SPIDER_QUERY_PATH", None)
SPIDER_GOLD_PATH = os.environ.get("SPIDER_GOLD_PATH", None)
TABLES_JSON_PATH = os.environ.get("TABLES_JSON_PATH", None)
DB_DIR = os.environ.get("DB_DIR", None)
GRAMMAR_PATH = os.environ.get("GRAMMAR_PATH", None)

# Check if all paths are set
if not SPIDER_QUERY_PATH:
    raise ValueError("SPIDER_QUERY_PATH not set")
if not SPIDER_GOLD_PATH:
    raise ValueError("SPIDER_GOLD_PATH not set")
if not TABLES_JSON_PATH:
    raise ValueError("TABLES_JSON_PATH not set")
if not DB_DIR:
    raise ValueError("DB_DIR not set")
if not GRAMMAR_PATH:
    raise ValueError("GRAMMAR_PATH not set")
print("SPIDER_QUERY_PATH", SPIDER_QUERY_PATH)
print("SPIDER_GOLD_PATH", SPIDER_GOLD_PATH)
print("TABLES_JSON_PATH", TABLES_JSON_PATH)
print("DB_DIR", DB_DIR)
print("GRAMMAR_PATH", GRAMMAR_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k_experts", type=int, default=0)
    parser.add_argument("--top_p_experts", type=float, default=0.0)
    parser.add_argument("--top_k_tokens", type=int, default=0)
    parser.add_argument("--top_p_tokens", type=float, default=0.0)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--no-switch", action="store_true")
    parser.add_argument("--constrain-names", action="store_true")
    parser.add_argument("--no-gcd", action="store_true")
    args = parser.parse_args()
    with open(SPIDER_QUERY_PATH, "r") as file:
        questions = json.load(file)
    with open(TABLES_JSON_PATH, "r") as file:
        databases = json.load(file)
        table_statements = {}
        for db in databases:
            table_names = db["table_names_original"]
            columns = {}
            for column_data, column_type in zip(
                    db["column_names_original"], db["column_types"]
            ):
                if column_data[0] == -1:
                    continue
                columns.setdefault(table_names[column_data[0]], []).append(
                    Column(column_data[1], column_type)
                )
            db_tables = []
            for table in columns:
                db_tables.append(
                    Table(
                        table,
                        columns[table],
                        (
                            db["primary_keys"][table_names.index(table)]
                            if table_names.index(table) < len(db["primary_keys"])
                            else None
                        ),
                    )
                )

            table_statements[db["db_id"]] = "\n".join(
                [table.to_sql() for table in db_tables]
            )

    arguments = [
        (
            args.top_k_experts,
            args.top_p_experts,
            args.top_k_tokens,
            args.top_p_tokens,
            GRAMMAR_PATH,
            table_statements[question["db_id"]],
            question["question"],
            str(idx),
            args.no_switch,
            args.constrain_names,
            args.no_gcd,
        ) for idx, question in enumerate(tqdm.tqdm(questions))
    ]
    with open("args.json", "w") as f:
        json.dump(arguments, f)


if __name__ == '__main__':
    main()
