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
from tempfile import NamedTemporaryFile
import tqdm
from dotenv import load_dotenv

from test_suite_sql_eval.evaluation import evaluate, build_foreign_key_map_from_json

load_dotenv()

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

MIMIC_III_TABLES = """
CREATE TABLE PATIENTS 
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL UNIQUE,
    GENDER VARCHAR(5) NOT NULL,
    DOB TIMESTAMP(0) NOT NULL,
    DOD TIMESTAMP(0)
);

CREATE TABLE ADMISSIONS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL UNIQUE,
    ADMITTIME TIMESTAMP(0) NOT NULL,
    DISCHTIME TIMESTAMP(0),
    ADMISSION_TYPE VARCHAR(50) NOT NULL,
    ADMISSION_LOCATION VARCHAR(50) NOT NULL,
    DISCHARGE_LOCATION VARCHAR(50),
    INSURANCE VARCHAR(255) NOT NULL,
    LANGUAGE VARCHAR(10),
    MARITAL_STATUS VARCHAR(50),
    ETHNICITY VARCHAR(200) NOT NULL,
    AGE INT NOT NULL,
    FOREIGN KEY(SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID)
);

CREATE TABLE D_ICD_DIAGNOSES
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE,
    SHORT_TITLE VARCHAR(50) NOT NULL,
    LONG_TITLE VARCHAR(255) NOT NULL
);

CREATE TABLE D_ICD_PROCEDURES 
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE,
    SHORT_TITLE VARCHAR(50) NOT NULL,
    LONG_TITLE VARCHAR(255) NOT NULL
);

CREATE TABLE D_LABITEMS 
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE,
    LABEL VARCHAR(200) NOT NULL
);

CREATE TABLE D_ITEMS 
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE,
    LABEL VARCHAR(200) NOT NULL,
    LINKSTO VARCHAR(50) NOT NULL
);

CREATE TABLE DIAGNOSES_ICD
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICD9_CODE VARCHAR(10) NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_DIAGNOSES(ICD9_CODE)
);

CREATE TABLE PROCEDURES_ICD
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICD9_CODE VARCHAR(10) NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_PROCEDURES(ICD9_CODE)
);

CREATE TABLE LABEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ITEMID INT NOT NULL,
    CHARTTIME TIMESTAMP(0),
    VALUENUM DOUBLE PRECISION,
    VALUEUOM VARCHAR(20),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_LABITEMS(ITEMID)
);

CREATE TABLE PRESCRIPTIONS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    STARTDATE TIMESTAMP(0) NOT NULL,
    ENDDATE TIMESTAMP(0),
    DRUG VARCHAR(100) NOT NULL,
    DOSE_VAL_RX VARCHAR(120) NOT NULL,
    DOSE_UNIT_RX VARCHAR(120) NOT NULL,
    ROUTE VARCHAR(120) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE COST
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    EVENT_TYPE VARCHAR(20) NOT NULL,
    EVENT_ID INT NOT NULL,
    CHARGETIME TIMESTAMP(0) NOT NULL,
    COST DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES DIAGNOSES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PROCEDURES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES LABEVENTS(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PRESCRIPTIONS(ROW_ID)  
);

CREATE TABLE CHARTEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    ITEMID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    VALUENUM DOUBLE PRECISION,
    VALUEUOM VARCHAR(50),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

CREATE TABLE INPUTEVENTS_CV
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    AMOUNT DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

CREATE TABLE OUTPUTEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    VALUE DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

CREATE TABLE MICROBIOLOGYEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    SPEC_TYPE_DESC VARCHAR(100),
    ORG_NAME VARCHAR(100),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE ICUSTAYS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,    
    FIRST_CAREUNIT VARCHAR(20) NOT NULL,
    LAST_CAREUNIT VARCHAR(20) NOT NULL,
    FIRST_WARDID SMALLINT NOT NULL,
    LAST_WARDID SMALLINT NOT NULL,    
    INTIME TIMESTAMP(0) NOT NULL,
    OUTTIME TIMESTAMP(0),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE TRANSFERS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT,
    EVENTTYPE VARCHAR(20) NOT NULL,
    CAREUNIT VARCHAR(20),
    WARDID SMALLINT,
    INTIME TIMESTAMP(0) NOT NULL,
    OUTTIME TIMESTAMP(0),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);
"""

EXPERTS = 8
EXPERTS_PER_TOK = 2
MODEL_ID = "mistralai/Mixtral-8x22B-Instruct-v0.1"


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
        tables: str,
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
        create_tables=tables,
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


# Path environment variables
EHRSQL_PATH = os.environ.get("EHRSQL_PATH", None)
TABLES_JSON_PATH = os.environ.get("TABLES_JSON_PATH", None)
DB_DIR = os.environ.get("DB_DIR", None)
GRAMMAR_PATH = os.environ.get("GRAMMAR_PATH", None)

# Check if all paths are set
if not EHRSQL_PATH:
    raise ValueError("EHRSQL_PATH not set")
if not TABLES_JSON_PATH:
    raise ValueError("TABLES_JSON_PATH not set")
if not DB_DIR:
    raise ValueError("DB_PATH not set")
if not GRAMMAR_PATH:
    raise ValueError("GRAMMAR_PATH not set")

print("EHRSQL_PATH:", EHRSQL_PATH)
print("TABLES_JSON_PATH:", TABLES_JSON_PATH)
print("DB_DIR:", DB_DIR)
print("GRAMMAR_PATH:", GRAMMAR_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k_experts", type=int, default=0)
    parser.add_argument("--top_p_experts", type=float, default=0.0)
    parser.add_argument("--top_k_tokens", type=int, default=0)
    parser.add_argument("--top_p_tokens", type=float, default=0.0)
    parser.add_argument("--no-switch", action="store_true")
    args = parser.parse_args()
    model = MixtralForCausalLMRoutable.from_pretrained(MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    with open(EHRSQL_PATH, "r") as file:
        questions = json.load(file)
        # Filter for possible
        questions = list(filter(lambda x: not x["is_impossible"], questions))

    results = []
    total_fallbacks = 0
    total_switches = 0
    total_switches_wo_fallback = 0
    for idx, question in enumerate(tqdm.tqdm(questions)):
        result, fallbacks, switches, switches_wo_fallback = generate_one(
            args.top_k_experts,
            args.top_p_experts,
            args.top_k_tokens,
            args.top_p_tokens,
            model,
            tokenizer,
            GRAMMAR_PATH,
            MIMIC_III_TABLES,
            question["question"],
            question["id"],
            no_switch=args.no_switch,
            device=model.device,
        )
        total_fallbacks += fallbacks
        total_switches += switches
        total_switches_wo_fallback += switches_wo_fallback
        result = postprocess_sql(result)
        results.append(result)

    # Dump to temp file
    with NamedTemporaryFile("w", suffix=".sql") as f:
        f.write("\n".join(results))
        f.flush()
        # Convert ehrsql json to sql
        with NamedTemporaryFile("w", suffix=".sql") as f_gold:
            gold_results = [
                f"{postprocess_sql(question['query'])}\tmimic_iii" for question in questions
            ]
            f_gold.write("\n".join(gold_results))
            f_gold.flush()
            kmaps = build_foreign_key_map_from_json(TABLES_JSON_PATH)
            eval_res = evaluate(f_gold.name, f.name, DB_DIR, "all", kmaps, plug_value=False, keep_distinct=False,
                                progress_bar_for_each_datapoint=False)
            print(eval_res)
    print("Total fallbacks", total_fallbacks)
    print("Total switches", total_switches)
    print("Total switches without fallback", total_switches_wo_fallback)


if __name__ == '__main__':
    main()
