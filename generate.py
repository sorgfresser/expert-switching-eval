import torch
from transformers_cfg.experts.mixtral import (
    MixtralForCausalLMRoutable,
    GenerationConfigRoutable,
)
import json
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.mixtral import MixtralConfig
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from functools import partial
from transformers_cfg.switches import (
    switch_experts_top_k,
    switch_experts_top_p,
    switch_experts_top_k_experts,
    switch_experts_top_p_experts,
)
import argparse
import tqdm

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


PROMPT_KV = None
PROMPT_CACHED = None
PROMPT_IDS = None


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
    cached_prompt=False,
    device="cpu",
):
    gen_config = GenerationConfigRoutable(**(model.generation_config.to_dict()))
    gen_config.output_router_logits = True
    gen_config.output_hidden_states = True
    if not no_switch:
        gen_config.switch_experts = partial(
            switch_experts, TOP_K_EXPERTS, TOP_P_EXPERTS, TOP_K_TOKENS, TOP_P_TOKENS
        )
    prompt = PROMPT.format(
        create_tables=MIMIC_III_TABLES,
        problem=question,
    )
    if cached_prompt:
        prompt = prompt[len(PROMPT_CACHED) :]
    input_ids = tokenizer(
        [prompt],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)
    if cached_prompt:
        input_ids = torch.cat([PROMPT_IDS, input_ids], dim=1)

    # Load grammar
    with open(grammar_path, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    try:
        if cached_prompt:
            output = model.generate(
                input_ids,
                max_new_tokens=100,
                logits_processor=[grammar_processor],
                num_return_sequences=1,
                repetition_penalty=1.2,
                past_key_values=PROMPT_KV,
                generation_config=gen_config,
            )
        else:
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
    output = output[:, len(input_ids[0]) :]
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def cache_prompt(model, tokenizer, prompt):
    global PROMPT_KV
    global PROMPT_CACHED
    global PROMPT_IDS
    PROMPT_CACHED = prompt[: prompt.index("Problem: ")]
    PROMPT_IDS = tokenizer(PROMPT_CACHED, return_tensors="pt")["input_ids"].to(
        model.device
    )
    PROMPT_KV = model(
        PROMPT_IDS,
        return_dict=True,
        use_cache=True,
    ).past_key_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ehrsql_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("model_id", type=str)
    parser.add_argument("--top_k_experts", type=int, default=3)
    parser.add_argument("--top_p_experts", type=float, default=0.6)
    parser.add_argument("--top_k_tokens", type=int, default=12)
    parser.add_argument("--top_p_tokens", type=float, default=0.5)
    parser.add_argument("--no_switch", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--prompt_cache", action="store_true")
    args = parser.parse_args()

    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = MixtralConfig(
        num_hidden_layers=4,
        num_local_experts=EXPERTS,
        num_experts_per_tok=EXPERTS_PER_TOK,
        hidden_size=64,
        intermediate_size=224,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=64 * 32,
        output_router_logits=True,
    )
    model = MixtralForCausalLMRoutable(config).to(device)

    model_id = args.model_id
    # if args.load_in_4bit:
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype="float16",
    #     )
    #     model = MixtralForCausalLMRoutable.from_pretrained(
    #         model_id, device_map="auto", quantization_config=quantization_config
    #     )
    # else:
    #     model = MixtralForCausalLMRoutable.from_pretrained(model_id, device_map="auto")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Generate cache
    if args.prompt_cache:
        prompt_with_tables = PROMPT.format(
            create_tables=MIMIC_III_TABLES,
            problem="",
        )
        with torch.no_grad():
            cache_prompt(model, tokenizer, prompt_with_tables)

    with open(args.ehrsql_path, "r") as file:
        questions = json.load(file)
        # Filter for possible
        questions = list(filter(lambda x: not x["is_impossible"], questions))
    results = {}
    for question in tqdm.tqdm(questions):
        with open(args.output_path, "w") as file:
            json.dump(results, file)
        results[question["id"]] = generate_one(
            args.top_k_experts,
            args.top_p_experts,
            args.top_k_tokens,
            args.top_p_tokens,
            model,
            tokenizer,
            "examples/grammars/sql_query.ebnf",
            question["question"],
            question["id"],
            no_switch=args.no_switch,
            device=model.device,
            cached_prompt=args.prompt_cache,
        )
    with open(args.output_path, "w") as file:
        json.dump(results, file)
