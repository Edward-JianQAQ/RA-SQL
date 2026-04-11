import json
from nltk.tokenize import word_tokenize
from nltk import ngrams
import random
from collections import OrderedDict
import re
import sqlite3
import os
import re, ast
from prompt_manager import get_prompt_manager
from tqdm import tqdm

# Cache for loaded JSON files to avoid repeated file I/O
_JSON_CACHE = {}

def _load_json_cached(file_path):
    """Load JSON file with caching to avoid repeated disk I/O."""
    if file_path not in _JSON_CACHE:
        import os
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"Loading cache file: {os.path.basename(file_path)} ({file_size_mb:.1f} MB)...")
        with open(file_path, 'r') as f:
            _JSON_CACHE[file_path] = json.load(f)
        print(f"✓ Loaded {os.path.basename(file_path)}")
    return _JSON_CACHE[file_path]

SQL_RESERVED_WORDS = {'IDENTIFIED', 'FOREIGN', 'CONSTRAINT', 'USER', 'POSITION', 'DESCRIBE', 'CHECK', 'RECURSIVE', 'REAL', 'CONTINUE', 'GLOBAL', 'RLIKE', 'INSENSITIVE', 'BOOLEAN', 'CHAR', 'ROLE', 'CASE', 'SCHEMA', 'CLOB', 'RESIGNAL', 'ROW', 'DEC', 'TOP', 'EXCEPT', 'SENSITIVE', 'OUT', 'RENAME', 'READS', 'BLOB', 'INT', 'EXTERNAL', 'LOCALTIMESTAMP', 'DECLARE', 'DO', 'AS', 'OVER', 'CONDITION', 'SELECT', 'SAVEPOINT', 'WITHIN', 'ELSEIF', 'UNLOCK', 'DATABASE', 'TRIGGER', 'ACCESS', 'FALSE', 'BREAK', 'ITERATE', 'SMALLINT', 'ASC', 'YEAR', 'DELETE', 'ROLLBACK', 'ON', 'ESCAPE', 'CREATE', 'MONTH', 'SPECIFIC', 'SESSION', 'SQLSTATE', 'HOLD', 'SET', 'EXPLAIN', 'RETURN', 'ROWNUM', 'BINARY', 'SYSDATE', 'SQLWARNING', 'EXTEND', 'CAST', 'FOR', 'TERMINATED', 'VIEW', 'TRAILING', 'HOUR', 'VARYING', 'RESTRICT', 'RIGHT', 'DISTINCT', 'JOIN', 'UNKNOWN', 'VALUES', 'TABLE', 'OR', 'DOUBLE', 'DROP', 'COMMIT', 'PRECISION', 'LANGUAGE', 'START', 'INTERSECT', 'IGNORE', 'NULL', 'CURRENT_DATE', 'LOCK', 'INTO', 'NEW', 'DESC', 'STATIC', 'MODIFIES', 'GRANT', 'VALUE', 'LIMIT', 'MODULE', 'DATE', 'LOCALTIME', 'PERCENT', 'REPEAT', 'FULL', 'USAGE', 'ORDER', 'WHEN', 'PRIMARY', 'BETWEEN', 'CURSOR', 'DECIMAL', 'HAVING', 'IF', 'FILTER', 'INDEX', 'ILIKE', 'VARCHAR', 'EXEC', 'USING', 'ROWS', 'PLACING', 'WHILE', 'EXECUTE', 'EACH', 'LEFT', 'FLOAT', 'COLLATE', 'CURRENT_TIME', 'OPEN', 'RANGE', 'CROSS', 'FUNCTION', 'TIME', 'BOTH', 'NOT', 'CONVERT', 'NCHAR', 'KEY', 'DEFAULT', 'LIKE', 'ANALYZE', 'EXISTS', 'IN', 'BIT', 'INOUT', 'SUM', 'NUMERIC', 'AFTER', 'LEAVE', 'INSERT', 'TO', 'COUNT', 'THEN', 'BEFORE', 'OUTER', 'COLUMN', 'ONLY', 'END', 'PROCEDURE', 'OFFSET', 'ADD', 'INNER', 'RELEASE', 'FROM', 'DAY', 'NO', 'CALL', 'BY', 'LOCAL', 'ZONE', 'TRUE', 'EXIT', 'LEADING', 'INTEGER', 'MERGE', 'OLD', 'AVG', 'MIN', 'SQL', 'LOOP', 'SIGNAL', 'REFERENCES', 'MINUTE', 'UNIQUE', 'GENERATED', 'ALL', 'MATCH', 'CASCADE', 'UNION', 'COMMENT', 'FETCH', 'UNDO', 'UPDATE', 'WHERE', 'ELSE', 'PARTITION', 'BIGINT', 'CHARACTER', 'CURRENT_TIMESTAMP', 'ALTER', 'INTERVAL', 'REVOKE', 'CONNECT', 'WITH', 'TIMESTAMP', 'GROUP', 'BEGIN', 'CURRENT', 'REGEXP', 'NATURAL', 'SOME', 'SQLEXCEPTION', 'MAX', 'SUBSTRING', 'OF', 'AND', 'REPLACE', 'IS'}
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_]')

def obtain_n_grams(sequence, max_n):
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_n_grams = []
    for n in range(1, max_n + 1):
        all_n_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_n_grams

def deduplicate_dicts(dict_list):
    seen = set()
    unique_dicts = []
    
    for d in dict_list:
        dict_tuple = frozenset(d.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)
    
    return unique_dicts

def calculate_substring_match_percentage(query, target):
    query = query.lower()
    target = target.lower()
    
    substrings = []
    for i in range(len(query)):
        for j in range(i + 1, len(query) + 1):
            substrings.append(query[i:j])
    max_matched_substring_len = max([len(substring) for substring in substrings if substring in target])
    return max_matched_substring_len/len(query)

def retrieve_question_related_db_values(hits, question):
    high_score_hits = []
    for idx, hit in enumerate(hits):
        table_name, column_name, c_id = hit["id"].split("-**-")
        score = calculate_substring_match_percentage(hit["contents"], question)
        if score > 0.85:
            high_score_hits.append(
                {
                    "table_dot_column_lower_case": f"{table_name}.{column_name}".lower(), 
                    "db_value": hit["contents"],
                    "score": score,
                    "index": idx,
                }
            )
    high_score_hits = sorted(high_score_hits, key=lambda x: (x["score"], len(x["db_value"]), x["index"]), reverse=True)
    high_score_hits = high_score_hits[:20] # remain top 20 db values
    
    relavant_db_values_dict = dict()
    for hit in high_score_hits:
        if hit["table_dot_column_lower_case"] in relavant_db_values_dict:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]].append(hit["db_value"])
        else:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]] = [hit["db_value"]]

    return relavant_db_values_dict

def obtain_pk_fk_column_idx(db_info):
    pk_fk_column_idx_list = []
    for primary_keys_idx in db_info["primary_keys"]:
        if isinstance(primary_keys_idx, int):
            pk_fk_column_idx_list.append(primary_keys_idx)
        elif isinstance(primary_keys_idx, list):
            pk_fk_column_idx_list.extend(primary_keys_idx)
    for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
        pk_fk_column_idx_list.append(source_column_idx)
        pk_fk_column_idx_list.append(target_column_idx)
    return pk_fk_column_idx_list


def needs_backticks(identifier):
    if identifier.upper() in SQL_RESERVED_WORDS:
        return True
    if SPECIAL_CHARS_PATTERN.search(identifier):
        return True
    return False


def format_identifier(identifier):
    if needs_backticks(identifier):
        return f'`{identifier}`'
    return identifier


def obtain_db_details(db_info, data_source, sampled_db_values_dict, relavant_db_values_dict, output_seq, mode, question):
    db_details = []
    assert len(db_info["column_names_original"]) == len(db_info["column_names"]) == len(db_info["column_types"])
    
    if mode == "train":
        '''
        to increase training data's diversity, the input database schema includes: 
        [PK, FK, output sequence-used columns, random sampled unused columns]
        '''
        # remain primary and foreign key columns
        used_column_idx_list = obtain_pk_fk_column_idx(db_info)
        # remain SQL-used columns
        for column_idx, (inner_table_idx, column_name) in enumerate(db_info["column_names_original"]):
            if column_name.lower() in output_seq.lower():
                used_column_idx_list.append(column_idx)
        
        used_column_idx_list = list(set(used_column_idx_list))
        used_column_num = len(used_column_idx_list)
        all_column_idx_list = list(range(len(db_info["column_names_original"])))
        unused_column_idx_list = [idx for idx in all_column_idx_list if idx not in used_column_idx_list]
        
        # random select some unused columns to mimic noise in the input sequence
        if unused_column_idx_list:
            unused_column_prob = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            sample_size = int(unused_column_prob * len(unused_column_idx_list))

            max_column_num = 225
            if used_column_num > max_column_num:
                sample_size = 0
            elif used_column_num + sample_size > max_column_num:
                sample_size = max_column_num - used_column_num
            else:
                sample_size = sample_size

            used_column_idx_list.extend(random.sample(unused_column_idx_list, sample_size))
    else:
        # put all tables and columns in the prompt
        used_column_idx_list = list(range(len(db_info["column_names_original"])))

    # print(used_column_idx_list)
    for outer_table_idx, table_name in enumerate(db_info["table_names_original"]):
        column_info_list = []
        pk_columns = []
        fk_info = []
        
        column_comment_prob = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        for column_idx, ((inner_table_idx, column_name), (_, column_comment), column_type) in enumerate(zip(
            db_info["column_names_original"], db_info["column_names"], db_info["column_types"]
        )):
            if inner_table_idx == outer_table_idx:
                if column_idx not in used_column_idx_list:
                    continue

                column_values = []
                if f"{table_name}.{column_name}".lower() in relavant_db_values_dict:
                    column_values.extend(relavant_db_values_dict[f"{table_name}.{column_name}".lower()])
                if f"{table_name}.{column_name}".lower() in sampled_db_values_dict:
                    column_values.extend(sampled_db_values_dict[f"{table_name}.{column_name}".lower()])
                column_values = list(dict.fromkeys(column_values)) # dedup (reserve order)
                column_values = column_values[:6]

                if data_source == "synthetic":
                    if random.random() < column_comment_prob:
                        column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                        if len(column_values) > 0:
                            column_info += f", example: {column_values}"
                    else: # simulate some columns do not have comment
                        column_info = f'    {format_identifier(column_name)} {column_type},'
                        if len(column_values) > 0:
                            column_info += f" -- example: {column_values}"
                else:
                    if column_name.lower() in [column_comment.lower(), column_comment.lower().replace(" ", "_"), column_comment.lower().replace(" ", "")] \
                        or column_comment.strip() == "":
                        column_info = f'    {format_identifier(column_name)} {column_type},'
                        if len(column_values) > 0:
                            column_info += f" -- example: {column_values}"
                    else:
                        column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                        if len(column_values) > 0:
                            column_info += f", example: {column_values}"
                
                column_info_list.append(column_info)
                
                for primary_keys_idx in db_info["primary_keys"]:
                    if isinstance(primary_keys_idx, int):
                        if column_idx == primary_keys_idx:
                            pk_columns.append(column_name) # f'    PRIMARY KEY ("{ }")'
                    elif isinstance(primary_keys_idx, list):
                        if column_idx in primary_keys_idx:
                            pk_columns.append(column_name)

                for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
                    if column_idx == source_column_idx:
                        source_table_idx = db_info["column_names_original"][source_column_idx][0]
                        source_table_name = db_info["table_names_original"][source_table_idx]
                        source_column_name = db_info["column_names_original"][source_column_idx][1]
                        target_table_idx = db_info["column_names_original"][target_column_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx]
                        target_column_name = db_info["column_names_original"][target_column_idx][1]
                        fk_info.append(f'    CONSTRAINT fk_{source_table_name.lower().replace(" ", "_")}_{source_column_name.lower().replace(" ", "_")} FOREIGN KEY ({format_identifier(source_column_name)}) REFERENCES {format_identifier(target_table_name)} ({format_identifier(target_column_name)}),')
                
        if len(column_info_list) > 0:
            pk_columns = list(OrderedDict.fromkeys(pk_columns))
            if len(pk_columns) > 0:
                pk_info = ['    PRIMARY KEY (' + ', '.join([f'{format_identifier(column_name)}' for column_name in pk_columns]) + '),']
            else:
                pk_info = []
            fk_info = list(OrderedDict.fromkeys(fk_info))

            table_ddl = ""
            table_ddl += f'CREATE TABLE {format_identifier(table_name)} (\n'
            table_ddl += "\n".join(column_info_list + pk_info + fk_info)
            if table_ddl.endswith(","):
                table_ddl = table_ddl[:-1] # remove extra commas
            table_ddl += "\n);"

            db_details.append(table_ddl)

    if mode == "train":
        random.shuffle(db_details)

    db_details = "\n\n".join(db_details)

    # double check
    for column_idx, (_, column_name) in enumerate(db_info["column_names_original"]):
        if column_name == "*":
            continue
        if column_idx in used_column_idx_list:
            assert column_name.lower() in db_details.lower()

    return db_details

def prepare_input_output_pairs(data, ek_key, db_id2relevant_hits, sampled_db_values_dict, db_info, source, output_key, mode, prompt=None, cot=False):
    if data[ek_key].strip() == "":
        question = data["question"]
    else:
        question = data[ek_key] + "\n" + data["question"]

    relavant_db_values_dict = dict()
    if db_id2relevant_hits: # retrieve matched values from the databases
        queries = obtain_n_grams(question, 8) + [question]
        queries = list(dict.fromkeys(queries))
        hits = []
        for query in queries:
            hits.extend(db_id2relevant_hits[data["db_id"]][query])
        hits = deduplicate_dicts(hits)
        relavant_db_values_dict = retrieve_question_related_db_values(hits, question)

    db_details = obtain_db_details(
        db_info, source, sampled_db_values_dict, relavant_db_values_dict, 
        data[output_key], mode, question
    )

    if prompt == None:
        # Use prompt manager to get template
        prompt_manager = get_prompt_manager()
        input_seq = prompt_manager.format_sql_prompt(
            db_engine="SQLite",
            db_details=db_details,
            question=question,
            cot=cot
        )
        return {"input_seq": input_seq, "output_seq": data[output_key]}
    else:
        input_seq = prompt.format(
            schema = db_details,
            external_knowledge=data[ek_key],
            sql_complexity=data['sql_complexity'],
            question = data["question"]
        )
        return {"input_seq": input_seq, "output_seq": data[output_key]}


def prepare_input_output_pairs_ra(
    data, ek_key, db_id2relevant_hits, sampled_db_values_dict, db_info, source, output_key, mode, prompt=None, cot=False
):
    # Build question with optional external knowledge
    if data[ek_key].strip() == "":
        question = data["question"]
    else:
        question = data[ek_key] + "\n" + data["question"]

    # Retrieve possibly relevant DB values (unchanged)
    relavant_db_values_dict = dict()
    if db_id2relevant_hits:
        queries = obtain_n_grams(question, 8) + [question]
        queries = list(dict.fromkeys(queries))
        hits = []
        for query in queries:
            hits.extend(db_id2relevant_hits[data["db_id"]][query])
        hits = deduplicate_dicts(hits)
        relavant_db_values_dict = retrieve_question_related_db_values(hits, question)

    # Assemble schema details for the prompt (unchanged)
    db_details = obtain_db_details(
        db_info, source, sampled_db_values_dict, relavant_db_values_dict,
        data[output_key], mode, question
    )

    if prompt is None:
        # Use prompt manager to get template
        prompt_manager = get_prompt_manager()
        input_seq = prompt_manager.format_ra_prompt(
            db_details=db_details,
            question=question,
            cot=cot,
            include_head=True
        )
        return {"input_seq": input_seq, "output_seq": data[output_key]}
    else:
        # Backward-compatible custom template support
        input_seq = prompt.format(
            schema=db_details,
            external_knowledge=data[ek_key],
            sql_complexity=data.get('sql_complexity', ''),
            question=data["question"]
        )
        return {"input_seq": input_seq, "output_seq": data[output_key]}


def get_db_schema(
    data, ek_key, db_id2relevant_hits, sampled_db_values_dict, db_info, source, output_key, mode, prompt=None
):
    # Build question with optional external knowledge
    if data[ek_key].strip() == "":
        question = data["question"]
    else:
        question = data[ek_key] + "\n" + data["question"]

    # Retrieve possibly relevant DB values (unchanged)
    relavant_db_values_dict = dict()
    if db_id2relevant_hits:
        queries = obtain_n_grams(question, 8) + [question]
        queries = list(dict.fromkeys(queries))
        hits = []
        for query in queries:
            hits.extend(db_id2relevant_hits[data["db_id"]][query])
        hits = deduplicate_dicts(hits)
        relavant_db_values_dict = retrieve_question_related_db_values(hits, question)

    # Assemble schema details for the prompt (unchanged)
    db_details = obtain_db_details(
        db_info, source, sampled_db_values_dict, relavant_db_values_dict,
        data[output_key], mode, question
    )

    return db_details

def get_input_seq(data, database_path, dataset_name, table_value_cache_path, table_info_cache_path, input_prompt=None, mode='dev', cot=False):

    if dataset_name == 'bird':
        output_key = "SQL"
        ek_key = "evidence"
    elif dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
        # Spider and Spider variants (including OmniSQL version)
        output_key = "query"
        ek_key = "external_knowledge"
        data[ek_key] = ""
    elif dataset_name in ['spider2', 'ehrsql', 'sciencebenchmark']:
        # OmniSQL benchmarks with BIRD-like format
        output_key = "sql"
        ek_key = "external_knowledge"
        data[ek_key] = data.get("evidence", "")  # Some may have evidence
    elif dataset_name == 'synsql':
        output_key = "sql"
        ek_key = "external_knowledge"
        prompt = input_prompt

    if dataset_name == 'spider-syn' and 'SpiderSynQuestion' in data:
        # Handle raw spider-syn format; processed format already has 'question' set
        data['question'] = data['SpiderSynQuestion']

    db_id2db_info = _load_json_cached(table_info_cache_path)
    db_id2sampled_db_values = _load_json_cached(table_value_cache_path)

    data["db_id"] = data["db_id"].replace('\n', '')

    input_seq_dict = prepare_input_output_pairs(
                    data=data, 
                    ek_key=ek_key, 
                    db_id2relevant_hits=None, 
                    sampled_db_values_dict=db_id2sampled_db_values[data["db_id"]], 
                    db_info=db_id2db_info[data["db_id"]], 
                    source=dataset_name, 
                    output_key=output_key, 
                    mode=mode,
                    prompt=prompt if dataset_name == 'synsql' else None,
                    cot=cot
                )
    
    input_seq = input_seq_dict['input_seq']

    return input_seq

def get_input_seq_ra(data, database_path, dataset_name, table_value_cache_path, table_info_cache_path, input_prompt=None, mode='dev', cot=False):

    if dataset_name == 'bird':
        output_key = "SQL"
        ek_key = "evidence"
    elif dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
        # Spider and Spider variants (including OmniSQL version)
        output_key = "query"
        ek_key = "external_knowledge"
        data[ek_key] = ""
    elif dataset_name in ['spider2', 'ehrsql', 'sciencebenchmark']:
        # OmniSQL benchmarks with BIRD-like format
        output_key = "sql"
        ek_key = "external_knowledge"
        data[ek_key] = data.get("evidence", "")  # Some may have evidence
    elif dataset_name == 'synsql':
        output_key = "sql"
        ek_key = "external_knowledge"
        prompt = input_prompt

    if dataset_name == 'spider-syn' and 'SpiderSynQuestion' in data:
        # Handle raw spider-syn format; processed format already has 'question' set
        data['question'] = data['SpiderSynQuestion']

    db_id2db_info = _load_json_cached(table_info_cache_path)
    db_id2sampled_db_values = _load_json_cached(table_value_cache_path)

    data["db_id"] = data["db_id"].replace('\n', '')

    input_seq_dict = prepare_input_output_pairs_ra(
                    data=data, 
                    ek_key=ek_key, 
                    db_id2relevant_hits=None, 
                    sampled_db_values_dict=db_id2sampled_db_values[data["db_id"]], 
                    db_info=db_id2db_info[data["db_id"]], 
                    source=dataset_name, 
                    output_key=output_key, 
                    mode=mode,
                    prompt=prompt if dataset_name == 'synsql' else None,
                    cot=cot
                )
    
    input_seq = input_seq_dict['input_seq']

    return input_seq


def get_input_seq_ra_sql(data, database_path, dataset_name, table_value_cache_path, table_info_cache_path, input_prompt=None, mode='dev', cot=False):
    """Generate prompt for RA+SQL mode where RA is used as thinking steps."""

    prompt = input_prompt  # Initialize prompt variable

    if dataset_name == 'bird':
        output_key = "SQL"
        ek_key = "evidence"
    elif dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
        # Spider and Spider variants (including OmniSQL version)
        output_key = "query"
        ek_key = "external_knowledge"
        data[ek_key] = ""
    elif dataset_name in ['spider2', 'ehrsql', 'sciencebenchmark']:
        # OmniSQL benchmarks with BIRD-like format
        output_key = "sql"
        ek_key = "external_knowledge"
        data[ek_key] = data.get("evidence", "")  # Some may have evidence
    elif dataset_name == 'synsql':
        output_key = "sql"
        ek_key = "external_knowledge"
        prompt = input_prompt

    if dataset_name == 'spider-syn' and 'SpiderSynQuestion' in data:
        # Handle raw spider-syn format; processed format already has 'question' set
        data['question'] = data['SpiderSynQuestion']

    db_id2db_info = _load_json_cached(table_info_cache_path)
    db_id2sampled_db_values = _load_json_cached(table_value_cache_path)

    data["db_id"] = data["db_id"].replace('\n', '')
    
    # Build question with optional external knowledge
    if data[ek_key].strip() == "":
        question = data["question"]
    else:
        question = data[ek_key] + "\n" + data["question"]

    # Retrieve possibly relevant DB values
    relavant_db_values_dict = dict()
    
    # Assemble schema details for the prompt
    db_details = obtain_db_details(
        db_id2db_info[data["db_id"]], dataset_name, 
        db_id2sampled_db_values[data["db_id"]], relavant_db_values_dict,
        data[output_key], mode, question
    )

    if prompt is None:
        # Use prompt manager to get template
        prompt_manager = get_prompt_manager()
        input_seq = prompt_manager.format_ra_sql_prompt(
            db_details=db_details,
            question=question,
            cot=cot
        )
        return input_seq
    else:
        # Backward-compatible custom template support
        input_seq = prompt.format(
            schema=db_details,
            external_knowledge=data[ek_key],
            sql_complexity=data.get('sql_complexity', ''),
            question=data["question"]
        )
        return input_seq


def get_db_schema_input(data, database_path, dataset_name, table_value_cache_path, table_info_cache_path, input_prompt=None, mode='dev'):

    if dataset_name == 'bird':
        output_key = "SQL"
        ek_key = "evidence"
    elif dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
        # Spider and Spider variants (including OmniSQL version)
        output_key = "query"
        ek_key = "external_knowledge"
        data[ek_key] = ""
    elif dataset_name in ['spider2', 'ehrsql', 'sciencebenchmark']:
        # OmniSQL benchmarks with BIRD-like format
        output_key = "sql"
        ek_key = "external_knowledge"
        data[ek_key] = data.get("evidence", "")  # Some may have evidence
    elif dataset_name == 'synsql':
        output_key = "sql"
        ek_key = "external_knowledge"
        prompt = input_prompt

    if dataset_name == 'spider-syn' and 'SpiderSynQuestion' in data:
        # Handle raw spider-syn format; processed format already has 'question' set
        data['question'] = data['SpiderSynQuestion']

    db_id2db_info = _load_json_cached(table_info_cache_path)
    db_id2sampled_db_values = _load_json_cached(table_value_cache_path)

    data["db_id"] = data["db_id"].replace('\n', '')
    db_schema = get_db_schema(
                    data=data, 
                    ek_key=ek_key, 
                    db_id2relevant_hits=None, 
                    sampled_db_values_dict=db_id2sampled_db_values[data["db_id"]], 
                    db_info=db_id2db_info[data["db_id"]], 
                    source=dataset_name, 
                    output_key=output_key, 
                    mode=mode,
                    prompt=prompt if dataset_name == 'synsql' else None
                )
    return db_schema

def sample_table_values(db_file_dir, table_names, limit_num):
    db_values_dict = dict()
    
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        for column_name in column_names:
            # cursor.execute(f"SELECT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {limit_num};")
            query = f"""
            SELECT `{column_name}` 
            FROM (
                SELECT DISTINCT `{column_name}` 
                FROM `{table_name}` 
                WHERE `{column_name}` IS NOT NULL and `{column_name}` != ''
            ) AS unique_values
            LIMIT {limit_num};
            """
            cursor.execute(query)
            values = cursor.fetchall()
            values = [value[0] for value in values]

            # truncate too long strings
            for idx in range(len(values)):
                if isinstance(values[idx], str):
                    values[idx] = values[idx][:40]

            if len(values) > 0:
                db_values_dict[f"{table_name}.{column_name}".lower()] = values
    
    cursor.close()
    conn.close()

    return db_values_dict

def add_id_as_key(db_file, output_path):
    """
    Add an 'id' key to each table entry in the JSON file.
    The 'id' is a combination of the table name and the column name.
    """
    with open(db_file, 'r') as f:
        data = json.load(f)

    new_data = {}

    for db in data:
        id = db['db_id']
        new_data[id] = db

    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)


def get_sampled_value_from_db(db_root_dir, id2db_info_path, limit_num, output_path):
    id2db_info = json.load(open(id2db_info_path, 'r'))
    db_sampled_value = {}
    for id, db_info in id2db_info.items():
        table_names = db_info['table_names']
        db_file_dir = os.path.join(db_root_dir, id, id + '.sqlite')
        sampled_values = sample_table_values(db_file_dir, table_names, limit_num)
        db_sampled_value[id] = sampled_values

    with open(output_path, 'w') as f:
        json.dump(db_sampled_value, f, indent=4)




def _strip_comments(s: str) -> str:
    # Remove //... and /* ... */ just in case
    return re.sub(r"(//[^\n]*|/\*.*?\*/)", "", s, flags=re.DOTALL)

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(?=\s*[\]}])", "", s)

def extract_json_from_response(response: str):
    # Prefer the fenced block inside <answer> if present
    m = re.search(
        r"<answer>.*?```(?:json|jsonc|js|javascript)?\s*(.+?)\s*```.*?</answer>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        # Fallback: any fenced block
        m = re.search(r"```(?:json|jsonc|js|javascript)?\s*(.+?)\s*```",
                      response, re.DOTALL | re.IGNORECASE)
    if not m:
        # Last resort: first balanced {...} or [...]
        m = re.search(r"(\{.*\}|\[.*\])", response, re.DOTALL)
    if not m:
        raise ValueError("No JSON-like block found.")

    raw = m.group(1).strip()
    raw = _strip_comments(raw)
    raw = _strip_trailing_commas(raw)

    # 1) Try strict JSON after a minimal normalization
    try:
        j = raw
        # normalize smart quotes -> straight
        j = j.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        # common tokens
        j = re.sub(r"\bTrue\b", "true", j)
        j = re.sub(r"\bFalse\b", "false", j)
        j = re.sub(r"\bNone\b", "null", j)
        j = re.sub(r"\bNaN\b", "null", j)
        j = re.sub(r"\b-?Infinity\b", "null", j)
        # most LLMs output single-quoted dicts; convert to double
        j = j.replace("'", '"')
        return json.loads(j)
    except Exception:
        pass

    # 2) Fall back to Python literal (works for single-quoted dicts like your example)
    try:
        p = raw
        p = p.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        p = re.sub(r"\btrue\b", "True", p)
        p = re.sub(r"\bfalse\b", "False", p)
        p = re.sub(r"\bnull\b", "None", p)
        return ast.literal_eval(p)
    except Exception as e:
        # Surface a helpful snippet
        snippet = raw[:300] + ("..." if len(raw) > 300 else "")
        raise ValueError(f"Could not parse block. Last error: {e}\nBlock snippet:\n{snippet}")


def build_relalg_prompt(schema: str, nl_question: str, gt_ra: str, hint=None) -> str:
    """
    Returns a single prompt string you can send to an LLM.

    Parameters
    ----------
    schema : str        # Full DDL or textual schema (tables, columns, PK/FK, types, constraints)
    nl_question : str   # Natural-language question to answer
    hint : str | None   # Optional extra cues (e.g., disambiguations, paraphrases, constraints)

    Output
    ------
    A prompt that asks the model to produce (1) brief, numbered reasoning steps
    and (2) a relational-algebra plan encoded in a strict JSON tree DSL compatible
    with downstream execution/validation.
    """
    # Use prompt manager to get and format template
    prompt_manager = get_prompt_manager()
    return prompt_manager.format_build_relalg_prompt(
        schema=schema,
        nl_question=nl_question,
        gt_ra=gt_ra,
        hint=hint
    )