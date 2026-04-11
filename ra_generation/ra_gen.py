import argparse
import json
import multiprocessing
import traceback
from pathlib import Path

import jpype
from jpype.types import JException
import sqlglot
from sqlglot import exp
from tqdm import tqdm

DEFAULT_MODEL_TEMPLATE = Path("./model.json")
DATASET_CONFIGS = {
    "spider_train": {
        "data_path": Path("./data/spider/train_spider.json"),
        "database_folder": Path("./data/spider/database"),
        "sql_key": "query",
        "output_correct": Path("./data/spider/train_spider_ra_correct.json"),
        "output_error": Path("./data/spider/train_spider_ra_error.json"),
    },
    "spider_dev": {
        "data_path": Path("./data/spider/dev.json"),
        "database_folder": Path("./data/spider/database"),
        "sql_key": "query",
        "output_correct": Path("./data/spider/dev_spider_ra_correct.json"),
        "output_error": Path("./data/spider/dev_spider_ra_error.json"),
    },
    "bird_train": {
        "data_path": Path("./data/bird/train/train.json"),
        "database_folder": Path("./data/bird/train/train_databases"),
        "sql_key": "SQL",
        "output_correct": Path("./data/bird/train/train_bird_ra_correct.json"),
        "output_error": Path("./data/bird/train/train_bird_ra_error.json"),
    },
    "bird_dev": {
        "data_path": Path("./data/bird/dev/dev.json"),
        "database_folder": Path("./data/bird/dev/dev_databases"),
        "sql_key": "SQL",
        "output_correct": Path("./data/bird/dev/dev_bird_ra_correct.json"),
        "output_error": Path("./data/bird/dev/dev_bird_ra_error.json"),
    },
    "synsql": {
        "data_path": Path("./data/synsql/data.json"),
        "database_folder": Path("./data/synsql/databases"),
        "sql_key": "sql",
        "parallel": True,
        "num_chunks": 40,
        "output_correct": Path("./data/synsql/synsql_ra_correct.json"),
        "output_error": Path("./data/synsql/synsql_ra_error.json"),
    },
}

PLAN_TO_JSON_CLASS = None


def get_plan_to_json_class():
    global PLAN_TO_JSON_CLASS
    if PLAN_TO_JSON_CLASS is None:
        if not jpype.isJVMStarted():
            raise RuntimeError("JVM must be started before accessing PlanToJson.")
        PLAN_TO_JSON_CLASS = jpype.JClass("myapp.PlanToJson")
    return PLAN_TO_JSON_CLASS


def load_json(path: Path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def prepare_model_files(data, base_model, database_folder, model_filename="model.json"):
    """
    Prepare a model template for each database in the dataset.

    Args:
        data (list): The dataset containing database information.
        base_model (dict): The base model template.
        database_folder (str): The folder where database files are located.
        model_filename (str): The name of the model file to save. Default is "model.json".
    """
    db_root = Path(database_folder)
    model_filename = Path(model_filename)

    for item in tqdm(data, desc="Creating model files"):
        db = item["db_id"]
        db_path = (db_root / db).resolve()
        base_model["schemas"][0]["jdbcUrl"] = f"jdbc:sqlite:{db_path}/{db}.sqlite"
        # Save the model under the database folder
        with open(db_path / model_filename, "w") as f:
            json.dump(base_model, f, indent=4)


_COMPARISON_PARENT_NAMES = {
    "EQ",
    "NEQ",
    "GT",
    "GTE",
    "LT",
    "LTE",
    "IN",
    "LIKE",
    "ILIKE",
    "REGEXLIKE",
    "BETWEEN",
    "IS",
    "NOT",
}


def _is_value_context_by_name(parent: exp.Expression | None) -> bool:
    if parent is None:
        return False
    # Be robust across sqlglot versions (Eq vs EQ, etc.)
    return type(parent).__name__.upper() in _COMPARISON_PARENT_NAMES


def convert_values(n: exp._Expression):
    # This is the main logic fix:
    # We check for a Column node directly.
    if isinstance(n, exp.Column):
        # print(f"Found Column: {n} | Its type: {type(n)}")
        # print(f"Accessing n.this: {n.this} | Its type: {type(n.this)}")

        # print(f"Parent Node: {n.parent} | Its type: {type(n.parent)}")

        # Check if it contains a quoted identifier AND its parent is a comparison operator.
        identifier = n.this
        if (
            isinstance(identifier, exp.Identifier)
            and identifier.quoted
            and _is_value_context_by_name(n.parent)
        ):
            # If so, convert the whole Column expression into a string Literal.
            return exp.Literal.string(identifier.this)

    # Canonicalize any existing string literals to single-quoted form.
    # This is useful if the input SQL already had 'value1'.
    if isinstance(n, exp.Literal) and n.is_string:
        return exp.Literal.string(n.this)

    return n


# The limitation of the following approach is that it assumes columns are not quoted.
# If a column name is quoted (e.g., "col1"), it will be treated as a string literal.
def normalize_for_calcite(sql: str) -> str:
    """
    1) Parse as SQLite (accepts double-quoted tokens as identifiers).
    2) Convert double-quoted Identifiers wrapped in Columns (used as values) into string Literals.
    3) Render with single-quoted strings and unquoted identifiers.
    """
    # Parse with SQLite rules, so "x" becomes an Identifier
    tree = sqlglot.parse_one(sql, read="sqlite")

    # Transform the tree with our corrected logic
    tree = tree.transform(convert_values, copy=False)

    # Render: strings will be single-quoted by default; disable identifier quoting
    return tree.sql(dialect="sqlite", identify=False)


def generate_relational_algebra(
    data,
    database_folder,
    sql_key,
    db_key="db_id",
    model_filename="model.json",
    chunk_id=1,
):
    """
    Process queries in the dataset and generate relational algebra.

    Args:
        data (list): The dataset containing database information and queries.
        database_folder (str): The folder where database files are located.
        model_filename (str): The name of the model file. Default is "model.json".
        chunk_id (int): The chunk ID for processing in parallel. Default is 1.
    Returns:
        tuple: A tuple containing lists of correct items and error items.
    """
    plan_to_json = get_plan_to_json_class()

    error_items = []
    correct_items = []

    if chunk_id == 1:
        # print tqdm progress bar only for chunk 1; in the case of sinlge chunk, always show progress bar; in the case of multiple chunks, only chunk 1 shows progress bar
        iterator = tqdm(data, desc="Processing queries")
    else:
        iterator = data

    for item in iterator:
        db = item[db_key]
        sql = item[sql_key]

        model_path = str(Path(f"{database_folder}/{db}/{model_filename}").resolve())

        try:
            sql = normalize_for_calcite(sql)
            result = str(plan_to_json.runOnce(model_path, sql))
            result = json.loads(result)
            item["relational_algebra"] = result["relational_algebra"]
            correct_items.append(item)

        except JException as ex:  # catches Java exceptions raised via JPype
            result = f"Java exception: {ex}"
            item["relational_algebra_error"] = result
            error_items.append(item)
        except Exception as e:  # any other Python-side issues
            result = f"Python exception: {e}"
            item["relational_algebra_error"] = result
            error_items.append(item)

    return correct_items, error_items


def process_chunk_safe(
    chunk, chunk_id, database_folder, sql_key, db_key, model_filename
):
    """Process a single chunk and capture exceptions so the pool can continue."""
    try:
        return generate_relational_algebra(
            chunk,
            database_folder=database_folder,
            sql_key=sql_key,
            db_key=db_key,
            model_filename=model_filename,
            chunk_id=chunk_id,
        )
    except Exception as e:
        error_msg = f"[Chunk {chunk_id}] Error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return ([], [{"chunk_id": chunk_id, "error": str(e)}])


# def run_process_chunk_safe(args):
#     """Wrapper for multiprocessing to unpack tuple arguments."""
#     return process_chunk_safe(*args)


def run_process_chunk_safe(args):
    """
    Each subprocess starts and stops its own JVM instance.
    """
    jar_path = str(
        Path(__file__).resolve().parent / "calcite-plan/target/calcite-plan-1.0-SNAPSHOT.jar"
    )

    try:
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jar_path])
            print(
                f"[{multiprocessing.current_process().name}] JVM started:",
                jpype.isJVMStarted(),
            )

        result = process_chunk_safe(*args)

    except Exception as e:
        print(f"[Process {multiprocessing.current_process().name}] error: {e}")
        result = ([], [{"error": str(e)}])

    finally:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

    return result


def parallel_generate(
    data,
    *,
    database_folder,
    sql_key,
    db_key="db_id",
    model_filename="model.json",
    num_chunks=40,
):
    if not data:
        return [], []

    num_chunks = max(1, num_chunks)
    num_processes = min(num_chunks, multiprocessing.cpu_count())

    chunk_size = max(1, len(data) // num_chunks + 1)
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    tasks = [
        (chunk, idx + 1, database_folder, sql_key, db_key, model_filename)
        for idx, chunk in enumerate(chunks)
    ]

    print(f"Starting {num_processes} processes across {len(tasks)} chunk(s)...")

    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for r in tqdm(
            pool.imap_unordered(run_process_chunk_safe, tasks),
            total=len(tasks),
            desc="Processing chunks",
            dynamic_ncols=True,
        ):
            results.append(r)

    if not results:
        return [], []

    correct_batches, error_batches = zip(*results)
    correct = [item for sublist in correct_batches for item in sublist]
    errors = [item for sublist in error_batches for item in sublist]

    print(f"Done. Success: {len(correct)} | Errors: {len(errors)}")
    return correct, errors


def process_dataset(dataset_name, config, generate_models):
    print(f"\nProcessing dataset: {dataset_name}")
    data = load_json(config["data_path"])
    db_folder = Path(config["database_folder"])
    model_filename = config.get("model_filename", "model.json")
    db_key = config.get("db_key", "db_id")

    if generate_models:
        model_template_path = config.get("model_template", DEFAULT_MODEL_TEMPLATE)
        base_model = load_json(model_template_path)
        prepare_model_files(
            data,
            base_model=base_model,
            database_folder=str(db_folder),
            model_filename=model_filename,
        )

    generation_kwargs = {
        "database_folder": str(db_folder),
        "sql_key": config["sql_key"],
        "db_key": db_key,
        "model_filename": model_filename,
    }

    if config.get("parallel"):
        correct, errors = parallel_generate(
            data,
            num_chunks=config.get("num_chunks", multiprocessing.cpu_count()),
            **generation_kwargs,
        )
    else:
        correct, errors = generate_relational_algebra(data, **generation_kwargs)

    if config.get("output_correct"):
        dump_json(config["output_correct"], correct)
    if config.get("output_error"):
        dump_json(config["output_error"], errors)

    print(f"{dataset_name}: {len(correct)} success | {len(errors)} errors")
    return correct, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate relational algebra for the selected datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS.keys()),
        required=True,
        help="List of datasets to process.",
    )
    parser.add_argument(
        "--generate-models",
        action="store_true",
        help="Generate per-database model.json files before processing queries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    jar_path = str(
        Path(__file__).resolve().parent / "calcite-plan/target/calcite-plan-1.0-SNAPSHOT.jar"
    )
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[jar_path])

    try:
        for dataset_name in args.datasets:
            config = DATASET_CONFIGS[dataset_name]
            process_dataset(dataset_name, config, args.generate_models)
    finally:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()


if __name__ == "__main__":
    main()
