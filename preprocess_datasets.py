import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

import ir_datasets
import pyterrier as pt
from negatives_miner import NegativesMiner
from pyterrier.transformer import TransformerBase
from sudachipy import dictionary, tokenizer
from tqdm import tqdm

if not pt.started():
    pt.init()

logger = logging.getLogger(__name__)


def load_query_table(dataset: ir_datasets.Dataset) -> dict[str, str]:
    logger.info("loading queries ...")
    queries_store = {}
    for query in dataset.queries_iter():
        queries_store[query.query_id] = query.text
    logger.info(f"queries num: {len(queries_store)}")
    return queries_store


def gen_tokenize_func():
    sudachi_tokenizer = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A

    def tokenize_text(text):
        atok = " ".join([m.surface() for m in sudachi_tokenizer.tokenize(text, mode)])
        return atok

    return tokenize_text


def load_contriever(
    index_path: Path, model_path: str = "facebook/mcontriever-msmarco"
) -> TransformerBase:
    contriever_loader_path = Path(__file__).parent.joinpath("../..")
    sys.path.append(str(contriever_loader_path))

    from models.pt_contriever import ContrieverIndexer, ContrieverRetrieval

    index_path = index_path.joinpath(model_path.split("/")[1])

    return (
        ContrieverIndexer(index_path, model_path=model_path, verbose=False),
        ContrieverRetrieval(index_path, model_path=model_path),
    )


def load_ntcir17_dataset(set_name: str):
    contriever_loader_path = Path(__file__).parent.joinpath("../../")
    sys.path.append(str(contriever_loader_path))
    import ntcir_datasets.ntcir_transfer

    dataset_key = f"ntcir-transfer/1/{set_name}"

    logger.info(f"loading dataset: {dataset_key}")
    dataset = ir_datasets.load(dataset_key)
    return dataset


def mine_negatives(
    dataset: ir_datasets.Dataset,
    indexer: TransformerBase,
    retriever: TransformerBase,
    num_negatives: int = 30,
) -> dict[str, list]:
    logger.info(f"mine negatives from {indexer.__class__}, {retriever.__class__}")
    negative_miner = NegativesMiner(
        dataset, num_negatives=num_negatives, tokenizer=gen_tokenize_func()
    )
    logger.info("indexing ...")
    negative_miner.index(indexer)
    logger.info("mining negatives ...")
    negatives = negative_miner.mine(retriever)
    return negatives


def mine_negatives_from_bm25(
    dataset: ir_datasets.Dataset, index_path: Path, num_negatives: int = 30
):
    negative_miner = NegativesMiner(
        dataset, num_negatives=num_negatives, tokenizer=gen_tokenize_func()
    )
    if not index_path.joinpath("data.properties").exists():
        logger.info("Index does not existed. Indexing...")
        indexer = pt.IterDictIndexer(
            str(index_path), threads=16, blocks=True, overwrite=True
        )
        indexer.setProperty("tokeniser", "UTFTokeniser")
        indexer.setProperty("termpipelines", "")
        negative_miner.index(indexer)
    else:
        logger.info(f"loading index from {index_path}")

    logger.info("mining negatives...")
    negatives = negative_miner.mine(
        pt.BatchRetrieve(str(index_path), num_results=num_negatives * 2, wmodel="BM25")
    )
    return negatives


def build_negatives_from_qrels(dataset: ir_datasets.Dataset) -> dict[str, list]:
    negatives = defaultdict(list)
    for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
        if qrel.relevance > 0:
            continue
        negatives[qrel.query_id].append(qrel.doc_id)
    return negatives


def load_dataset(
    dataset_key: str,
    language: str,
    set_name: str,
) -> ir_datasets.Dataset:
    irds_dataset_split_label = f"{dataset_key}/{language}/{set_name}"

    logger.info(f"loading dataset from {irds_dataset_split_label}")
    dataset = ir_datasets.load(irds_dataset_split_label)
    return dataset


def create_example(
    query,
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    for_tevatron: bool = False,
    query_id: Optional[str] = None,
) -> dict:
    if for_tevatron:
        if query_id is None:
            raise Exception("query_id is required to generate dataset for tevatron")
        example = {
            "query_id": query_id,
            "query": query,
            "positive_passages": positives,
            "negative_passages": negatives,
        }
    else:
        example = {
            "question": query,
            "positive_ctxs": positives,
            "negative_ctxs": negatives,
        }
    return example


def preprocess_finetuning_dataset(
    dataset: ir_datasets.Dataset,
    output_path: Path,
    index_path: Path,
    num_negatives: int = 10,
    tevatron: bool = False,
) -> None:
    def gen_docs_to_ctxs(docs: list[NamedTuple]) -> list[dict[str, str]]:
        negative_ctxs = []
        for doc in docs:
            if hasattr(doc, "text") and hasattr(doc, "doc_id"):
                negative = {"text": doc.text, "docid": doc.doc_id}
            else:
                raise Exception(
                    f"given doc {doc} is expected to have attribute 'text' and 'doc_id'"
                )
            if hasattr(doc, "title"):
                negative["title"] = doc.title
            negative_ctxs.append(negative)
        return negative_ctxs

    docstore = dataset.docs_store()
    queries_store = load_query_table(dataset)

    indexer, retriever = load_contriever(index_path)
    negatives = mine_negatives(dataset, indexer, retriever, num_negatives=num_negatives)
    # negatives = mine_negatives_from_bm25(
    #     dataset, index_path, num_negatives=num_negatives
    # )
    # negatives = build_negatives_from_qrels(dataset)

    logger.info(f"writing out train triples to {output_path} ...")
    write_count = 0
    with output_path.open("w") as fw:
        for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
            query_id = qrel.query_id
            if qrel.relevance <= 0:
                continue
            query = queries_store[query_id]
            try:
                positive_doc = docstore.get(qrel.doc_id)
            except KeyError:
                logger.warning(
                    f"positive document (id: {qrel.doc_id}) does not found. skipped."
                )
                continue

            negative_docids = negatives[query_id][:num_negatives]
            negative_docs = docstore.get_many(negative_docids).values()
            negative_ctxs = gen_docs_to_ctxs(negative_docs)

            positive_ctxs = gen_docs_to_ctxs([positive_doc])
            example = create_example(
                query,
                positive_ctxs,
                negative_ctxs,
                for_tevatron=tevatron,
                query_id=query_id,
            )

            jsonl = json.dumps(example, ensure_ascii=False)
            fw.write(jsonl + "\n")
            write_count += 1

    print(write_count)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="preprocess hf datasets")
    parser.add_argument(
        "--dataset_key",
        default="mr-tydi",
        help="dataset key string for huggingface datasets",
    )
    parser.add_argument("--lang", default="ja", help="dataset laungage")
    parser.add_argument("--set_name", default="train", help="dataset split label")
    parser.add_argument(
        "--num_negatives", default=10, type=int, help="dataset split label"
    )
    parser.add_argument(
        "--tevatron",
        action="store_true",
        default=False,
        help="if specified, generate dataset with tevatron format",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    dataset_key = args.dataset_key
    language = args.lang
    set_name = args.set_name

    logger.info(args)

    # dataset = load_dataset(dataset_key, language, set_name)
    dataset = load_ntcir17_dataset(set_name)
    dataset_key = "ntcir-transfer"

    output_dir = Path(__file__).parent.joinpath("dataset")
    output_dir = output_dir.joinpath(dataset_key.replace("/", "_"), language)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir.joinpath(f"{set_name}.jsonl")
    index_path = Path(__file__).parent.joinpath("index", f"{dataset_key}/{language}")
    preprocess_finetuning_dataset(
        dataset,
        output_path,
        index_path,
        num_negatives=args.num_negatives,
        tevatron=args.tevatron,
    )
