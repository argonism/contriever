import logging
import random
import re
from collections import defaultdict
from typing import Callable, Iterable, Optional

import ir_datasets
import pandas as pd
from pyterrier.transformer import TransformerBase

logger = logging.getLogger(__name__)


class NegativesMiner(object):
    def __init__(
        self,
        dataset: ir_datasets.Dataset,
        num_negatives: int = 30,
        tokenizer: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.num_negatives = num_negatives
        self.tokenizer = tokenizer

    def corpus_iter(self) -> Iterable[dict[str, str]]:
        for doc in self.dataset.docs_iter():
            text = doc.text
            if self.tokenizer is not None:
                text = self.tokenizer(text)
            yield {"docno": doc.doc_id, "text": text}

    def random_negatives(self, positive_ids: set) -> list[str]:
        if not hasattr(self, "doc_ids"):
            self.doc_ids = []
            for doc in self.dataset.docs_iter():
                self.doc_ids.append(doc.doc_id)
        random_docids = random.choices(
            self.doc_ids, k=(self.num_negatives + len(positive_ids))
        )
        random_docids = [docid for docid in random_docids if docid not in positive_ids]
        return random_docids[: self.num_negatives]

    def queries(self) -> pd.DataFrame:
        code = re.compile(
            "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
        )

        data: dict[str, list] = {"qid": [], "query": []}
        for query in self.dataset.queries_iter():
            text = query.text if self.tokenizer is None else self.tokenizer(query.text)
            data["qid"].append(query.query_id)
            data["query"].append(code.sub("", text))

        frame = pd.DataFrame.from_dict(data)
        return frame

    def qrels(self) -> dict[str, set]:
        qrels_table = defaultdict(set)
        for qrel in self.dataset.qrels_iter():
            qrels_table[qrel.query_id].add(qrel.doc_id)
        return qrels_table

    def index(
        self, indexer: TransformerBase, custum_corpus_iter: Optional[Callable] = None
    ) -> str:
        corpus_iter = (
            self.corpus_iter() if custum_corpus_iter is None else custum_corpus_iter()
        )
        return indexer.index(corpus_iter)

    def mine(self, retriever: TransformerBase, fill: bool = True) -> dict[str, list]:
        queries = self.queries()
        qrels = self.qrels()

        result_df = retriever.transform(queries)

        result: defaultdict = defaultdict(dict)
        qids = set()
        for _, row in result_df.iterrows():
            qid = row["qid"]
            qids.add(int(qid))
            doc_id = row["docno"]
            if not row["docno"] in qrels[qid]:
                result[qid][doc_id] = row["score"]
        # print(sorted(list(qids)))

        sorted_result: dict[str, list] = {}
        for _, row in queries.iterrows():
            qid = row["qid"]
            if len(result[qid]) > 0:
                sorted_result[qid] = sorted(
                    result[qid].keys(), key=lambda x: result[qid][x], reverse=True
                )[: self.num_negatives]
            else:
                sorted_result[qid] = self.random_negatives(qrels[qid])

        return sorted_result
