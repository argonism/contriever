import copy
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from src.contriever import Contriever
from src.utils import save_pretrained
from tevatron.arguments import DataArguments, ModelArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.data import QPCollator, TrainDataset
from tevatron.datasets import HFTrainDataset
from tevatron.datasets.dataset import DEFAULT_PROCESSORS
from tevatron.datasets.preprocessor import (
    CorpusPreProcessor,
    QueryPreProcessor,
    TrainPreProcessor,
)
from tevatron.modeling import DenseModel
from tevatron.modeling.encoder import EncoderOutput
from tevatron.trainer import GCTrainer
from tevatron.trainer import TevatronTrainer as Trainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)

logger = logging.getLogger(__name__)

import torch
from torch import Tensor, nn


class TevatronContriever(Contriever):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.training = True
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def encode(self, inputs):
        return super(TevatronContriever, self).forward(**inputs)

    def save(self, output_dir: str) -> None:
        self.save_pretrained(output_dir)

    def forward(
        self,
        query: Optional[dict[str, Tensor]] = None,
        passage: Optional[dict[str, Tensor]] = None,
    ):
        # print("passage:", passage)
        # print("query:", query)
        q_reps, p_reps = None, None
        if query is not None:
            q_reps = self.encode(query)
        if passage is not None:
            p_reps = self.encode(passage)
        # print("q_reps:", q_reps)
        # print("p_reps:", p_reps)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )


class CustomHFTrainDataset(HFTrainDataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str
    ):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_language,
            data_files=data_files,
            cache_dir=cache_dir,
            use_auth_token=True,
        )[data_args.dataset_split]
        self.preprocessor = DEFAULT_PROCESSORS[0]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = TevatronContriever.from_pretrained(model_args.model_name_or_path)

    train_dataset = CustomHFTrainDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
