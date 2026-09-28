"""Дообучение bi-encoder на парах (запрос → выбранное объявление) из A с in-batch негативами (MNRL)."""
import pickle, sys, time, numpy as np, pandas as pd, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
from cg.config import ART, DENSE_MODEL, SEED
from cg.pipeline import dense_item_text, dense_query_text
t0 = time.time()
n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
A = pickle.load(open(ART/"val_split.pkl", "rb"))["A"]
pairs = pd.DataFrame({"anchor": "query: " + dense_query_text(A), "positive": "passage: " + dense_item_text(A)}).drop_duplicates()
pairs = pairs.sample(min(n_pairs, len(pairs)), random_state=SEED)
print("pairs", len(pairs), flush=True)
model = SentenceTransformer(DENSE_MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
model.max_seq_length = 128
args = SentenceTransformerTrainingArguments(output_dir=str(ART/"ft_tmp"), num_train_epochs=1, per_device_train_batch_size=64,
    learning_rate=3e-5, warmup_ratio=0.05, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=200, save_strategy="no",
    report_to="none", dataloader_drop_last=True, seed=SEED, fp16=False, bf16=False)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_pandas(pairs, preserve_index=False),
                                     loss=losses.MultipleNegativesRankingLoss(model))
trainer.train()
out = ART/"models/e5s_ft_v1"; model.save(str(out))
print("saved", out, round(time.time()-t0), flush=True)
