from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

df = pd.read_csv("news_summary.csv")  # columns: text, summary
dataset = Dataset.from_pandas(df)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="summarizer_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train()
model.save_pretrained("models/summarizer_model")
tokenizer.save_pretrained("models/summarizer_model")
