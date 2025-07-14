import os
import csv
import gc
import torch
from transformers import Trainer, TrainingArguments, LlamaConfig, LlamaForCausalLM
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
from huggingface_hub import login
import sentencepiece

#CSVtoLlamaModel By Deccatron

print(sentencepiece.__version__)

csv.field_size_limit(2**20)  

# Login to Hugging Face
login(token="EDIT KEY", add_to_git_credential=True)

def create_corpus_from_csv(csv_file, output_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        with open(output_file, mode='w') as text_file:
            for row in reader:
                word = row['word']
                definition = row['def']
                text_file.write(f"{word}: {definition}\n")

create_corpus_from_csv('english Dictionary.csv', 'corpus.txt')

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.train(files=["corpus.txt"], trainer=trainer)

os.makedirs("llama_tokenizer", exist_ok=True)
tokenizer.save("llama_tokenizer/tokenizer.json")

tokenizer = PreTrainedTokenizerFast.from_pretrained("llama_tokenizer")

special_tokens_dict = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
}

num_added_toks = tokenizer.add_tokens(list(special_tokens_dict.values()))

tokenizer.pad_token = "<pad>"

dataset = load_dataset("text", data_files={"train": "corpus.txt"})

def tokenize_function(examples):
    encodings = tokenizer.batch_encode_plus(
        examples["text"],
        truncation=True,
        padding="max_length", 
        max_length=80,         
        return_tensors='pt'   
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone(), 
    }

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Model config stuff
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512, 
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
)

model = LlamaForCausalLM(config)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator, 
)

trainer.train()

# Save the trained model
trainer.save_model("trained_llama_model")

# Clean up memory
gc.collect()
torch.cuda.empty_cache()
