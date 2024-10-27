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

#CSVtoLlamaModel Developed By Deccatron

print(sentencepiece.__version__)

# Increase the field size limit for the CSV reader
csv.field_size_limit(2**20)  # Setting the limit to 1MB (1048576 bytes)

# Login to Hugging Face
login(token="EDIT KEY", add_to_git_credential=True)

# Convert CSV into a plain text corpus
def create_corpus_from_csv(csv_file, output_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        with open(output_file, mode='w') as text_file:
            for row in reader:
                word = row['word']
                definition = row['def']
                text_file.write(f"{word}: {definition}\n")

create_corpus_from_csv('english Dictionary.csv', 'corpus.txt')

# Initialize a Byte-Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize the trainer for the tokenizer
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

# Train the tokenizer on your corpus
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Save the tokenizer in the directory
os.makedirs("llama_tokenizer", exist_ok=True)
tokenizer.save("llama_tokenizer/tokenizer.json")

# Load the tokenizer from the directory
tokenizer = PreTrainedTokenizerFast.from_pretrained("llama_tokenizer")

# Add special tokens to the tokenizer after loading
special_tokens_dict = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
}

# Add special tokens to the tokenizer
num_added_toks = tokenizer.add_tokens(list(special_tokens_dict.values()))

# Set the pad token
tokenizer.pad_token = "<pad>"

# Load the corpus
dataset = load_dataset("text", data_files={"train": "corpus.txt"})

# Tokenization function
def tokenize_function(examples):
    encodings = tokenizer.batch_encode_plus(
        examples["text"],
        truncation=True,
        padding="max_length",  # Pad to maximum length
        max_length=80,         # Set a maximum length
        return_tensors='pt'   # Optional: to return tensors (if needed)
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone(),  # Add labels for loss computation
    }

# Apply the tokenization function
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Model Configuration
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,  # You can modify this as needed
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
)

# Initialize the model
model = LlamaForCausalLM(config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,  # Use the data collator here
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("trained_llama_model")

# Clean up memory
gc.collect()
torch.cuda.empty_cache()
