import json
import random
from tqdm import tqdm


with open(r'D:\external\LLM\data\data\tamil_data.txt', "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if len(line.strip()) > 10]

def generate_question(sentence):
    words = sentence.split()
    main_word = words[0] if words else "இது"
    question_templates = [
        "{} பற்றி என்ன தெரியும்?",
        "{} யார்?",
        "{} எங்கு உள்ளது?",
        "{} எப்போது நடைபெற்றது?",
        "{} முக்கியத்துவம் என்ன?",
        "{} யாரால் நிகழ்ந்தது?",
        "{} என்பதன் விளக்கம் என்ன?",
        "{} எதற்காக பிரபலமானது?",
        "{} பற்றிய தகவல்கள் என்ன?",
        "{} எப்போது ஆரம்பமானது?"
    ]
    template = random.choice(question_templates)
    question = template.format(main_word)
    return question
dataset = []

for _ in tqdm(range(1000)):
    sentence = random.choice(lines)
    question = generate_question(sentence)

    entry = {
        "instruction": "Generate a question in Tamil based on the given text.",
        "input": sentence,
        "output": question
    }
    dataset.append(entry)

output_path = "tamil_question_generation_dataset.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(dataset, json_file, ensure_ascii=False, indent=4)
print(f"Generated {len(dataset)} samples — Saved at: {output_path}")
import sentencepiece as spm
input_file = r'D:\external\LLM\data\data\tamil_data.txt'
model_prefix = "tamil_spm"
vocab_size = 300
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.9995,
    model_type='bpe'
)
sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
from indicnlp.tokenize import indic_tokenize
from indicnlp import common
from indicnlp.morph import unsupervised_morph
INDIC_NLP_RESOURCES = '/content/indic_nlp_resources'
common.set_resources_path(INDIC_NLP_RESOURCES)
morph_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('ta')
def extract_main_word(sentence):
    tokens = indic_tokenize.trivial_tokenize(sentence)
    morphs = morph_analyzer.morph_analyze_document(tokens)
    for word, morph in zip(tokens, morphs):
        if 'Noun' in morph:
            return word
    return tokens[0] if tokens else "இது"
for _ in tqdm(range(1000)):
    sentence = random.choice(lines)
    main_word = extract_main_word(sentence)
    question = random.choice(question_templates).format(main_word)
    dataset.append({
        "instruction": "Generate a question in Tamil based on the given text.",
        "input": sentence,
        "output": question
    })
with open("tamil_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
from datasets import Dataset
hf_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in dataset],
    "input": [d["input"] for d in dataset],
    "output": [d["output"] for d in dataset]
})
hf_dataset.save_to_disk("tamil_qa_dataset_hf")
hf_dataset.push_to_hub('saivimenthan/tamil-qa-dataset')
# Decoder-Only Model Setup
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import json

config = {
    "model_type": "gpt2",
}

with open("config.json", "w") as f:
    json.dump(config, f)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Create a new BPE tokenizer
bpe_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
bpe_tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer from the existing SentencePiece model
trainer = BpeTrainer(special_tokens=["<s>", "</s>", "<pad>", "<unk>"], vocab_size=300)
bpe_tokenizer.train_from_iterator(iter(lines), trainer=trainer)

# Save the tokenizer to tokenizer.json
bpe_tokenizer.save("tokenizer.json")

# Load the tokenizer using PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)
# Create model config
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
model = GPT2LMHeadModel(config)
dataset = load_dataset("json", data_files="tamil_qa_dataset.json")
# Tokenization function
def tokenize(batch):
    # Concatenate instruction, input, and output
    text = [f"{i} {j} {k}" for i, j, k in zip(batch['instruction'], batch['input'], batch['output'])]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])
# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
# Training configuration
training_args = TrainingArguments(
    output_dir="./decoder_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()
