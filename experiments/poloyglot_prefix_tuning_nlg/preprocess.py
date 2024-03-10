import os
import pandas as pd
from args import Args
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset


process_dir = "processed_data"


def execute_clone_repository():
    os.system('git clone https://github.com/songys/Chatbot_data')


def custom_train_test_split():
    df = pd.read_excel("comments.xlsx")

    df = df[["댓글", "대댓글"]].rename(
        columns={
            "댓글": "source",
            "대댓글": "target"
        }
    )[:20000]

    df_train, df_test = train_test_split(
        df,
        test_size=0.1
    )
    df_train, df_valid = train_test_split(
        df,
        test_size=0.1
    )

    
    if not os.path.exists(process_dir):
        os.mkdir(process_dir)
    
    df_train.to_csv(process_dir + "_train.tsv", sep="\t", index=None)
    df_valid.to_csv(process_dir + "_valid.tsv", sep="\t", index=None)
    df_test.to_csv(process_dir + "_test.tsv", sep="\t", index=None)


def batch_tokenize_preprocess_decoder(batch, tokenizer, max_length):
    source, target = batch["source"], batch["target"]

    # For GPT-2
    # input_sents = ['<|startoftext|>'+  s + '<|endoftext|><|startoftext|>' + t + '<|endoftext|>' for s,t in zip(source, target)]
    # input_sents = ['<s>'+  s + ' [A] ' + t + '</s>' for s, t in zip(source, target)]
    input_sents = [tokenizer.bos_token +  s + tokenizer.eos_token+tokenizer.bos_token + t + tokenizer.eos_token for s, t in zip(source, target)]
    
    tokenized = tokenizer(
        input_sents, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        add_special_tokens=True
    )
    
    # batch = {"input_ids": tokenized["input_ids"]}
    batch = tokenized
    batch["source"] = source
    batch["target"] = target
    return batch


def dataset(tokenizer, max_length=128):
    df_train = pd.read_csv(process_dir + "_train.tsv", sep="\t")
    df_valid = pd.read_csv(process_dir + "_valid.tsv", sep="\t")

    ds_train = Dataset.from_pandas(df_train)
    ds_valid = Dataset.from_pandas(df_valid)

    ds_train = ds_train.map(
        lambda batch: batch_tokenize_preprocess_decoder(
            batch=batch,
            tokenizer=tokenizer,
            max_length=max_length
        ),
        batched=True
    )

    ds_valid = ds_valid.map(
        lambda batch: batch_tokenize_preprocess_decoder(
            batch=batch,
            tokenizer=tokenizer,
            max_length=max_length
        ),
        batched=True
    )

    return ds_train, ds_valid


def get_dataset():
    # Github repository clone
    # if not os.path.exists("Chatbot_data"):
        # execute_clone_repository()
    
    # split train & test
    custom_train_test_split()

    args = Args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_checkpoint,
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
    return dataset(
        tokenizer=tokenizer
    )
