import os
import torch
from args import Args
from preprocess import get_dataset
from prefix_gptneox_model import PrefixGPTNeoXLMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def get_trainer(model, ds_train, ds_valid, rank):
    # Train Params
    training_args = TrainingArguments(
        run_name = "prefix_gptneox_chatbot",

        ## Steps/Epochs
        num_train_epochs = 3,

        ## LR
        learning_rate = 5e-5,

        ## Batch
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 1,

        ## ETC
        # label_smoothing_factor = config["label_smoothing_factor"],

        # Checkpointing, Saving
        output_dir = os.path.join("weights", "checkpoints"),
        save_strategy = "steps", # steps, epoch
        save_steps = 80,
        save_total_limit = 1,
        load_best_model_at_end = True,
        overwrite_output_dir=True,

        # Evaluating
        evaluation_strategy = "steps",
        metric_for_best_model = "eval_loss",

        # Logging
        logging_dir = "weights",
        logging_steps = 80,
        disable_tqdm = False,
        report_to = "none",
        # predict_with_generate = True,

        # System
        # seed = random_seed,
        fp16 = False,
        bf16 = False,
        # sharded_ddp=True,
        local_rank=rank
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(model.tokenizer, mlm=False),
        train_dataset=ds_train,
        eval_dataset=ds_valid
    )


def train_ddp(rank, world_size, model, ds_train, ds_valid):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)    
    
    trainer = get_trainer(
        model=model,
        ds_train=ds_train,
        ds_valid=ds_valid,
        rank=rank
    )
    trainer.train()

if __name__ == "__main__":
    ds_train, ds_valid = get_dataset()
    world_size = 2
    args = Args()
    model = PrefixGPTNeoXLMHeadModel(args)
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size, model, ds_train, ds_valid),
        nprocs=world_size,
        join=True
    )
