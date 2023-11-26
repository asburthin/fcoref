from fastcoref import TrainingArgs, CorefTrainer
import spacy
import wandb

nlp = spacy.blank("th")

wandb.login()


def train(config):
    args = TrainingArgs(
        output_dir="wangchan",
        overwrite_output_dir=True,
        device="cuda:0",
        logging_steps=100,
        eval_steps=100,
        max_tokens_in_batch=8000,
        epochs=600,
        **config,
    )

    trainer = CorefTrainer(
        args=args,
        nlp=nlp,
        train_file="/home/poomphob/Desktop/Thesis/s2e_coref/data/17_10_2023_doccano/train_tokens.jsonl",
        dev_file="/home/poomphob/Desktop/Thesis/s2e_coref/data/17_10_2023_doccano/val_tokens.jsonl",  # optional
    )

    trainer.train()

    return trainer.evaluate()


# 1: Define objective/training function
def objective(config):
    score = train(config)["f1"]
    return score


def main():
    wandb.init(project="coref-wangchan-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "top_lambda": {"max": 0.9, "min": 0.3},
        "max_span_length": {"max": 50, "min": 20, "distribution": "int_uniform"},
        # "adam_epsilon": {"max": 0.000001, "min": 0.0000001},
        "ffnn_size": {"max": 2048, "min": 512, "distribution": "int_uniform"},
        "model_name_or_path": {
            "values": [
                "xlm-roberta-base",
                "airesearch/wangchanberta-base-att-spm-uncased",
            ],
            "distribution": "categorical",
        },
        # "adam_beta1": {"max": 1.8, "min": 0.45},
        # "adam_beta2": {"max": 1.8, "min": 0.45},
        # "adam_epsilon": {"max": 0.000002, "min": 5e-7},
        # "learning_rate": {"max": 0.0001, "min": 0.000001},
        # "head_learning_rate": {"max": 0.0005, "min": 0.000001},
        "dropout_prob": {"max": 0.6, "min": 0.1},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="coref-sweep-f1-500-data")

wandb.agent(sweep_id, function=main, count=150)
