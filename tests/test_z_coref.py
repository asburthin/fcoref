import pytest


def test_coref():
    from fastcoref import TrainingArgs, CorefTrainer
    import spacy
    import spacy_pythainlp.core

    nlp = spacy.blank("th")

    args = TrainingArgs(
        output_dir="test_training",
        overwrite_output_dir=True,
        model_name_or_path="airesearch/wangchanberta-base-att-spm-uncased",
        device="cuda:0",
        epochs=20,
        logging_steps=3,
        eval_steps=3,
        dropout_prob=0.4,
        max_segment_len=512,
        max_tokens_in_batch=8000,
    )

    trainer = CorefTrainer(
        args=args,
        nlp=nlp,
        train_file="/home/poomphob/Desktop/Thesis/fastcoref/tests/test_files/train_tokens.jsonl",
        dev_file="/home/poomphob/Desktop/Thesis/fastcoref/tests/test_files/train_tokens.jsonl",  # optional
        # test_file='/home/poomphob/Desktop/Thesis/fastcoref/test.py'   # optional
    )

    # trainer = CorefTrainer(
    #     args=args,
    #     nlp=nlp,
    #     train_file="/home/poomphob/Desktop/Thesis/s2e_coref/raw_data/hann_coref/val.json",
    #     dev_file="/home/poomphob/Desktop/Thesis/s2e_coref/raw_data/hann_coref/test.json",  # optional
    #     # test_file='/home/poomphob/Desktop/Thesis/fastcoref/test.py'   # optional
    # )

    trainer.train()
