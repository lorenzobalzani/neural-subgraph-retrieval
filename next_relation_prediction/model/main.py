import datetime
import os

from arg_parser import parse_arguments
from datasets import load_dataset
from evaluation import compute_metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def fine_tune_model(
    pretrained_model_name: str,
    model_huggingface_hub_url: str,
    dataset_huggingface_hub_url: str,
    n_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    early_stopping_patience: int,
    stats_folder: str,
):
    # Load dataset
    dataset = load_dataset(dataset_huggingface_hub_url)

    # Remove the examples with the label "scale_type_of" as it is not a valid
    # label
    dataset["test"] = dataset["test"].filter(
        lambda example: example["output"] != "scale_type_of"
    )

    # Encode the labels
    dataset["train"] = dataset["train"].class_encode_column("output")
    class_label_feature = dataset["train"].features["output"]
    dataset["eval"] = dataset["eval"].cast_column(
        "output", class_label_feature)
    dataset["test"] = dataset["test"].cast_column(
        "output", class_label_feature)
    dataset = dataset.rename_column("output", "labels")

    # Load the pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=len(class_label_feature.names),
        id2label={
            idx: label for idx,
            label in enumerate(
                class_label_feature.names)},
        label2id={
            label: idx for idx,
            label in enumerate(
                class_label_feature.names)},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
    )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["input"],
            truncation=True,
            max_length=model.config.max_position_embeddings,
        ),
        batched=True,
        remove_columns=["input"],
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors="pt")

    # Define the training arguments
    training_args = TrainingArguments(
        num_train_epochs=n_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        weight_decay=1e-2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=True,
        hub_model_id=model_huggingface_hub_url,
        output_dir=model_huggingface_hub_url,
        logging_dir=os.path.join(stats_folder, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience)],
    )

    # Train the model
    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    trainer.push_to_hub()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(
        os.path.join(stats_folder, f"test_metrics_{timestamp}.info"), "w"
    ) as file:
        file.write(str(test_metrics))


if __name__ == "__main__":
    args = parse_arguments()
    fine_tune_model(
        args.pretrained_model_name,
        args.model_huggingface_hub_url,
        args.dataset_huggingface_hub_url,
        args.n_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.early_stopping_patience,
        args.stats_folder,
    )
