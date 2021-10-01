from datasets import load_dataset
from transformers import DataCollatorWithPadding
from tokenization_kobert import KoBertTokenizer
from torch.utils.data import DataLoader

checkpoint = "monologg/kobert"
tokenizer = KoBertTokenizer.from_pretrained(checkpoint)


def klue_ynat():
    raw_datasets = load_dataset("klue", "ynat")

    def tokenize_function(example):
        return tokenizer(example["title"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['guid', 'title', 'url', 'date']
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # print(tokenized_datasets["train"].column_names)
    # print(tokenized_datasets["train"][0])
    # print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def klue_nli():
    raw_datasets = load_dataset("klue", "nli")
    # print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['guid', 'source', 'premise', 'hypothesis']
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # print(tokenized_datasets["train"].column_names)
    # print(tokenized_datasets["train"][0])
    # print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def klue_sts():
    raw_datasets = load_dataset("klue", "sts")
    # print(raw_datasets)
    # print(raw_datasets['train'][0])
    
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    def make_labels(example):
        example['labels'] = example['labels']['binary-label']
        return example

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True).map(make_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['guid', 'source', 'sentence1', 'sentence2']
    )
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # print(tokenized_datasets["train"].column_names)
    # print(tokenized_datasets["train"][0])
    # print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def klue_re():
    raw_datasets = load_dataset("klue", "re")
    print(raw_datasets)
    print(raw_datasets['train'][0])
    
    subject_start_marker = "<subj>"
    subject_end_marker = "</subj>"
    object_start_marker = "<obj>"
    object_end_marker = "</obj>"
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                subject_start_marker,
                subject_end_marker,
                object_start_marker,
                object_end_marker,
            ]
        }
    )
    def _mark_entity_spans(
        text,
        subject_range,
        object_range
    ):
        if subject_range < object_range:
            segments = [
                text[: subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 : object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 :],
            ]
        elif subject_range > object_range:
            segments = [
                text[: object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 : subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 :],
            ]
        else:
            raise ValueError("Entity boundaries overlap.")

        marked_text = "".join(segments)

        return marked_text


    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    def make_data(example):
        example['sentence'] = _mark_entity_spans(
            example['sentence'],
            (example['subject_entity']['start_idx'], example['subject_entity']['end_idx']),
            (example['object_entity']['start_idx'], example['object_entity']['end_idx'])
        )
        return example

    tokenized_datasets = raw_datasets.map(make_data).map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['guid', 'sentence', 'subject_entity', 'object_entity', 'source']
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


data_loaders = {
    'ynat':klue_ynat,
    'nli':klue_nli,
    'sts':klue_sts,
    're':klue_re
}

if __name__ == '__main__':
    train_dataloader, eval_dataloader = klue_re()