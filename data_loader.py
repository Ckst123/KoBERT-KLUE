from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForTokenClassification
from tokenization_kobert import KoBertTokenizer
from torch.utils.data import DataLoader

checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


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


def klue_ner():
    raw_datasets = load_dataset("klue", "ner")
    print(raw_datasets)
    print(raw_datasets['train'][0])
    
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True, padding='max_length', max_length=128)

    def make_labels(example):
        # if self.tokenizer_type == "xlm-sp":
        #     strip_char = "???"
        # elif self.tokenizer_type == "bert-wp":
        #     strip_char = "##"
        strip_char = "##"

        original_clean_tokens = []
        original_clean_labels = []
        sentence = ""
        for token, tag in zip(example['tokens'], example['ner_tags']):
            sentence += token
            if token == " ":
                continue
            original_clean_tokens.append(token)
            original_clean_labels.append(tag)
        
        sent_words = sentence.split(" ")
        modi_labels = []
        # modi_labels.append(12)
        char_idx = 0

        for word in sent_words:
            # ??????, ?????????
            correct_syllable_num = len(word)
            tokenized_word = tokenizer.tokenize(word)
            # case1: ?????? tokenizer --> [???, ##???]
            # case2: wp tokenizer --> [??????]
            # case3: ??????, wp tokenizer?????? unk --> [unk]
            # unk?????? --> ????????? ????????? unk??? ??????, ???, ????????? ??????
            contain_unk = True if tokenizer.unk_token in tokenized_word else False
            for i, token in enumerate(tokenized_word):
                token = token.replace(strip_char, "")
                if not token:
                    modi_labels.append(12)
                    continue
                modi_labels.append(original_clean_labels[char_idx])
                if not contain_unk:
                    char_idx += len(token)
            if contain_unk:
                char_idx += correct_syllable_num
        # modi_labels.append(12)
        # print(sentence, modi_labels)
        example['sentence'] = sentence
        example['labels'] = modi_labels
        return example

    def make_padding_label(example):
        l = len(example["input_ids"])
        labels = example['labels']
        labels = [-100] + labels + ( [-100] * (l - len(labels) - 1))
        example['labels'] = labels
        return example
        

    tokenized_datasets = raw_datasets.map(make_labels).map(tokenize_function, batched=True).map(make_padding_label)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['sentence', 'tokens', 'ner_tags']
    )
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))
    print(tokenized_datasets["train"][0]["labels"])


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
    're':klue_re,
    'ner':klue_ner,
}

if __name__ == '__main__':
    train_dataloader, eval_dataloader = klue_ner()