from data_loader import data_loaders
from eval_metric import metrics
from transformers import AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW

import sys

task = sys.argv[1] if len(sys.argv) > 1 else 'ynat'

if task not in ['ynat']:
    exit()

train_dataloader, eval_dataloader = data_loaders[task]()

model_classes = {
    'ynat': AutoModelForSequenceClassification
}

num_classes = {
    'ynat': 7
}

def train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device):
    
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def eval(model, eval_dataloader, metric, device):
    model.eval()
    preds = []
    targets = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.append(predictions)
        targets.append(batch["labels"])
    preds = torch.cat(preds, dim=-1).cpu().numpy()
    targets = torch.cat(targets, dim=-1).cpu().numpy()
    for k, v in metric(preds, targets).items():
        print(k, v)


def main():
    checkpoint = "monologg/kobert"
    train_dataloader, eval_dataloader = data_loaders[task]()
    model = model_classes[task].from_pretrained(checkpoint, num_labels=num_classes[task])
    metric = metrics[task]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device)
    print()

    eval(model, eval_dataloader, metric, device)
    


if __name__ == '__main__':
    main()