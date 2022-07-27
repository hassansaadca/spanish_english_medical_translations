import os
import numpy as np

from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments
from datasets import load_metric

import torch

SOURCE_LANGUAGE = 'es'
TARGET_LANGUAGE = 'en'



def load_model(source_language, target_language):
  model_checkpoint = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
  print(f'Model Checkpoint Name: {model_checkpoint}')
  lang = {'es':'Spanish', 'en':'English'}
  print(f'Translation: {lang[source_language]} to {lang[target_language]}')
  return tokenizer, model, model_checkpoint


tokenizer, model, model_checkpoint = load_model(source_language = SOURCE_LANGUAGE, target_language = TARGET_LANGUAGE)



def load_data(source_language, corpus = '/home/ec2-user/data/UFAL/medical.train_val'):

  with open(corpus) as f:
      l = f.readlines()

  pairs = []
  for i in l:
      pairs.append(i.strip().split('\t'))

  total_lines = len(l)
  split = int(total_lines * .8)

  if source_language == 'es':
    train_sources= [i[0] for i in pairs[0:split]] 
    train_targets = [i[1] for i in pairs[0:split]]
    val_sources = [i[0] for i in pairs[split:]]
    val_targets = [i[1] for i in pairs[split:]]
  else:
    train_sources= [i[1] for i in pairs[-1:-1* split:-1]] 
    train_targets = [i[0] for i in pairs[-1:-1* split:-1]]
    val_sources = [i[1] for i in pairs[-1*split::-1]]
    val_targets = [i[0] for i in pairs[-1*split::-1]]

  print('Sample Sources:\n----------')
  for i in train_sources[:2]:
    print(i)
  print('\nSample Targets:\n----------')
  for i in train_targets[:2]:
    print(i)
  return train_sources, train_targets, val_sources, val_targets


train_sources, train_targets, val_sources, val_targets = load_data(source_language = SOURCE_LANGUAGE)



assert len(train_targets)==len(train_sources)
assert len(val_targets)==len(val_sources)



class BatchData(torch.utils.data.Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length= 512):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_token_length = max_length


    def convert_data(self):
        self.model_inputs = self.tokenizer(self.source_texts, max_length = self.max_token_length, truncation = True) #, padding = True)
        with tokenizer.as_target_tokenizer():
            self.labels = tokenizer(self.target_texts, max_length= self.max_token_length, truncation = True) #, padding = True)
        self.model_inputs['labels'] = self.labels['input_ids']


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.model_inputs.items()}
        return item

    def __len__(self):
        return len(self.model_inputs['labels'])

train_dataset = BatchData(train_sources, train_targets, tokenizer = tokenizer)
val_dataset = BatchData(val_sources, val_targets, tokenizer = tokenizer)

train_dataset.convert_data()
val_dataset.convert_data()


BATCH_SIZE = 8

assert len(train_dataset) == len(train_sources)
assert len(val_dataset) == len(val_sources)

model_name = model_checkpoint.split("/")[-1]


#https://huggingface.co/docs/transformers/main_classes/trainer
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{SOURCE_LANGUAGE}-to-{TARGET_LANGUAGE}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps = 50,
    num_train_epochs=1,
    predict_with_generate=True   
)

print('\nFine-tuned Model Directory:')
print(f"{model_name}-finetuned-{SOURCE_LANGUAGE}-to-{TARGET_LANGUAGE}/")



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

