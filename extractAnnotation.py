import sys
import csv
import jsonlines
 

test_jsonl = './esprit/human_annotations/test/test.jsonl'
train_jsonl = './esprit/human_annotations/train/train.jsonl'
val_jsonl = './esprit/human_annotations/test/test.jsonl'

ann_csv = './table_annotations/ann.csv'

icount=0
with jsonlines.open(test_jsonl) as reader, open(ann_csv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "annotation"])
    
    for sample in reader:
        id = sample["id"]
        anns = sample["salient_events_description"]
        icount = icount + 1
        print(icount)
        writer.writerow([id, anns])

with jsonlines.open(train_jsonl) as reader, open(ann_csv, 'a') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(["id", "annotation"])
    for sample in reader:
        id = sample["id"]
        anns = sample["salient_events_description"]
        icount = icount + 1
        print(icount)
        writer.writerow([id, anns])

with jsonlines.open(val_jsonl) as reader, open(ann_csv, 'a') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(["id", "annotation"])
    for sample in reader:
        id = sample["id"]
        anns = sample["salient_events_description"]
        icount = icount + 1
        print(icount)
        writer.writerow([id, anns])