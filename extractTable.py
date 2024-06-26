import sys
import csv
import jsonlines

# human annotations
test_ann_jsonl = './esprit/human_annotations/test/test.jsonl'
train_ann_jsonl = './esprit/human_annotations/train/train.jsonl'
val_ann_jsonl = './esprit/human_annotations/test/test.jsonl'

# detailed table information
test_table_csv = './esprit/data_tables/salient_events/test/csv/'
train_table_csv = './esprit/data_tables/salient_events/train/csv/'
val_table_csv = './esprit/data_tables/salient_events/val/csv/'

new_table_path = './table_annotations/table/'

# collect salient frames in each video
event_frames = {}
with jsonlines.open(test_ann_jsonl) as reader:

    for sample in reader:
        id = sample["id"]
        anns = sample["salient_frames"]
        
        event_frames[id] = anns

with jsonlines.open(train_ann_jsonl) as reader:

    for sample in reader:
        id = sample["id"]
        anns = sample["salient_frames"]
        
        event_frames[id] = anns

with jsonlines.open(val_ann_jsonl) as reader:

    for sample in reader:
        id = sample["id"]
        anns = sample["salient_frames"]
        
        event_frames[id] = anns

icount = 0

import glob
csvFiles = glob.glob(test_table_csv+"*.csv")
for csvfile in csvFiles:
    with open(csvfile, 'r') as readfile:
        reader = csv.reader(readfile)

        new_table_csv_path = new_table_path + csvfile.split('/')[-1]
        with open(new_table_csv_path, 'w') as writerfile:
            writer = csv.writer(writerfile)
            writer.writerow(["step", "is_collision", "kind", "id_1", "type_1", "color_1", "x_1","y_1","x_vel_1", "y_vel_1", "angle_1", "id_2", "type_2", "color_2", "x_2", "y_2", "x_vel_2", "y_vel_2", "angle_2"])
            
            icount = icount+1
            print(icount)
            #import pdb
            #pdb.set_trace()
            #print('debug')

            for sample in reader:
                if sample[0]=='in_solved_state':
                    continue
                if sample[11]=='True':
                    writer.writerow([sample[1], sample[11], sample[12]]+sample[13:29])

csvFiles = glob.glob(train_table_csv+"*.csv")
for csvfile in csvFiles:
    with open(csvfile, 'r') as readfile:
        reader = csv.reader(readfile)

        new_table_csv_path = new_table_path + csvfile.split('/')[-1]
        with open(new_table_csv_path, 'w') as writerfile:
            writer = csv.writer(writerfile)
            writer.writerow(["step", "is_collision", "kind", "id_1", "type_1", "color_1", "x_1","y_1","x_vel_1", "y_vel_1", "angle_1", "id_2", "type_2", "color_2", "x_2", "y_2", "x_vel_2", "y_vel_2", "angle_2"])

            icount = icount+1
            print(icount)
            #import pdb
            #pdb.set_trace()
            #print('debug')

            for sample in reader:
                if sample[0]=='in_solved_state':
                    continue
                if sample[11]=='True':
                    writer.writerow([sample[1], sample[11], sample[12]]+sample[13:29])

csvFiles = glob.glob(val_table_csv+"*.csv")
for csvfile in csvFiles:
    with open(csvfile, 'r') as readfile:
        reader = csv.reader(readfile)

        new_table_csv_path = new_table_path + csvfile.split('/')[-1]
        with open(new_table_csv_path, 'w') as writerfile:
            writer = csv.writer(writerfile)
            writer.writerow(["step", "is_collision", "kind", "id_1", "type_1", "color_1", "x_1","y_1","x_vel_1", "y_vel_1", "angle_1", "id_2", "type_2", "color_2", "x_2", "y_2", "x_vel_2", "y_vel_2", "angle_2"])

            icount = icount+1
            print(icount)
            #import pdb
            #pdb.set_trace()
            #print('debug')

            for sample in reader:
                if sample[0]=='in_solved_state':
                    continue
                if sample[11]=='True':
                    writer.writerow([sample[1], sample[11], sample[12]]+sample[13:29])


