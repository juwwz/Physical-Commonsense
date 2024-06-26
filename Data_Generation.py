import os
import csv
import re
import numpy as np
from tqdm import tqdm
from T5AutoEncoder import T5AutoEncoder


static_radius = 1000
jar_radius = 1
max_samples_per_class = 10000


class MovingObject:
    def __init__(self, x, y, vx, vy, r, type):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.type = type
        self.v = np.linalg.norm(np.array([self.vx, self.vy]))


def get_object_from_row(start_column, row, rs):
    return MovingObject(x=float(row[start_column + 3]), y=float(row[start_column + 4]), vx=float(row[start_column + 5]), vy=float(row[start_column + 6]),
                        r=rs[row[start_column]], type=row[start_column + 1])


def get_object_attributes(start_column, row, rs):
    return np.array([float(row[start_column + 3]), float(row[start_column + 4]), float(row[start_column + 5]), float(row[start_column + 6]), rs[row[start_column]]])


def get_radius_from_initial_file(file_path):
    rs = {}
    with open(file_path, newline='') as f:
        csvreader = csv.reader(f)
        _ = next(csvreader)
        for row in csvreader:
            if row[1] == "boundary" or row[1] == "bar":
                rs[row[0]] = static_radius
            elif row[1] == "jar":
                rs[row[0]] = jar_radius
            else:
                rs[row[0]] = float(row[-1])
    return rs


def fill_in_templates(object1, object2):
    if object1.v > object2.v:
        objtmp = object1
        object1 = object2
        object2 = objtmp
    if object1.type == "boundary" or object1.type == "bar":
        return "A ball will hit the boundary", 3
    else:
        if object2.r > object1.r:
            return "A larger ball is moving to a smaller ball.", 1
        elif object2.r < object1.r:
            return "A smaller ball is moving to a larger ball.", 0
        else:
            return "A ball is moving to a ball with the same mass.", 2


if __name__ == "__main__":
    autoencoder = T5AutoEncoder()
    data_samples = {"samples": [], "target_domain": autoencoder.rule_embeddings[:-1]}
    salient_event_dir = os.path.join(".", "esprit", "data_tables", "salient_events", "train")
    counter = [0 for _ in range(len(autoencoder.rules) - 1)]
    for root, dirs, files in os.walk(salient_event_dir):
        for file in tqdm(files):
            if file.endswith(".csv"):
                initial_event_dir = re.sub("salient_events", "initial_state", root)
                rs = get_radius_from_initial_file(os.path.join(initial_event_dir, file))
                with open(os.path.join(root, file), newline='') as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        if len(row) >= 13 and row[11] == "True" and row[12] == "begin":
                            description, rule_id = fill_in_templates(get_object_from_row(13, row), get_object_from_row(21, row))
                            # if counter[rule_id] >= max_samples_per_class:
                            #     continue
                            # embedding = autoencoder.encode([description])
                            embedding = np.concatenate([get_object_attributes(13, row), get_object_attributes(21, row)], axis=0)
                            all_rule_ids = list(range(len(autoencoder.rules)))
                            all_rule_ids.pop(rule_id)
                            data_samples["samples"].append({"x": embedding, "label": rule_id})
                            counter[rule_id] += 1
    np.save("data.npy", data_samples)
