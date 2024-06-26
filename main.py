import os
import csv
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from Pretrain_Encoder import MLP
from T5AutoEncoder import T5AutoEncoder
from Data_Generation import get_object_from_row, get_object_attributes, fill_in_templates, get_radius_from_initial_file
from HopfieldNetworks import retrieve_from_hopfield_network
from gpt_utils import table_to_text, send_messages_to_gpt, prompt_wrapper


encoder_path = "encoder.pt"
data_path = os.path.join(".", "esprit", "data_tables")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hn_beta = 10
selected_experiment_ids = list(map(lambda x: "%05d" % x, [0, 1, 2, 7, 11, 14, 15, 17]))


prompt_template = """
The following is an example of integrating physics rules into the event description to interpret the description. You don't need to point out which rule you use. The rule library is as follows.\n

Physics rules:\n

Consider a collision between two balls Ball 1 and Ball 2. With respect to the reference frame in which Ball 2 is static before the collision.\n

1. Ball one will be bounced off and Ball two will move in the previous Ball one's direction if Ball one's mass is less than Ball two's.\n

2. Ball one and ball two will move in the same direction after the crash if Ball one's mass is larger than Ball two.\n

3. Ball one will stay static if its mass is equal to Ball two, Ball two will start moving in the previous Ball one’s moving direction after the crash.\n

4. If a ball hits the platform it will be bounced off.\n

5. A ball will fall due to gravity in the case of free fall.\n

Here is an added interpretation example.\n

Event:\n

The red ball lands in the cubby and the green ball lands on top and a little to the right, sending the green ball right. It rolls over the short black wall off the edge and onto the floor, 
where it keeps rolling right towards the purple goal. As a result of the impact with the red ball, the green ball moves towards the right, on the short black platform, and continues rolling to the 
right. It continues rolling until it reaches the purple floor on the right.\n 

Interpretation example:\n

The red ball falls into the cubby due to the gravity from above and the green ball lands on top and a little to the right. In the reference frame of the green ball, the red ball moves towards the 
right and its mass is relatively larger than the green ball. So the red ball keeps rolling right, and the green ball moves in the previous direction of the red ball. The green ball moves towards 
the right, on the short black platform, and continues rolling to the right. It continues rolling until it reaches the purple floor on the right.\n 

You don't need to point out which rule you use. But you should point out the mass relation if the rule you used is related to the relative masses. The same for moving direction. Now try to 
integrate physics rules into the following event description: \n """


naive_prompt = "Integrate some physics interpretation to the following event description in one paragraph: \n"


def get_object_name(start_column, row):
    if row[start_column + 1] == "boundary" or row[start_column + 1] == "bar":
        return f"the {row[start_column + 1]}"
    elif row[start_column + 1] == "jar":
        return "a jar"
    elif row[start_column + 1].find("ircle") >= 0:
        return f"the {row[start_column + 2]} circle"
    else:
        print(row[start_column + 1])
        raise NotImplementedError


if __name__ == "__main__":
    autoencoder = T5AutoEncoder()
    salient_event_dir = os.path.join(data_path, "salient_events", "train")
    event_encoder = torch.load(encoder_path).to(device)
    event_encoder.eval()
    rule_embeddings = torch.from_numpy(autoencoder.rule_embeddings).to(device)
    for root, dirs, files in os.walk(salient_event_dir):
        for file in tqdm(files):
            experiment_id = re.findall("\d{5}", file)[0]
            if experiment_id in selected_experiment_ids and file.endswith(".csv"):
                prompt = prompt_template
                initial_event_dir = re.sub("salient_events", "initial_state", root)
                rs = get_radius_from_initial_file(os.path.join(initial_event_dir, file))
                table = ""
                embeddings_from_table = []
                steps = []
                object_names_1 = []
                object_names_2 = []
                checkpoint_time = time.time()
                with open(os.path.join(root, file), newline='') as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        if len(row) >= 13 and row[11] == "True" and row[12] == "begin":
                            step_id = int(row[1])
                            gt_rule, rule_id = fill_in_templates(get_object_from_row(13, row, rs), get_object_from_row(21, row, rs))
                            embedding = np.concatenate([get_object_attributes(13, row, rs), get_object_attributes(21, row, rs)], axis=0)
                            embeddings_from_table.append(embedding)
                            object_name_1 = get_object_name(13, row)
                            object_name_2 = get_object_name(21, row)
                            steps.append(step_id)
                            object_names_1.append(object_name_1)
                            object_names_2.append(object_name_2)
                            table += "| " + " | ".join(row) + " |\n"
                table_to_text_description = table_to_text(table)
                print("Table to text time is ", time.time() - checkpoint_time)
                checkpoint_time = time.time()

                naive_output = send_messages_to_gpt(prompt_wrapper(naive_prompt + table_to_text_description))
                print("Naive output time is ", time.time() - checkpoint_time)
                checkpoint_time = time.time()

                embedding = np.stack(embeddings_from_table)
                embedding = torch.from_numpy(embedding).float().to(device)
                embedding, _ = event_encoder(embedding)
                embedding = retrieve_from_hopfield_network(rule_embeddings, embedding, hn_beta)
                embedding = embedding.reshape(embedding.shape[0], autoencoder.rule_max_length, -1)
                decoded_rule = autoencoder.decode(embedding)
                prompt += "The rules in the event is: \n"
                for i in range(len(decoded_rule)):
                    prompt += f"At step {steps[i]}, the event between {object_names_1[i]} ands {object_names_2[i]} can be interpreted by physics rule {decoded_rule[i]}\n"

                prompt += "And also A ball will fall due to gravity in the case of free fall.\n"
                prompt += "The event to integrate is as follow:\n" + table_to_text_description
                print("Encoding, extraction and decoding time is ", time.time() - checkpoint_time)
                checkpoint_time = time.time()

                output = send_messages_to_gpt(prompt_wrapper(prompt))
                print("Our experiment time is ", time.time() - checkpoint_time)
                with open(os.path.join(".", "output_text.txt"), "a+") as f:
                    f.write(f"""Original output for {file}: {table_to_text_description}\n
                            Naive physics explain for {file}: {naive_output}\n
                            Our output for {file}: {output}\n""")
