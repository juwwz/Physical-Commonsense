import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.losses import SupConLoss
from torchmetrics.functional import pairwise_cosine_similarity


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        outputs = self.layers(x)
        embeddings = nn.functional.normalize(outputs, dim=1, p=2)
        return outputs, embeddings


if __name__ == "__main__":
    current_time = "%02d%02d%02d%02d" % (time.localtime().tm_mon, time.localtime().tm_mday, time.localtime().tm_hour, time.localtime().tm_min)
    for main_dir in ["pre_models", "images", "train_log"]:
        if not os.path.isdir(f"./{main_dir}/{current_time}"):
            os.makedirs(f"./{main_dir}/{current_time}")

    # Load data
    data = np.load('data.npy', allow_pickle=True).item()
    samples = data['samples']
    target_domain = torch.Tensor(data['target_domain'][:4])

    # Extract x and labels
    x_values = [torch.Tensor(sample['x']) for sample in samples]
    labels = [sample['label'] for sample in samples]

    # Convert to tensors
    x_tensor = torch.stack(x_values)
    labels_tensor = torch.tensor(labels)

    # Split data
    train_x, test_x, train_labels, test_labels = train_test_split(x_tensor, labels_tensor, test_size=0.2, random_state=42)
    train_x, val_x, train_labels, val_labels = train_test_split(train_x, train_labels, test_size=0.25, random_state=42)

    # Create TensorDataset
    train_dataset = TensorDataset(train_x, train_labels)
    val_dataset = TensorDataset(val_x, val_labels)
    test_dataset = TensorDataset(test_x, test_labels)

    # Create MPerClassSampler
    train_sampler = MPerClassSampler(train_labels, m=16, batch_size=64, length_before_new_iter=len(train_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = len(samples[0]['x'])
    hidden_dim = 128
    output_dim = target_domain.size(1)
    model = MLP(input_dim, hidden_dim, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_domain = target_domain.to(device)

    criterion_mse = nn.MSELoss()
    supcl = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda_supcl = 0.1
    lambda_cos = 1

    writer = SummaryWriter(f"./train_log/{current_time}")


    def test_accuracy_per_class(model, test_loader, target_domain, timestep=0):
        model.eval()
        all_predicted_labels = []
        all_true_labels = []

        with torch.no_grad():
            for source, labels in test_loader:
                source, labels = source.to(device).float(), labels.to(device)
                outputs, _ = model(source)

                # Compute cosine similarity between outputs and target domain
                similarity = cosine_similarity(outputs.cpu().numpy(), target_domain.cpu().numpy())

                # Find the index of the maximum similarity for each sample
                predicted_labels = np.argmax(similarity, axis=1)

                # Collect all predicted and true labels
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(labels.cpu().numpy())

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

        # Calculate and print accuracy for each class
        correct_predictions = 0
        total_samples = 0
        for i in range(target_domain.size(0)):
            correct = conf_matrix[i, i]
            total = sum(conf_matrix[i])
            class_accuracy = correct / total if total > 0 else 0
            writer.add_scalar(f"Val/acc_{i}", class_accuracy, timestep)
            print(f"Accuracy for class {i}: {class_accuracy * 100:.2f}%")
            correct_predictions += correct
            total_samples += total

        # Calculate and print overall accuracy
        overall_accuracy = correct_predictions / total_samples
        writer.add_scalar(f"Val/acc_total", overall_accuracy, timestep)
        print(f"Overall Nearest Neighbor Accuracy: {overall_accuracy * 100:.2f}%")

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'./images/{current_time}/confusion_matrix_{timestep}.png')
        plt.show()

        return overall_accuracy


    num_epochs = 200
    best_overall_accuracy = 0
    best_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        stats = {"mse": 0, "supcl": 0, "cos": 0, "loss": 0}
        model.train()
        for source, labels in train_loader:
            source, labels = source.to(device).float(), labels.to(device)
            target = target_domain[labels].float()
            outputs, embeddings = model(source)
            mse_loss = criterion_mse(outputs, target)
            scl_loss = supcl(embeddings, labels)
            cos_loss = F.cross_entropy(F.softmax(pairwise_cosine_similarity(outputs, target_domain), dim=1), labels)
            loss = mse_loss + lambda_supcl * scl_loss + lambda_cos * cos_loss
            stats["mse"] += mse_loss.item()
            stats["supcl"] += scl_loss.item()
            stats["cos"] += cos_loss.item()
            stats["loss"] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in stats.items():
            writer.add_scalar(f"Train/{k}", v / len(train_loader), epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for source, labels in val_loader:
                source, labels = source.to(device).float(), labels.to(device)
                target = target_domain[labels].float()
                outputs, _ = model(source)
                val_loss += criterion_mse(outputs, target).item()
        val_loss /= len(val_loader)
        writer.add_scalar("Val/loss", val_loss, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss}")
        overall_accuracy = test_accuracy_per_class(model, val_loader, target_domain, epoch)
        if overall_accuracy > best_overall_accuracy:
            best_overall_accuracy = overall_accuracy
            best_model.load_state_dict(model.state_dict())
            torch.save(best_model, f"./pre_models/{current_time}/best_encoder.pt")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for source, labels in test_loader:
            source, labels = source.to(device).float(), labels.to(device)
            target = target_domain[labels].float()
            outputs, _ = model(source)
            test_loss += criterion_mse(outputs, target).item()
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss}")
    test_accuracy_per_class(model, test_loader, target_domain, num_epochs)
    torch.save(model, f"./pre_models/{current_time}/last_encoder.pt")
    torch.save(best_model, f"./pre_models/{current_time}/best_encoder.pt")
