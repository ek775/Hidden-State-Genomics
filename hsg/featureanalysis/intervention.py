import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

from biocommons.seqrepo import SeqRepo
from hsg.cisplatinRNA.CNNhead import CNNHead
from hsg.cisplatinRNA.CNNtrain import prepare_data
from hsg.stattools.features import get_latent_model

from google.cloud import storage
from tqdm import tqdm
import os, math

from dotenv import load_dotenv
load_dotenv()



def test(feature: int, intervention_value: int, sequences: list[str, torch.Tensor], cnn, sae) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test the effect of intervening on a specific feature in the latent space.

    Args:
        feature (int): The feature index to intervene on.
        intervention_value (int): The value to increase the feature by.
        sequences (list[str, torch.Tensor]): List of tuples containing sequence strings and their corresponding tensors.
        cnn (CNNHead): The pre-trained CNN model for feature extraction.
        sae (torch.nn.Module): The pre-trained SAE model for latent representation.

    Returns:
        predictions (torch.Tensor): The model class prediction probabilities after intervention.
        labels (torch.Tensor): The true labels for the input sequences.
    """
    results = []
    labels = []
    for seq, label in tqdm(sequences):
        with torch.no_grad():
            latent = torch.squeeze(sae.forward(seq))
            intervention_vec = torch.zeros_like(latent)
            intervention_vec[:, feature] = intervention_value
            modified_latent = latent + intervention_vec
            output = cnn.forward(cnn.pad_sequence(modified_latent, max_length=cnn.seq_length).unsqueeze(0))
            results.append(output.squeeze(0))
            labels.append(torch.Tensor(label))

    return torch.stack(results), torch.stack(labels)


def generate_markdown_report(feature: int, intervention_value: int, probas: torch.Tensor, labels: torch.Tensor, base_probas: torch.Tensor, 
                             base_labels: torch.Tensor, save_dir: str = None) -> str:
    """
    Generate a markdown report comparing the intervention and baseline results.

    Args:
        feature (int): The feature index that was intervened on.
        intervention_value (int): The value that was added to the feature.
        preds (torch.Tensor): The model predictions after intervention.
        labels (torch.Tensor): The true labels for the input sequences.
        base_preds (torch.Tensor): The model predictions without intervention (baseline).
        base_labels (torch.Tensor): The true labels for the baseline.

    Returns:
        report (str): A markdown formatted string summarizing the results.
    """
    # create save directory if it doesn't exist, use default naming
    if save_dir is None:
        save_dir = f"intervention_reports/f{feature}_{intervention_value}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # simple metrics
    preds = torch.argmax(probas, dim=1).cpu().numpy()
    labels_1d = torch.argmax(labels, dim=1).cpu().numpy()
    base_preds = torch.argmax(base_probas, dim=1).cpu().numpy()
    base_labels_1d = torch.argmax(base_labels, dim=1).cpu().numpy()
    # data to cpu
    labels = labels.cpu().numpy()
    base_labels = base_labels.cpu().numpy()
    probas = probas.cpu().numpy()
    base_probas = base_probas.cpu().numpy()

    accuracy = accuracy_score(labels_1d, preds)
    base_accuracy = accuracy_score(base_labels_1d, base_preds)

    roc_auc = roc_auc_score(labels[:, 1], probas[:, 1])
    base_roc_auc = roc_auc_score(base_labels[:, 1], base_probas[:, 1])

    conf_matrix = confusion_matrix(labels_1d, preds)
    base_conf_matrix = confusion_matrix(base_labels_1d, base_preds)
    # charts
    fpr, tpr, _ = roc_curve(labels[:, 1], probas[:, 1])
    base_fpr, base_tpr, _ = roc_curve(base_labels[:, 1], base_probas[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Intervention (AUC = {:.2f})'.format(roc_auc))
    plt.plot(base_fpr, base_tpr, label='Baseline (AUC = {:.2f})'.format(base_roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # plot the probability distributions
    plt.figure()
    plt.hist(probas[:, 1], bins=30, alpha=0.5, label='Intervention', color='blue')
    plt.hist(base_probas[:, 1], bins=30, alpha=0.5, label='Baseline', color='orange')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Predicted Probability Distributions')
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(save_dir, 'probability_distributions.png'))
    plt.close()

    # markdown report
    report = f"""
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | {accuracy:.4f}     | {base_accuracy:.4f}   |
| ROC AUC          | {roc_auc:.4f}     | {base_roc_auc:.4f}   |

## Detailed Classification Report (Intervention)

```
{conf_matrix}
{classification_report(labels_1d, preds)}
```

## Detailed Classification Report (Baseline)

```
{base_conf_matrix}
{classification_report(base_labels_1d, base_preds)}
```

## ROC Curve

![ROC Curve](/{os.path.join(save_dir, 'roc_curve.png')})

## Predicted Probability Distributions

![Probability Distributions](/{os.path.join(save_dir, 'probability_distributions.png')})
"""

    return report


def main(feature: int, intervention_value: int, cnn_path: str, sae_path: str, cisplatin_positive: str, cisplatin_negative: str):

    # check inputs and download models if needed
    if not os.path.exists(cisplatin_positive):
        raise FileNotFoundError(f"Cisplatin positive BED file not found: {cisplatin_positive}")
    if not os.path.exists(cisplatin_negative):
        raise FileNotFoundError(f"Cisplatin negative BED file not found: {cisplatin_negative}")
    
    if cnn_path.startswith("gs://"):
        local_cnn_path = f"/tmp/{os.path.basename(cnn_path)}"
        if not os.path.exists(local_cnn_path):
            print(f"Downloading CNN model from {cnn_path} to {local_cnn_path}...")
            storage.Client().bucket(cnn_path.split("/")[2]).blob("/".join(cnn_path.split("/")[3:])).download_to_filename(local_cnn_path)
        cnn_path = local_cnn_path
    if sae_path.startswith("gs://"):
        local_sae_path = f"/tmp/{os.path.basename(sae_path)}"
        if not os.path.exists(local_sae_path):
            print(f"Downloading SAE model from {sae_path} to {local_sae_path}...")
            storage.Client().bucket(sae_path.split("/")[2]).blob("/".join(sae_path.split("/")[3:])).download_to_filename(local_sae_path)
        sae_path = local_sae_path

    print("Loading models...")
    cnn_model = torch.load(cnn_path)
    sae_model = get_latent_model(parent_model_path=os.environ["NT_MODEL"], layer_idx=23, sae_path=sae_path)

    _, _, test_data = prepare_data(cisplatin_positive, cisplatin_negative)
    test_data = test_data[:100]  # limit to 1000 samples for testing

    # intervention
    print("------------ Intervention -----------")
    probas, labels = test(feature, intervention_value, test_data, cnn_model, sae_model)

    # baseline
    print("------------ Baseline -----------")
    base_probas, base_labels = test(0, 0, test_data, cnn_model, sae_model)

    # generate report
    print("Generating report...")
    report = generate_markdown_report(feature, intervention_value, probas, labels, base_probas, base_labels)
    report_path = f"intervention_reports/f{feature}_{intervention_value}/report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the effect of interventions on sequence features using a pre-trained model.")
    parser.add_argument("--feature", type=int, required=True, help="Feature index to intervene on (e.g. 3378, 791, 4096, etc.).")
    parser.add_argument("--intervention_value", type=int, required=True, help="Value to increase the feature by (e.g. +1, +2, +10, etc.).")
    parser.add_argument("--cnn", type=str, default="gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/features.pt", help="Path to the CNN feature model file.")
    parser.add_argument("--sae", type=str, default="gs://hidden-state-genomics/ef8/sae/layer_23.pt", help="Path to the SAE model file.")
    parser.add_argument("--cisplatin_positive", type=str, default="data/A2780_Cisplatin_Binding/cisplatin_pos.bed", help="Path to the positive cisplatin BED file.")
    parser.add_argument("--cisplatin_negative", type=str, default="data/A2780_Cisplatin_Binding/cisplatin_neg_45k.bed", help="Path to the negative cisplatin BED file.")

    args = parser.parse_args()

    main(args.feature, args.intervention_value, args.cnn, args.sae, args.cisplatin_positive, args.cisplatin_negative)