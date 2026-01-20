import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from hsg.cisplatinRNA.CNNtrain import prepare_data
from hsg.stattools.features import get_latent_model

from google.cloud import storage
from tqdm import tqdm
import os

from dotenv import load_dotenv
load_dotenv()



def test(feature: int, feat_min: int, act_factor: int, sequences: list[str, torch.Tensor], cnn, sae, control: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test the effect of intervening on a specific feature in the latent space.

    Args:
        feature (int): The feature index to intervene on.
        feat_min (int): The minimum value for the feature.
        act_factor (int): The activation factor to multiply the feature by, and suppress others by its inverse.
        sequences (list[str, torch.Tensor]): List of tuples containing sequence strings and their corresponding tensors.
        cnn (CNNHead): The pre-trained CNN model for feature extraction.
        sae (torch.nn.Module): The pre-trained SAE model for latent representation.
        control (bool): Whether to run the control test without intervention.

    Returns:
        predictions (torch.Tensor): The model class prediction probabilities after intervention.
        labels (torch.Tensor): The true labels for the input sequences.
    """
    results = []
    labels = []
    for seq, label in tqdm(sequences):
        with torch.no_grad():
            latent = torch.squeeze(sae.forward(seq))
            if control:
                modified_latent = latent
            else:
                # amplify feature signal and suppress others
                latent[:, feature] = torch.clamp(latent[:, feature], min=feat_min) # allow actual activation, but avoid zeroing out
                intervention_vec = torch.zeros_like(latent) # feature vector
                intervention_vec[:, :] = 1/act_factor if act_factor != 0 else 0 # suppression rate
                intervention_vec[:, feature] = act_factor  if act_factor != 0 else 1 # feature weight
                modified_latent = latent * intervention_vec # element-wise multiplication
                
            # generate embeddings or use raw features depending on CNN head
            if modified_latent.size(1) != cnn.input_dim:
                decoder = sae.sae.decoder
                modified_latent = decoder(modified_latent)
                
            # get predictions
            output = cnn.forward(cnn.pad_sequence(modified_latent, max_length=cnn.seq_length).unsqueeze(0))
            results.append(output.squeeze(0))
            labels.append(torch.Tensor(label))

    return torch.stack(results), torch.stack(labels)


def generate_markdown_report(feature: int, feat_min: float, act_factor: float, probas: torch.Tensor, labels: torch.Tensor, base_probas: torch.Tensor, 
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
        save_dir = f"intervention_reports/f{feature}_m{feat_min}_a{act_factor}"
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
    plt.hist(probas[:, 1], bins=50, alpha=0.5, label='Intervention', color='blue')
    plt.hist(base_probas[:, 1], bins=50, alpha=0.5, label='Baseline', color='orange')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Predicted Probability Distributions')
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(save_dir, 'probability_distributions.png'))
    plt.close()

    # plot confusion matrices
    plt.figure()
    plt.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Intervention)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for (i, j), value in np.ndenumerate(conf_matrix):
        plt.text(j, i, value, ha='center', va='center', color='red')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_intervention.png'))
    plt.close()

    plt.figure()
    plt.matshow(base_conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Baseline)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for (i, j), value in np.ndenumerate(base_conf_matrix):
        plt.text(j, i, value, ha='center', va='center', color='red')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_baseline.png'))
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
{classification_report(labels_1d, preds)}
```
![Confusion Matrix (Intervention)](/{os.path.join(save_dir, 'confusion_matrix_intervention.png')})

## Detailed Classification Report (Baseline)

```
{classification_report(base_labels_1d, base_preds)}
```
![Confusion Matrix (Baseline)](/{os.path.join(save_dir, 'confusion_matrix_baseline.png')})

## ROC Curve

![ROC Curve](/{os.path.join(save_dir, 'roc_curve.png')})

## Predicted Probability Distributions

![Probability Distributions](/{os.path.join(save_dir, 'probability_distributions.png')})
"""

    return report


def main(feature: int, feat_min: int, act_factor: int, cnn_path: str, sae_path: str, 
         cisplatin_positive: str, cisplatin_negative: str, folder_name: str = "intervention_reports"):

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
    test_data = test_data[:1000]  # limit to 1000 samples for time / resources

    # intervention
    print("------------ Intervention -----------")
    probas, labels = test(feature, feat_min, act_factor, test_data, cnn_model, sae_model, control=False)

    # baseline
    print("------------ Baseline -----------")
    base_probas, base_labels = test(0, 0, 0, test_data, cnn_model, sae_model, control=True)
    # generate report
    print("Generating report...")
    save_dir = f"data/{folder_name}/f{feature}_m{feat_min}_a{act_factor}"
    report = generate_markdown_report(feature, feat_min, act_factor, probas, labels, 
                                      base_probas, base_labels, save_dir)
    report_path = f"{save_dir}/report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the effect of interventions on sequence features using a pre-trained model.")
    parser.add_argument("--feature", type=int, required=True, help="Feature index to intervene on (e.g. 3378, 791, 4096, etc.).")
    parser.add_argument("--min_act", type=float, default=1.0, help="Minimum activation value for the feature during intervention.")
    parser.add_argument("--act_factor", type=float, default=50.0, help="Activation factor to multiply the feature by during intervention.")
    parser.add_argument("--cnn", type=str, default="gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/features.pt", help="Path to the CNN feature model file.")
    parser.add_argument("--sae", type=str, default="gs://hidden-state-genomics/ef8/sae/layer_23.pt", help="Path to the SAE model file.")
    parser.add_argument("--cisplatin_positive", type=str, default="data/A2780_Cisplatin_Binding/cisplatin_pos.bed", help="Path to the positive cisplatin BED file.")
    parser.add_argument("--cisplatin_negative", type=str, default="data/A2780_Cisplatin_Binding/cisplatin_neg_45k.bed", help="Path to the negative cisplatin BED file.")
    parser.add_argument("--folder_name", type=str, default="intervention_reports", help="Folder name to save intervention reports.")

    args = parser.parse_args()

    print("---------- Intervention Parameters ----------")
    print(f"Feature Index: {args.feature}")
    print(f"Minimum Activation: {args.min_act}")
    print(f"Activation Factor: {args.act_factor}")
    print(f"CNN Model Path: {args.cnn}")
    print(f"SAE Model Path: {args.sae}")
    print(f"Cisplatin Positive BED: {args.cisplatin_positive}")
    print(f"Cisplatin Negative BED: {args.cisplatin_negative}")
    print(f"Saving Results to {args.folder_name}")
    print("---------------------------------------------")

    main(args.feature, args.min_act, args.act_factor, args.cnn, args.sae, 
         args.cisplatin_positive, args.cisplatin_negative, args.folder_name)