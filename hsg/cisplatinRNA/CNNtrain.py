import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from hsg.sae.protocol.checkpoint import History
from hsg.cisplatinRNA.CNNhead import CNNHead
from hsg.featureanalysis.regelementcorr import read_bed_file, get_sequences_from_dataframe
from hsg.stattools.features import get_latent_model

from biocommons.seqrepo import SeqRepo
from google.cloud import storage
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
import os, math

load_dotenv()

### Functions for data preparation and model training

def prepare_data(cisplatin_positive, cisplatin_negative) -> tuple[list, list, list]:

    print(" --- Reading cisplatin BED files --- ")
    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    positive_df = read_bed_file(cisplatin_positive, max_columns=6,) # limit=1000) # limit set for debugging
    print("Retrieving sequences from positive BED file...")
    positive_sequences = get_sequences_from_dataframe(positive_df, seqrepo=seqrepo, pad_size=0)
#    limit = len(positive_sequences) # there are a lot of negative sequences, so we limit how many we read to save time/memory
    positive_sequences = list(set(positive_sequences))  # remove duplicates
    print(f"Found {len(positive_sequences)} unique positive sequences.")
    del positive_df  # free memory

    negative_df = read_bed_file(cisplatin_negative, max_columns=6,) # limit=1000)  # limit set for debugging
    print(f"Retrieving sequences from negative BED file...")
    negative_sequences = get_sequences_from_dataframe(negative_df, seqrepo=seqrepo, pad_size=0)
    negative_sequences = list(set(negative_sequences))  # remove duplicates
    print(f"Found {len(negative_sequences)} unique negative sequences.")
    del negative_df  # free memory

    data = [(seq, [0, 1]) for seq in positive_sequences] + [(seq, [1, 0]) for seq in negative_sequences]
    del positive_sequences, negative_sequences  # free memory
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)

    return train_data, validation_data, test_data





def process_input(seqs, condition, prediction_head, upstream_model, device, vectorizer=None):
            if condition == "features":
                seq_tensor = torch.stack([prediction_head.pad_sequence(upstream_model(s), 1000) for s in seqs]).to(device)
            elif condition == "raw_tokens":
                stack = [upstream_model.tokenizer.encode_plus(s, padding="max_length", truncation=True, return_tensors="pt", max_length=1000)["input_ids"] for s in seqs]
                stack = vectorizer.vectorize_tokens([s.squeeze() for s in stack])
                seq_tensor = torch.stack(stack).float().to(device)
            else:  # condition == "embeddings"
                seq_tensor = torch.stack([prediction_head.pad_sequence(upstream_model(s, return_hidden_states=True)[1], 1000) for s in seqs]).to(device)
            
            return seq_tensor





def train(upstream_model, prediction_head, train, validate, condition:str, 
          layer_idx, early_stop_patience=10, epochs=100, batch_size=32, learning_rate=0.001, output_dir:str=None, vectorizer=None):
    """
    Train a CNN model for RNA sequence classification.
    """
    if condition not in ["embeddings", "features", "raw_tokens"]:
        raise ValueError("Condition must be either 'embeddings', 'features', or 'raw_tokens'.")

    print(f"Training {condition} model for layer {layer_idx}...")
    # for early stopping
    tracker = History(patience=early_stop_patience, layer=layer_idx)
    log_dir = os.path.join(output_dir, condition)

    # for tensorboard logging
    train_log_writer = SummaryWriter(log_dir=log_dir) 
    layout = {
        f"layer_{layer_idx}": {
            "Accuracy": ["Multiline", ["Accuracy/Train", "Accuracy/Val"]],
            "Loss": ["Multiline", ["Loss/Train", "Loss/Val"]],
        },
    }
    train_log_writer.add_custom_scalars(layout)

    # try to load prediction_head on GPU, if not enough memory, train on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        prediction_head.to(device)
    except torch.OutOfMemoryError:
        print("GPU memory insufficient to host both parent model and prediction_head. Attempting to train prediction_head on CPU.")
        device = torch.device("cpu")
        prediction_head.to(device)

    optimizer = torch.optim.Adam(prediction_head.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # since there's a lot of data, fragment dataset into sets for each epoch
    train_frag_size = len(train) // epochs + 1
    train_set = [train[i:i + train_frag_size] for i in range(0, len(train), train_frag_size)]
    val_frag_size = len(validate) // epochs + 1
    validate_set = [validate[i:i + val_frag_size] for i in range(0, len(validate), val_frag_size)]

    ### Main training loop ###

    for epoch in range(epochs):

        train: list = train_set[epoch % len(train_set)]
        validate: list = validate_set[epoch % len(validate_set)]

        # train & backprop
        train_losses = []
        labels = []
        predictions = []
        for sample in tqdm([train[i:i + batch_size] for i in range(0, len(train), batch_size)], desc=f"Training Epoch {epoch+1}/{epochs} - {condition}"):
            seqs = [s[0] for s in sample]
            seq_labels = [s[1] for s in sample]

            # get sequence tensor from upstream model, pad to max length
            seq_tensor = process_input(seqs, condition, prediction_head, upstream_model, device, vectorizer=vectorizer)
            label_tensor = torch.stack([torch.as_tensor(l, dtype=torch.float) for l in seq_labels]).to(device)

            prediction_head.train()
            optimizer.zero_grad()
            output = prediction_head(seq_tensor)
            loss = loss_function(output, label_tensor)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            predicted = torch.argmax(output, dim=1).tolist()
            label = torch.argmax(label_tensor, dim=1).tolist()
            predictions.extend(predicted)
            labels.extend(label)

            # memory cleanup fail-safe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # training epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = accuracy_score(labels, predictions)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {avg_train_accuracy:.4f}")

        # validation
        with torch.no_grad():
            prediction_head.eval()
            val_losses = []
            labels = []
            predictions = []

            for sample in tqdm([validate[i:i + batch_size] for i in range(0, len(validate), batch_size)], desc=f"Validating Epoch {epoch+1}/{epochs} - {condition}"):

                seqs = [s[0] for s in sample]
                seq_labels = [s[1] for s in sample]

                seq_tensor = process_input(seqs, condition, prediction_head, upstream_model, device, vectorizer=vectorizer)
                label_tensor = torch.stack([torch.as_tensor(l, dtype=torch.float) for l in seq_labels]).to(device)

                output = prediction_head(seq_tensor)
                loss = loss_function(output, label_tensor)
                val_losses.append(loss.item())

                predicted = torch.argmax(output, dim=1).tolist()
                label = torch.argmax(label_tensor, dim=1).tolist()
                predictions.extend(predicted)
                labels.extend(label)

                # memory cleanup fail-safe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # validation epoch summary
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_accuracy = accuracy_score(labels, predictions)
            print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_accuracy:.4f}")
        
        # checkpoint logic
        tracker.update(prediction_head, avg_train_loss, avg_val_loss, epoch)
        if tracker.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break
        # log to tensorboard
        train_log_writer.add_scalar(f"Loss/Train", avg_train_loss, epoch)
        train_log_writer.add_scalar(f"Loss/Val", avg_val_loss, epoch)
        train_log_writer.add_scalar(f"Accuracy/Train", avg_train_accuracy, epoch)
        train_log_writer.add_scalar(f"Accuracy/Val", avg_val_accuracy, epoch)
        train_log_writer.flush()

    ### Save Results ###

    print("Saving results...")

    # upload to GCS
    if output_dir.startswith("gs://"):
        client = storage.Client()
        bucket_name = output_dir.split("/")[2]
        blob_name = '/'.join(output_dir.split("/")[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'{blob_name}{condition}.pt')
        with blob.open("wb", ignore_flush=True) as f:
            torch.save(prediction_head, f)
        print(f"Model saved to {output_dir} as {blob_name}{condition}.pt")

    else:
        # save to local directory
        if os.path.exists(output_dir):
            torch.save(prediction_head, os.path.join(output_dir, f'{condition}.pt'))
        else:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(prediction_head, os.path.join(output_dir, f'{condition}.pt'))

    print("=== Done ===")
    print(f"Results saved to {output_dir}")

    return prediction_head





def evaluate(upstream_model, prediction_head, test_data, condition:str, batch_size:int, output_dir:str=None, vectorizer=None):
    print("Evaluating model...")
    with torch.no_grad():
        prediction_head.eval()
        device = prediction_head.fc.weight.device
        test_losses = []
        labels = []
        predictions = []
        results = []
        
        for sample in tqdm([test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)], desc=f"Evaluating {condition} model"):
            seqs = [s[0] for s in sample]
            seq_labels = [s[1] for s in sample]
            # get sequence tensor from upstream model, pad to max length
            seq_tensor = process_input(seqs, condition, prediction_head, upstream_model, device, vectorizer=vectorizer)
            label_tensor = torch.stack([torch.as_tensor(l, dtype=torch.float) for l in seq_labels]).to(device)

            output = prediction_head(seq_tensor)
            loss = nn.CrossEntropyLoss()(output, label_tensor)
            test_losses.append(loss.item())

            predicted = torch.argmax(output, dim=1).tolist()
            label = torch.argmax(label_tensor, dim=1).tolist()
            predictions.extend(predicted)
            labels.extend(label)

            results.append({
                "sequence": seqs,
                "predicted": predicted,
                "actual": label,
                "loss": loss.item(),
            })

            # memory cleanup fail-safe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # calculate overall accuracy and average loss
        accuracy = accuracy_score(labels, predictions)
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Average Test Accuracy: {accuracy:.4f}")

        print("Classification Report:")
        y_true = []
        y_pred = []
        for r in results:
            y_true.extend(r['actual'])
            y_pred.extend(r['predicted'])
        print(classification_report(y_true, y_pred))
        df = pd.DataFrame(results)

    # upload to GCS
    if output_dir.startswith("gs://"):
        client = storage.Client()
        bucket_name = output_dir.split("/")[2]
        blob_name = '/'.join(output_dir.split("/")[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'{blob_name}{condition}_results.csv')
        with blob.open("wb", ignore_flush=True) as f:
            df.to_csv(f, index=True)
        print(f"Results saved to {output_dir} as {blob_name}{condition}_results.csv")

    else:
        # save to local directory
        if os.path.exists(output_dir):
            df.to_csv(os.path.join(output_dir, f'{condition}_results.csv'), index=True)
        else:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, f'{condition}_results.csv'), index=True)

    # remove the prediction head from memory
    prediction_head.to("cpu")
    del prediction_head
    import gc
    gc.collect()

    print("=== Done ===")
    print(f"Results saved to {output_dir}")





### Compose everything into a main function

def main(cisplatin_positive, cisplatin_negative, layer_idx=23, exp_factor=8, early_stop_patience=10, 
         epochs=100, batch_size=32, learning_rate=0.001, sae_dir=None, output_path=None, condition="all"):
    """
    Conduct training and evaluation of a CNN model for RNA sequence classification.
    """
    # set default paths
    if sae_dir is None:
        sae_path = f"/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef{exp_factor}/sae/layer_{layer_idx}.pt"
    else:
        sae_path = f"{sae_dir}/layer_{layer_idx}.pt"

    if output_path is None:
        output_path = f"gs://hidden-state-genomics/cisplatinCNNheads/ef{exp_factor}/layer_{layer_idx}/"
    
    # Load data
    train_data, validation_data, test_data = prepare_data(cisplatin_positive, cisplatin_negative)
    # Initialize NTv2 model + SAE
    upstream_model = get_latent_model(os.environ["NT_MODEL"], layer_idx, sae_path=sae_path)
    # Freeze the upstream model weights
    for param in upstream_model.parameters():
        param.requires_grad = False

    # Train embedding model
    if condition in ["embeddings", "all"]:
        embedding_head = CNNHead(input_dim=upstream_model.parent_model.esm.encoder.layer[layer_idx].output.dense.out_features, seq_length=1000, output_dim=2)
        embedding_head = train(
            upstream_model, 
            embedding_head, 
            train_data, 
            validation_data, 
            condition="embeddings", 
            layer_idx=layer_idx,
            early_stop_patience=early_stop_patience,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            output_dir=output_path
        )

        evaluate(upstream_model=upstream_model, prediction_head=embedding_head, test_data=test_data, condition="embeddings", 
                batch_size=batch_size, output_dir=output_path)

    # Train feature model
    if condition in ["features", "all"]:
        feature_head = CNNHead(input_dim=upstream_model.sae.dict_size, seq_length=1000, output_dim=2)
        feature_head = train(
            upstream_model, 
            feature_head, 
            train_data, 
            validation_data, 
            condition="features", 
            layer_idx=layer_idx,
            early_stop_patience=early_stop_patience,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            output_dir=output_path
        )

        evaluate(upstream_model=upstream_model, prediction_head=feature_head, test_data=test_data, condition="features", 
                batch_size=batch_size, output_dir=output_path)

    # Train raw_tokens model
    if condition in ["raw_tokens", "all"]:
        from hsg.cisplatinRNA.vectorizer import Vectorizer
        vectorizer = Vectorizer()
        raw_tokens_head = CNNHead(input_dim=upstream_model.sae.dict_size, seq_length=1000, output_dim=2)
        raw_tokens_head = train(
            upstream_model, 
            raw_tokens_head, 
            train_data, 
            validation_data, 
            condition="raw_tokens", 
            layer_idx=layer_idx,
            early_stop_patience=early_stop_patience,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            output_dir=output_path,
            vectorizer=vectorizer,
        )

        evaluate(upstream_model=upstream_model, prediction_head=raw_tokens_head, test_data=test_data, condition="raw_tokens", 
                batch_size=batch_size, output_dir=output_path, vectorizer=vectorizer)





# Entry point for the script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate CNN models for RNA sequence classification.")
    parser.add_argument("--cisplatin_positive", type=str, required=True, help="Path to the positive cisplatin BED file.")
    parser.add_argument("--cisplatin_negative", type=str, required=True, help="Path to the negative cisplatin BED file.")
    parser.add_argument("--layer_idx", type=int, required=False, default=23, help="Layer index for the model.")
    parser.add_argument("--exp_factor", type=int, default=8, help="Expansion factor for the model.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--sae_dir", type=str, default=None, help="Directory containing SAE models.")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for saving results.")
    parser.add_argument("--condition", type=str, default="all", help="Condition to train: 'embeddings', 'features', 'raw_tokens', or 'all'.")

    args = parser.parse_args()

    main(
        args.cisplatin_positive,
        args.cisplatin_negative,
        args.layer_idx,
        exp_factor=args.exp_factor,
        early_stop_patience=args.early_stop_patience,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sae_dir=args.sae_dir,
        output_path=args.output_path,
        condition=args.condition
    )