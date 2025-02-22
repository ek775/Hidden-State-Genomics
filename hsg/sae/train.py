# load environment variables
from dotenv import load_dotenv
load_dotenv()

# built ins
import os, logging, sys
from random import shuffle

# external libs
import torch
from torch.utils.tensorboard import SummaryWriter
from biocommons.seqrepo import SeqRepo
from tqdm import tqdm
from tap import tapify

# hsg modules
from hsg.pipelines.hidden_state import load_model, get_unique_refseqs
from hsg.pipelines.variantmap import DNAVariantProcessor
from hsg.sae.dictionary import AutoEncoder
from hsg.sae.protocol.etl import extract_hidden_states
from hsg.sae.protocol.epoch import train, validate
from hsg.sae.math.loss import mse_reconstruction_loss
from hsg.sae.protocol.checkpoint import History

#####################################################################################################
# Main Functions
#####################################################################################################
def train_sae(
        parent_model, 
        tokenizer,
        train_seqs: list,
        layer_idx: int, 
        log_dir: str, 
        expansion_factor: int, 
        device:str, 
        num_epochs: int,
        shard_size: int,
        learning_rate: float,
        l1_penalty: float,
        l1_annealing_steps: int,
        early_stop_patience: int = 100,
        silent: bool = False
    ) -> AutoEncoder:
    """
    Train a SAE on the output of a layer of a parent model.
    """

    ### initializing stuffs ###

    # for early stopping
    tracker = History(patience=early_stop_patience, layer=layer_idx)

    # for tensorboard logging
    train_log_writer = SummaryWriter(log_dir=os.path.join(log_dir, f"layer_{layer_idx}")) 
    layout = {
        f"layer_{layer_idx}": {
            "Accuracy": ["Multiline", ["Accuracy/Train", "Accuracy/Val"]],
            "Loss": ["Multiline", ["Loss/Train", "Loss/Val"]],
        },
    }
    train_log_writer.add_custom_scalars(layout)

    # config SAE
    activation_dim = parent_model.esm.encoder.layer[layer_idx].output.dense.out_features
    num_latents = activation_dim * expansion_factor

    # try to load SAE on GPU, if not enough memory, train on CPU
    SAE = AutoEncoder(activation_dim=activation_dim, dict_size=num_latents)
    try:
        SAE.to(device)
    except torch.OutOfMemoryError:
        logging.error("GPU memory insufficient to host both parent model and SAE. Attempting to train SAE on CPU.")
        device = torch.device("cpu")
        SAE.to(device)

    # optimizer and loss function
    optimizer = torch.optim.Adam(SAE.parameters(), lr=learning_rate)
    loss_fn = mse_reconstruction_loss
    
    # create training, validation sets with 10% hold out,
    # shard the data due to immense size of dataset and resource/time limitations
    shuffle(train_seqs)
    split = len(train_seqs) // 10 # 10% validation split

    train_shards = train_seqs[:len(train_seqs)-split]
    train_shards = [train_shards[i:i + shard_size] for i in range(0, len(train_shards), shard_size)]

    val_shards = train_seqs[len(train_seqs)-split:]
    val_shards = [val_shards[i:i + shard_size] for i in range(0, len(val_shards), shard_size)]

    del train_seqs # memory management

    ### MAIN LOOP ###
    for epoch in range(1, num_epochs + 1):
        # select shards, allowing for looping if large shard size
        train_loc = epoch % len(train_shards)
        val_loc = epoch % len(val_shards)
        train_set = train_shards[train_loc]
        val_set = val_shards[val_loc]

        # train, treating each sequence as an SAE batch
        train_loss = 0
        train_acc = 0
        current_l1_penalty = l1_penalty * min(1, epoch / l1_annealing_steps)

        logging.info(f"Training SAE on layer {layer_idx} - Epoch {epoch} - L1 Penalty: {current_l1_penalty} - Shards: T={len(train_shards)} V={len(val_shards)}")
        for seq in tqdm(train_set, disable=silent):
            batch = extract_hidden_states(model=parent_model, sequence=seq, tokenizer=tokenizer, layer=layer_idx, device=device)
            train_loss, train_acc = train(SAE, batch, optimizer=optimizer, loss_fn=loss_fn, l1_penalty=current_l1_penalty)

        # validate
        logging.info("Validating...")
        val_loss = []
        val_acc = []
        for seq in tqdm(val_set, disable=silent):
            batch = extract_hidden_states(model=parent_model, sequence=seq, tokenizer=tokenizer, layer=layer_idx, device=device)
            loss, accuracy = validate(SAE, batch, loss_fn=loss_fn, l1_penalty=current_l1_penalty)
            val_loss.append(loss)
            val_acc.append(accuracy)

        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_acc = sum(val_acc) / len(val_acc)

        # checkpoint logic
        tracker.update(SAE, train_acc, avg_val_acc, epoch)
        if tracker.early_stop:
            logging.info(f"Early stopping at epoch {epoch}.")
            SAE = tracker.reload_checkpoint(SAE)
            train_log_writer.add_text("Best Restored Model:", str(tracker.best_metrics()))
            train_log_writer.flush()
            break

        # logging
        logging.info(f"Epoch {epoch} - Train Acc: {train_acc}, Val Acc: {avg_val_acc} - Train Loss: {loss}, Val Loss: {avg_val_loss}")

        train_log_writer.add_scalar("Accuracy/Train", train_acc, epoch)
        train_log_writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)
        train_log_writer.add_scalar("Loss/Train", train_loss, epoch)
        train_log_writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        train_log_writer.add_scalar("L1 Penalty", current_l1_penalty, epoch)
        train_log_writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        train_log_writer.flush()

        # server memory management
        torch.cuda.empty_cache()
        
    # wrap up
    train_log_writer.close()
    return SAE

#####################################################################################################

def train_all_layers(
        # logistics
        silent: bool = False,
        restart: bool = False,
        parent_model: str = os.environ["NT_MODEL"], 
        SAE_directory: str = "./data/sae", 
        log_dir: str = "./data/train_logs",
        variant_data: str = os.environ["CLIN_VAR_CSV"],
        early_stop_patience: int = 50,
        # parameters
        epochs: int = 1000,
        shard_size: int = 256,
        learning_rate: float = 0.0001,
        l1_penalty: float = 0.001,
        l1_annealing_steps: int = 100,
        dict_expansion_factor: int = 8,
        **kwargs
    ):
    """
    Command line interface to train a series of SAEs on the output of each layer of a parent model. Uses GPU if available.

    Args:
        silent (bool): If True, suppresses logging output. Useful for training on a server.
        parent_model (str): The name of the parent model to be loaded from huggingface.
        SAE_directory (str): The directory to save the trained SAEs to.
        log_dir (str): The directory to save the training logs to.
        variant_data (str): The path to the ClinVar VCF file to be used for training.
        epochs (int): The number of epochs to train for.
        shard_size (int): Number of sequences to train on per epoch.
        learning_rate (float): The learning rate to use for training.
        l1_penalty (float): The L1 penalty to use for training.
        l1_annealing_steps (int): The number of steps to anneal the L1 penalty over.
        dict_expansion_factor (int): The factor by which to expand the dictionary size.
    """

    # logging
    if silent:
        import warnings
        warnings.filterwarnings("ignore")
        logging.disable(logging.INFO)
        logging.disable(logging.WARNING)
        logging.disable(logging.DEBUG)
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("main").addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("main").addHandler(logging.StreamHandler(sys.stdout))

    # gather objects
    print("Gathering Resources...")

    model, tokenizer, device = load_model(parent_model)
    accessions = get_unique_refseqs(variant_data, DNAVariantProcessor())
    seq_repo = SeqRepo(os.environ["SEQREPO_PATH"])
    n_layers = len(model.esm.encoder.layer)

    print("===============================================================")
    print(f"Preparing to extract features from: {parent_model}")
    print(f"Training {n_layers} SAEs on {len(accessions)} sequences.")
    print(f"Logs will be written to: {log_dir}")
    print(f"SAEs will be saved to: {SAE_directory}\n")
    print("--- Parameters --- \n")
    print(f"Max Epochs: {epochs}")
    print(f"Shard Size: {shard_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"L1 Penalty: {l1_penalty}")
    print(f"L1 Annealing Steps: {l1_annealing_steps}")
    print(f"Dictionary Expansion Factor: {dict_expansion_factor}")
    print("===============================================================")

    # get sequences as strings
    print("Loading Sequences...")
    train_seqs = []
    for acc in tqdm(accessions, disable=silent):
        try:
            proxy = seq_repo[f"refseq:{acc}"]
            seq = proxy.__str__()
        except KeyError:
            logging.warning(f"Accession {acc} not found in SeqRepo. Skipping.")
            continue

        train_seqs.append(seq)
    print("Beginning Training...")

    # restart logic
    if restart:
        start = len(os.listdir(SAE_directory))
        print()
    else:
        start = 0

    # core training loop - may attempt multiprocessing later
    for layer in range(start, n_layers):

        sae = train_sae(
            parent_model=model, 
            train_seqs=train_seqs,
            tokenizer=tokenizer,
            layer_idx=layer, 
            log_dir=log_dir, 
            expansion_factor=dict_expansion_factor,
            device=device,
            num_epochs=epochs,
            shard_size=shard_size,
            learning_rate=learning_rate,
            l1_penalty=l1_penalty,
            l1_annealing_steps=l1_annealing_steps,
            silent=silent,
            early_stop_patience=early_stop_patience
        )

        # save trained SAEs
        save_path = os.path.join(SAE_directory, f"layer_{layer}.pt")

        if not os.path.exists(SAE_directory):
            os.makedirs(SAE_directory)
            torch.save(sae.state_dict(), save_path)

        # check for overwrites
        elif os.path.exists(save_path):

            logging.info(f"Layer {layer} already exists. attempting to save copy.")

            # if we find more than 10 copies of a trained SAE, ok to start overwriting.
            for i in range(1, 11):
                if not os.path.exists(os.path.join(SAE_directory, f"layer_{layer}({i}).pt")):
                    torch.save(sae.state_dict(), os.path.join(SAE_directory, f"layer_{layer}({i}).pt"))
                    break
                elif i == 10:
                    logging.info(f"Too many copies of layer {layer} exist. Overwriting.")
                    torch.save(sae.state_dict(), save_path)
                else:
                    continue
        # save or overwrite
        else:
            torch.save(sae.state_dict(), save_path)

### run module as cmdline tool ###
if __name__ == "__main__":
    tapify(train_all_layers)