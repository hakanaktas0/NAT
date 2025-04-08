import time
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
from tqdm import tqdm
import wandb
from zeus.monitor import ZeusMonitor

from embedding_hypernetwork.rnn_model import DynamicRNNModel
from embedding_hypernetwork.embeddings import (
    EmbeddingsDataset,
)


def collate_with_lengths(batch):
    """
    Custom collate function that returns padded sequences along with their lengths.

    Args:
        batch: List of tuples (sequence, target)

    Returns:
        padded_sequences: Tensor with padded sequences
        targets: Tensor with target values
        sequence_lengths: List of original sequence lengths
    """
    # Sort batch by sequence length (descending) - required for pack_padded_sequence
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    # Get sequences, targets and lengths
    sequences = [item[0] for item in batch]
    targets = [item[1].unsqueeze(0) for item in batch]
    sequence_lengths = [seq.size(0) for seq in sequences]

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    targets = torch.cat(targets, dim=0)

    return padded_sequences, targets, sequence_lengths


def embedding_reconstruction_loss(original_embeddings, predicted_embeddings, alpha=0.5):
    """
    Compute the reconstruction loss between original and predicted embeddings.

    Args:
        original_embeddings: Original embeddings
        predicted_embeddings: Predicted embeddings
    Returns:
        loss: Computed loss
    """

    return alpha * nn.MSELoss()(original_embeddings, predicted_embeddings) + (
        1 - alpha
    ) * nn.CosineEmbeddingLoss()(
        original_embeddings,
        predicted_embeddings,
        torch.ones(original_embeddings.shape[0]).to(original_embeddings.device),
    )


def load_model(model, args):
    # load the model state dict
    model.load_state_dict(
        torch.load(
            args.trained_model_path,
            weights_only=False,
            map_location=args.device,
        )["model_state_dict"]
    )


def train_rnn(
    model,
    train_dataset,
    valid_dataset=None,
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    device="cpu",
    save_dir="checkpoints",
    use_wandb=False,
    wandb_project="rnn-training",
    wandb_entity=None,
    wandb_name=None,
    combined_loss_alpha=1,
):
    """
    Train an RNN model on the provided EmbeddingsDataset.

    Args:
        model: The RNN model to train
        train_dataset: An EmbeddingsDataset instance for training
        valid_dataset: Optional EmbeddingsDataset for validation
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        use_wandb: Whether to use Weights & Biases tracking
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team name)
        wandb_name: WandB run name
        combined_loss_alpha: Alpha value for combined loss. 0 makes it cosine similarity, 1 makes it MSE

    Returns:
        Trained model
    """
    # Initialize wandb if enabled
    if use_wandb:
        config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "model_type": model.__class__.__name__,
            "hidden_dim": model.hidden_dim if hasattr(model, "hidden_dim") else None,
            "num_layers": model.num_layers if hasattr(model, "num_layers") else None,
            "rnn_type": model.rnn_type if hasattr(model, "rnn_type") else None,
            "task_type": "regression",
            "combined_loss_alpha": combined_loss_alpha,
        }
        wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_name, config=config
        )
        # Log model summary
        wandb.watch(model)

    # Initialize Zeus Monitor for overall training
    overall_monitor = ZeusMonitor(
        gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True
    )

    # Create data loaders with the new collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_lengths,
    )

    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_with_lengths,
        )

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()

    criterion = lambda x, y: embedding_reconstruction_loss(
        x,
        y,
        alpha=combined_loss_alpha,
    )

    # Add L1 loss for MAE calculation
    mae_metric = nn.L1Loss()
    mse_metric = nn.MSELoss()
    cosine_metric = nn.CosineEmbeddingLoss()

    # Ensure model is on the specified device
    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    model = model.to(device)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    # Start overall energy monitoring
    overall_monitor.begin_window("overall_training")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_maes = []  # Track MAE instead of accuracy

        overall_monitor.begin_window(f"epoch_{epoch+1}")

        # For tracking iteration energy
        iteration_energies = []

        start_time = time.time()

        for i, (inputs, targets, lengths) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            overall_monitor.begin_window(f"iteration_{i+1}")

            inputs, targets = inputs.to(device), targets.to(device)

            # Pack the padded sequences
            packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass with packed sequences
            outputs = model(packed_inputs, lengths)

            # Compute loss only on the target elements
            loss = criterion(
                outputs,
                targets,
            )

            # Compute MAE for tracking regression performance
            mae = mae_metric(outputs, targets)
            train_maes.append(mae.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            train_losses.append(loss.item())

            # End iteration energy monitoring
            energy_stats = overall_monitor.end_window(f"iteration_{i+1}")
            iteration_energy = energy_stats.total_energy
            iteration_energies.append(iteration_energy)

        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_maes)

        # Calculate average iteration energy
        avg_iter_energy = np.mean(iteration_energies)

        # Validation phase
        if valid_dataset:
            model.eval()
            val_losses = []
            val_maes = []  # Track MAE instead of accuracy
            val_mses = []
            val_cosines = []

            overall_monitor.begin_window(f"validation_epoch_{epoch+1}")

            with torch.no_grad():
                for inputs, targets, lengths in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Pack the padded sequences
                    packed_inputs = pack_padded_sequence(
                        inputs, lengths, batch_first=True
                    )

                    # Forward pass with packed sequences
                    outputs = model(packed_inputs, lengths)

                    loss = criterion(
                        outputs,
                        targets,
                    )
                    mae = mae_metric(outputs, targets)
                    mse = mse_metric(outputs, targets)
                    cosine_loss = cosine_metric(
                        outputs,
                        targets,
                        torch.ones(outputs.shape[0]).to(outputs.device),
                    )

                    # Track statistics
                    val_losses.append(loss.item())
                    val_maes.append(mae.item())
                    val_mses.append(mse.item())
                    val_cosines.append(cosine_loss.item())

            # Calculate validation metrics
            val_loss = np.mean(val_losses)
            val_mae = np.mean(val_maes)
            val_mse = np.mean(val_mses)
            val_cosine = np.mean(val_cosines)

            # End validation energy monitoring
            val_energy_stats = overall_monitor.end_window(f"validation_epoch_{epoch+1}")
            val_energy = val_energy_stats.total_energy

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                    },
                    f"{save_dir}/best_model.pt",
                )

            epoch_time = time.time() - start_time

            # End epoch energy monitoring
            epoch_energy_stats = overall_monitor.end_window(f"epoch_{epoch+1}")
            epoch_energy = epoch_energy_stats.total_energy

            print(
                f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.8f} - "
                f"Train MAE: {train_mae:.8f} - Val Loss: {val_loss:.8f} - Val MAE: {val_mae:.8f} - "
                f"Energy: {epoch_energy:.2f}J - Val Energy: {val_energy:.2f}J - Avg Iter Energy: {avg_iter_energy:.2f}J"
            )

            # Log metrics to wandb
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/mae": train_mae,
                        "val/loss": val_loss,
                        "val/mae": val_mae,
                        "val/mse": val_mse,
                        "val/cosine_similarity": val_cosine,
                        "energy/epoch": epoch_energy,
                        "energy/validation": val_energy,
                        "energy/avg_iteration": avg_iter_energy,
                        "epoch": epoch,
                        "epoch_time": epoch_time,
                    }
                )
        else:
            epoch_time = time.time() - start_time

            # End epoch energy monitoring (no validation)
            epoch_energy_stats = overall_monitor.end_window(f"epoch_{epoch+1}")
            epoch_energy = epoch_energy_stats.total_energy

            print(
                f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - "
                f"Train MAE: {train_mae:.4f} - Energy: {epoch_energy:.2f}J - Avg Iter Energy: {avg_iter_energy:.2f}J"
            )

            # Log metrics to wandb (training only with energy)
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/mae": train_mae,
                        "energy/epoch": epoch_energy,
                        "energy/avg_iteration": avg_iter_energy,
                        "epoch": epoch,
                        "epoch_time": epoch_time,
                    }
                )

    # Save checkpoint every epoch
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
        },
        f"{save_dir}/final_model.pt",
    )

    # End overall energy monitoring
    overall_energy_stats = overall_monitor.end_window("overall_training")
    total_energy = overall_energy_stats.total_energy

    if use_wandb:
        wandb.run.summary["total_training_energy"] = total_energy
        wandb.log({"energy/total": total_energy})

    print(f"Total training energy consumption: {total_energy:.2f}J")

    # Finish the wandb run
    if use_wandb:
        wandb.finish()

    return model


def validate_rnn(
    model,
    val_dataset,
    batch_size=32,
    device="cpu",
):
    """
    Validate the RNN model on the provided EmbeddingsDataset.

    Args:
        model: The RNN model to validate
        val_dataset: An EmbeddingsDataset instance for validation
        batch_size: Batch size for validation
        device: Device to validate on ('cuda' or 'cpu')

    Returns:
        Validation loss and MAE
    """
    # Create data loader with the new collate function
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_lengths,
    )

    # Initialize loss function
    criterion = nn.MSELoss()

    # Ensure model is on the specified device
    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    model = model.to(device)

    # Validation phase
    model.eval()
    val_losses = []
    val_maes = []  # Track MAE instead of accuracy
    cosine_losses = []

    with torch.no_grad():
        for inputs, targets, lengths in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Pack the padded sequences
            packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)

            # Forward pass with packed sequences
            outputs = model(packed_inputs, lengths)

            loss = criterion(outputs, targets)
            mae = nn.L1Loss()(outputs, targets)
            cosine_loss = nn.CosineEmbeddingLoss()(
                outputs,
                targets,
                torch.ones(outputs.shape[0]).to(outputs.device),
            )

            # Track statistics
            val_losses.append(loss.item())
            val_maes.append(mae.item())
            cosine_losses.append(cosine_loss.item())

    # Calculate validation metrics
    val_loss = np.mean(val_losses)
    val_mae = np.mean(val_maes)
    cosine_loss = np.mean(cosine_losses)

    print(
        f"Validation Loss: {val_loss:.8f} - Validation MAE: {val_mae:.8f}, Cosine Loss: {cosine_loss:.8f}"
    )

    return val_loss, val_mae, cosine_loss


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train RNN model on embeddings")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode to run the script in ('train', 'val' or 'test')",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the model files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for tracking",
    )
    parser.add_argument(
        "--input_dim", type=int, help="Input dimension for the RNN model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, help="Hidden dimension for the RNN model"
    )
    parser.add_argument(
        "--output_dim", type=int, help="Output dimension for the RNN model"
    )
    parser.add_argument(
        "--num_layers", type=int, help="Number of layers for the RNN model"
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        choices=["lstm", "gru", "rnn"],
        help="Type of RNN to use",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the RNN model",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--combined_loss_alpha",
        type=float,
        default=1,
        help="Alpha value for combined loss. 0 makes it cosine similarity, 1 makes it MSE",
    )
    args = parser.parse_args()

    # Create model and datasets
    model = DynamicRNNModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    )  # Initialize your model

    if args.mode == "train":
        summary(model, input_size=(1, args.input_dim), batch_dim=0)
        train_dataset = EmbeddingsDataset(
            args.model_dir, split="train", split_files_path="."
        )
        val_dataset = EmbeddingsDataset(
            args.model_dir, split="val", split_files_path="."
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Train the model
        trained_model = train_rnn(
            model,
            train_dataset,
            val_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            save_dir=f"{args.save_dir}-{timestamp}",
            use_wandb=args.use_wandb,  # Enable wandb tracking
            wandb_project="embedding-rnn",  # Set project name
            wandb_name=f"rnn-training-{timestamp}",  # Create unique run name
            combined_loss_alpha=args.combined_loss_alpha,
        )

        print("Training completed!")

    elif args.mode == "val":
        val_dataset = EmbeddingsDataset(
            args.model_dir, split="val", split_files_path="."
        )

        load_model(model, args)

        # Validate the model
        val_loss, val_mae, cosine_loss = validate_rnn(
            model,
            val_dataset,
            batch_size=args.batch_size,
            device=args.device,
        )

    elif args.mode == "test":
        test_dataset = EmbeddingsDataset(
            args.model_dir, split="test", split_files_path="."
        )

        load_model(model, args)

        # Validate the model
        test_loss, test_mae, cosine_loss = validate_rnn(
            model,
            test_dataset,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        print("Invalid mode. Please choose 'train', 'validate' or 'test'.")
        exit(1)
