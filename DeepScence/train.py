import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from DeepScence import logger
from torch.optim import Adam, RMSprop
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.stats import pearsonr


def train(
    model,
    adata,
    learning_rate=0.03,
    epochs=200,
    validation_split=0.1,
    early_stop=10,
    reduce_lr=10,
    batch_size=None,
    verbose=False,
):
    # get input normalized matrix
    if issparse(adata.X):
        X_input = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    else:
        X_input = torch.tensor(adata.X, dtype=torch.float32)

    # get output raw count matrix
    if issparse(adata.layers["raw_counts"]):
        raw_output = torch.tensor(
            adata.layers["raw_counts"].toarray(), dtype=torch.float32
        )
    else:
        raw_output = torch.tensor(adata.layers["raw_counts"], dtype=torch.float32)

    # get size factor, batch matrix
    sf = torch.tensor(adata.obs["size_factors"].values, dtype=torch.float32)
    batch_labels = adata.obs["batch"].unique()
    batch_matrix = np.zeros((adata.n_obs, len(batch_labels)))
    for i, batch_label in enumerate(batch_labels):
        batch_indices = adata.obs["batch"] == batch_label
        batch_matrix[batch_indices, i] = 1
    batch_matrix = torch.tensor(batch_matrix, dtype=torch.float32)

    dataset = TensorDataset(X_input, sf, batch_matrix, raw_output)

    if batch_size is None:  # default run without minibatching
        batch_size = adata.n_obs

    total_samples = len(dataset)
    val_size = int(total_samples * validation_split)
    train_size = total_samples - val_size

    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Training on {train_size} cells, validate on {val_size} cells.")
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Training on {train_size} cells, no validation.")

    # trainning loop
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = early_stop
    patience_counter = 0
    lr_patience = reduce_lr
    best_model_state = None  # save lowest val_loss weights

    for epoch in tqdm(range(epochs)):
        # training
        model.train()
        train_loss = 0
        for X, sf, batch_matrix, targets in train_loader:
            inputs = (X, sf, batch_matrix)
            optimizer.zero_grad()
            output = model(inputs)
            loss = model.loss(targets, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)  # divide by 1 if no minibatch
        train_losses.append(train_loss)

        # validation
        if val_size > 0:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0
                for X, sf, batch_matrix, targets in val_loader:
                    inputs = (X, sf, batch_matrix)
                    output = model(inputs)
                    loss = model.loss(targets, output)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
            val_losses.append(val_loss)
        else:
            val_loss = np.nan

        # record correlation
        encoded_scores = model.encoded_scores.detach().cpu().numpy()
        pearson_corr, _ = pearsonr(encoded_scores[:, 0], encoded_scores[:, 1])

        # if verbose:
        #     print(
        #         f"Epoch {epoch}, train_loss: {round(train_loss, 5)}, "
        #         f"val_loss: {round(val_loss, 5)}, "
        #         f"zinb_loss: {round(total_zinb_loss, 5)}, "
        #         f"ortho_loss: {round(total_ortho_loss, 5)}, "
        #         f"pearson r: {round(pearson_corr, 5)}"
        #     )

        # Early stopping logic
        if early_stop is not None and reduce_lr is not None:
            ## temporarily disabled - follow up
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            # Reduce learning rate if no improvement seen over lr_patience epochs
            min_lr = 1e-6
            if patience_counter > lr_patience:
                current_lr = optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * 0.5, min_lr)
                if current_lr > min_lr:
                    if verbose:
                        print(f"Reducing learning rate from {current_lr} to {new_lr}")
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr

            if patience_counter > patience:
                if verbose:
                    print(f"Stopping early at epoch {epoch + 1}")
                break
