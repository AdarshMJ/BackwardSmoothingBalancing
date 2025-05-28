import argparse
import torch
import torch.nn.functional as F # Added for loss computation
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from dataset import CustomDataset
from model import NodeGCN as StandardNodeGCN 
from balancedGCN import BalancedGCN, initialize_gcn_balanced,initialize_gcn_orthogonal_balanced
from train import train, test, get_model as get_standard_model, set_seed
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)  # Set a global seed for reproducibility
SYN_CORA_ROOT = "syn-cora"
SYN_CORA_NAME = "h0.70-r1" 

MODEL_TYPE_ANALYSIS = 'gcn' 
HIDDEN_DIM = 32
NUM_LAYERS_GNN = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
SEED = 42

def calculate_pairwise_differences_E_squared(signal_matrix_numpy):
    if signal_matrix_numpy is None or not isinstance(signal_matrix_numpy, np.ndarray) or signal_matrix_numpy.shape[0] < 2:
        return np.nan
    if signal_matrix_numpy.ndim == 1:
        return np.var(signal_matrix_numpy)
    node_variances_per_feature = np.var(signal_matrix_numpy, axis=0)
    return np.sum(node_variances_per_feature)

def train_and_analyze_model(args, X_torch, edge_index_torch, labels_torch, 
                            train_mask_torch, val_mask_torch, test_mask_torch,
                            device, use_balanced_init=False, 
                            is_orthogonal_too=False, # New flag
                            beta_val=2.0):
    num_features = X_torch.shape[1]
    num_classes = labels_torch.max().item() + 1
    model = BalancedGCN(num_features, num_classes, args.hidden_dim, args.num_layers).to(device)

    if use_balanced_init:
        if is_orthogonal_too:
            print(f"\n--- Training GCN with ORTHOGONAL+BALANCED Initialization (beta_sq={beta_val:.1f}) ---")
            # beta_val for ortho is often 1.0 if rows are made unit norm then scaled to beta
            # or it's inherent from a construction like LLOrtho (e.g. 2.0)
            initialize_gcn_orthogonal_balanced(model, beta_first_layer_norm_sq=beta_val, 
                                             init_method_before_ortho='randn_small_std')
        else:
            print(f"\n--- Training GCN with BALANCED Initialization (beta_sq={beta_val:.1f}) ---")
            initialize_gcn_balanced(model, beta_first_layer_norm_sq=beta_val, init_method='xavier')
    else:
        print("\n--- Training GCN with STANDARD Initialization ---")
        model.reset_parameters_standard() # Ensure it's standard Glorot
        
        if not hasattr(BalancedGCN, 'get_all_intermediate_states_and_retain_grads'):
            def _get_all_intermediate_states_and_retain_grads(self, x, edge_index):
                self.eval()
                H_states = []
                X_states = [x.clone().detach()]
                current_x_prop = x
                for i, conv in enumerate(self.convs):
                    h_k = conv(current_x_prop, edge_index)
                    h_k.retain_grad()
                    H_states.append(h_k)
                    if i < len(self.convs) - 1: current_x_activated = F.relu(h_k)
                    else: current_x_activated = h_k
                    X_states.append(current_x_activated.clone().detach())
                    current_x_prop = current_x_activated
                return H_states, X_states
            BalancedGCN.get_all_intermediate_states_and_retain_grads = _get_all_intermediate_states_and_retain_grads

        model = BalancedGCN(num_features, num_classes, 
                            args.hidden_dim, args.num_layers).to(device)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    best_model_state = model.state_dict()

    for epoch in range(1, args.epochs + 1):
        loss_val_train = train(model, X_torch, edge_index_torch, train_mask_torch, labels_torch, optimizer, device)
        val_acc = test(model, X_torch, edge_index_torch, val_mask_torch, labels_torch, device)
        if val_acc > best_val_acc: best_val_acc = val_acc; best_model_state = model.state_dict()
        if epoch % 20 == 0: print(f'Epoch: {epoch:03d}, Loss: {loss_val_train:.4f}, Val Acc: {val_acc:.4f}')
    model.load_state_dict(best_model_state)
    final_test_acc = test(model, X_torch, edge_index_torch, test_mask_torch, labels_torch, device)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}, Final Test Acc: {final_test_acc:.4f}")
    model.eval()
    H_states_torch, X_states_torch = model.get_all_intermediate_states_and_retain_grads(X_torch, edge_index_torch)
    output_logits = H_states_torch[-1]
    loss_for_backward = F.cross_entropy(output_logits[train_mask_torch], labels_torch[train_mask_torch])
    optimizer.zero_grad(); loss_for_backward.backward()
    B_states_np_actual = []
    for h_k_tensor in H_states_torch:
        if h_k_tensor.grad is not None: B_states_np_actual.append(h_k_tensor.grad.clone().cpu().detach().numpy())
        else: B_states_np_actual.append(np.full_like(h_k_tensor.cpu().detach().numpy(), np.nan))
    pairwise_diff_forward_Esq = [calculate_pairwise_differences_E_squared(x_k.cpu().numpy()) for x_k in X_states_torch]
    pairwise_diff_backward_Esq_actual = [calculate_pairwise_differences_E_squared(b_k) for b_k in B_states_np_actual]
    return pairwise_diff_forward_Esq, pairwise_diff_backward_Esq_actual, final_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=SYN_CORA_NAME)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS_GNN)
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--beta_balanced', type=float, default=BETA_BALANCED_INIT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Data ---
    print(f"Loading dataset: {args.dataset_name} from {SYN_CORA_ROOT}")
    syn_dataset_obj = CustomDataset(root=SYN_CORA_ROOT, name=args.dataset_name, setting='gcn', seed=args.seed)
    if sparse.issparse(syn_dataset_obj.features): features_np = syn_dataset_obj.features.toarray()
    else: features_np = syn_dataset_obj.features
    labels_np = syn_dataset_obj.labels
    X_torch_data = torch.FloatTensor(features_np).to(device)
    labels_torch_data = torch.LongTensor(labels_np).to(device)
    edge_index_coo = syn_dataset_obj.adj.tocoo()
    edge_index_torch_data = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long).to(device)
    train_mask_torch_data = torch.zeros(X_torch_data.shape[0], dtype=torch.bool).to(device); train_mask_torch_data[syn_dataset_obj.idx_train] = True
    val_mask_torch_data = torch.zeros(X_torch_data.shape[0], dtype=torch.bool).to(device); val_mask_torch_data[syn_dataset_obj.idx_val] = True
    test_mask_torch_data = torch.zeros(X_torch_data.shape[0], dtype=torch.bool).to(device); test_mask_torch_data[syn_dataset_obj.idx_test] = True
    print(f"Dataset '{args.dataset_name}' loaded. Nodes: {X_torch_data.shape[0]}")

    # --- Run 1: Standard Initialization ---
    fwd_std, bwd_std, acc_std = train_and_analyze_model(
        args, X_torch_data, edge_index_torch_data, labels_torch_data,
        train_mask_torch_data, val_mask_torch_data, test_mask_torch_data,
        device, use_balanced_init=False
    )

    # --- Run 2: Balanced Initialization ---
    fwd_bal, bwd_bal, acc_bal = train_and_analyze_model(
        args, X_torch_data, edge_index_torch_data, labels_torch_data,
        train_mask_torch_data, val_mask_torch_data, test_mask_torch_data,
        device, use_balanced_init=True, beta_val=args.beta_balanced
    )

    fwd_ortho_bal, bwd_ortho_bal, acc_ortho_bal = train_and_analyze_model(
    args, X_torch_data, edge_index_torch_data, labels_torch_data,
    train_mask_torch_data, val_mask_torch_data, test_mask_torch_data,
    device, use_balanced_init=True, # Still True, but init function changes
    is_orthogonal_too=True, # New flag
    beta_val=args.beta_balanced # Or a specific beta for ortho, e.g., 1.0
)

    # --- Plotting Comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    num_X_states = args.num_layers + 1
    layer_indices_fwd = np.arange(num_X_states)
    num_B_states = args.num_layers
    layer_indices_bwd = np.arange(num_B_states)

    # Forward Plot
    axes[0].plot(layer_indices_fwd, fwd_std, marker='o', linestyle='-', label=f'Standard Init - Fwd $E(X^{{(k)}})^2$ (Acc: {acc_std:.3f})')
    axes[0].plot(layer_indices_fwd, fwd_bal, marker='s', linestyle='--', label=f'Balanced Init - Fwd $E(X^{{(k)}})^2$ (Acc: {acc_bal:.3f})')
    axes[0].plot(layer_indices_fwd, fwd_ortho_bal, marker='D', linestyle=':', label=f'Ortho+Bal Init - Fwd $E(X^{{(k)}})^2$ (Acc: {acc_ortho_bal:.3f})')
    axes[0].set_ylabel('Pairwise Diff. $E(X)^2$', fontsize=14)
    axes[0].set_title(f'Forward Signal Smoothing Comparison on {args.dataset_name} ({args.num_layers} Layers)', fontsize=16)
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", ls="--", alpha=0.7)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis='y', labelsize=12)


    # Backward Plot
    axes[1].plot(layer_indices_bwd, bwd_std, marker='x', linestyle='-', label=f'Standard Init - Bwd $E(B^{{(k)}})^2$')
    axes[1].plot(layer_indices_bwd, bwd_bal, marker='P', linestyle='--', label=f'Balanced Init - Bwd $E(B^{{(k)}})^2$')
    axes[1].plot(layer_indices_bwd, bwd_ortho_bal, marker='v', linestyle=':', label=f'Ortho+Bal Init - Bwd $E(B^{{(k)}})^2$')
    axes[1].set_xlabel('Layer Index k', fontsize=14)
    axes[1].set_ylabel('Pairwise Diff. $E(B)^2$', fontsize=14)
    axes[1].set_title(f'Backward Signal Smoothing Comparison (Actual $dL/dH^{{(k)}}$)', fontsize=16)
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", ls="--", alpha=0.7)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(axis='both', labelsize=12)


    plt.tight_layout()
    plt.savefig(f"{args.dataset_name}_{args.num_layers}L_{args.beta_balanced}.png")
    plt.show()

if __name__ == '__main__':
    main()
    
