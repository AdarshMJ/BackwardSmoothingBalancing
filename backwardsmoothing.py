# import argparse
# import torch
# import torch.nn.functional as F
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import sparse

# from dataset import CustomDataset
# from model import NodeGCN, NodeGAT, NodeGraphConv # Ensure these have the new method
# from train import train, test, get_model, set_seed

# # --- Configuration ---
# SYN_CORA_ROOT = "syn-cora"
# SYN_CORA_NAME = "h0.30-r1" 

# MODEL_TYPE = 'gcn' 
# HIDDEN_DIM = 32
# NUM_LAYERS_GNN = 15
# GAT_HEADS = 4
# LEARNING_RATE = 0.01
# WEIGHT_DECAY = 5e-4
# EPOCHS = 100 
# SEED = 42

# # --- Helper Functions (Pairwise Diffs, etc. - can be reused) ---
# def calculate_pairwise_differences_E_squared(signal_matrix_numpy):
#     if signal_matrix_numpy is None or signal_matrix_numpy.shape[0] < 2:
#         return np.nan
#     if signal_matrix_numpy.ndim == 1:
#         return np.var(signal_matrix_numpy)
#     node_variances_per_feature = np.var(signal_matrix_numpy, axis=0)
#     return np.sum(node_variances_per_feature)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_name', type=str, default=SYN_CORA_NAME)
#     parser.add_argument('--model_type', type=str, default=MODEL_TYPE, choices=['gcn', 'gat', 'graphconv'])
#     parser.add_argument('--num_layers', type=int, default=NUM_LAYERS_GNN)
#     # ... (other args)
#     args = parser.parse_args() # Use defaults if not passed

#     set_seed(args.seed if hasattr(args,'seed') else SEED)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # --- Load Data (same as before) ---
#     print(f"Loading dataset: {args.dataset_name} from {SYN_CORA_ROOT}")
#     syn_dataset_obj = CustomDataset(root=SYN_CORA_ROOT, name=args.dataset_name, setting='gcn', 
#                                     seed=(args.seed if hasattr(args,'seed') else SEED))
#     if sparse.issparse(syn_dataset_obj.features): features_np = syn_dataset_obj.features.toarray()
#     else: features_np = syn_dataset_obj.features
#     labels_np = syn_dataset_obj.labels
#     X_torch = torch.FloatTensor(features_np).to(device)
#     labels_torch = torch.LongTensor(labels_np).to(device)
#     edge_index_coo = syn_dataset_obj.adj.tocoo()
#     edge_index_torch = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long).to(device)
#     train_mask_torch = torch.zeros(X_torch.shape[0], dtype=torch.bool).to(device)
#     val_mask_torch = torch.zeros(X_torch.shape[0], dtype=torch.bool).to(device)
#     test_mask_torch = torch.zeros(X_torch.shape[0], dtype=torch.bool).to(device)
#     train_mask_torch[syn_dataset_obj.idx_train] = True
#     val_mask_torch[syn_dataset_obj.idx_val] = True
#     test_mask_torch[syn_dataset_obj.idx_test] = True
#     print(f"Dataset '{args.dataset_name}' loaded. Nodes: {X_torch.shape[0]}")

#     # --- Train GNN Model (same as before) ---
#     num_features = X_torch.shape[1]
#     num_classes = labels_torch.max().item() + 1
#     model = get_model(args.model_type, num_features, num_classes,
#                       args.hidden_dim if hasattr(args,'hidden_dim') else HIDDEN_DIM, 
#                       args.num_layers, 
#                       args.gat_heads if hasattr(args,'gat_heads') else GAT_HEADS).to(device)
#     print(f"\nTraining {args.model_type.upper()} model with {args.num_layers} layers...")
#     optimizer = torch.optim.Adam(model.parameters(), 
#                                  lr=(args.lr if hasattr(args,'lr') else LEARNING_RATE), 
#                                  weight_decay=(args.weight_decay if hasattr(args,'weight_decay') else WEIGHT_DECAY))
#     best_val_acc = 0
#     best_model_state = model.state_dict() # Initialize with current state

#     for epoch in range(1, (args.epochs if hasattr(args,'epochs') else EPOCHS) + 1):
#         loss_val = train(model, X_torch, edge_index_torch, train_mask_torch, labels_torch, optimizer, device)
#         val_acc = test(model, X_torch, edge_index_torch, val_mask_torch, labels_torch, device)
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict()
#         if epoch % 20 == 0: print(f'Epoch: {epoch:03d}, Loss: {loss_val:.4f}, Val Acc: {val_acc:.4f}')
#     model.load_state_dict(best_model_state)
#     final_test_acc = test(model, X_torch, edge_index_torch, test_mask_torch, labels_torch, device)
#     print(f"Training complete. Best Val Acc: {best_val_acc:.4f}, Final Test Acc: {final_test_acc:.4f}")

#     # --- Analysis with Actual Gradients ---
#     model.eval()
    
#     # 1. Forward pass to get H(k) states and call retain_grad()
#     # H_states_torch_analysis will contain the H(k) tensors that require gradients
#     H_states_torch_analysis, X_states_torch_analysis = model.get_all_intermediate_states_and_retain_grads(X_torch, edge_index_torch)

#     # 2. Compute a loss (e.g., on training data)
#     # The output of the last H state is the logits
#     output_logits = H_states_torch_analysis[-1] # This is H(L-1)
#     # We need to compute loss on some nodes, e.g., training nodes
#     loss_for_backward = F.cross_entropy(output_logits[train_mask_torch], labels_torch[train_mask_torch])
    
#     # 3. Perform backward pass to populate .grad attributes
#     optimizer.zero_grad() # Clear any old gradients from model parameters
#     loss_for_backward.backward()

#     # 4. Collect B(k) = dL/dH(k) which are now in H_state.grad
#     B_states_np_actual = []
#     for k_idx, h_k_tensor in enumerate(H_states_torch_analysis):
#         if h_k_tensor.grad is not None:
#             B_states_np_actual.append(h_k_tensor.grad.clone().cpu().detach().numpy())
#             # print(f"Collected B({k_idx}) (dL/dH({k_idx})) shape: {B_states_np_actual[-1].shape}")
#         else:
#             # This might happen if H(k) was not part of the computation graph for the loss
#             # (e.g., if a layer's output was detached unintentionally earlier, or if it's an input layer)
#             # Or if the H(k) is the final output layer itself, its grad w.r.t. itself is 1 for sum, but dL/dH(L-1) is what we get.
#             print(f"Warning: No gradient found for H({k_idx}).")
#             B_states_np_actual.append(np.full_like(h_k_tensor.cpu().detach().numpy(), np.nan))


#     # --- Calculate Pairwise Differences ---
#     # Forward signals (X(k) are post-activation)
#     # X_states_torch_analysis[0] is X0, X_states_torch_analysis[k] is X(k) for k>0
#     pairwise_diff_forward_Esq = []
#     for x_k_tensor in X_states_torch_analysis:
#         pairwise_diff_forward_Esq.append(calculate_pairwise_differences_E_squared(x_k_tensor.cpu().numpy()))

#     # Backward signals (B(k) = dL/dH(k))
#     # B_states_np_actual[k] is dL/dH(k)
#     pairwise_diff_backward_Esq_actual = []
#     for b_k_np in B_states_np_actual:
#         pairwise_diff_backward_Esq_actual.append(calculate_pairwise_differences_E_squared(b_k_np))


#     # --- Plotting ---
#     fig, axes = plt.subplots(1, 1, figsize=(12, 7))
    
#     # Forward: X_states_torch_analysis has L+1 elements (X0 to XL=Logits)
#     num_X_states = len(pairwise_diff_forward_Esq)
#     layer_indices_forward_plot = np.arange(num_X_states) 
#     axes.plot(layer_indices_forward_plot, pairwise_diff_forward_Esq, marker='o', linestyle='-', 
#               label=f'Forward $E(X^{{(k)}})^2$ (Post-Activation)')

#     # Backward: B_states_np_actual has L elements (dL/dH(0) to dL/dH(L-1))
#     num_B_states = len(pairwise_diff_backward_Esq_actual)
#     layer_indices_backward_plot = np.arange(num_B_states)
#     axes.plot(layer_indices_backward_plot, pairwise_diff_backward_Esq_actual, marker='x', linestyle='--', color='r', 
#               label=f'Backward $E(B^{{(k)}})^2$ (Actual $dL/dH^{{(k)}}$)')

#     axes.set_xlabel('Layer Index k', fontsize=14)
#     axes.set_ylabel('Pairwise Difference Metric $E(Signal)^2$', fontsize=14)
#     title_str = f'Signal Smoothness (Actual Errors) on {args.dataset_name}\n(Trained {args.model_type.upper()}, {args.num_layers} Layers, Test Acc: {final_test_acc:.3f})'
#     axes.set_title(title_str, fontsize=16)
#     axes.set_yscale('log')
#     axes.grid(True, which="both", ls="--", alpha=0.7)
#     axes.legend(fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"oversmoothing_actual_errors_{args.model_type}_{args.dataset_name}_{args.num_layers}L.png")
#     plt.show()

# if __name__ == '__main__':
#     main()
import argparse
import torch
import torch.nn.functional as F # Added for loss computation
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from dataset import CustomDataset
# Assuming model.py contains the original NodeGCN, NodeGAT, NodeGraphConv
# And gcn_balanced_init.py contains BalancedGCN and initialize_gcn_balanced
from model import NodeGCN as StandardNodeGCN # Alias to avoid name clash
from balancedGCN import BalancedGCN, initialize_gcn_balanced,initialize_gcn_orthogonal_balanced
from train import train, test, get_model as get_standard_model, set_seed # get_model might need adjustment or we use direct instantiation
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

# --- Configuration (can be overridden by argparse) ---
SYN_CORA_ROOT = "syn-cora"
SYN_CORA_NAME = "h0.70-r1" 

MODEL_TYPE_ANALYSIS = 'gcn' # We will focus on GCN for balanced init comparison
HIDDEN_DIM = 32
NUM_LAYERS_GNN = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
SEED = 42
BETA_BALANCED_INIT = 2.0 # For balanced initialization

# --- Helper Functions (Pairwise Diffs, etc. - can be reused) ---
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
    set_seed(args.seed)
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
        # Use StandardNodeGCN if it has the get_all_intermediate_states_and_retain_grads method
        # Or adapt BalancedGCN to have it too, and just control init method.
        # For simplicity, let's assume BalancedGCN can also be used for standard by not calling initialize_gcn_balanced
        # but its reset_parameters_standard() is called by default.
        # We need get_all_intermediate_states_and_retain_grads method in the model used.
        # Let's assume BalancedGCN has it or we add it similar to the previous NodeGCN modification.
        
        # To ensure 'get_all_intermediate_states_and_retain_grads' exists for standard GCN:
        # We can either:
        # 1. Use StandardNodeGCN and ensure it has this method (as modified in previous step)
        # 2. Modify BalancedGCN to also have this method and use it for both cases.
        # Let's assume we modify BalancedGCN to have this method for consistency.
        
        # Add get_all_intermediate_states_and_retain_grads to BalancedGCN if not already there
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
        # model.reset_parameters_standard() # Called by __init__, ensure it's Glorot

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
    