import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class BalancedGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(num_features, num_classes))
        else:
            self.convs.append(GCNConv(num_features, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))
        
        self.reset_parameters_standard() 

    def reset_parameters_standard(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index): # ... (same as before)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1: x = F.relu(x)
        return x

    def get_weights_list(self): # ... (same as before)
        weights = []
        for conv in self.convs:
            if hasattr(conv, 'lin') and hasattr(conv.lin, 'weight'):
                 weights.append(conv.lin.weight.data) 
            elif hasattr(conv, 'weight'): 
                 weights.append(conv.weight.data)
            else: raise ValueError("Could not find weight tensor.")
        return weights
        
    # Add the analysis method here for consistency if using this class in the main script
    def get_all_intermediate_states_and_retain_grads(self, x, edge_index):
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


# --- Orthogonal + Balanced Initialization Function ---
def make_orthogonal(weight_matrix):
    """Makes a matrix orthogonal using SVD. Handles rectangular matrices."""
    # weight_matrix shape: [out_features, in_features]
    if weight_matrix.numel() == 0: return weight_matrix # Empty tensor
    rows, cols = weight_matrix.shape
    if rows == 0 or cols == 0: return weight_matrix

    with torch.no_grad():
        try:
            u, _, vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            # For W = U S Vh, W_ortho = U Vh
            # If rows < cols, U is (rows, rows), Vh is (rows, cols). U @ Vh is (rows, cols)
            # If rows > cols, U is (rows, cols), Vh is (cols, cols). U @ Vh is (rows, cols)
            # If rows == cols, U is (rows, rows), Vh is (rows, rows). U @ Vh is (rows, rows)
            orthogonal_matrix = u @ vh
            # Ensure the output shape is the same as input
            if orthogonal_matrix.shape != weight_matrix.shape:
                 # This can happen if full_matrices=True and matrix is non-square
                 # With full_matrices=False, U shape is (M, K) and Vh shape is (K, N) where K=min(M,N)
                 # For W_ortho = U @ Vh, if M<N, U is MxM, Vh is MxN. Result is MxN. OK.
                 # If M>N, U is MxN, Vh is NxN. Result is MxN. OK.
                 # So this should generally be fine.
                 print(f"Shape mismatch after SVD: Original {weight_matrix.shape}, Ortho {orthogonal_matrix.shape}")
                 return weight_matrix # Fallback
            return orthogonal_matrix
        except torch.linalg.LinAlgError as e:
            print(f"SVD failed for matrix of shape {weight_matrix.shape}: {e}. Returning original.")
            return weight_matrix # Fallback if SVD fails (e.g., all zeros)
        
# # --- Balanced Initialization Function ---
def initialize_gcn_balanced(model: BalancedGCN, beta_first_layer_norm_sq=2.0, init_method='xavier'):
    """
    Applies balanced initialization to the GCN model.
    The principle is || W_in_i ||^2 = || W_out_i ||^2 for each neuron i.

    Args:
        model: The BalancedGCN model instance.
        beta_first_layer_norm_sq: The target squared L2 norm for weights incoming
                                  to each neuron in the first *hidden* layer.
        init_method: 'xavier', 'kaiming', or 'randn' for initial random weights before scaling.
    """
    print(f"Applying balanced initialization with beta_first_layer_norm_sq = {beta_first_layer_norm_sq}")

    if model.num_layers == 1:
        print("Skipping balancing for 1-layer GCN. Using standard init.")
        # For a 1-layer GCN, the concept of balancing incoming/outgoing norms
        # for hidden neurons doesn't directly apply in the same way.
        # Standard init (e.g., Glorot) is often sufficient.
        # If you want to control its norm, you'd scale W^0[i,:] or W^0[:,j] directly.
        # Let's ensure its standard init is called.
        model.convs[0].reset_parameters()
        return

    # 1. Initialize all weights randomly first (to get non-zero values)
    for conv_idx, conv_layer in enumerate(model.convs):
        # Access the linear layer within GCNConv
        lin_layer = None
        if hasattr(conv_layer, 'lin') and isinstance(conv_layer.lin, torch.nn.Linear):
            lin_layer = conv_layer.lin
        elif isinstance(conv_layer, torch.nn.Linear): # Should not happen for GCNConv
            lin_layer = conv_layer
        else:
            # GCNConv might store weight directly if no bias or other specific configs
            # For PyG's default GCNConv, it has a self.lin
            try:
                conv_layer.reset_parameters() # Fallback to default init if lin not found
                if hasattr(conv_layer, 'lin'): lin_layer = conv_layer.lin
                else:
                    print(f"Warning: Could not find 'lin' sublayer in conv {conv_idx} for custom init. Using default reset.")
                    continue

            except Exception as e:
                 print(f"Error re-initializing conv {conv_idx}: {e}")
                 continue
        
        if lin_layer is None: continue

        if init_method == 'xavier':
            torch.nn.init.xavier_uniform_(lin_layer.weight)
        elif init_method == 'kaiming': # Good for ReLU
            torch.nn.init.kaiming_uniform_(lin_layer.weight, nonlinearity='relu')
        elif init_method == 'randn':
            torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=0.01) # Small std for randn
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        if lin_layer.bias is not None:
            torch.nn.init.zeros_(lin_layer.bias)

    # Balancing loop (from first hidden layer to output layer)
    # For GCN, layer l has weights W_l.
    # Layer 0: W_0 (input_dim, hidden_dim_0)
    # Layer 1: W_1 (hidden_dim_0, hidden_dim_1)
    # ...
    # Layer L-1: W_{L-1} (hidden_dim_{L-2}, num_classes)

    # Step 1: Scale incoming weights to the first hidden layer (model.convs[0])
    # W_0.weight is [hidden_channels, num_features]
    # We want || W_0[i, :] ||^2 = beta_first_layer_norm_sq (for each neuron i in the first hidden layer)
    W0_weight = model.convs[0].lin.weight # Shape: [out_dim_0, in_dim_0]
    num_neurons_layer0_out = W0_weight.shape[0]
    for i in range(num_neurons_layer0_out): # For each output neuron of the first GCN layer
        row_i = W0_weight[i, :]
        current_norm_sq = torch.sum(row_i**2)
        scale_factor = torch.sqrt(beta_first_layer_norm_sq / (current_norm_sq + 1e-9)) # Add epsilon
        with torch.no_grad():
            W0_weight[i, :] *= scale_factor
    
    # Step 2: Iteratively balance subsequent layers
    # For layer l (convs[l]), W_l.weight is [dim_out_l, dim_in_l]
    #   dim_in_l is dim_out_{l-1}
    # We need || W_l[:, j] ||^2 = || W_{l-1}[j, :] ||^2
    #   where j is an index for neurons in layer l-1 (output of convs[l-1])
    #   W_l[:, j] means the j-th column of W_l.weight (weights from neuron j of prev layer to all neurons of current layer)
    #   W_{l-1}[j, :] means j-th row of W_{l-1}.weight (weights from all inputs of prev layer to neuron j of prev layer)

    for l_idx in range(model.num_layers - 1): # l_idx goes from 0 to L-2
        # W_prev_layer is W_l (convs[l_idx].lin.weight)
        # W_curr_layer is W_{l+1} (convs[l_idx+1].lin.weight)
        W_prev_lin_weight = model.convs[l_idx].lin.weight # Shape [out_prev, in_prev]
        W_curr_lin_weight = model.convs[l_idx+1].lin.weight # Shape [out_curr, in_curr], where in_curr = out_prev

        num_neurons_intermediate = W_prev_lin_weight.shape[0] # Number of output neurons in layer l_idx
                                                            # (which is input neurons for layer l_idx+1)
        assert W_curr_lin_weight.shape[1] == num_neurons_intermediate # Sanity check

        for j in range(num_neurons_intermediate): # For each neuron j in the output of layer l_idx
            # Target norm_sq is || W_prev_lin_weight[j, :] ||^2 (norm of row j of W_prev)
            target_norm_sq_for_outgoing_j = torch.sum(W_prev_lin_weight[j, :]**2)
            
            # Column j of W_curr_lin_weight corresponds to weights *outgoing* from neuron j (of prev layer's output)
            # to all neurons in the current layer's output.
            col_j_curr = W_curr_lin_weight[:, j]
            current_norm_sq_outgoing_j = torch.sum(col_j_curr**2)
            
            scale_factor = torch.sqrt(target_norm_sq_for_outgoing_j / (current_norm_sq_outgoing_j + 1e-9))
            with torch.no_grad():
                W_curr_lin_weight[:, j] *= scale_factor

    print("Balanced initialization applied.")

def initialize_gcn_orthogonal_balanced(model: BalancedGCN, beta_first_layer_norm_sq=1.0, 
                                     init_method_before_ortho='randn_small_std'):
    """
    Applies Orthogonalization THEN Balanced Norm scaling.
    1. Initialize weights (e.g., randn).
    2. Make each weight matrix orthogonal using SVD.
    3. Scale rows of the first layer's orthogonalized W to have norm^2 = beta.
    4. Iteratively balance subsequent layers based on the previous layer's row norms.
       This subsequent balancing will likely break perfect orthogonality of those layers.
    """
    print(f"Applying ORTHOGONAL + BALANCED initialization with beta_first_layer_norm_sq = {beta_first_layer_norm_sq}")

    if model.num_layers == 1:
        print("Applying orthogonal init to 1-layer GCN.")
        W0_lin = model.convs[0].lin
        if init_method_before_ortho == 'randn_small_std':
            torch.nn.init.normal_(W0_lin.weight, mean=0.0, std=0.02) # Small std before ortho
        else: # xavier
            torch.nn.init.xavier_uniform_(W0_lin.weight)

        W0_lin.weight.data = make_orthogonal(W0_lin.weight.data)
        # Optionally scale rows of this single layer to beta_first_layer_norm_sq
        num_neurons_out = W0_lin.weight.shape[0]
        for i in range(num_neurons_out):
            row_i = W0_lin.weight.data[i, :]
            current_norm_sq = torch.sum(row_i**2)
            scale_factor = torch.sqrt(beta_first_layer_norm_sq / (current_norm_sq + 1e-9))
            W0_lin.weight.data[i, :] *= scale_factor
        if W0_lin.bias is not None: torch.nn.init.zeros_(W0_lin.bias)
        return

    # 1. & 2. Initialize and Orthogonalize all layers first
    for conv_idx, conv_layer in enumerate(model.convs):
        lin_layer = conv_layer.lin
        if init_method_before_ortho == 'randn_small_std':
            torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=0.02) # std matters less before SVD ortho
        elif init_method_before_ortho == 'xavier':
            torch.nn.init.xavier_uniform_(lin_layer.weight)
        else: # kaiming
            torch.nn.init.kaiming_uniform_(lin_layer.weight, nonlinearity='relu')

        if lin_layer.bias is not None:
            torch.nn.init.zeros_(lin_layer.bias)
        
        # Make it orthogonal
        lin_layer.weight.data = make_orthogonal(lin_layer.weight.data)
        # After this, rows of W (if out_dim <= in_dim) should have approx norm_sq = 1 (if in_dim is large enough for U to be dense)
        # or more generally, sum of squared singular values is rank.
        # For W_ortho = U @ Vh, if U is M x K and Vh is K x N (K=min(M,N))
        # If M <= N (out_dim <= in_dim): rows of U are orthonormal if K=M. Then rows of U@Vh have norm 1.
        # If M > N (out_dim > in_dim): cols of Vh are orthonormal if K=N. Then cols of U@Vh have norm 1.

    # 3. Scale incoming weights to the first hidden layer (model.convs[0])
    # W0_weight is [hidden_channels, num_features]
    W0_lin_weight = model.convs[0].lin.weight # Shape: [out_dim_0, in_dim_0]
    num_neurons_layer0_out = W0_lin_weight.shape[0]
    for i in range(num_neurons_layer0_out): 
        row_i = W0_lin_weight.data[i, :]
        current_norm_sq = torch.sum(row_i**2)
        # If it was made perfectly orthogonal and out_dim <= in_dim, current_norm_sq should be 1.
        # We scale it to beta_first_layer_norm_sq regardless.
        scale_factor = torch.sqrt(beta_first_layer_norm_sq / (current_norm_sq + 1e-9))
        W0_lin_weight.data[i, :] *= scale_factor
    
    # 4. Iteratively balance subsequent layers (this will disrupt orthogonality of W_curr)
    for l_idx in range(model.num_layers - 1): 
        W_prev_lin_weight = model.convs[l_idx].lin.weight.data # Shape [out_prev, in_prev]
        W_curr_lin_weight = model.convs[l_idx+1].lin.weight.data # Shape [out_curr, in_curr]

        num_neurons_intermediate = W_prev_lin_weight.shape[0]
        
        for j in range(num_neurons_intermediate): 
            target_norm_sq_for_outgoing_j = torch.sum(W_prev_lin_weight[j, :]**2) # Norm of row j of W_prev
            col_j_curr = W_curr_lin_weight[:, j]
            current_norm_sq_outgoing_j = torch.sum(col_j_curr**2)
            scale_factor = torch.sqrt(target_norm_sq_for_outgoing_j / (current_norm_sq_outgoing_j + 1e-9))
            W_curr_lin_weight[:, j] *= scale_factor

    print("Orthogonal + Balanced initialization applied.")


# --- Example Usage (Verification function remains the same) ---
def verify_balancing(model: BalancedGCN, beta_first_layer_norm_sq=1.0):
    # ... (same as before)
    if model.num_layers == 1: print("Verification skipped for 1-layer GCN."); return
    print("\nVerifying ORTHO+BALANCED norms:")
    W0_weight = model.convs[0].lin.weight
    print(f"Layer 0 (Input to Hidden 0): Target incoming norm_sq per neuron ~ {beta_first_layer_norm_sq:.4f}")
    for i in range(min(5, W0_weight.shape[0])):
        norm_sq = torch.sum(W0_weight[i, :]**2).item()
        print(f"  Neuron {i} incoming norm_sq: {norm_sq:.4f}")
        if not np.isclose(norm_sq, beta_first_layer_norm_sq, atol=1e-5):
             print(f"    WARNING: Neuron {i} norm_sq {norm_sq} deviates from beta {beta_first_layer_norm_sq}")


    for l_idx in range(model.num_layers - 1):
        W_prev_lin_weight = model.convs[l_idx].lin.weight
        W_curr_lin_weight = model.convs[l_idx+1].lin.weight
        num_neurons_intermediate = W_prev_lin_weight.shape[0]
        print(f"\nBalancing between Layer {l_idx} (output) and Layer {l_idx+1} (input from Layer {l_idx}):")
        for j in range(min(5, num_neurons_intermediate)):
            norm_sq_incoming_to_j_prev = torch.sum(W_prev_lin_weight[j, :]**2).item()
            norm_sq_outgoing_from_j_curr = torch.sum(W_curr_lin_weight[:, j]**2).item()
            print(f"  Neuron {j} of Layer {l_idx}'s output:")
            print(f"    || W^{l_idx}[{j},:] ||^2 (inc. to it): {norm_sq_incoming_to_j_prev:.4f}")
            print(f"    || W^{l_idx+1}[:,{j}] ||^2 (out. from it): {norm_sq_outgoing_from_j_curr:.4f}")
            assert np.isclose(norm_sq_incoming_to_j_prev, norm_sq_outgoing_from_j_curr, atol=1e-5), \
                   f"Mismatch for Layer {l_idx} neuron {j}!"
    print("Norm balancing verification successful (after orthogonalization).")


