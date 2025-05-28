# model.py (partial - showing changes for NodeGCN for hook registration)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GraphConv as PyGGraphConv

class NodeGCN(torch.nn.Module):
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
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        # ... (original forward method remains the same for training) ...
        post_activation_embeddings = [] 
        current_x = x
        for i, conv in enumerate(self.convs):
            current_x = conv(current_x, edge_index)
            if i < len(self.convs) - 1:
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, p=0.0, training=self.training) 
                post_activation_embeddings.append(current_x)
        final_output = current_x
        if return_node_emb:
            if not post_activation_embeddings and self.num_layers > 1 :
                 raise ValueError("No hidden embeddings available.")
            if embedding_layer is not None:
                if embedding_layer >= len(post_activation_embeddings):
                    raise ValueError(f"embedding_layer index out of bounds.")
                return final_output, post_activation_embeddings[embedding_layer]
            if post_activation_embeddings:
                return final_output, post_activation_embeddings[-1]
            else: 
                return final_output, x 
        return final_output


    def get_all_intermediate_states_and_register_hooks(self, x, edge_index, backward_gradients_H):
        """
        Performs a forward pass, stores H(k) and X(k), and registers backward hooks on H(k).
        backward_gradients_H: A list to store the captured dL/dH(k) gradients.
        """
        self.eval()
        H_states_for_analysis = [] # Store H(k)
        X_states_for_analysis = [x.clone().detach()] # X(0)

        # Clear previous gradients list for this new forward-backward pass
        backward_gradients_H.clear() 

        current_x_prop = x
        for i, conv in enumerate(self.convs):
            # H(i) = W(i) @ (P @ X(i)) is computed inside conv
            # We need H(i) *before* activation for dL/dH(i)
            
            # To get H(i), we need to "intercept" it if GCNConv applies activation internally.
            # Standard GCNConv from PyG does: Aggr(XW). So the output of conv IS H(i).
            h_k = conv(current_x_prop, edge_index)
            h_k.retain_grad() # Ensure grad is saved for this intermediate tensor H(k)
            H_states_for_analysis.append(h_k) # Store H(k) before detaching for hook

            # Register hook for dL/dH(k)
            # The hook function will be called with the gradient of the output of this H_k tensor
            def hook_fn(grad, k_idx=i): # Use default arg to capture k_idx at definition time
                # print(f"Hook called for H({k_idx}), grad shape: {grad.shape if grad is not None else 'None'}")
                if grad is not None:
                    backward_gradients_H.append(grad.clone().detach()) 
                else:
                    backward_gradients_H.append(None) 


            # h_k.register_hook(hook_fn) 
            # The above hook is on the gradient of h_k *itself* which is dL/dh_k.
            # This is what we want: B(k) = dL/dH(k)
            
            # --- Activation and next X ---
            if i < len(self.convs) - 1: # Not the last GCN layer
                current_x_activated = F.relu(h_k)
            else: # Last layer, output is H(L-1) (logits)
                current_x_activated = h_k 
            
            X_states_for_analysis.append(current_x_activated.clone().detach())
            current_x_prop = current_x_activated # Input for next layer is the activated output
            
        return H_states_for_analysis, X_states_for_analysis # H_states now has .grad populated after backward()

    def get_all_intermediate_states_and_retain_grads(self, x, edge_index):
        """
        Performs a forward pass, stores H(k) and X(k), and calls retain_grad() on H(k).
        Returns H_states and X_states for gradient analysis.
        """
        self.eval()
        H_states_for_analysis = []
        X_states_for_analysis = [x.clone().detach()]
        
        current_x_prop = x
        for i, conv in enumerate(self.convs):
            h_k = conv(current_x_prop, edge_index)
            h_k.retain_grad() # Key step
            H_states_for_analysis.append(h_k)
            
            if i < len(self.convs) - 1:
                current_x_activated = F.relu(h_k)
            else:
                current_x_activated = h_k
            
            X_states_for_analysis.append(current_x_activated.clone().detach())
            current_x_prop = current_x_activated
        
        return H_states_for_analysis, X_states_for_analysis

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class NodeGAT(torch.nn.Module): # Simplified example, focusing on retain_grad
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.heads = heads
        if num_layers == 1:
            self.convs.append(GATv2Conv(num_features, num_classes, heads=1, concat=False))
        else:
            self.convs.append(GATv2Conv(num_features, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.convs.append(GATv2Conv(hidden_channels * heads, num_classes, heads=1, concat=False))
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        # ... (original forward) ...
        post_activation_embeddings = []
        current_x = x
        for i, conv in enumerate(self.convs):
            current_x = conv(current_x, edge_index)
            if i < len(self.convs) - 1:
                current_x = F.elu(current_x)
                post_activation_embeddings.append(current_x)
        final_output = current_x
        if return_node_emb:
            if not post_activation_embeddings and self.num_layers > 1 : raise ValueError("No hidden embeddings available.")
            if embedding_layer is not None:
                if embedding_layer >= len(post_activation_embeddings): raise ValueError(f"embedding_layer index out of bounds.")
                return final_output, post_activation_embeddings[embedding_layer]
            if post_activation_embeddings: return final_output, post_activation_embeddings[-1]
            else: return final_output, x 
        return final_output


    def get_all_intermediate_states_and_retain_grads(self, x, edge_index):
        self.eval()
        H_states_for_analysis = []
        X_states_for_analysis = [x.clone().detach()]
        current_x_prop = x
        for i, conv in enumerate(self.convs):
            h_k = conv(current_x_prop, edge_index)
            h_k.retain_grad() # Key step
            H_states_for_analysis.append(h_k)
            if i < len(self.convs) - 1:
                current_x_activated = F.elu(h_k)
            else:
                current_x_activated = h_k
            X_states_for_analysis.append(current_x_activated.clone().detach())
            current_x_prop = current_x_activated
        return H_states_for_analysis, X_states_for_analysis

    def reset_parameters(self): # ...
        for conv in self.convs: conv.reset_parameters()

class NodeGraphConv(torch.nn.Module): # Simplified example, focusing on retain_grad
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, aggr='add'):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        if num_layers == 1: self.layers.append(PyGGraphConv(num_features, num_classes, aggr=aggr))
        else:
            self.layers.append(PyGGraphConv(num_features, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2): self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
            self.layers.append(PyGGraphConv(hidden_channels, num_classes, aggr=aggr))
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False, embedding_layer=None):
        post_activation_embeddings = []
        current_x = x
        for i, layer_conv in enumerate(self.layers):
            current_x = layer_conv(current_x, edge_index)
            if i < len(self.layers) - 1:
                current_x = F.relu(current_x)
                post_activation_embeddings.append(current_x)
        final_output = current_x
        if return_node_emb:
            if not post_activation_embeddings and self.num_layers > 1: 
                raise ValueError("No hidden embeddings available.")
            if embedding_layer is not None:
                if embedding_layer >= len(post_activation_embeddings): 
                    raise ValueError(f"embedding_layer index out of bounds.")
                return final_output, post_activation_embeddings[embedding_layer]
            if post_activation_embeddings: 
                return final_output, post_activation_embeddings[-1]
            else: 
                return final_output, x
        return final_output

    def get_all_intermediate_states_and_retain_grads(self, x, edge_index):
        self.eval()
        H_states_for_analysis = []
        X_states_for_analysis = [x.clone().detach()]
        current_x_prop = x
        for i, layer_conv in enumerate(self.layers):
            h_k = layer_conv(current_x_prop, edge_index)
            h_k.retain_grad() # Key step
            H_states_for_analysis.append(h_k)
            if i < len(self.layers) - 1:
                current_x_activated = F.relu(h_k)
            else:
                current_x_activated = h_k
            X_states_for_analysis.append(current_x_activated.clone().detach())
            current_x_prop = current_x_activated
        return H_states_for_analysis, X_states_for_analysis
        
    def reset_parameters(self): # ...
        for layer in self.layers: layer.reset_parameters()
