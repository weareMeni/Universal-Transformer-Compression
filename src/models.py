import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# SHARED UTILITIES & CLOUD PREDICTOR
# ==========================================
class SwiGLU_FFN(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier=4):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, xq_.size(-1))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CloudPredictor(nn.Module):
    """Simulates the remote model decoding the compressed vector."""
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_classes)
        )
    def forward(self, z):
        return self.net(z)

# ==========================================
# ATTENTION & LAYERS
# ==========================================
class RoPEAttention(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nhead, self.head_dim)
        k = self.wk(x).view(B, T, self.nhead, self.head_dim)
        v = self.wv(x).view(B, T, self.nhead, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(attn_out)

class StandardTransformerLayer(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)
        self.ffwd = SwiGLU_FFN(dim)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffwd(self.norm2(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([StandardTransformerLayer(dim, nhead) for _ in range(num_layers)])

    def forward(self, x, freqs_cis):
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return x

# ==========================================
# BASELINE: STATIC COMPRESSOR
# ==========================================
class StandardLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, dim, nhead, num_layers, max_seq_len=5000):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, dim)
        self.block = TransformerBlock(dim, nhead, num_layers)
        self.layer_norm = nn.LayerNorm(dim)
        self.predictor = CloudPredictor(dim, num_classes)
        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // nhead, max_seq_len * 2), persistent=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)
        freqs_cis = self.freqs_cis[:T].to(x.device)
        
        h = self.block(h, freqs_cis)
        h_norm = self.layer_norm(h)
        
        # --- BOTTLENECK: Extract final valid token representation ---
        lengths = (x != 0).sum(dim=1).clamp(min=1) - 1
        z = h_norm[torch.arange(B), lengths, :] 
        
        logits = self.predictor(z)
        
        # Return static ponder cost for fair plotting
        static_ponder = torch.tensor(float(self.num_layers), device=x.device)
        return logits, static_ponder

# ==========================================
# DYNAMIC COMPRESSOR (UNIVERSAL)
# ==========================================
class UniversalACTWrapper(nn.Module):
    def __init__(self, layer, dim, max_steps=20, dropout=0.1): 
        super().__init__()
        self.layer = layer
        self.max_steps = max_steps
        self.dropout = nn.Dropout(p=dropout)
        
        self.rnn_controller = nn.GRUCell(dim, dim)
        self.halting_classifier = nn.Linear(dim, 1)
        self.halting_classifier.bias.data.fill_(-1.0) 
        
        # --- NEW: Projection to inject RNN state back into the sequence ---
        self.state_proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cis, pad_mask):
        B, T, C = x.shape
        device = x.device

        accumulated_probs = torch.zeros(B, T, 1, device=device)
        updates = torch.zeros(B, T, 1, device=device)
        active_mask = pad_mask.clone()
        
        output_state = torch.zeros_like(x)
        current_x = x  
        h_rnn = torch.zeros(B * T, C, device=device)
        
        for ponder_step in range(1, self.max_steps + 1):
            
            # --- NEW: Feedback loop for temporal/state awareness ---
            if ponder_step > 1:
                # Reshape h_rnn back to sequence dimensions and project
                rnn_feedback = self.state_proj(h_rnn.view(B, T, C))
                step_input = current_x + rnn_feedback
            else:
                step_input = current_x

            # Pass the injected sequence to the transformer layer
            layer_out = self.layer(step_input, freqs_cis)
            layer_out = self.dropout(layer_out)
            
            flat_in = layer_out.detach().view(-1, C)
            h_rnn = self.rnn_controller(flat_in, h_rnn)
            p_t = torch.sigmoid(self.halting_classifier(h_rnn)).view(B, T, 1)
            
            is_halting_now = active_mask if ponder_step == self.max_steps else ((accumulated_probs + p_t) >= 1.0) & active_mask
            
            step_weight = torch.where(is_halting_now, 1.0 - accumulated_probs, p_t) * active_mask.float()
            output_state = output_state + (step_weight * layer_out)
            accumulated_probs = accumulated_probs + step_weight
            updates = updates + active_mask.float()
            
            active_mask = active_mask & ~is_halting_now
            
            # Update current_x with the raw layer output for the next step's baseline
            current_x = torch.where(active_mask, layer_out, current_x)
            
            if not active_mask.any():
                break
                
        valid_tokens = pad_mask.float().sum().clamp(min=1)
        ponder_cost = (updates * pad_mask.float()).sum() / valid_tokens
        
        return output_state, ponder_cost

class UniversalLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, dim, nhead, max_steps=20, max_seq_len=5000, core_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        
        # UT traditionally shares 1 core block across time steps
        core_block = TransformerBlock(dim, nhead, num_layers=core_layers)
        self.universal_layer = UniversalACTWrapper(core_block, dim, max_steps=max_steps)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.predictor = CloudPredictor(dim, num_classes)
        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // nhead, max_seq_len * 2), persistent=False)

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x != 0).unsqueeze(-1)
        
        h = self.embed(x)
        freqs_cis = self.freqs_cis[:T].to(x.device)
        
        h, ponder_cost = self.universal_layer(h, freqs_cis, pad_mask)
        h_norm = self.layer_norm(h)
        
        # --- BOTTLENECK: Extract final valid token representation ---
        lengths = (x != 0).sum(dim=1).clamp(min=1) - 1
        z = h_norm[torch.arange(B), lengths, :] 
        
        logits = self.predictor(z)
        return logits, ponder_cost