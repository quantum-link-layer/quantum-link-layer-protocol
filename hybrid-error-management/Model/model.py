import math
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
X-Decoder
'''
                

class PositionalEncoding(nn.Module):
    """Positional Encoding for sequence input to provide relative position information."""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x: Tensor of shape [B, L, D] - Input tensor

        Returns:
            Tensor of shape [B, L, D] - Output with positional encoding added
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return x
    
    
class StabilizerEmbedder(nn.Module):
    """Embed the measurement and event inputs to a higher-dimensional space."""
    def __init__(self, d_model=128):
        super(StabilizerEmbedder, self).__init__()
        # self.measure_linear = nn.Linear(1, d_model)  # Embed measurement input: [B, L, 1] -> [B, L, d_model]
        self.event_linear = nn.Linear(1, d_model)    # Embed event input: [B, L, 1] -> [B, L, d_model]
        
        self.resnet = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model, track_running_stats=False),
            nn.ReLU()
        )
        self.pos_encoding = PositionalEncoding(d_model)

        # Learnable embedding for the buffer round
        self.buffer_round_embedding = nn.Parameter(torch.zeros(1, 1, d_model))  # Shape: [1, 1, d_model]

    def forward(self, event):
        """
        Embed the measurement and event input sequences and apply position encoding.

        Args:
            measurement: Tensor of shape [B, L, 1] - Measurement input
            event: Tensor of shape [B, L, 1] - Event input

        Returns:
            Tensor of shape [B, L, D] - Embedded and position-encoded output
        """
        # m_embed = self.measure_linear(measurement)  # Measurement embedding: [B, L, d_model]
        e_embed = self.event_linear(event)          # Event embedding: [B, L, d_model]
        x = e_embed
        
        # [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)
        x = x + self.resnet(x)  # Apply residual connection
        # [B, L, D]
        x = x.permute(0, 2, 1)
        
        # # Add learnable embedding if this is the buffer region
        # if is_buffer:
        #     x = x + self.buffer_round_embedding  # Broadcasted addition
        
        x = self.pos_encoding(x)  # Add positional encoding
        return x
    
    
class SyndromeTransformerLayer(nn.Module):
    """
    Syndrome Transformer Layer:
    Combines self-attention with gated dense block, followed by
    dilated convolutions, and 2D scatter-gather operations.
    """
    def __init__(self, d, d_model=128, nhead=4, dim_feedforward=128*5, dropout=0.1):
        super(SyndromeTransformerLayer, self).__init__()
        self.d_model = d_model
        self.d = d

        # Multi-head self-attention with attention bias
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dilated 2D convolutions
        self.dilated_convs = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(d_model, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(d_model, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(d_model, track_running_stats=False),
            nn.ReLU()
        )

        # Residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_bias=None):
        """
        Args:
            x: Tensor of shape [B, L, D]
            attention_bias: Optional tensor to add bias in attention computation.

        Returns:
            Tensor of shape [B, L, D]
        """
        B, L, D = x.size()

        # Step 1: Self-Attention with Attention Bias
        attn_output, _ = self.self_attn(x, x, x)  # [B, L, D]
        if attention_bias is not None:
            attn_output += attention_bias
        x = self.norm1(x + attn_output)  # Residual connection + normalization

        # Step 2: Gated Dense Block
        # Feedforward network
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))  # [B, L, D]
        x = self.norm2(x + ff_output)

        # Step 3: Reshape [B, L, D] to [B, D, H, W]
        H = self.d + 1
        W = self.d + 1
        x_2d = x.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Step 4: Dilated Convolutions
        x_2d = self.dilated_convs(x_2d)  # [B, D, H, W]

        # Step 5: Gather back to 1D format [B, L, D]
        x_out = x_2d.permute(0, 2, 3, 1).reshape(B, L, D)  # [B, L, D]

        # Final residual connection
        x = x + x_out
        return x


class RNNCore(nn.Module):
    """RNN core that projects input sequence and passes through transformer layers."""
    def __init__(self, d, num_layers=3, d_model=128, nhead=4, dropout=0.1):
        super(RNNCore, self).__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.transformer_layers = nn.ModuleList([
            SyndromeTransformerLayer(d=d, d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, prev_decoder_state, stabilizer_embed):
        """
        Compute the next state by combining previous state and stabilizer embedder output.

        Args:
            prev_decoder_state: Tensor of shape [B, L, D] - Previous decoder state
            stabilizer_embed: Tensor of shape [B, L, D] - Stabilizer embedding

        Returns:
            Tensor of shape [B, L, D] - Updated decoder state
        """
        x = (prev_decoder_state + stabilizer_embed) * 0.707  # Combine and scale
        x = self.proj(x)  # Project to model dimension

        for layer in self.transformer_layers:
            x = layer(x)

        return x


class Readout(nn.Module):
    """Final readout layer to compute logits for classification."""
    def __init__(self, d, d_model=128):
        super(Readout, self).__init__()
        self.linear = nn.Linear(d_model * d, 1)
        self.conv_to_data = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )
        self.d = d

    def forward(self, decoder_state):
        """
        Compute the logit for classification.

        Args:
            decoder_state: Tensor of shape [B, L, D] - Decoder state output

        Returns:
            Tensor of shape [B, 1] - Logits for classification
        """
        B, L, D = decoder_state.size()
        # Reshape [B, L, D] to [B, D, H, W]
        H = self.d + 1
        W = self.d + 1
        decoder_state = decoder_state.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # Apply 2x2 convolution with padding to maintain dimensions
        decoder_state = self.conv_to_data(decoder_state) # Output shape: [B, D, d, d]
        
        pooled = decoder_state.mean(dim=3)  # Average pooling over sequence length perpendicular to the logical operator [B, D, d, d] -> [B, D, d, 1]
        
        # Flatten pooled output to [B, D * d]
        pooled = pooled.view(B, -1)
        
        # linear regression: [B, D * d] -> [B, 1]
        logit = self.linear(pooled)
        return torch.sigmoid(logit)


class NN(nn.Module):
    def __init__(self, d, d_model=192, nhead=8, num_layers=3, dropout=0.1):
        super(NN, self).__init__()
        self.d = d
        self.d_model = d_model
        self.nhead = nhead
        
        self.embedder = StabilizerEmbedder(d_model)
        self.rnn_core = RNNCore(d=d, d_model=d_model, nhead=nhead, dropout=dropout, num_layers=num_layers)
        self.readout  = Readout(d=d, d_model=d_model)
        
    def forward(self, x):
        # x shape: (B, N, H, W)
        B, N, H, W = x.shape
        assert N == 3, "All syndrome measurement round mismatchs 3 here"
        L = H * W
        x = x.view(B, N, L, 1)
        
        device = x.device
        decoder_state = torch.zeros(B, L, self.d_model, device=device)  # (B, L, D)
        
        for t in range(N):
            # wt = self.time_weights[:, t:t+1, :].to(device)  # [1,1,1]
            event_t = x[:, t]
            stab_embed = self.embedder(event_t)
            decoder_state = self.rnn_core(decoder_state, stab_embed)
            
        logit = self.readout(decoder_state)
        return logit


'''
Risk / reliability scoring network (syndrome-only).

Given the detector event tensor (same input as the decoder), this network
predicts the probability that a *fixed* backend decoder will fail on the shot.

Intended use: train on labels L = 1[ decoder(s) != true_observable ], then use
the score for post-selection at a target acceptance rate via quantile
thresholding.

Notes:
  - This is a lightweight CNN designed for small surface-code distances.
  - It does NOT consume the decoder's soft outputs as an input feature.
'''