import numpy as np
from layers import *

class Pythia:
    def __init__(self, vocab_size_or_config, d_model=None, n_layer=None, max_len=None):
        # Check if the first argument is a dictionary
        if isinstance(vocab_size_or_config, dict):
            config = vocab_size_or_config
            vocab_size = config['vocab_size']
            d_model    = config['d_model']
            n_layer    = config['n_layer']
            max_len    = config['max_len']
        else:
            vocab_size = vocab_size_or_config
            
        self.token_emb = (np.random.randn(vocab_size, d_model) * 0.02).astype(np.float32)
        self.pos_emb = (np.random.randn(max_len, d_model) * 0.02).astype(np.float32)
        
        self.d_token_emb = np.zeros_like(self.token_emb)
        self.d_pos_emb = np.zeros_like(self.pos_emb)

        self.blocks = [Block(d_model, max_len) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)


    def collect_params(self):
        """Registry: Returns a list of all (param, grad) tuples."""
        params = []
        # Embeddings
        params.append((self.token_emb, self.d_token_emb))
        params.append((self.pos_emb, self.d_pos_emb))
        # Blocks
        for b in self.blocks:
            # LN1
            params.append((b.ln1.gamma, b.ln1.dgamma))
            params.append((b.ln1.beta, b.ln1.dbeta))
            # Attn
            params.append((b.attn.c_attn.W, b.attn.c_attn.dW))
            params.append((b.attn.c_attn.b, b.attn.c_attn.db))
            params.append((b.attn.c_proj.W, b.attn.c_proj.dW))
            params.append((b.attn.c_proj.b, b.attn.c_proj.db))
            # LN2
            params.append((b.ln2.gamma, b.ln2.dgamma))
            params.append((b.ln2.beta, b.ln2.dbeta))
            # MLP
            params.append((b.mlp.net1.W, b.mlp.net1.dW))
            params.append((b.mlp.net1.b, b.mlp.net1.db))
            params.append((b.mlp.net2.W, b.mlp.net2.dW))
            params.append((b.mlp.net2.b, b.mlp.net2.db))
        # Final
        params.append((self.ln_f.gamma, self.ln_f.dgamma))
        params.append((self.ln_f.beta, self.ln_f.dbeta))
        params.append((self.head.W, self.head.dW))
        params.append((self.head.b, self.head.db))
        return params

    def zero_grad(self):
        """Resets all gradient buffers to zero."""
        # We iterate over the parameter registry to cleanly zero everything
        for _, grad in self.collect_params():
            grad.fill(0)

    def forward(self, idx, targets=None):
        # 1. Dynamic Context Cropping (The Fix)
        # We derive max_len from the shape of the positional embeddings
        max_len = self.pos_emb.shape[0]
        
        # If input exceeds capacity, crop to the last 'max_len' tokens
        if idx.shape[1] > max_len:
            idx = idx[:, -max_len:]
            if targets is not None:
                targets = targets[:, -max_len:]

        B, T = idx.shape
        
        # Now T is guaranteed to be <= max_len
        x = self.token_emb[idx] + self.pos_emb[:T]
        
        for block in self.blocks:
            x = block.forward(x)
            
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)
        
        if targets is None: return logits
        
        # Softmax & Loss Calculation
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        
        flat_probs = probs.reshape(-1, probs.shape[-1])
        # Use the potentially cropped targets for loss
        flat_targets = targets.reshape(-1)
        loss = -np.log(flat_probs[np.arange(B*T), flat_targets] + 1e-9).mean()
        return logits, loss, probs
        
    def train_step(self, idx, targets):
        # 1. Reset gradients from previous step
        self.zero_grad()
        
        # 2. Forward Pass
        logits, loss, probs = self.forward(idx, targets)
        B, T, V = probs.shape
        
        # 3. Backward Pass (Gradients for Head & Layers)
        dlogits = probs.copy()
        dlogits.reshape(-1, V)[np.arange(B*T), targets.reshape(-1)] -= 1
        dlogits /= (B*T)
        
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
            
        # 4. Backward Pass (Embeddings)
        # Token Embeddings: Accumulate gradients for repeated tokens
        np.add.at(self.d_token_emb, idx, dx)
        
        # Positional Embeddings: Sum across batch, apply only to valid length T
        self.d_pos_emb[:T] += np.sum(dx, axis=0)
        
        return loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            logits = logits[:, -1, :] / temperature
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            # Sampling
            idx_next = np.array([[np.random.choice(len(probs[0]), p=probs[0])]])
            idx = np.concatenate((idx, idx_next), axis=1)
        return idx