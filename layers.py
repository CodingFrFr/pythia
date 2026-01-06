import numpy as np

class LayerNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.eps = np.float32(1e-5)

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        N = self.x.shape[-1]
        # Accumulate gradients
        self.dgamma += np.sum(dout * self.x_hat, axis=(0, 1))
        self.dbeta += np.sum(dout, axis=(0, 1))
        
        dx_hat = dout * self.gamma
        inv_std = 1.0 / np.sqrt(self.var + self.eps)
        
        dx = (1.0 / N) * inv_std * (
            N * dx_hat - 
            np.sum(dx_hat, axis=-1, keepdims=True) - 
            self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )
        return dx

class Linear:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2 / in_dim)
        self.W = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, dout):
        if len(self.x.shape) == 3:
            self.dW += np.sum(self.x.transpose(0, 2, 1) @ dout, axis=0)
            self.db += np.sum(dout, axis=(0, 1))
        else:
            self.dW += self.x.T @ dout
            self.db += np.sum(dout, axis=0)
        return dout @ self.W.T

class CausalSelfAttention:
    def __init__(self, d_model, max_len):
        self.c_attn = Linear(d_model, 3 * d_model)
        self.c_proj = Linear(d_model, d_model)
        self.mask = np.tril(np.ones((max_len, max_len), dtype=np.float32)).reshape(1, max_len, max_len)
        self.scale = np.float32(1.0 / np.sqrt(d_model))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)

        att = (q @ k.transpose(0, 2, 1)) * self.scale
        att = np.where(self.mask[:, :T, :T], att, np.float32(-1e9))
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        self.att_weights = att / np.sum(att, axis=-1, keepdims=True)
        
        y = self.att_weights @ v
        self.cache = (x, q, k, v)
        return self.c_proj.forward(y)

    def backward(self, dout):
        dout_proj = self.c_proj.backward(dout)
        x, q, k, v = self.cache
        B, T, C = x.shape
        
        d_v = self.att_weights.transpose(0, 2, 1) @ dout_proj
        d_att = dout_proj @ v.transpose(0, 2, 1)
        d_att_scores = self.att_weights * (d_att - np.sum(d_att * self.att_weights, axis=-1, keepdims=True))
        d_att_scores *= self.scale
        
        d_q = d_att_scores @ k
        d_k = d_att_scores.transpose(0, 2, 1) @ q
        d_qkv = np.concatenate((d_q, d_k, d_v), axis=-1)
        return self.c_attn.backward(d_qkv)

class FeedForward:
    def __init__(self, d_model):
        self.net1 = Linear(d_model, 4 * d_model)
        self.net2 = Linear(4 * d_model, d_model)
        
    def forward(self, x):
        self.h = np.maximum(0, self.net1.forward(x))
        return self.net2.forward(self.h)
    
    def backward(self, dout):
        d_net2 = self.net2.backward(dout)
        d_relu = d_net2 * (self.h > 0)
        return self.net1.backward(d_relu)

class Block:
    # Block model for Pythia class
    def __init__(self, d_model, max_len):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, max_len)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, dout):
        d_mlp_out = dout
        d_mlp_in = self.ln2.backward(self.mlp.backward(d_mlp_out))
        dout = dout + d_mlp_in
        d_attn_out = dout 
        d_attn_in = self.ln1.backward(self.attn.backward(d_attn_out))
        return dout + d_attn_in
