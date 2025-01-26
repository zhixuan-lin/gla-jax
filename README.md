# GLA kernels in JAX

Experimental implementation of [GLA](https://arxiv.org/abs/2312.06635)
in JAX and pallas. Three implementations are available in `gla.py`

* `recurrent_gla_naive`: naive and slow recurrent implementation using `jax.lax.scan`.
* `recurrent_gla`: recurrent implementation that avoids materialization of hidden
  states
* `chunk_gla` : a variant of the chunk-wise algorithm described in the GLA paper with
  special multi-scale secondary chunking

## Setup

Only earlier versions of JAX are supported for now.

```bash
conda create jax-playground python=3.10
conda activate jax-playground

pip install "jax[cuda12_pip]==0.4.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install einops
pip install matplotlib
```

## Benchmarking and Testing

```bash
python gla.py
```

## Usage

```python
from gla import chunk_gla
import jax
import jax.numpy as jnp

batch_size = 4
num_heads = 6
head_dim = 128
seq_len = 4096
dtype = 'bfloat16'
rng = jax.random.PRNGKey(0)

hidden_init = jax.random.normal(key=rng, shape=(num_heads, head_dim, head_dim), dtype=dtype)
value = jax.random.normal(key=rng, shape=(seq_len, num_heads, head_dim), dtype=dtype)
query = jax.random.normal(key=rng, shape=(seq_len, num_heads, head_dim), dtype=dtype)
log_fgate = jax.nn.log_sigmoid(5.0 + jax.random.normal(key=rng, shape=(seq_len, num_heads, head_dim), dtype=jnp.float32))
key = jax.random.normal(key=rng, shape=(seq_len, num_heads, head_dim), dtype=dtype)

hidden_final, out = chunk_gla(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    chunk_size=head_dim,  # Recommended, but other values also work
)
```
