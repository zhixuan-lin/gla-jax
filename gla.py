import time
import numpy as np
from typing import Optional, NamedTuple
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import timeit
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import functools
import math
import einops
import itertools
from ops import join, cumsum_recursive, rnn_recursive

# https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


def factors(n):
    return set(functools.reduce(list.__add__, 
              ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def recurrent_gla_naive(hidden, log_fgate, value, key, query, reverse):
    # vmap over heads
    @jax.vmap
    def f(hidden, inputs):
        # hidden (key, value)
        fgate_t, value_t, key_t, query_t = inputs
        assert hidden.ndim == 2
        assert fgate_t.ndim == value_t.ndim == key_t.ndim == query_t.ndim == 1
        # (key,) * (key, value) + outer(key, value)
        hidden = fgate_t[:, None] * hidden + jnp.outer(key_t, value_t.astype(jnp.float32))
        # out = hidden @ query_t
        out = jnp.sum(query_t[:, None] * hidden, axis=0)
        return hidden, out

    fgate = jnp.exp(log_fgate)
    inputs = (fgate, value, key, query)
    chunk_size = key.shape[0]
    hidden_dtype = hidden.dtype
    hidden = hidden.astype(jnp.float32)
    hidden, out = jax.lax.scan(f, hidden, inputs, reverse=reverse, unroll=min(32, chunk_size))
    hidden = hidden.astype(hidden_dtype)
    return hidden, out


def recurrent_gla_forward_kernel(
    hidden_init_ref,
    log_fgate_ref,
    value_ref,
    key_ref,
    query_ref,

    hidden_final_ref,
    out_ref
):
    # This kernel processes multi-head attention. The reason is it is we need
    # to do experiment with head dim one, in that case it makes sense to group
    # many heads into one block

    # Shape of every thing except carry (T, H, D)
    # carry shape (H, D, D)
    T, H, D = log_fgate_ref.shape

    def for_loop_body(i, val):
        last_hidden = val
        @jax.vmap
        def single_step(log_fgate, value, key, query, last_hidden):
            fgate = jnp.exp(log_fgate)
            assert fgate.ndim == value.ndim == key.ndim == query.ndim == 1
            assert last_hidden.ndim == 2
            hidden = fgate[:, None] * last_hidden + jnp.outer(key, value.astype(jnp.float32))
            # It looks like @ does not work with matrix-vector multiplication on Triton
            out = jnp.sum(hidden * query[:, None], axis=0).astype(out_ref.dtype)
            return hidden, out
        hidden, out_ref[i] = single_step(
            log_fgate_ref[i],
            value_ref[i],
            key_ref[i],
            query_ref[i],
            last_hidden
        )
        val = hidden
        return val

    val = hidden_init_ref[:].astype(jnp.float32)
    val = jax.lax.fori_loop(
        lower=0,
        upper=T,
        body_fun=for_loop_body,
        init_val=val,
    )
    hidden_final_ref[:] = val.astype(hidden_final_ref.dtype)


def recurrent_gla_forward(


    hidden_init,
    log_fgate,
    value,
    key,
    query,

    value_block_size: int = 32,
    key_block_size: int = 32,
    head_block_size: int = 1,
    num_warps: int = 1,
    num_stages: int = 1,
    interpret: bool = False,
):
    # Shape: (B, T, H, D)
    # Except hidden_init: (B, H, D, D)

    T, H, D_key = key.shape
    T, H, D_value = value.shape
    assert D_key == D_value

    value_block_size = min(D_value, value_block_size)
    key_block_size = min(D_key, key_block_size)

    assert D_key % key_block_size == 0, (D_key, key_block_size)
    assert D_value % value_block_size == 0, (D_value, value_block_size)
    assert H % head_block_size == 0, (H, head_block_size)

    N_key = D_key // key_block_size
    N_value = D_value // value_block_size
    N_head = H // head_block_size
    # Shape of every thing except carry (T, D)
    # carry shape (D, D)
    func = pl.pallas_call(
        recurrent_gla_forward_kernel,
        interpret=interpret,
        out_shape=[
            jax.ShapeDtypeStruct(shape=hidden_init.shape, dtype=hidden_init.dtype),
            # Note the N_key
            jax.ShapeDtypeStruct(shape=(N_key, T, H, D_value), dtype=value.dtype),
        ],
        in_specs=[
            pl.BlockSpec(lambda h, k, v: (h, k, v), (head_block_size, key_block_size, value_block_size)),
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
            pl.BlockSpec(lambda h, k, v: (0, h, v), (T, head_block_size, value_block_size)),
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
        ],
        out_specs=[
            pl.BlockSpec(lambda h, k, v: (h, k, v), (head_block_size, key_block_size, value_block_size)),
            pl.BlockSpec(lambda h, k, v: (k, 0, h, v), (None, T, head_block_size, value_block_size))
        ],
        grid=(N_head, N_key, N_value),
        compiler_params=dict(
            num_warps=num_warps,
            num_stages=num_stages
        )
    )
    hidden_final, out = func(
        hidden_init,
        log_fgate,
        value,
        key,
        query,
    )
    out = out.sum(axis=0)
    assert out.shape == (T, H, D_value)
    assert hidden_final.shape == (H, D_value, D_key)
    res = (hidden_init, log_fgate, value, key, query, hidden_final)
    return (hidden_final, out), res

def recurrent_gla_backward(
    value_block_size,
    key_block_size,
    head_block_size,
    num_warps,
    num_stages,
    interpret,
    res,
    g
):
    # Shape: (B, T, H, D)
    # Except hidden_init: (B, H, D, D)
    # hidden_final is not used unless we compute dlog_fgate outside the kernel
    (hidden_init, log_fgate, value, key, query, hidden_final) = res
    dcarry_final, dout = g

    T, H, D_key = key.shape
    T, H, D_value = value.shape

    value_block_size = min(D_key, value_block_size)
    key_block_size = min(D_key, key_block_size)

    assert D_key % key_block_size == 0
    assert D_value % value_block_size == 0
    assert H % head_block_size == 0

    N_key = D_key // key_block_size
    N_value = D_value // value_block_size
    N_head = H // head_block_size
    # Shape of every thing except carry (T, D)
    # carry shape (D, D)
    func = pl.pallas_call(
        recurrent_gla_backward_kernel,
        interpret=interpret,
        out_shape=[
            # hidden
            jax.ShapeDtypeStruct(shape=hidden_init.shape, dtype=hidden_init.dtype),
            # value
            jax.ShapeDtypeStruct(shape=(N_key, T, H, D_value), dtype=jnp.float32),
            # key
            jax.ShapeDtypeStruct(shape=(N_value, T, H, D_key), dtype=jnp.float32),
            # query
            jax.ShapeDtypeStruct(shape=(N_value, T, H, D_key), dtype=jnp.float32),
        ],
        in_specs=[
            # hidden
            pl.BlockSpec(lambda h, k, v: (h, k, v), (head_block_size, key_block_size, value_block_size)),
            # fgate
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
            # value
            pl.BlockSpec(lambda h, k, v: (0, h, v), (T, head_block_size, value_block_size)),
            # key
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
            # query
            pl.BlockSpec(lambda h, k, v: (0, h, k), (T, head_block_size, key_block_size)),
            # dcarry_final
            pl.BlockSpec(lambda h, k, v: (h, k, v), (head_block_size, key_block_size, value_block_size)),
            # dout
            pl.BlockSpec(lambda h, k, v: (0, h, v), (T, head_block_size, value_block_size)),
        ],
        out_specs=[
            # hidden
            pl.BlockSpec(lambda h, k, v: (h, k, v), (head_block_size, key_block_size, value_block_size)),
            # value
            pl.BlockSpec(lambda h, k, v: (k, 0, h, v), (None, T, head_block_size, value_block_size)),
            # key
            pl.BlockSpec(lambda h, k, v: (v, 0, h, k), (None, T, head_block_size, key_block_size)),
            # query
            pl.BlockSpec(lambda h, k, v: (v, 0, h, k), (None, T, head_block_size, key_block_size)),
        ],
        compiler_params=dict(
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        grid=(N_head, N_key, N_value),
    )
    dhidden_init, dvalue, dkey, dquery = func(
        hidden_init,
        log_fgate,
        value,
        key,
        query,
        dcarry_final,
        dout
    )
    dvalue = dvalue.sum(axis=0)
    dkey = dkey.sum(axis=0)
    dquery = dquery.sum(axis=0)

    # computing this here saves memory because there are N_value copies of dquery
    # in the kernel and we need to compute N_value copies of dlog_fgate
    dlog_fgate = jax.lax.cumsum(dquery * query - dkey * key, axis=0, reverse=True) + (dcarry_final * hidden_final).sum(axis=-1)

    dvalue = dvalue.astype(value.dtype)
    dquery = dquery.astype(query.dtype)
    dkey = dkey.astype(key.dtype)
    dhidden_init = dhidden_init.astype(hidden_init.dtype)
    return (dhidden_init, dlog_fgate, dvalue, dkey, dquery)

def recurrent_gla_backward_kernel(
    hidden_init_ref,
    log_fgate_ref,
    value_ref,
    key_ref,
    query_ref,

    dcarry_final_ref,
    dout_ref,

    dhidden_init_ref,
    dvalue_ref,
    dkey_ref,
    dquery_ref,
):
    # This kernel processes multi-head attention. The reason is it is we need
    # to do experiment with head dim one, in that case it makes sense to group
    # many heads into one program

    # Shape of every thing except hidden (T, H, D)
    # hidden shape (H, D, D)

    # hidden and carry are different nodes in the computational graph connected
    # by the identity function. However, out[t] does not depend on carry[t]
    # and hidden[t + 1] depends on carry[t] but not hidden[t]
    T, H, D = log_fgate_ref.shape
    def forward_body(i, val):
        # vmap over head
        @jax.vmap
        def single_step(log_fgate, value, key, query, last_hidden, dout):
            fgate = jnp.exp(log_fgate)
            assert fgate.ndim == value.ndim == key.ndim == query.ndim == dout.ndim == 1
            assert last_hidden.ndim == 2
            hidden = fgate[:, None] * last_hidden + jnp.outer(key, value.astype(jnp.float32))
            # It looks like @ does not work with matrix-vector multiplication on Triton
            dquery = jnp.sum(hidden * dout, axis=-1).astype(dquery_ref.dtype)
            return hidden, dquery
        last_hidden = val
        hidden, dquery_ref[i] = single_step(
            log_fgate_ref[i],
            value_ref[i],
            key_ref[i],
            query_ref[i],
            last_hidden,
            dout_ref[i]
        )
        return hidden

    val = hidden_init_ref[:].astype(jnp.float32)
    _ = jax.lax.fori_loop(
        lower=0,
        upper=T,
        body_fun=forward_body,
        init_val=val,
    )

    def backward_body(i, val):
        t = T - i - 1
        # vmap over head
        @jax.vmap
        def single_step(log_fgate, value, key, query, dcarry, dout):
            fgate = jnp.exp(log_fgate)
            assert fgate.ndim == value.ndim == key.ndim == query.ndim == dout.ndim == 1
            assert dcarry.ndim == 2
            # It looks like @ does not work with matrix-vector multiplication on Triton
            dhidden = dcarry + jnp.outer(query, dout.astype(jnp.float32))
            dvalue = jnp.sum(key[:, None] * dhidden, axis=0).astype(dvalue_ref.dtype)
            dkey = jnp.sum(dhidden * value, axis=-1).astype(dkey_ref.dtype)
            dprev_carry = fgate[:, None] * dhidden
            return dprev_carry, dvalue, dkey

        dcarry = val
        dprev_carry, dvalue_ref[t], dkey_ref[t] = single_step(
            log_fgate_ref[t],
            value_ref[t],
            key_ref[t],
            query_ref[t],
            dcarry,
            dout_ref[t],
        )
        return dprev_carry

    dcarry_final = dcarry_final_ref[:].astype(jnp.float32)
    dcarry_init = jax.lax.fori_loop(
        lower=0,
        upper=T,
        body_fun=backward_body,
        init_val=dcarry_final,
    )
    dhidden_init_ref[:] = dcarry_init.astype(dhidden_init_ref.dtype)

@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def recurrent_gla_forward_backward(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    value_block_size: int = 32,
    key_block_size: int = 32,
    head_block_size: int = 1,
    num_warps: int = 1,
    num_stages: int = 1,
    interpret: bool = False
):
    return recurrent_gla_forward(
        hidden_init,
        log_fgate,
        value,
        key,
        query,
        value_block_size,
        key_block_size,
        head_block_size,
        num_warps,
        num_stages,
        interpret,
    )[0]

def auto_block_size(value_dim: int, key_dim: int, num_heads: int, max_block_size):
    assert value_dim == key_dim, 'Not supported yet'
    value_block_size = min(max_block_size, value_dim)
    key_block_size = min(max_block_size, key_dim)
    # Largest factor of num_heads such that the following is true
    head_block_size = None
    for c in sorted(factors(num_heads)):
        if c * value_block_size <= max_block_size and c * key_block_size <= max_block_size:
            head_block_size = c
        else:
            break
    assert head_block_size is not None, 'This is impossible bro...'
    print(f'(v, k, h) = ({value_block_size}, {key_block_size}, {head_block_size})')
    return value_block_size, key_block_size, head_block_size

def recurrent_gla(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    reverse: bool = False,
    max_block_size: int = 64,
    num_warps: int = 1,
    num_stages: int = 1,
    interpret: bool = False,
):
    assert value.shape[-2] == key.shape[-2]
    value_block_size, key_block_size, head_block_size = auto_block_size(
        value.shape[-1], key.shape[-1], value.shape[-2], max_block_size
    )
    if reverse:
        (log_fgate, value, key, query) = [
            x[::-1, :, :] for x in (log_fgate, value, key, query)]
    hidden_final, out = recurrent_gla_forward_backward(
        hidden_init=hidden_init,
        log_fgate=log_fgate,
        value=value,
        key=key,
        query=query,
        value_block_size=value_block_size,
        key_block_size=key_block_size,
        head_block_size=head_block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    )
    if reverse:
        out = out[::-1, :, :]
    return hidden_final, out

recurrent_gla_forward_backward.defvjp(recurrent_gla_forward, recurrent_gla_backward)


def compute_score_forward(
    log_lambda,
    key,
    query,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):

    L, D = log_lambda.shape
    block_size_d = min(block_size_d, D)
    num_blocks_d = D // block_size_d

    func = pl.pallas_call(
        compute_score_forward_kernel,
        interpret=interpret,
        # f=functools.partial(compute_score_forward_kernel, block_size_d=block_size_d),
        # out_shape=jax.ShapeDtypeStruct((L, L), key.dtype),
        out_shape=jax.ShapeDtypeStruct((num_blocks_d, L, L), key.dtype),
        in_specs=[
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
        ],
        out_specs=pl.BlockSpec(lambda d: (d, 0, 0), (None, L, L)),
        grid=(num_blocks_d,),
        compiler_params=dict(
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    score = func(log_lambda, key, query)
    score = score.sum(axis=0, dtype=jnp.float32).astype(key.dtype)
    res = (log_lambda, key, query)
    return score, res


# @functools.partial(jax.vmap(in_axes=(0, 1, 1, 1, 1, None, None, None, None, None, None), out_axes=(0, 1)))
def chunk_gla_forward(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    block_size_t: int,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):
    # assert log_fgate.dtype == jnp.float32

    # log_fgate = log_fgate.astype(jnp.float32)
    N, C, D = log_fgate.shape
    assert log_fgate.shape == value.shape == key.shape == query.shape
    assert is_power_of_two(C), f'Chunk length must be power of 2, but got {C}'
    # (log_fgate, value, key, query) = [
    #     einops.rearrange(x, '(n c) h d -> h n c d', c=chunk_size) for x in (log_fgate, value, key, query)
    # ]
    # (log_fgate, value, key, query) = [
        # einops.rearrange(x, 'h (n c) d -> h n c d', c=chunk_size) for x in (log_fgate, value, key, query)
    # ]

    # For some reason vmap is faster
    log_lambda = cumsum_recursive(log_fgate, axis=-2)
    # print('WARNING: not using cumsum')
    # log_lambda = log_fgate
    log_fgate_chunk = log_lambda[:, -1, :]

    intra_contrib, score_list = compute_intra_contrib(
        log_lambda,
        value,
        key,
        query,
        block_size_t=block_size_t,
        block_size_d=block_size_d,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=num_stages

    )

    (hidden_final, inter_contrib), state = compute_inter_contrib(
        hidden_init,
        log_lambda,
        log_fgate_chunk,
        value,
        key,
        query
    )

    out = intra_contrib.at[:].add(inter_contrib)
    # out = einops.rearrange(out, 'h n c d -> h (n c) d', c=chunk_size)
    assert out.dtype == value.dtype
    out = out.astype(value.dtype)

    # For backward pass
    res = (
        log_fgate,
        value,
        key,
        query,
        score_list,
        state,
    )
    return (hidden_final, out), res

def chunk_gla_backward(
    # hidden_init,
    # log_fgate,
    # value,
    # key,
    # query,

    # dhidden_
    block_size_t: int,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int,

    res,
    g,
):
    (log_fgate, value, key,
        query, score_list, state) = res
    dhidden_final, dout = g
    # assert log_fgate.dtype == jnp.float32

    # log_fgate = log_fgate.astype(jnp.float32)
    N, C, D = log_fgate.shape
    assert is_power_of_two(C), f'Sequence length must be power of 2, but got {T}'
    # (log_fgate, value, key, query) = [
    #     einops.rearrange(x, '(n c) h d -> h n c d', c=chunk_size) for x in (log_fgate, value, key, query)
    # ]
    # dout = einops.rearrange(dout, 'h (n c) d -> h n c d', c=chunk_size)

    # For some reason vmap is faster
    log_lambda = cumsum_recursive(log_fgate, axis=-2)
    # print('WARNING: not using cumsum')
    # log_lambda = log_fgate
    log_fgate_chunk = log_lambda[:, -1, :]

    (dlog_lambda_intra, dvalue_intra,
     dkey_intra, dquery_intra) = compute_intra_contrib_backward(
            log_lambda,
            value,
            key,
            query,
            dout=dout,
            score_list=score_list,
            block_size_t=block_size_t,
            block_size_d=block_size_d,
            interpret=interpret,
            num_warps=num_warps,
            num_stages=num_stages
    )

    (dhidden_init, dlog_lambda_inter, dlog_fgate_chunk,
     dvalue_inter, dkey_inter, dquery_inter) = compute_inter_contrib_backward(
        log_lambda,
        log_fgate_chunk,
        value,
        key,
        query,
        dhidden_final=dhidden_final,
        dout=dout,
        state=state
    )

    dlog_lambda = dlog_lambda_inter.at[:].add(dlog_lambda_intra)
    dlog_fgate = cumsum_recursive(dlog_lambda, axis=-2, init=dlog_fgate_chunk,
                                  reverse=True)
    dvalue = dvalue_inter.at[:].add(dvalue_intra)
    dquery = dquery_inter.at[:].add(dquery_intra)
    dkey = dkey_inter.at[:].add(dkey_intra)

    # dlog_fgate = cumsum_recursive((query * dquery - key * dkey).reshape(N * C, D), axis=0, reverse=True) + (dhidden_final * hidden_final).sum(axis=-1)
    # dlog_fgate = dlog_fgate.reshape(N, C, D)

    return dhidden_init, dlog_fgate, dvalue, dkey, dquery


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def chunk_gla_forward_backward(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    block_size_t: int,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):
    (hidden_final, out), res = chunk_gla_forward(
        hidden_init=hidden_init,
        log_fgate=log_fgate,
        value=value,
        key=key,
        query=query,
        block_size_t=block_size_t,
        block_size_d=block_size_d,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return hidden_final, out

chunk_gla_forward_backward.defvjp(chunk_gla_forward, chunk_gla_backward)

def chunk_gla(
    hidden_init,
    log_fgate,
    value,
    key,
    query,
    chunk_size,
    block_size_t: int = 64,
    block_size_d: int = 64,
    interpret: bool = False,
    num_warps: int = 4,
    num_stages: int = 2
):
    # (log_fgate, value, key, query) = [
        # einops.rearrange(x, 'h (n c) d -> h n c d', c=chunk_size) for x in (log_fgate, value, key, query)
    # ]
    (log_fgate, value, key, query) = [
        einops.rearrange(x, '(n c) h d -> h n c d', c=chunk_size) for x in (log_fgate, value, key, query)
    ]
    hidden_final, out = jax.vmap(functools.partial(
        chunk_gla_forward_backward,
        block_size_t=block_size_t,
        block_size_d=block_size_d,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=num_stages
    ))(
        hidden_init=hidden_init,
        log_fgate=log_fgate,
        value=value,
        key=key,
        query=query,
    )
    # out = einops.rearrange(out, 'h n c d -> h (n c) d', c=chunk_size)
    out = einops.rearrange(out, 'h n c d -> (n c) h d', c=chunk_size)
    return hidden_final, out


def compute_inter_contrib(
    hidden_init,
    log_lambda,
    log_fgate_chunk,
    value,
    key,
    query,
):
    N, L, D = log_lambda.shape
    # (N, D)
    log_gamma = log_fgate_chunk[:, None, :] - log_lambda

    weighted_key = (key * jnp.exp(log_gamma)).astype(key.dtype)
    kv = weighted_key.mT @ value

    # Scan
    _, state = rnn_recursive(
        log_fgate_chunk,
        kv,
        jnp.zeros_like(log_fgate_chunk[0]),
        hidden_init,
    )


    state = state.astype(kv.dtype)
    result = jnp.concatenate([hidden_init[None, ...].astype(state.dtype), state], axis=0)
    # hidden_final = (jnp.exp(log_fgate_chunk[-1][:, None]) * state[-1] + kv[-1]).astype(hidden_init.dtype)
    # (N, D, D)
    state, hidden_final = result[:-1], result[-1]
    weighted_query = (query * jnp.exp(log_lambda)).astype(query.dtype)
    out = weighted_query @ state

    hidden_final = hidden_final.astype(hidden_init.dtype)

    return (hidden_final, out), state

def compute_inter_contrib_backward(
    log_lambda,
    log_fgate_chunk, # (N, D)
    value,
    key,
    query,

    dhidden_final,
    dout,

    state
):
    N, L, D = log_lambda.shape
    # (N, D)
    log_gamma = log_fgate_chunk[:, None, :] - log_lambda

    # (N, L, D)
    weighted_query = (query * jnp.exp(log_lambda)).astype(query.dtype)

    dstate = weighted_query.mT @ dout

    dweighted_query = matmul_fp32(dout, state.mT)
    dquery = (dweighted_query * jnp.exp(log_lambda)).astype(query.dtype)
    dlog_lambda_query = dweighted_query * weighted_query
    assert dlog_lambda_query.dtype == jnp.float32


    # You can verify this correct!
    _, dkv = rnn_recursive(
        log_fgate_chunk,
        dstate,
        jnp.zeros_like(log_fgate_chunk[0]),
        dhidden_final,
        reverse=True
    )
    dkv = dkv.astype(state)

    result = jnp.concatenate([dkv, dhidden_final[None, ...].astype(state.dtype)], axis=0)
    # hidden_final = (jnp.exp(log_fgate_chunk[-1][:, None]) * state[-1] + kv[-1]).astype(hidden_init.dtype)
    # (N, D, D)
    dhidden_init, dkv = result[0], result[1:]


    weighted_key = (key * jnp.exp(log_gamma)).astype(key.dtype)
    dvalue = weighted_key @ dkv
    dweighted_key = matmul_fp32(value, dkv.mT)

    dkey = dweighted_key * jnp.exp(log_gamma)
    # (N, L, D)
    dlog_lambda_key = -dweighted_key * weighted_key
    # (N, D)
    dlog_fgate_chunk = -dlog_lambda_key.sum(axis=1) + (state * jnp.exp(log_fgate_chunk)[:, :, None] * dkv).sum(axis=-1)
    # dlog_fgate_chunk = jnp.zeros((N, D))
    assert dlog_lambda_query.dtype == jnp.float32

    dlog_lambda = dlog_lambda_key + dlog_lambda_query

    dhidden_init = dhidden_init.astype(dhidden_final.dtype)

    return dhidden_init, dlog_lambda, dlog_fgate_chunk, dvalue, dkey, dquery

def compute_intra_contrib(
    log_lambda,
    value,
    key,
    query,
    block_size_t: int,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):
    """
    It is possible to pass log_fgate instead of log_lambda. That has better
    numerical precision but more trouble to implement.
    """

    # (L, D)
    N, L, D = log_lambda.shape
    assert is_power_of_two(L)

    @jax.vmap
    def attention(log_lambda_left, log_lambda_right, value, key, query):
        anchor = log_lambda_left[-1:]
        query_weight = jnp.exp(log_lambda_right - anchor)
        weighted_query = (query_weight * query).astype(query.dtype)

        key_weight = jnp.exp(anchor - log_lambda_left)
        weighted_key = (key * key_weight).astype(key.dtype)
        score = (weighted_query @ weighted_key.mT)
        contrib = score @ value
        return contrib, score

    score_list = []
    def _scan(log_lambda, value, key, query):
        assert log_lambda.shape == value.shape == key.shape == query.shape
        N, C, D = log_lambda.shape
        if C <= block_size_t:
            # If chunk fit into one thread block, we do it in SRAM
            # (N, C, C)
            batch_compute_score = jax.vmap(
                functools.partial(compute_score, block_size_d=block_size_d, interpret=interpret, num_warps=num_warps, num_stages=num_stages)
            )
            score = batch_compute_score(
                log_lambda,
                key,
                query,
            )
            out = score @ value
            # This is the accumulator
            score_list.append(score)
            return out
        else:
            log_lambda_next, value_next, key_next, query_next = [x.reshape(2 * N, C // 2, D) for x in (log_lambda, value, key, query)]
            contrib_intra = _scan(log_lambda_next, value_next, key_next, query_next)
            # assert contrib_intra.dtype == jnp.float32
            contrib_intra = contrib_intra.reshape(N, C, D)
            M = C // 2
            contrib_inter, score = attention(log_lambda[:, :M], log_lambda[:, M:], value[:, :M], key[:, :M], query[:, M:])
            score_list.append(score)
            out = contrib_intra.at[:, M:].add(contrib_inter)
            return out

    contrib = _scan(log_lambda, value, key, query)
    return contrib, score_list

def compute_intra_contrib_backward(
    log_lambda,
    value,
    key,
    query,
    dout,
    score_list,


    block_size_t: int,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):
    N, L, D = log_lambda.shape
    assert is_power_of_two(L)




    @jax.vmap
    def attention_backward(log_lambda_left, log_lambda_right, value, key, query, dout, score):
        dvalue = score.mT @ dout
        dscore = dout @ value.mT


        anchor = log_lambda_left[-1:]
        query_weight = jnp.exp(log_lambda_right - anchor)
        weighted_query = (query_weight * query).astype(query.dtype)

        key_weight = jnp.exp(anchor - log_lambda_left)
        weighted_key = (key * key_weight).astype(key.dtype)
        # contrib = score @ value

        dweighted_query = matmul_fp32(dscore, weighted_key)
        dquery = (dweighted_query * query_weight).astype(query.dtype)
        dlog_lambda_right = dweighted_query * weighted_query
        # dquery = dweighted_query.astype(query.dtype)
        # dlog_lambda_right = dweighted_query
        assert dlog_lambda_right.dtype == jnp.float32

        dweighted_key = matmul_fp32(dscore.mT, weighted_query)
        dkey = (dweighted_key * key_weight).astype(key.dtype)
        dlog_lambda_left = -dweighted_key * weighted_key
        # dkey = dweighted_key.astype(key.dtype)
        # dlog_lambda_left = dweighted_key
        assert dlog_lambda_left.dtype == jnp.float32

        return dlog_lambda_left, dlog_lambda_right, dvalue, dkey, dquery

    def _compute(log_lambda, value, key, query, dout):
        """
        Returns:
            dlog_lambda
            dvalue
            dkey
            dquery
        """
        N, C, D = log_lambda.shape
        score = score_list.pop()
        assert log_lambda.shape == value.shape == key.shape == query.shape == dout.shape == (N, C, D)
        if C <= block_size_t:
            assert score.shape == (N, C, C), (score.shape, (N, C, C))
            # If chunk fit into one thread block, we do it in SRAM
            # (N, C, C)
            batch_compute_score = jax.vmap(
                functools.partial(compute_score, block_size_d=block_size_d, interpret=interpret, num_warps=num_warps, num_stages=num_stages)
            )
            dvalue = score.mT @ dout
            dscore = dout @ value.mT
            dlog_lambda, dkey, dquery = jax.vjp(
                batch_compute_score,
                log_lambda,
                key,
                query
            )[-1](dscore)
            assert dlog_lambda.dtype == jnp.float32


            return dlog_lambda, dvalue, dkey, dquery
        else:
            assert score.shape == (N, C // 2, C // 2), (score.shape, (N, C // 2, C // 2))
            log_lambda_next, value_next, key_next, query_next, dout_next = [x.reshape(2 * N, C // 2, D) for x in (log_lambda, value, key, query, dout)]
            dlog_lambda_intra, dvalue_intra, dkey_intra, dquery_intra = _compute(log_lambda_next, value_next, key_next, query_next, dout_next)
            dlog_lambda_intra, dvalue_intra, dkey_intra, dquery_intra = [
                x.reshape(N, C, D) for x in (dlog_lambda_intra, dvalue_intra, dkey_intra, dquery_intra)
            ]
            M = C // 2
            (dlog_lambda_left, dlog_lambda_right,
             dvalue_inter, dkey_inter, dquery_inter) = attention_backward(
                 log_lambda[:, :M], log_lambda[:, M:], value[:, :M],
                 key[:, :M], query[:, M:], dout[:, M:], score)
            # TODO: maybe use fp32 here
            dlog_lambda = dlog_lambda_intra.at[:, :M].add(dlog_lambda_left)
            dlog_lambda = dlog_lambda.at[:, M:].add(dlog_lambda_right)

            dvalue = dvalue_intra.at[:, :M].add(dvalue_inter)
            dquery = dquery_intra.at[:, M:].add(dquery_inter)
            dkey = dkey_intra.at[:, :M].add(dkey_inter)
            return dlog_lambda, dvalue, dkey, dquery

    dlog_lambda, dvalue, dkey, dquery = _compute(log_lambda, value, key, query, dout)
    return dlog_lambda, dvalue, dkey, dquery



def compute_score_forward_kernel(
    log_lambda_ref,
    key_ref,
    query_ref,
    out_ref,
):
    # assert log_lambda_ref.dtype == jnp.float32
    L, D_q = query_ref.shape
    L, D_k = key_ref.shape
    assert D_q == D_k
    D = D_q
    assert D >= 16
    assert L >= 32
    assert is_power_of_two(L)

    # Fill upper triangular with zero
    indices = jnp.arange(L)
    pl.store(out_ref, slice(None), jnp.zeros_like(out_ref), mask=indices[:, None] < indices[None, :])
    # Fill diagonal entries
    out_ref[indices, indices] = (query_ref[:] * key_ref[:]).sum(axis=-1, dtype=jnp.float32).astype(out_ref.dtype)

    def compute(start, end):
        middle = (start + end) // 2
        anchor = log_lambda_ref[middle - 1:middle]
        log_lambda = (log_lambda_ref[middle:end] - anchor).astype(jnp.float32)
        log_gamma = (anchor - log_lambda_ref[start:middle]).astype(jnp.float32)

        q_weighted = (query_ref[middle:end] * jnp.exp(log_lambda)).astype(query_ref.dtype)
        k_weighted = (key_ref[start:middle] * jnp.exp(log_gamma)).astype(key_ref.dtype)
        out_ref[middle:end, start:middle] = q_weighted @ k_weighted.T
        if end - start > 32:
            compute(start, middle)
            compute(middle, end)
        else:
            assert end - start == 32

            for C in [16, 8, 4, 2]:
                # Tricky case: we group smaller matmul into larger matmul >= 16
                # This uses more FLOPs but due to tensor core it is actually faster
                # G * C = 32
                # (G, 1)
                G = 32 // C
                start_list = C * jnp.arange(G)[:, None] + start
                # (G, 1)
                anchor_indices = start_list + (C // 2 - 1)  # It is okay not to minus 1
                # (G, C // 2)
                k_indices = (start_list + jnp.arange(C // 2)[None, :])
                # (G, C // 2) -> 16
                q_indices = k_indices + C // 2
                mask = q_indices.ravel()[:, None] > k_indices.ravel()[None, :]
                # (16, 16)
                mask = mask & mask.T  # Block diagonal

                # (G, 1, D)
                anchor = log_lambda_ref[anchor_indices] # It is okay not to minus 1
                # (G, C // 2, D)
                log_lambda = (log_lambda_ref[q_indices] - anchor).astype(jnp.float32)
                log_gamma = (anchor - log_lambda_ref[k_indices]).astype(jnp.float32)
                # (16, D)
                q_weighted = (query_ref[q_indices] * jnp.exp(log_lambda)).astype(query_ref.dtype).reshape(16, D)
                k_weighted = (key_ref[k_indices] * jnp.exp(log_gamma)).astype(key_ref.dtype).reshape(16, D)

                result = q_weighted @ k_weighted.T
                pl.store(out_ref, (q_indices.ravel()[:, None], k_indices.ravel()[None, :]), result, mask=mask)
    
    compute(0, L)
    # fill_diagonal()


def matmul_fp32(a, b):
    return jnp.matmul(a, b, preferred_element_type=jnp.float32)

def compute_score_backward_kernel(
    log_lambda_ref,
    key_ref,
    query_ref,
    dscore_ref,

    dlog_lambda_ref,
    dkey_ref,
    dquery_ref,
):
    # assert log_lambda_ref.dtype == jnp.float32
    L, D_q = query_ref.shape
    L, D_k = key_ref.shape
    assert D_q == D_k
    D = D_q
    assert D >= 16
    assert is_power_of_two(L)

    # Fill upper triangular with zero
    # indices = jnp.arange(L)
    # pl.store(out_ref, slice(None), jnp.zeros_like(out_ref), mask=indices[:, None] < indices[None, :])

    # buffer = dict()


    def compute(start, end):
        middle = (start + end) // 2
        anchor = log_lambda_ref[middle - 1:middle]
        log_lambda = (log_lambda_ref[middle:end] - anchor).astype(jnp.float32)
        log_gamma = (anchor - log_lambda_ref[start:middle]).astype(jnp.float32)

        q_weighted = (query_ref[middle:end] * jnp.exp(log_lambda)).astype(query_ref.dtype)
        k_weighted = (key_ref[start:middle] * jnp.exp(log_gamma)).astype(key_ref.dtype)

        dscore = dscore_ref[middle:end, start:middle]

        # Query 
        # This must be in float32 to ensure numerical precision
        dq_weighted = matmul_fp32(dscore, k_weighted)
        assert dq_weighted.dtype == jnp.float32
        dquery_lower_left = dq_weighted * jnp.exp(log_lambda)
        # We must NOT reused dquery here for numerical precision
        dlog_lambda_query = dq_weighted * q_weighted
        # dlog_lambda_query = dquery_lower_left * query_ref[middle:end]

        # Key and log
        dk_weighted = matmul_fp32(dscore.T, q_weighted)
        assert dk_weighted.dtype == jnp.float32
        dkey_lower_left = dk_weighted * jnp.exp(log_gamma)
        # We must NOT reused dkey here for numerical precision
        dlog_lambda_key = -dk_weighted * k_weighted
        # dlog_lambda_key = -dkey_lower_left * key_ref[start:middle]

        if end - start > 32:
            dlog_lambda_upper_left, dkey_upper_left, dquery_upper_left = compute(start, middle)
            dlog_lambda_lower_right, dkey_lower_right, dquery_lower_right = compute(middle, end)

            dlog_lambda = join(
                dlog_lambda_key + dlog_lambda_upper_left,
                dlog_lambda_query + dlog_lambda_lower_right,
                axis=0
            )
            dkey = join(
                dkey_lower_left + dkey_upper_left,
                dkey_lower_right,
                axis=0
            )
            dquery = join(
                dquery_upper_left,
                dquery_lower_left + dquery_lower_right,
                axis=0
            )
            return dlog_lambda, dkey, dquery
        else:
            assert end - start == 32

            # (32, D)
            filler = jnp.zeros((16, D), dtype=jnp.float32)
            dkey = join(dkey_lower_left, filler, axis=0)
            dquery = join(filler, dquery_lower_left, axis=0)
            dlog_lambda = join(dlog_lambda_key, dlog_lambda_query, axis=0)
            for C in [16, 8, 4, 2]:
                # Tricky case: we group smaller matmul into larger matmul >= 16
                # This uses more FLOPs but due to tensor core it is actually faster
                # G * C = 32
                # (G, 1)
                G = 32 // C
                start_list = C * jnp.arange(G)[:, None] + start
                # (G, 1)
                anchor_indices = start_list + (C // 2 - 1)  # It is okay not to minus 1
                # (G, C // 2)
                k_indices = (start_list + jnp.arange(C // 2)[None, :])
                # (G, C // 2) -> 16
                q_indices = k_indices + C // 2
                mask = q_indices.ravel()[:, None] > k_indices.ravel()[None, :]
                # (16, 16)
                mask = mask & mask.T  # Block diagonal

                # (G, 1, D)
                anchor = log_lambda_ref[anchor_indices] # It is okay not to minus 1
                # (G, C // 2, D)
                log_lambda = (log_lambda_ref[q_indices] - anchor).astype(jnp.float32)
                log_gamma = (anchor - log_lambda_ref[k_indices]).astype(jnp.float32)
                # (16, D)
                q_weighted = (query_ref[q_indices] * jnp.exp(log_lambda)).astype(query_ref.dtype)
                k_weighted = (key_ref[k_indices] * jnp.exp(log_gamma)).astype(key_ref.dtype)

                # result = q_weighted @ k_weighted.T
                # pl.store(out_ref, (q_indices.ravel()[:, None], k_indices.ravel()[None, :]), result, mask=mask)
                # dscore = 
                # (16, 16)
                # Note this other=0.0 is extremely important
                dscore = pl.load(dscore_ref, (q_indices.ravel()[:, None], k_indices.ravel()[None, :]), mask=mask, other=0.0)

                filler = jnp.zeros((G, C // 2, D), dtype=jnp.float32)

                # Query
                # This must be in float32 to ensure numerical precision
                dq_weighted = matmul_fp32(dscore, k_weighted.reshape(16, D))
                dq_weighted = dq_weighted.reshape(G, C // 2, D)
                assert dq_weighted.dtype == jnp.float32
                # (G, C // 2, D)
                dquery_local = dq_weighted * jnp.exp(log_lambda)
                dquery += join(filler, dquery_local, axis=1).reshape(32, D)
                # We must NOT reused dquery here for numerical precision
                dlog_lambda_query = dq_weighted * q_weighted
                # dlog_lambda_query = dquery_local * query_ref[q_indices]

                # Key
                # This must be in float32 to ensure numerical precision
                dk_weighted = matmul_fp32(dscore.T, q_weighted.reshape(16, D))
                dk_weighted = dk_weighted.reshape(G, C // 2, D)
                assert dk_weighted.dtype == jnp.float32
                dkey_local = dk_weighted * jnp.exp(log_gamma)
                dkey += join(dkey_local, filler, axis=1).reshape(32, D)
                dlog_lambda_key = -dk_weighted * k_weighted
                # dlog_lambda_key = -dkey_local * key_ref[k_indices]

                dlog_lambda += join(
                    dlog_lambda_key, dlog_lambda_query, axis=1
                ).reshape(32, D)
            return dlog_lambda, dkey, dquery
    
    dlog_lambda, dkey, dquery = compute(0, L)

    indices = jnp.arange(L)
    dscore_diag = dscore_ref[indices, indices]
    dkey += dscore_diag[:, None] * query_ref[:]
    dquery += dscore_diag[:, None] * key_ref[:]
    # No need to update dlog_lambda here
    # return dlog_lambda, dkey, dquery
    dkey_ref[:] = dkey.astype(dkey_ref.dtype)
    dquery_ref[:] = dquery.astype(dquery_ref.dtype)
    dlog_lambda_ref[:] = dlog_lambda.astype(jnp.float32)

def compute_score_backward(

    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int,
    res,
    g
    # log_lambda,
    # key,
    # query,
    # dscore,
):
    log_lambda, key, query = res
    dscore = g

    L, D = log_lambda.shape
    block_size_d = min(block_size_d, D)
    num_blocks_d = D // block_size_d

    func = pl.pallas_call(
        compute_score_backward_kernel,
        interpret=interpret,
        # f=functools.partial(compute_score_backward_kernel, block_size_d=block_size_d),
        # out_shape=jax.ShapeDtypeStruct((L, L), key.dtype),
        out_shape= [
            jax.ShapeDtypeStruct((L, D), log_lambda.dtype),
            jax.ShapeDtypeStruct((L, D), key.dtype),
            jax.ShapeDtypeStruct((L, D), query.dtype),
        ],
        in_specs=[
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, 0), (L, L)),

        ],
        out_specs=[
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
            pl.BlockSpec(lambda d: (0, d), (L, block_size_d)),
        ],
        grid=(num_blocks_d,),
        compiler_params=dict(
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    dlog_lambda, dkey, dquery = func(log_lambda, key, query, dscore)
    return dlog_lambda, dkey, dquery

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def compute_score(
    log_lambda,
    key,
    query,
    block_size_d: int,
    interpret: bool,
    num_warps: int,
    num_stages: int
):
    score, res = compute_score_forward(
        log_lambda,
        key,
        query,
        block_size_d=block_size_d,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return score

compute_score.defvjp(compute_score_forward, compute_score_backward)






if __name__ == '__main__':



    INTERPRET = False
    def test_speed():
        variable = 'seq_len'
        suffix = 'bs1'
        # batch_size_list = [1, 4, 16, 32]
        # seq_len_list = [1024, 2048, 4096, 8192, 16384]
        # head_dim_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        # batch_size_list = [1]
        # seq_len_list = [16384]
        # seq_len_list = [1024]

        # batch_size_seq_len_list = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
        # batch_size_seq_len_list = [(1, 16384)]
        batch_size_seq_len_list = [(4, 4096)]
        # batch_size_seq_len_list = itertools.product(batch_size_list, seq_len_list)
        # head_dim_list = [256]
        # head_dim_list = [64, 128, 256, 512, 1024, 2048]
        head_dim_list = [128]
        max_block_size = 64
        num_warps = 2
        num_stages = 1
        key_dim = 2048
        # assert value_dim == key_dim
        reverse = False
        GRAD = True
        CHECK = True
        PLOT = False
        block_size_d = 64
        block_size_t = 64

        num_warps_fast = 4  # Set to 4 if head_dim < 128 
        num_stages_fast = 2

        methods = [
            'recurrent_gla_naive',
            'recurrent_gla',
            'chunk_gla',
        ]
        # number = 30

        result_dict = defaultdict(list)
        for ((batch_size, seq_len), head_dim) in itertools.product(batch_size_seq_len_list, head_dim_list):
            num_heads = key_dim // head_dim

            chunk_size = head_dim

            # value_block_size = min(max_value_block_size, head_dim)
            # key_block_size = min(max_key_block_size, head_dim)
            # if value_block_size < max_key_block_size:
            #     head_block_size = max_key_block_size // value_block_size
            # else:
            # head_block_size = 1




            def wrapper(func):
                # Head dim
                # func = jax.vmap(func, in_axes=(1, 1, 1, 1, 0), out_axes=(0, 1))
                    # Batch dim
                func = jax.vmap(func)
                def loss(hidden_init, log_fgate, value, key, query):
                    hidden_final, out = func(hidden_init, log_fgate, value, key, query)
                    return hidden_final.sum() + out.sum(), (hidden_final, out)
                if GRAD:
                    grad_func = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))
                else:
                    grad_func = jax.jit(lambda *args: func(*args))
                # grad_func = func
                return grad_func

            recurrent_gla_naive_wrapped = wrapper(functools.partial(
                recurrent_gla_naive,
                reverse=reverse))

            recurrent_gla_wrapped = wrapper(functools.partial(
                recurrent_gla,
                reverse=reverse,
                max_block_size=max_block_size,
                num_warps=num_warps,
                num_stages=num_stages,
                interpret=INTERPRET,
                # value_block_size=value_block_size,
                # key_block_size=key_block_size,
                # head_block_size=head_block_size
            ))

            chunk_gla_wrapped = wrapper(functools.partial(
                chunk_gla,
                block_size_d=block_size_d,
                block_size_t=block_size_t,
                chunk_size=chunk_size,
                interpret=INTERPRET,
                num_warps=num_warps_fast,
                num_stages=num_stages_fast,
            ))

            out_dict = {}
            for name, func in dict(
                chunk_gla=chunk_gla_wrapped,
                recurrent_gla=recurrent_gla_wrapped,
                recurrent_gla_naive=recurrent_gla_naive_wrapped,
            ).items():
                if name not in methods:
                    continue
                print('=' * 80)
                print(f'seq_len: {seq_len}')
                print(f'head_dim: {head_dim}')
                print(f'batch_size: {batch_size}')
                print(name)

                rng = jax.random.PRNGKey(0)
                dtype = 'bfloat16'
                hidden_init = jax.random.normal(key=rng, shape=(batch_size, num_heads, head_dim, head_dim), dtype=dtype)
                value = jax.random.normal(key=rng, shape=(batch_size, seq_len, num_heads, head_dim), dtype=dtype)
                query = jax.random.normal(key=rng, shape=(batch_size, seq_len, num_heads, head_dim), dtype=dtype)
                log_fgate = jax.nn.log_sigmoid(5.0 + jax.random.normal(key=rng, shape=(batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32))
                key = jax.random.normal(key=rng, shape=(batch_size, seq_len, num_heads, head_dim), dtype=dtype)



                hidden_init, value, key, query, log_fgate = [jax.device_put(x) for x in (hidden_init, value, key, query, log_fgate)]
                stmt = lambda: jax.tree_util.tree_map(lambda x: x.block_until_ready(), func(
                    hidden_init,
                    log_fgate,
                    value,
                    key,
                    query,
                ))
                start = time.perf_counter()
                if GRAD:
                    (dhidden_init, dlog_fgate, dvalue, dkey, dquery), (hidden_final, out)  = stmt()
                    out_dict[name] = dict(
                        hidden_final=hidden_final,
                        out=out,
                        dhidden_init=dhidden_init,
                        dlog_fgate=dlog_fgate,
                        dvalue=dvalue,
                        dkey=dkey,
                        dquery=dquery)
                else:
                    (hidden_final, out)  = stmt()
                    out_dict[name] = dict(
                        hidden_final=hidden_final,
                        out=out,
                    )

                end = time.perf_counter()
                compile_time = end - start
                print(f'Compilation time: {compile_time}s')

                # hidden_final_dict[name] = hidden_final
                # out_dict[name] = out

                # Execution time
                timer = timeit.Timer(stmt=stmt)
                iterations, exec_time = timer.autorange()
                # iterations = 100
                # exec_time = timer.timeit(number=iterations)
                exec_time = exec_time / iterations
                print(f'Length: {seq_len}')
                # print(r'Execution time:')
                # print(f'    Total: {exec_time}s')
                print(f'Iterations: {iterations}')
                print(f'Time: {exec_time * 1000}ms')
                print('=' * 80)
                result_dict[name].append(
                    dict(
                        exec_time=exec_time,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        key_dim=key_dim,
                        head_dim=head_dim,
                    )
                )


            if CHECK and len(out_dict) >= 2:
                for name_a, name_b in itertools.combinations(out_dict, 2):
                    print(f'Verifying {name_a} and {name_b}')
                    assert set(out_dict[name_a]) == set(out_dict[name_b])
                    for key in out_dict[name_a]:
                        value_a = out_dict[name_a][key]
                        value_b = out_dict[name_b][key]
                        if not jnp.allclose(value_a, value_b):
                            error = jnp.absolute(value_a - value_b)
                            # rel_error = error / jnp.abs(value_a)
                            rel_error = jnp.linalg.norm(value_a.astype(jnp.float32) - value_b) / jnp.linalg.norm(value_a.astype(jnp.float32))
                            print(f'  {name_a}->{key} and {name_b}->{key} do not match')
                            print(f'    Max abs error {error.max()}')
                            print(f'    Rel error {rel_error}')
                            # index = error.argmax()
                            # print(f'    Rel error at maximum abs error {rel_error.ravel()[index]}')
                            # index = rel_error.argmax()
                            # print(f'    Max rel error {rel_error.max()}')
                            # print(f'    Abs error at maximum rel error {error.ravel()[index]}')

        def get_data(result_list: list, variable: str, filter: callable = lambda *_: True):
            x = []
            y = []
            for entry in result_list:
                if filter(entry):
                    x.append(entry[variable])
                    y.append(entry['exec_time'] * 1000)
            return x, y


        if PLOT:
            fig, axes = plt.subplots(squeeze=False)
            for name, result_list in result_dict.items():
                ax = axes[0][0]
                x, y = get_data(result_list, variable)
                ax.plot(x, y, label=name)
                ax.set_xlabel(variable)
                ax.set_ylabel('exec_time (ms)')
                ax.legend()

            fig.tight_layout()
            FIGURE_DIR = Path('figures')
            FIGURE_DIR.mkdir(exist_ok=True, parents=True)
            path = FIGURE_DIR / f'plot_{variable}_{suffix}.png'
            fig.savefig(path)
            print(f'Result saved to {path}')

    test_speed()
