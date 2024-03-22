from einops import rearrange, repeat, reduce, pack, unpack
from typing import Callable
from torch import nn, einsum
import torch.nn.functional as F
import torch

import vector_quantize_pytorch
from vector_quantize_pytorch.vector_quantize_pytorch import default, partial, exists, distributed, log, orthogonal_loss_fn, identity, gumbel_noise, pack_one, unpack_one, uniform_init, sample_vectors_distributed, batched_sample_vectors, noop, kmeans, autocast, laplace_smoothing, ema_inplace, cdist, batched_embedding

def gumbel_sample_topk(
    logits,
    k,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    # ind = sampling_logits.argmax(dim = dim)
    ind = torch.topk(sampling_logits, k+1, dim = dim).indices[:, :, k] # TODO: Make this more efficient
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim = dim)
        π1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim = dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

class EuclideanCodebookTopK(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample_topk,
        sample_codebook_temp = 1.,
        ema_update = True,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask = None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased = False)

        # calculate codebook mean and variance

        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.training:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            
            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(
        self,
        x,
        k,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, k=k, dim = -1, temperature = sample_codebook_temp, training = self.training)

        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:

            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.

            cluster_size = embed_onehot.sum(dim = 1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, ps, 'h * d')

        return quantize, embed_ind, dist
    
class RerankVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0. # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        self.sync_update_v = sync_update_v

        # Modified codebook
        codebook_class = EuclideanCodebookTopK

        gumbel_sample_fn = partial(
            gumbel_sample_topk,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def forward_pre_quantize(
        self,
        x,
        k,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        # quantize

        quantize, embed_ind, distances = self._codebook(x, k=k)
        return quantize, embed_ind, distances

    def forward_post_quantize(self, x, quantize_overwrite, indices_overwrite, distances_overwrite, 
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False):

        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        # quantize
        quantize, embed_ind, distances = quantize_overwrite, indices_overwrite, distances_overwrite

        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again
            quantize, embed_ind, distances = quantize_overwrite, indices_overwrite, distances_overwrite

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)            

            # straight through

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(
                rearrange(mask, '... -> ... 1'),
                quantize,
                orig_input
            )

        return quantize, embed_ind, loss
    
    def forward_topk(self, x, topk, search_func, seq_len):
        # TODO: Beam search
        quantized_topk = []
        embed_ind_topk = []
        distances_topk = []
        for k in range(topk):
            quantize, embed_ind, distances = self.forward_pre_quantize(x, k)
            # quantize: [batch_size * seq_len, 1, dim]
            # embed_ind: [batch_size * seq_len, 1]
            # distances: [1, batch_size * seq_len, 1, codebook_size]
            quantized_topk.append(quantize.unsqueeze(0))
            embed_ind_topk.append(embed_ind.unsqueeze(0))
            distances_topk.append(distances.unsqueeze(0))

        quantized_topk = torch.cat(quantized_topk, dim=0).reshape([topk, -1, seq_len, 1, quantize.shape[-1]]) # [topk, batch_size, seq_len, 1, dim]
        embed_ind_topk = torch.cat(embed_ind_topk, dim=0).reshape([topk, -1, seq_len, 1]) # [topk, batch_size, seq_len, 1]
        distances_topk = torch.cat(distances_topk, dim=0).reshape([topk, 1, -1, seq_len, 1, distances.shape[-1]]) # [topk, 1, batch_size, seq_len, 1, codebook_size]
        
        # print(quantized_topk.shape, embed_ind_topk.shape, distances_topk.shape)
        best_quantize, best_embed_ind, best_distances = search_func(quantized_topk, embed_ind_topk, distances_topk)

        quantize, embed_ind, loss = self.forward_post_quantize(x, best_quantize, best_embed_ind, best_distances)
        return quantize, embed_ind, loss