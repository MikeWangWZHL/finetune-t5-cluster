"""
adapt from https://github.com/lucidrains/flamingo-pytorch
"""


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_time_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.time_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


# gated cross attention

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x,
        media,
        media_locations = None,
        aug_exist_idx = None,
    ):
        b, t, m = media.shape[:3]
        # print("media: b, t, m:",b,t,m)
        h = self.heads

        x = self.norm(x)

        # print("x:", x.shape)

        q = self.to_q(x)
        
        # print("q:", q.shape)

        if len(media.shape) == 4:
            media = rearrange(media, 'b t m d -> b (t m) d')

        # print("media:", media.shape)

        k, v = self.to_kv(media).chunk(2, dim = -1)
        
        # print("k,v", k.shape, v.shape)
        
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        
        # print("q,k,v:", q.shape, k.shape, v.shape)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        # print("sim:", sim.shape)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        # print("aug_exist_idx:", aug_exist_idx)

        if exists(aug_exist_idx):
            aug_exist_idx = aug_exist_idx.detach()
            # print(aug_exist_idx)
            expended_idx = torch.repeat_interleave(aug_exist_idx, m, dim = -1)
            # print("expended_idx:", expended_idx.shape)
            expended_idx = repeat(expended_idx, 'b d -> b h n d', b=b, h=h, n=sim.shape[2])
            assert tuple(expended_idx.shape) == tuple(sim.shape)
            # print("expended_idx:", expended_idx.shape)
            ones = torch.ones_like(expended_idx)
            attend_to_exist_aug_mask = torch.eq(ones, expended_idx)
            # print(attend_to_exist_aug_mask)
            # print("attend_to_exist_aug_mask:",attend_to_exist_aug_mask.shape)
            sim = sim.masked_fill(~attend_to_exist_aug_mask, -torch.finfo(sim.dtype).max)
            # print(attend_to_exist_aug_mask[0][0][0])
            # print(attend_to_exist_aug_mask[16][0][0])

        # print('============================================================')

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        # print(sim[0][0][0])
        # print(sim[16][0][0])

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = False # zhenhailong: by default attend to all retrieved examples
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None,  # boolean tensor indicating positions of media - (batch, sequence)
        aug_exist_idx = None
    ):
        x = self.attn(x, media, media_locations = media_locations, aug_exist_idx = aug_exist_idx) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

def ReweightingFeedForward(in_dim, out_dim, mult = 0.5):
    inner_dim = int(in_dim * mult)
    return nn.Sequential(
        nn.Linear(in_dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, out_dim, bias = False),
        nn.Tanh()
    )

class DoubleGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = False, # zhenhailong: by default attend to all retrieved examples
        num_aug_source = 5
    ):
        super().__init__()
        self.reweighting_gate = nn.Parameter(torch.tensor([2. for i in range(num_aug_source)])) # init with almost full pass

        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))


    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None,  # boolean tensor indicating positions of media - (batch, sequence)
        aug_exist_idx = None
    ):
        ### get weighting for each aug source
        reweighting_factors = self.reweighting_gate[:media.shape[1]]
        # print(reweighting_factors)
        reweighting_factors = repeat(reweighting_factors, 't -> b t 1 1', b = media.shape[0])
        # print(reweighting_factors)
        media = media * reweighting_factors.tanh()
        # print(media.shape)

        ### cross attention
        x = self.attn(x, media, media_locations = media_locations, aug_exist_idx = aug_exist_idx) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

# ## July 18th 
# class DoubleGatedCrossAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_head = 64,
#         heads = 8,
#         ff_mult = 4,
#         only_attend_immediate_media = False, # zhenhailong: by default attend to all retrieved examples
#     ):
#         super().__init__()
#         self.reweighting_gate = ReweightingFeedForward(dim, 1)

#         self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
#         self.attn_gate = nn.Parameter(torch.tensor([0.]))

#         self.ff = FeedForward(dim, mult = ff_mult)
#         self.ff_gate = nn.Parameter(torch.tensor([0.]))


#     def forward(
#         self,
#         x,
#         media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
#         media_locations = None,  # boolean tensor indicating positions of media - (batch, sequence)
#         aug_exist_idx = None
#     ):
#         ### get weighting for each aug source
#         # print(media.shape)
#         media_cls = reduce(media, 'b t n d -> b t d', 'mean')
#         # print(media_cls.shape)
#         reweighting_factors = self.reweighting_gate(media_cls) # (b t 1)
#         # print(reweighting_factors.shape)
#         reweighting_factors = rearrange(reweighting_factors, 'b t 1 -> b t 1 1')
#         # print(reweighting_factors.shape)
#         media = media * reweighting_factors
#         # print(media.shape)

#         ### cross attention
#         x = self.attn(x, media, media_locations = media_locations, aug_exist_idx = aug_exist_idx) * self.attn_gate.tanh() + x
#         x = self.ff(x) * self.ff_gate.tanh() + x
#         return x


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

###############################
###############################


# from transformers import AutoModelForSeq2SeqLM,  AutoConfig, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM,  AutoConfig, AutoTokenizer, T5EncoderModel
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

class AugmentationResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_aug_sources = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(num_aug_sources, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.time_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class PerceiverResamplerWithT5(nn.Module):
    def __init__(
        self,
        *,
        encoder_path_or_name,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_ret_rank_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        
        # frozen T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(encoder_path_or_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # perceiver layers
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.ret_rank_emb = nn.Parameter(torch.randn(num_ret_rank_embeds, 1, dim)) # num of retrieved examples

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

        self.model_parallel = False
        self.device_map = None
    
    def parallelize(self, device_map=None):
        self.model_parallel = True
        self.encoder.parallelize(device_map=None)
        self.device_map = (
            get_device_map(len(self.layers), range(torch.cuda.device_count())) if device_map is None else device_map
        )

        assert_device_map(self.device_map, len(self.layers))
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.layers[layer] = self.layers[layer].to(cuda_device)
        
        # norm to last device
        self.norm = self.norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.encoder.deparallelize()
        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i].to("cpu")
        # self.latents = self.latents.to("cpu")
        # self.ret_rank_emb = self.ret_rank_emb.to("cpu")
        self.norm = self.norm.to("cpu")
        torch.cuda.empty_cache()

    def forward(
        self,
        *,
        input_ids, # (b, num_ret_examples, n) 
        attention_mask, # (b, num_ret_examples, n)
        **kwargs
    ):

        # get hidden states
        if input_ids.ndim == 2:
            assert attention_mask.ndim == 2
            input_ids = rearrange(input_ids, 'b n -> b 1 n')
            attention_mask = rearrange(attention_mask, 'b n -> b 1 n')
        
        b, m, n = input_ids.shape
        input_ids = rearrange(input_ids, 'b m n -> (b m) n') # group into a larger batch
        attention_mask = rearrange(attention_mask, 'b m n -> (b m) n')

        
        if self.model_parallel:
            if input_ids is not None:
                input_ids = input_ids.to(self.encoder.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.encoder.device)

        x = self.encoder(input_ids = input_ids, attention_mask = attention_mask, return_dict=True, **kwargs)
        x = x.last_hidden_state # (b m) n d
        x = rearrange(x, '(b m) n d -> b m n d', b = b)

        # preceiver forward
        assert x.ndim == 4

        num_ret_examples = x.shape[1]
        assert num_ret_examples == m
        
        # TODO: better implement model parallel
        if self.model_parallel:
            x = x + self.ret_rank_emb[:num_ret_examples].to(x.device)
        else:
            x = x + self.ret_rank_emb[:num_ret_examples]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            if self.model_parallel:
                attn = attn.to(x.device)
                ff = ff.to(x.device)
                latents = attn(x, latents.to(x.device)) + latents.to(x.device)
                latents = ff(latents) + latents
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        if self.model_parallel:
            self.norm = self.norm.to(latents.device)

        output = self.norm(latents)
        
        return output



if __name__ == "__main__":

    def set_up_device(gpu_index):
        # single gpu
        if torch.cuda.is_available():
            dev = f"cuda:{gpu_index}"
        else:
            dev = "cpu"
        return gpu_index, torch.device(dev)

    ## set up device
    gpu_index, device = set_up_device(7)
    
    encoder_model_name = "google/t5-base-lm-adapt"
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    input_texts = ['hello world', 'how are you', "I am fine thank you"]
    tokenized_inputs = tokenizer(
        input_texts,
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)

    # print(tokenized_inputs.input_ids.shape)

    model = PerceiverResamplerWithT5(
        encoder_path_or_name = encoder_model_name,
        dim = 768,
        depth = 2,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_ret_rank_embeds = 4,
        ff_mult = 4
    )
    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('trainable param:',name)

    ret_encoded = model(**tokenized_inputs)
    ret_encoded.to(device)

    xatten = GatedCrossAttentionBlock(
        dim = 768,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = False
    )
    xatten.to(device)

    h = torch.randn(3, 1024, 768) # b n d
    h = h.to(device)

    attn_output = xatten(h, ret_encoded)
    print('attn output shape:', attn_output.shape) # b n d


