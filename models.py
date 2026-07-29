import math
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from addict import Dict

try:
    import flash_attn
except ImportError:
    flash_attn = None

@torch.no_grad()
def _z_order_encode(coord, depth):
    x, y, z = coord[:, 0].long(), coord[:, 1].long(), coord[:, 2].long()
    code = torch.zeros_like(x)
    for i in range(depth):
        code |= ((x >> i) & 1) << (3 * i)
        code |= ((y >> i) & 1) << (3 * i + 1)
        code |= ((z >> i) & 1) << (3 * i + 2)
    return code

@torch.no_grad()
def _hilbert_encode(coord, depth):
    x = coord.long().clone()
    num_dims = 3
    m = 1 << (depth - 1)
    q = m
    while q > 1:
        p = q - 1
        mask = (x[:, 0] & q) != 0
        x[:, 0] = torch.where(mask, x[:, 0] ^ p, x[:, 0])
        for dim in range(1, num_dims):
            mask = (x[:, dim] & q) != 0
            x0 = x[:, 0].clone()
            xd = x[:, dim].clone()
            t = (x0 ^ xd) & p
            x[:, 0] = torch.where(mask, x0 ^ p, x0 ^ t)
            x[:, dim] = torch.where(mask, xd, xd ^ t)
        q >>= 1
    for dim in range(1, num_dims):
        x[:, dim] ^= x[:, dim - 1]
    t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    q = m
    while q > 1:
        t ^= ((x[:, num_dims - 1] & q) != 0).long() * (q - 1)
        q >>= 1
    x ^= t.unsqueeze(1)
    code = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for bit in reversed(range(depth)):
        for dim in range(num_dims):
            code = (code << 1) | ((x[:, dim] >> bit) & 1)
    return code

@torch.no_grad()
def encode(grid_coord, batch, depth, order="z"):
    coord = grid_coord.long().clone()
    if order == "z":
        code = _z_order_encode(coord, depth)
    elif order == "z-trans":
        code = _z_order_encode(coord[:, [1, 0, 2]], depth)
    elif order == "hilbert":
        code = _hilbert_encode(coord, depth)
    elif order == "hilbert-trans":
        code = _hilbert_encode(coord[:, [1, 0, 2]], depth)
    else:
        raise ValueError(f"Unknown order: {order}")
    return (batch.long() << (3 * depth)) | code

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))

@torch.inference_mode()
def offset2batch(offset):
    bc = offset2bincount(offset)
    return torch.arange(len(bc), device=offset.device, dtype=torch.long).repeat_interleave(bc)

@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

FIXED_SPARSE_SHAPE = [192, 192, 256]

class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc").int()
        if depth is None:
            depth = max(1, int(self.grid_coord.max()).bit_length())
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63 and depth <= 16
        code = torch.stack([encode(self.grid_coord, self.batch, depth, order=o) for o in order])
        order_idx = torch.argsort(code)
        inverse = torch.zeros_like(order_idx).scatter_(
            dim=1, index=order_idx,
            src=torch.arange(0, code.shape[1], device=order_idx.device).repeat(code.shape[0], 1))
        if shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code, order_idx, inverse = code[perm], order_idx[perm], inverse[perm]
        self["serialized_code"] = code
        self["serialized_order"] = order_idx
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc").int()
        sparse_shape = FIXED_SPARSE_SHAPE
        self["grid_coord"] = self["grid_coord"].clamp(min=0, max=min(sparse_shape) - 1)
        self["sparse_conv_feat"] = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape, batch_size=self.batch[-1].tolist() + 1)
        self["sparse_shape"] = sparse_shape

class PointModule(nn.Module):
    pass

class PointSequential(PointModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items(): self.add_module(key, module)
        else:
            for idx, module in enumerate(args): self.add_module(str(idx), module)
        for name, module in kwargs.items(): self.add_module(name, module)
    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)): raise IndexError(f"index {idx} out of range")
        if idx < 0: idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx): next(it)
        return next(it)
    def __len__(self): return len(self._modules)
    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules: raise KeyError("name exists")
        self.add_module(name, module)
    def forward(self, input):
        for k, module in self._modules.items():
            if isinstance(module, PointModule): input = module(input)
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else: input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0: input = input.replace_feature(module(input.features))
                else: input = module(input)
        return input

class GeometrySelectiveSSM(nn.Module):


    ATTR_DIM = 3
    EPS = 1e-6

    def __init__(
        self,
        channels,
        state_dim=16,
        expand=1,
        num_scan_orders=2,
        use_geometry_modulation=True,
        use_boundary_modulation=True,
        bidirectional=True,
        residual_gate_init=0.0,
    ):
        super().__init__()
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        if num_scan_orders < 1:
            raise ValueError("num_scan_orders must be at least 1")
        self.selective_scan_fn = selective_scan_fn
        self.channels = channels
        self.state_dim = state_dim
        self.num_scan_orders = int(num_scan_orders)
        self.inner_dim = channels * expand
        self.use_geometry_modulation = bool(use_geometry_modulation)
        self.use_boundary_modulation = bool(use_boundary_modulation)
        self.bidirectional = bool(bidirectional)

        self.in_proj = nn.Linear(channels, self.inner_dim * 2, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.inner_dim,
                self.inner_dim,
                4,
                padding=3,
                groups=self.inner_dim,
            )
            for _ in range(self.num_scan_orders)
        ])
        self.x_projs = nn.ModuleList([
            nn.Linear(
                self.inner_dim,
                state_dim * 2 + self.inner_dim,
                bias=False,
            )
            for _ in range(self.num_scan_orders)
        ])

        target_weight = torch.full(
            (self.inner_dim, self.inner_dim),
            1.0 / self.inner_dim,
        )
        target_weight.diagonal().fill_(1.0)
        raw_weight = torch.log(torch.expm1(target_weight))
        self.dt_weights_raw = nn.ParameterList([
            nn.Parameter(raw_weight.clone())
            for _ in range(self.num_scan_orders)
        ])
        self.dt_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.inner_dim))
            for _ in range(self.num_scan_orders)
        ])

        if self.use_geometry_modulation:
            self.delta_geo = nn.Sequential(
                nn.Linear(self.ATTR_DIM, self.inner_dim // 4),
                nn.GELU(),
                nn.Linear(self.inner_dim // 4, self.inner_dim),
            )
            nn.init.zeros_(self.delta_geo[-1].weight)
            nn.init.zeros_(self.delta_geo[-1].bias)
            gain_raw = math.log(math.expm1(0.01))
            self.coord_gain_raw = nn.Parameter(
                torch.full((self.inner_dim,), gain_raw)
            )
        else:
            self.delta_geo = None
            self.register_parameter("coord_gain_raw", None)

        if self.use_boundary_modulation:
            self.edge_predictor = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, 1),
            )
            gain_raw = math.log(math.expm1(0.01))
            alpha_raw = math.log(math.expm1(1.0))
            self.boundary_gain_raw = nn.Parameter(
                torch.full((self.inner_dim,), gain_raw)
            )
            self.edge_alpha = nn.Parameter(torch.tensor(alpha_raw))
        else:
            self.edge_predictor = None
            self.register_parameter("boundary_gain_raw", None)
            self.register_parameter("edge_alpha", None)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A.unsqueeze(0).expand(self.inner_dim, -1))
        )
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.scan_weights = nn.Parameter(
            torch.full((self.num_scan_orders,), 1.0 / self.num_scan_orders)
        )
        self.out_proj = nn.Linear(self.inner_dim, channels, bias=False)

        self.residual_gate = nn.Parameter(
            torch.tensor([float(residual_gate_init)], dtype=torch.float32)
        )
        self.norm = nn.LayerNorm(channels)

    @torch.no_grad()
    def _get_attribute_stats(self, coord, batch, normals=None, color_grad=None):
        del normals, color_grad
        stats = torch.zeros(
            coord.shape[0],
            self.ATTR_DIM,
            device=coord.device,
            dtype=coord.dtype,
        )
        for batch_id in range(int(batch.max().item()) + 1):
            mask = batch == batch_id
            block_coord = coord[mask]
            lower = block_coord.amin(dim=0)
            scale = (block_coord.amax(dim=0) - lower).clamp_min(self.EPS)
            stats[mask] = (block_coord - lower) / scale
        return stats

    @staticmethod
    def _coordinate_change(ordered_coord):
        change = ordered_coord.new_zeros((ordered_coord.shape[0], 1))
        if ordered_coord.shape[0] > 1:
            change[1:] = torch.linalg.vector_norm(
                ordered_coord[1:] - ordered_coord[:-1],
                dim=1,
                keepdim=True,
            ) / math.sqrt(3.0)
        return change.clamp_(0.0, 1.0)

    def _compute_delta(
        self,
        dt_raw,
        geo_base,
        coordinate_change,
        edge_score,
        scan_idx,
    ):
        conditioned = dt_raw
        if self.use_geometry_modulation:
            coord_gain = F.softplus(self.coord_gain_raw).to(dt_raw)
            geo_mod = (
                geo_base.to(dt_raw)
                + coordinate_change.to(dt_raw) * coord_gain.unsqueeze(0)
            )
            conditioned = conditioned + geo_mod
        if self.use_boundary_modulation:
            boundary_gain = F.softplus(self.boundary_gain_raw).to(dt_raw)
            alpha = F.softplus(self.edge_alpha).to(dt_raw)
            boundary_mod = edge_score.to(dt_raw) * boundary_gain.unsqueeze(0)
            conditioned = conditioned + alpha * boundary_mod

        weight = F.softplus(self.dt_weights_raw[scan_idx]).to(dt_raw)
        bias = self.dt_biases[scan_idx].to(dt_raw)
        delta = F.linear(conditioned, weight, bias)
        return F.softplus(delta).clamp_max_(2.0)

    def _scan_core(
        self,
        x,
        geo_base,
        ordered_coord,
        edge_score,
        scan_idx,
        reverse_flag,
    ):
        length = x.shape[2]
        if reverse_flag:
            x = x.flip(-1)
            geo_base = geo_base.flip(0)
            ordered_coord = ordered_coord.flip(0)
            edge_score = edge_score.flip(0)

        x_conv = F.silu(self.convs[scan_idx](x)[..., :length])
        projected = self.x_projs[scan_idx](
            x_conv.squeeze(0).transpose(0, 1)
        )
        B_param, C_param, dt_raw = torch.split(
            projected,
            (self.state_dim, self.state_dim, self.inner_dim),
            dim=1,
        )
        if self.use_geometry_modulation:
            coordinate_change = self._coordinate_change(ordered_coord)
        else:
            coordinate_change = ordered_coord.new_zeros((length, 1))
        delta = self._compute_delta(
            dt_raw,
            geo_base,
            coordinate_change,
            edge_score,
            scan_idx,
        )
        A = -torch.exp(self.A_log.float().clamp(min=-8, max=2))
        output = self.selective_scan_fn(
            x_conv.contiguous(),
            delta.transpose(0, 1).unsqueeze(0).contiguous(),
            A.contiguous(),
            B_param.transpose(0, 1).unsqueeze(0).unsqueeze(1).contiguous(),
            C_param.transpose(0, 1).unsqueeze(0).unsqueeze(1).contiguous(),
            self.D.float().contiguous(),
            z=None,
            delta_bias=None,
            delta_softplus=False,
        )
        return output.flip(-1) if reverse_flag else output

    def _scan_one_direction(
        self,
        x,
        geo_base,
        ordered_coord,
        edge_score,
        scan_idx,
        reverse=False,
    ):
        if self.training:
            return checkpoint(
                self._scan_core,
                x,
                geo_base,
                ordered_coord,
                edge_score,
                scan_idx,
                reverse,
                use_reentrant=False,
            )
        return self._scan_core(
            x,
            geo_base,
            ordered_coord,
            edge_score,
            scan_idx,
            reverse,
        )

    def forward(
        self,
        feat,
        coord,
        batch,
        serialized_order=None,
        normals=None,
        color_grad=None,
    ):
        normalized_coord = self._get_attribute_stats(
            coord,
            batch,
            normals,
            color_grad,
        )
        if self.use_geometry_modulation:
            geo_base = self.delta_geo(normalized_coord)
        else:
            geo_base = feat.new_zeros((feat.shape[0], self.inner_dim))

        if self.use_boundary_modulation:
            edge_logits = self.edge_predictor(feat).squeeze(-1)
            edge_score = torch.sigmoid(edge_logits).unsqueeze(-1)
        else:
            edge_logits = None
            edge_score = feat.new_zeros((feat.shape[0], 1))

        x, z = self.in_proj(feat).chunk(2, dim=-1)
        z = F.silu(z)
        output = torch.zeros_like(x)
        available_orders = (
            serialized_order.shape[0] if serialized_order is not None else 1
        )
        num_scans = min(self.num_scan_orders, available_orders)
        scan_weight = F.softmax(self.scan_weights[:num_scans], dim=0)

        for batch_id in range(int(batch.max().item()) + 1):
            mask = batch == batch_id
            global_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
            count = global_ids.numel()
            if count < 2:
                output[global_ids] = x[global_ids]
                continue

            batch_x = x[global_ids]
            batch_geo = geo_base[global_ids]
            batch_coord = normalized_coord[global_ids]
            batch_edge = edge_score[global_ids]
            batch_output = torch.zeros_like(batch_x)

            global_to_local = torch.full(
                (feat.shape[0],),
                -1,
                dtype=torch.long,
                device=feat.device,
            )
            global_to_local[global_ids] = torch.arange(
                count,
                device=feat.device,
            )

            for scan_idx in range(num_scans):
                if serialized_order is None:
                    local_order = torch.arange(count, device=feat.device)
                else:
                    global_order = serialized_order[scan_idx]
                    global_order = global_order[batch[global_order] == batch_id]
                    local_order = global_to_local[global_order]

                ordered_x = batch_x[local_order]
                ordered_geo = batch_geo[local_order]
                ordered_coord = batch_coord[local_order]
                ordered_edge = batch_edge[local_order]
                scan_input = ordered_x.transpose(0, 1).unsqueeze(0).contiguous()

                forward_output = self._scan_one_direction(
                    scan_input,
                    ordered_geo,
                    ordered_coord,
                    ordered_edge,
                    scan_idx,
                    reverse=False,
                )
                if self.bidirectional:
                    backward_output = self._scan_one_direction(
                        scan_input,
                        ordered_geo,
                        ordered_coord,
                        ordered_edge,
                        scan_idx,
                        reverse=True,
                    )
                    scan_output = (forward_output + backward_output) * 0.5
                else:
                    scan_output = forward_output
                scan_output = scan_output.squeeze(0).transpose(0, 1)

                inverse_order = torch.empty_like(local_order)
                inverse_order[local_order] = torch.arange(
                    count,
                    device=feat.device,
                )
                batch_output.add_(
                    scan_weight[scan_idx] * scan_output[inverse_order]
                )
            output[global_ids] = batch_output

        output = self.out_proj(output * z)
        output = feat + torch.sigmoid(self.residual_gate) * self.norm(output)
        return output, edge_logits


class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size; self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1/3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)
    def forward(self, coord):
        idx = (coord.clamp(-self.pos_bnd, self.pos_bnd) + self.pos_bnd
               + torch.arange(3, device=coord.device) * self.rpe_num)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        return out.view(idx.shape + (-1,)).sum(3).permute(0, 3, 1, 2)

class SerializedAttention(PointModule):
    def __init__(self, channels, num_heads, patch_size, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, order_index=0, enable_rpe=False,
                 enable_flash=True, upcast_attention=True, upcast_softmax=True):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels; self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention; self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe; self.enable_flash = enable_flash
        if enable_flash:
            assert not enable_rpe and not upcast_attention and not upcast_softmax
            assert flash_attn is not None
            self.patch_size = patch_size; self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size; self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels); self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pk, uk, ck = "pad", "unpad", "cu_seqlens_key"
        if pk not in point.keys() or uk not in point.keys() or ck not in point.keys():
            offset = point.offset; bc = offset2bincount(offset)
            bcp = (torch.div(bc + self.patch_size - 1, self.patch_size,
                             rounding_mode="trunc") * self.patch_size)
            mask = bc > self.patch_size; bcp = ~mask * bc + mask * bcp
            _o = nn.functional.pad(offset, (1, 0))
            _op = nn.functional.pad(torch.cumsum(bcp, dim=0), (1, 0))
            pad = torch.arange(_op[-1], device=offset.device)
            unpad = torch.arange(_o[-1], device=offset.device)
            cu = []
            for i in range(len(offset)):
                unpad[_o[i]:_o[i+1]] += _op[i] - _o[i]
                if bc[i] != bcp[i]:
                    r = bc[i] % self.patch_size
                    pad[_op[i+1]-self.patch_size+r:_op[i+1]] = \
                        pad[_op[i+1]-2*self.patch_size+r:_op[i+1]-self.patch_size]
                pad[_op[i]:_op[i+1]] -= _op[i] - _o[i]
                cu.append(torch.arange(_op[i], _op[i+1], step=self.patch_size,
                                       dtype=torch.int32, device=offset.device))
            point[pk] = pad; point[uk] = unpad
            point[ck] = nn.functional.pad(torch.concat(cu), (0, 1), value=_op[-1])
        return point[pk], point[uk], point[ck]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        H, K, C = self.num_heads, self.patch_size, self.channels
        pad, unpad, cu = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        if not self.enable_flash:
            q, k, v = qkv.reshape(-1, K, 3, H, C//H).permute(2,0,3,1,4).unbind(0)
            if self.upcast_attention: q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                key = f"rel_pos_{self.order_index}"
                if key not in point.keys():
                    gc = point.grid_coord[order].reshape(-1, K, 3)
                    point[key] = gc.unsqueeze(2) - gc.unsqueeze(1)
                attn = attn + self.rpe(point[key])
            if self.upcast_softmax: attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C//H), cu, max_seqlen=K,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale).reshape(-1, C).to(qkv.dtype)
        point.feat = self.proj_drop(self.proj(feat[inverse]))
        return point

class MLP(nn.Module):
    def __init__(self, inc, hc=None, outc=None, act=nn.GELU, drop=0.0):
        super().__init__()
        outc = outc or inc; hc = hc or inc
        self.fc1 = nn.Linear(inc, hc); self.act = act()
        self.fc2 = nn.Linear(hc, outc); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Block(PointModule):
    def __init__(self, channels, num_heads, patch_size=48, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 pre_norm=True, order_index=0, cpe_indice_key=None,
                 enable_rpe=False, enable_flash=True,
                 upcast_attention=True, upcast_softmax=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True,
                              indice_key=cpe_indice_key),
            nn.Linear(channels, channels), norm_layer(channels))
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels, num_heads, patch_size, qkv_bias, qk_scale, attn_drop,
            proj_drop, order_index, enable_rpe, enable_flash,
            upcast_attention, upcast_softmax)
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(channels, int(channels * mlp_ratio), channels, act_layer, proj_drop))
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0 else nn.Identity())

    def forward(self, point: Point):
        sc = point.feat; point = self.cpe(point); point.feat = sc + point.feat
        sc = point.feat
        if self.pre_norm: point = self.norm1(point)
        point = self.drop_path(self.attn(point)); point.feat = sc + point.feat
        if not self.pre_norm: point = self.norm1(point)
        sc = point.feat
        if self.pre_norm: point = self.norm2(point)
        point = self.drop_path(self.mlp(point)); point.feat = sc + point.feat
        if not self.pre_norm: point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class GSSMBlock(PointModule):
    def __init__(self, channels, num_heads, patch_size=48, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 pre_norm=True, order_index=0, cpe_indice_key=None,
                 enable_rpe=False, enable_flash=True,
                 upcast_attention=True, upcast_softmax=True,
                 ssm_state_dim=16, ssm_expand=1, num_scan_orders=2,
                 use_geometry_modulation=True,
                 use_boundary_modulation=True, bidirectional=True,
                 residual_gate_init=0.0):
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True,
                              indice_key=cpe_indice_key),
            nn.Linear(channels, channels), norm_layer(channels))
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels, num_heads, patch_size, qkv_bias, qk_scale, attn_drop,
            proj_drop, order_index, enable_rpe, enable_flash,
            upcast_attention, upcast_softmax)
        self.gssm = GeometrySelectiveSSM(
            channels,
            state_dim=ssm_state_dim,
            expand=ssm_expand,
            num_scan_orders=num_scan_orders,
            use_geometry_modulation=use_geometry_modulation,
            use_boundary_modulation=use_boundary_modulation,
            bidirectional=bidirectional,
            residual_gate_init=residual_gate_init,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(channels, int(channels * mlp_ratio), channels, act_layer, proj_drop))
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0 else nn.Identity())

    def forward(self, point: Point):
        sc = point.feat; point = self.cpe(point); point.feat = sc + point.feat
        sc = point.feat
        if self.pre_norm: point = self.norm1(point)
        point = self.drop_path(self.attn(point)); point.feat = sc + point.feat
        if not self.pre_norm: point = self.norm1(point)
        ser_order = point.serialized_order if "serialized_order" in point.keys() else None
        normals = point.normals if "normals" in point.keys() else None
        color_grad = point.color_grad if "color_grad" in point.keys() else None
        point.feat, gssm_edge = self.gssm(
            point.feat, point.coord, point.batch, ser_order,
            normals=normals, color_grad=color_grad
        )
        if gssm_edge is not None:
            edge_list = point.get("gssm_edge_logits", [])
            edge_list.append(gssm_edge)
            point["gssm_edge_logits"] = edge_list
            if "boundary" in point.keys():
                boundary_list = point.get("gssm_boundary_targets", [])
                boundary_list.append(point.boundary)
                point["gssm_boundary_targets"] = boundary_list
        sc = point.feat
        if self.pre_norm: point = self.norm2(point)
        point = self.drop_path(self.mlp(point)); point.feat = sc + point.feat
        if not self.pre_norm: point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(self, inc, outc, stride=2, norm_layer=None, act_layer=None,
                 reduce="max", shuffle_orders=True, traceable=True):
        super().__init__()
        self.stride = stride; self.reduce = reduce
        self.shuffle_orders = shuffle_orders; self.traceable = traceable
        self.proj = nn.Linear(inc, outc)
        self.norm = PointSequential(norm_layer(outc)) if norm_layer else None
        self.act = PointSequential(act_layer()) if act_layer else None

    def forward(self, point: Point):
        pd = (math.ceil(self.stride)-1).bit_length()
        if pd > point.serialized_depth: pd = 0
        code = point.serialized_code >> pd * 3
        _, cluster, counts = torch.unique(code[0], sorted=True,
                                           return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        hi = indices[idx_ptr[:-1]]
        code = code[:, hi]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code, order, inverse = code[perm], order[perm], inverse[perm]
        pd_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[hi] >> pd,
            serialized_code=code, serialized_order=order, serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pd, batch=point.batch[hi])
        if "normals" in point.keys():
            pd_dict["normals"] = torch_scatter.segment_csr(
                point.normals[indices], idx_ptr, reduce="mean")
        if "color_grad" in point.keys():
            pd_dict["color_grad"] = torch_scatter.segment_csr(
                point.color_grad[indices], idx_ptr, reduce="mean")
        if "boundary" in point.keys():
            pd_dict["boundary"] = torch_scatter.segment_csr(
                point.boundary[indices], idx_ptr, reduce="max")
        if "condition" in point.keys(): pd_dict["condition"] = point.condition
        if "context" in point.keys(): pd_dict["context"] = point.context
        if self.traceable:
            pd_dict["pooling_inverse"] = cluster; pd_dict["pooling_parent"] = point
        point = Point(pd_dict)
        if self.norm: point = self.norm(point)
        if self.act: point = self.act(point)
        point.sparsify()
        return point

class StandardUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer:
            self.proj.add(norm_layer(out_channels)); self.proj_skip.add(norm_layer(out_channels))
        if act_layer:
            self.proj.add(act_layer()); self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        parent = point.pop("pooling_parent"); inverse = point.pop("pooling_inverse")
        point = self.proj(point); parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        if self.traceable: parent["unpooling_parent"] = point
        return parent

class Embedding(PointModule):
    def __init__(self, inc, ec, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(inc, ec, kernel_size=5, padding=1,
                                    bias=False, indice_key="stem"))
        if norm_layer: self.stem.add(norm_layer(ec), name="norm")
        if act_layer: self.stem.add(act_layer(), name="act")
    def forward(self, point: Point):
        return self.stem(point)

class PointTransformerV3(PointModule):
    def __init__(self, in_channels=10,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 stride=(2, 2, 2),
                 enc_depths=(2, 2, 6, 2), enc_channels=(48, 96, 192, 384),
                 enc_num_head=(3, 6, 12, 24), enc_patch_size=(1024,)*4,
                 dec_depths=(2, 2, 2), dec_channels=(48, 96, 192),
                 dec_num_head=(3, 6, 12), dec_patch_size=(1024,)*3,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
                 pre_norm=True, shuffle_orders=True,
                 enable_rpe=False, enable_flash=True,
                 upcast_attention=False, upcast_softmax=False,
                 cls_mode=False, use_gssm=True, gssm_stages=(3,),
                 use_geometry_modulation=True,
                 use_boundary_modulation=True, num_scan_orders=2,
                 bidirectional=True, residual_gate_init=0.0):
        super().__init__()
        ns = len(enc_depths); self.num_stages = ns
        self.spatial_orders = [order] if isinstance(order, str) else list(order)
        self.num_orders = len(self.spatial_orders)
        if num_scan_orders > self.num_orders:
            raise ValueError(
                f"num_scan_orders={num_scan_orders} exceeds available orders={self.num_orders}"
            )
        self.cls_mode = cls_mode; self.shuffle_orders = shuffle_orders
        self.gssm_stages = set(gssm_stages) if use_gssm else set()
        invalid_stages = [s for s in self.gssm_stages if s < 0 or s >= ns]
        if invalid_stages:
            raise ValueError(f"Invalid GSSM stages: {invalid_stages}")
        bn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln = nn.LayerNorm; act = nn.GELU
        self.embedding = Embedding(in_channels, enc_channels[0], bn, act)

        edp = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(ns):
            edp_ = edp[sum(enc_depths[:s]):sum(enc_depths[:s+1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(
                    enc_channels[s-1], enc_channels[s], stride[s-1], bn, act,
                    shuffle_orders=self.shuffle_orders), name="down")
            BlockClass = GSSMBlock if s in self.gssm_stages else Block
            for i in range(enc_depths[s]):
                block_args = dict(
                    channels=enc_channels[s], num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=proj_drop,
                    drop_path=edp_[i], norm_layer=ln, act_layer=act,
                    pre_norm=pre_norm, order_index=i % self.num_orders,
                    cpe_indice_key=f"stage{s}", enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
                if BlockClass is GSSMBlock:
                    block_args.update(
                        num_scan_orders=num_scan_orders,
                        use_geometry_modulation=use_geometry_modulation,
                        use_boundary_modulation=use_boundary_modulation,
                        bidirectional=bidirectional,
                        residual_gate_init=residual_gate_init,
                    )
                enc.add(BlockClass(**block_args), name=f"block{i}")
            if len(enc): self.enc.add(enc, name=f"enc{s}")

        if not cls_mode:
            ddp = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dc = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(ns - 1)):
                ddp_ = ddp[sum(dec_depths[:s]):sum(dec_depths[:s+1])]; ddp_.reverse()
                dec = PointSequential()
                dec.add(StandardUnpooling(dc[s+1], enc_channels[s], dc[s], bn, act),
                        name="up")
                for i in range(dec_depths[s]):
                    dec.add(Block(dc[s], dec_num_head[s], dec_patch_size[s],
                        mlp_ratio, qkv_bias, qk_scale, attn_drop, proj_drop,
                        ddp_[i], ln, act, pre_norm,
                        i % self.num_orders,
                        f"stage{s}", enable_rpe, enable_flash,
                        upcast_attention, upcast_softmax), name=f"block{i}")
                self.dec.add(dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.spatial_orders, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        edge_logits = point.get("gssm_edge_logits", [])
        boundary_targets = point.get("gssm_boundary_targets", [])
        if not self.cls_mode:
            multi_scale_feats = []
            multi_scale_inverses = []
            current = point
            dec_modules = list(self.dec._modules.values())
            for dec_stage in dec_modules:
                for name, module in dec_stage._modules.items():
                    if isinstance(module, StandardUnpooling):
                        inv = current.pooling_inverse.clone()
                        current = module(current)
                        multi_scale_inverses.append(inv)
                    else:
                        current = module(current)
                multi_scale_feats.append(current.feat.clone())
            point = current
            point["multi_scale_feats"] = multi_scale_feats
            point["multi_scale_inverses"] = multi_scale_inverses
        point["gssm_edge_logits"] = edge_logits
        point["gssm_boundary_targets"] = boundary_targets
        return point


class ClassAwareMultiScalePrediction(nn.Module):
    def __init__(self, dec_channels, num_classes, fusion_mode="class_aware"):
        super().__init__()
        if fusion_mode not in {"shared", "class_aware"}:
            raise ValueError(f"Unsupported CAMP fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.scale_heads = nn.ModuleList()
        for ch in reversed(dec_channels):
            self.scale_heads.append(nn.Sequential(
                nn.Linear(ch, ch // 2), nn.GELU(),
                nn.Linear(ch // 2, num_classes)))
        self.num_scales = len(dec_channels)
        if fusion_mode == "class_aware":
            weight_shape = (self.num_scales, num_classes)
        else:
            weight_shape = (self.num_scales, 1)
        self.scale_weight = nn.Parameter(torch.zeros(weight_shape))

    def forward(self, multi_scale_feats, multi_scale_inverses):
        if len(multi_scale_feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} decoder scales, got {len(multi_scale_feats)}"
            )
        N_fine = multi_scale_feats[-1].shape[0]
        device = multi_scale_feats[-1].device
        w = F.softmax(self.scale_weight, dim=0)
        num_classes = self.scale_heads[-1][-1].out_features
        if self.fusion_mode == "shared":
            w = w.expand(-1, num_classes)
        combined = torch.zeros(N_fine, num_classes, device=device)
        for s, (feat, head) in enumerate(zip(multi_scale_feats, self.scale_heads)):
            logits_up = head(feat)
            for inv in multi_scale_inverses[s+1:]:
                logits_up = logits_up[inv]
            combined = combined + w[s].unsqueeze(0) * logits_up
        return combined


class PTv3SegWrapper(nn.Module):
    def __init__(self, in_channels=10, num_classes=4, grid_size=0.01,
                 enable_flash=True, fusion_mode="class_aware", **ptv3_kwargs):
        super().__init__()
        if fusion_mode not in {"none", "shared", "class_aware"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.grid_size = float(grid_size)
        self.in_channels = in_channels
        self.fusion_mode = fusion_mode
        self.use_boundary_modulation = bool(
            ptv3_kwargs.get("use_boundary_modulation", True)
        )
        if enable_flash and flash_attn is None:
            print("[PTv3] flash_attn not found, using non-flash"); enable_flash = False
        self.backbone = PointTransformerV3(
            in_channels=in_channels, enable_flash=enable_flash,
            upcast_attention=not enable_flash, upcast_softmax=not enable_flash,
            **ptv3_kwargs)
        dec_ch = ptv3_kwargs.get("dec_channels", (48, 96, 192))
        self.seg_head = nn.Sequential(
            nn.Linear(dec_ch[0], dec_ch[0]), nn.BatchNorm1d(dec_ch[0]),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(dec_ch[0], num_classes))
        self.camp = None
        if fusion_mode != "none":
            self.camp = ClassAwareMultiScalePrediction(
                dec_channels=list(dec_ch),
                num_classes=num_classes,
                fusion_mode=fusion_mode,
            )

    def forward(self, features, xyz, labels=None, boundary=None):
        del labels
        B, N, C = features.shape; device = features.device
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")
        feat_flat = features.reshape(B * N, C).float()
        coord_flat = xyz.reshape(B * N, 3).float()
        batch_idx = torch.arange(B, device=device, dtype=torch.long
            ).unsqueeze(1).expand(B, N).reshape(B * N)
        offset = torch.arange(1, B + 1, device=device, dtype=torch.long) * N

        normals_flat = None
        color_grad_flat = None
        if C >= 10:
            normals_flat = feat_flat[:, 6:9].contiguous()
            color_grad_flat = feat_flat[:, 9].contiguous()

        gc_list = []
        for b in range(B):
            s, e = b * N, (b + 1) * N
            bc = coord_flat[s:e]
            gc = torch.div(
                bc - bc.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
            gc = gc.clamp(0, FIXED_SPARSE_SHAPE[0] - 1)
            gc_list.append(gc)
        grid_coord = torch.cat(gc_list, dim=0)
        data_dict = dict(feat=feat_flat, coord=coord_flat,
                         grid_coord=grid_coord, batch=batch_idx, offset=offset)
        if normals_flat is not None:
            data_dict["normals"] = normals_flat
            data_dict["color_grad"] = color_grad_flat
        if boundary is not None and self.use_boundary_modulation:
            data_dict["boundary"] = boundary.reshape(B * N).float()

        point = self.backbone(data_dict)
        fine_logits = self.seg_head(point.feat)
        if self.camp is not None:
            if "multi_scale_feats" not in point.keys():
                raise RuntimeError("CAMP requires decoder multi-scale features")
            camp_logits = self.camp(
                point.multi_scale_feats,
                point.multi_scale_inverses)
            logits = fine_logits + camp_logits
        else:
            logits = fine_logits
        logits = logits.reshape(B, N, -1)
        edge_logits = point.get("gssm_edge_logits", [])
        boundary_targets = point.get("gssm_boundary_targets", [])
        return logits, edge_logits, boundary_targets, None, None


def create_model(in_channels=10, num_classes=4, model_size="base", **kwargs):

    kwargs.pop("num_anchors", None)
    kwargs.pop("sub_proto_counts", None)
    architecture = kwargs.pop("architecture", "gssformer")
    if architecture not in {"gssformer", "ptv3_c"}:
        raise ValueError(f"Unsupported architecture: {architecture}")

    configs = {
        "small": dict(enc_depths=(2,2,4,2), enc_channels=(32,64,128,256),
            enc_num_head=(2,4,8,16), enc_patch_size=(512,)*4,
            dec_depths=(2,2,2), dec_channels=(32,64,128),
            dec_num_head=(2,4,8), dec_patch_size=(512,)*3, drop_path=0.2),
        "base": dict(enc_depths=(2,2,6,2), enc_channels=(48,96,192,384),
            enc_num_head=(3,6,12,24), enc_patch_size=(1024,)*4,
            dec_depths=(2,2,2), dec_channels=(48,96,192),
            dec_num_head=(3,6,12), dec_patch_size=(1024,)*3, drop_path=0.3),
        "large": dict(enc_depths=(2,2,6,2), enc_channels=(64,128,256,512),
            enc_num_head=(4,8,16,32), enc_patch_size=(1024,)*4,
            dec_depths=(2,2,2), dec_channels=(64,128,256),
            dec_num_head=(4,8,16), dec_patch_size=(1024,)*3, drop_path=0.4),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}")

    allowed = {
        "enable_flash", "grid_size", "use_gssm",
        "use_geometry_modulation", "use_boundary_modulation",
        "fusion_mode", "num_scan_orders", "bidirectional",
        "random_scan_orders", "residual_gate_init", "gssm_stages",
    }
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"Unsupported model options: {', '.join(unknown)}")

    enable_flash = bool(kwargs.pop("enable_flash", True))
    grid_size = float(kwargs.pop("grid_size", 0.01))
    use_gssm = bool(kwargs.pop("use_gssm", True))
    use_geometry = bool(kwargs.pop("use_geometry_modulation", True))
    use_boundary = bool(kwargs.pop("use_boundary_modulation", True))
    fusion_mode = kwargs.pop("fusion_mode", "class_aware")
    num_scan_orders = int(kwargs.pop("num_scan_orders", 2))
    bidirectional = bool(kwargs.pop("bidirectional", True))
    random_scan_orders = bool(kwargs.pop("random_scan_orders", True))
    residual_gate_init = float(kwargs.pop("residual_gate_init", 0.0))
    gssm_stages = tuple(kwargs.pop("gssm_stages", (3,)))

    if not use_gssm:
        use_geometry = False
        use_boundary = False
    if not kwargs == {}:
        raise AssertionError(f"Unconsumed model options: {kwargs}")

    backbone_config = dict(configs[model_size])
    backbone_config.update(
        use_gssm=use_gssm,
        gssm_stages=gssm_stages,
        use_geometry_modulation=use_geometry,
        use_boundary_modulation=use_boundary,
        num_scan_orders=num_scan_orders,
        bidirectional=bidirectional,
        shuffle_orders=random_scan_orders,
        residual_gate_init=residual_gate_init,
    )
    return PTv3SegWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        grid_size=grid_size,
        enable_flash=enable_flash,
        fusion_mode=fusion_mode,
        **backbone_config,
    )
