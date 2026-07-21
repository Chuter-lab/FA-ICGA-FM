"""Model definitions for FA-ICGA-FM.

Implements:
  A1  - MAE (Masked Autoencoder) pretraining on FA/ICGA
  A3  - Multi-task supervised pretraining
  B1  - ViT-B/16 backbone
  B2  - Swin-B backbone
  B3  - ConvNeXt-B backbone
  B4  - RETFound fine-tuning
  B5  - BiomedCLIP fine-tuning
  B6  - DINOv2 linear probe + fine-tune
  C1  - FA+ICGA cross-modal contrastive
  D1  - Phase-conditioned single-frame embedding
  E3  - CORAL ordinal regression head
  F2  - Two-stage hierarchical classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


# ─── MAE (item A1) ────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        self.mask_ratio = mask_ratio
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        pos = torch.zeros(self.pos_embed.shape)
        d = self.pos_embed.shape[-1]
        num_p = self.pos_embed.shape[1] - 1
        for p in range(num_p):
            for i in range(0, d, 2):
                pos[0, p + 1, i]     = math.sin(p / 10000 ** (i / d))
                pos[0, p + 1, i + 1] = math.cos(p / 10000 ** (i / d))
        self.pos_embed.data.copy_(pos)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def encode_full(self, x):
        """No masking — for linear probe / downstream use."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, encoder_dim=768, decoder_dim=512,
                 depth=8, num_heads=16, patch_size=16):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size  = patch_size
        self.embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim),
                                      requires_grad=False)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=decoder_dim * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * 3, bias=True)

    def forward(self, x, ids_restore):
        x = self.embed(x)
        B, N_vis, D = x[:, 1:, :].shape
        N = self.num_patches
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.pos_embed
        x = self.decoder(x)
        x = self.norm(x)
        x = self.pred(x[:, 1:, :])
        return x


class MAEModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mask_ratio=0.75):
        super().__init__()
        self.encoder = MAEEncoder(img_size, patch_size, mask_ratio=mask_ratio)
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder = MAEDecoder(num_patches, patch_size=patch_size)

    def patchify(self, imgs):
        p = self.encoder.patch_embed.proj.kernel_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(imgs.shape[0], h * w, p * p * 3)

    def forward(self, x):
        latent, mask, ids_restore = self.encoder(x)
        pred = self.decoder(latent, ids_restore)
        target = self.patchify(x)
        mean = target.mean(dim=-1, keepdim=True)
        var  = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss, pred, mask


# ─── External backbones (B4=RETFound, B5=BiomedCLIP, B6=DINOv2) ──────────────

class RETFoundWrapper(nn.Module):
    """RETFound ViT-L fine-tuning wrapper (B4).

    Loads from torch.hub if weights available, else falls back to timm ViT-L.
    """

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        try:
            import huggingface_hub
            self.backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=False, num_classes=0
            )
            weights_path = _try_load_retfound()
            if weights_path and pretrained:
                state = torch.load(weights_path, map_location="cpu")
                sd = state.get("model", state)
                self.backbone.load_state_dict(sd, strict=False)
        except Exception:
            self.backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=pretrained, num_classes=0
            )
        self.head = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def features(self, x):
        return self.backbone(x)


def _try_load_retfound():
    from pathlib import Path
    CACHE = Path("/dartfs-hpc/scratch/f008pp2/.cache/retfound")
    candidates = [
        CACHE / "RETFound_cfp_weights.pth",
        CACHE / "RETFound_mae_weights.pth",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    try:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(
            repo_id="rmaphoh/RETFound-MAE", filename="RETFound_cfp_weights.pth",
            cache_dir=str(CACHE)
        )
        return ckpt
    except Exception:
        return None


class BiomedCLIPWrapper(nn.Module):
    """BiomedCLIP vision encoder fine-tuning wrapper (B5)."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = None
        self.feat_dim = 512
        if pretrained:
            try:
                import open_clip
                model, _, _ = open_clip.create_model_and_transforms(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                self.backbone = model.visual
                self.feat_dim = 512
            except Exception:
                pass
        if self.backbone is None:
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=0
            )
            self.feat_dim = self.backbone.num_features
        self.head = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x):
        if hasattr(self.backbone, "encode_image"):
            feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat[:, 0] if feat.shape[1] > 1 else feat.squeeze(1)
        return self.head(feat)

    def features(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat[:, 0] if feat.shape[1] > 1 else feat.squeeze(1)
        return feat


class DINOv2Wrapper(nn.Module):
    """DINOv2 ViT-B/14 linear probe + fine-tune wrapper (B6)."""

    def __init__(self, n_classes, frozen=True):
        super().__init__()
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                source="github"
            )
        except Exception:
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
        self.feat_dim = getattr(self.backbone, "embed_dim",
                                getattr(self.backbone, "num_features", 768))
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x):
        with torch.set_grad_enabled(not all(not p.requires_grad
                                             for p in self.backbone.parameters())):
            feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat[:, 0]
        return self.head(feat)

    def features(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat[:, 0]
        return feat.detach()

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


class SwinBWrapper(nn.Module):
    """Swin-B backbone for FA/ICGA (B2)."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        return self.backbone(x)


class ConvNeXtBWrapper(nn.Module):
    """ConvNeXt-B backbone (B3) — fast, GradCAM-friendly."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base", pretrained=pretrained, num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        return self.backbone(x)


class ViTBWrapper(nn.Module):
    """ViT-B/16 primary backbone (B1)."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        return self.backbone(x)


# ─── Cross-modal contrastive (C1) ─────────────────────────────────────────────

class CrossModalContrastive(nn.Module):
    """FA+ICGA cross-modal contrastive alignment (C1).

    FA and ICGA images from the same patient form positive pairs.
    Uses a shared ViT-B/16 backbone + modality-specific projection heads.
    """

    def __init__(self, pretrained=True, proj_dim=256, temperature=0.07):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.fa_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        self.icga_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def encode_fa(self, x):
        return F.normalize(self.fa_proj(self.backbone(x)), dim=-1)

    def encode_icga(self, x):
        return F.normalize(self.icga_proj(self.backbone(x)), dim=-1)

    def forward(self, fa, icga):
        fa_z  = self.encode_fa(fa)
        icg_z = self.encode_icga(icga)
        logits = fa_z @ icg_z.T / self.temperature.abs()
        labels = torch.arange(fa_z.shape[0], device=fa_z.device)
        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean()
        return loss, acc


# ─── Phase-conditioned embedding (D1) ─────────────────────────────────────────

PHASE_NAMES = ["early", "arterial", "venous", "late"]

class PhaseConditionedViT(nn.Module):
    """ViT-B with learned phase-conditioning token (D1).

    Injects FA phase label (0-3) as an additive embedding to CLS token.
    """

    def __init__(self, n_classes, n_phases=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.phase_embed = nn.Embedding(n_phases, feat_dim)
        nn.init.normal_(self.phase_embed.weight, std=0.02)
        self.head = nn.Linear(feat_dim, n_classes)
        self.phase_head = nn.Linear(feat_dim, n_phases)

    def forward(self, x, phase=None):
        feat = self.backbone(x)
        if phase is not None:
            feat = feat + self.phase_embed(phase)
        cls_logits   = self.head(feat)
        phase_logits = self.phase_head(feat)
        return cls_logits, phase_logits

    def features(self, x):
        return self.backbone(x)


# ─── CORAL ordinal head (E3) ──────────────────────────────────────────────────

class CORALHead(nn.Module):
    """CORAL ordinal regression head for disease severity (E3).

    Compatible with any backbone returning a feature vector.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, x):
        logits = self.fc(x) + self.bias
        return logits

    @staticmethod
    def coral_loss(logits, y):
        sets = [torch.clamp(y - i, 0, 1) for i in range(logits.shape[1])]
        labels = torch.stack(sets, dim=1).float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    @staticmethod
    def predict(logits):
        return (logits.sigmoid() > 0.5).sum(dim=1)


class BackboneWithCORAL(nn.Module):
    def __init__(self, backbone_name, n_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.coral = CORALHead(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.coral(self.backbone(x))


# ─── Two-stage hierarchical classifier (F2) ───────────────────────────────────

class HierarchicalClassifier(nn.Module):
    """Two-stage coarse-to-fine classifier (F2).

    Stage 1: disease category (e.g., normal / DR / AMD / other)
    Stage 2: subcategory within each coarse group
    """

    def __init__(self, in_features, coarse_classes, fine_classes_per_coarse):
        super().__init__()
        self.coarse_head = nn.Linear(in_features, coarse_classes)
        self.fine_heads = nn.ModuleList([
            nn.Linear(in_features, n) for n in fine_classes_per_coarse
        ])

    def forward(self, feat):
        coarse_logits = self.coarse_head(feat)
        fine_logits = [h(feat) for h in self.fine_heads]
        return coarse_logits, fine_logits


# ─── Segmentation heads (E2, E8) ──────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """Lightweight UNet-style decoder attached to ViT backbone features (E2, E8)."""

    def __init__(self, in_channels=768, out_channels=1, img_size=224, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.img_size   = img_size
        self.h = self.w = img_size // patch_size
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        x = patch_tokens.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)
        return self.proj(x)


class ViTSegmentation(nn.Module):
    """ViT-B + UNet decoder for vessel/lesion segmentation (E2, E8)."""

    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained,
            num_classes=0, global_pool=""
        )
        self.decoder = UNetDecoder(in_channels=768, out_channels=out_channels)

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        patch_tokens = tokens[:, 1:]
        return self.decoder(patch_tokens)


# ─── MTL pretraining backbone (A3) ────────────────────────────────────────────

class MTLModel(nn.Module):
    """Multi-task learning pretraining on APTOS 24 conditions + auxiliary tasks (A3)."""

    def __init__(self, n_conditions=24, n_phases=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        feat = self.backbone.num_features
        self.condition_head = nn.Linear(feat, n_conditions)
        self.phase_head     = nn.Linear(feat, n_phases)
        self.quality_head   = nn.Linear(feat, 2)  # E6: IQA binary
        self.severity_head  = nn.Linear(feat, 3)  # DR severity Normal/NPDR/PDR

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "condition": self.condition_head(feat),
            "phase":     self.phase_head(feat),
            "quality":   self.quality_head(feat),
            "severity":  self.severity_head(feat),
        }
