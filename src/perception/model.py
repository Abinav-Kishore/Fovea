"""SqueezeSeg model: architecture + pretrained-weight loading.

The architecture below is a line-for-line reimplementation of
PRBonn/lidar-bonnetal's `train/backbones/squeezeseg.py` and
`train/tasks/semantic/decoders/squeezeseg.py` (MIT license) -- fetched and
read directly from the source at integration time, not reconstructed from
guesswork. Every parameter name below is chosen to match the pretrained
checkpoint's state_dict keys exactly, and weights are loaded with
`strict=True`: if the architecture were wrong, loading would raise
immediately rather than silently producing garbage predictions. This
repo verified that load succeeds (see tests/test_perception.py).

Architecture: SqueezeNet-style encoder (Fire modules) -> symmetric decoder
(FireUp modules with additive skip connections) -> a small conv head ->
per-pixel class logits over a spherical range-image projection of the
point cloud. Original design: Bichen Wu et al., "SqueezeSeg" (2017);
this specific checkpoint is the Bonnetal/RangeNet++ project's SemanticKITTI-
trained version (see models/squeezeseg/NOTICE.md for full provenance).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.perception.base import RawPrediction
from src.perception.preprocessing import RangeImageProjection, RangeImageProjector

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "squeezeseg"


# ---------------------------------------------------------------------------
# Architecture (matches the pretrained checkpoint's state_dict exactly)
# ---------------------------------------------------------------------------

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        return torch.cat([self.activation(self.expand1x1(x)), self.activation(self.expand3x3(x))], 1)


class FireUp(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, stride):
        super().__init__()
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        if self.stride == 2:
            self.upconv = nn.ConvTranspose2d(
                squeeze_planes, squeeze_planes, kernel_size=[1, 4], stride=[1, 2], padding=[0, 1]
            )
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        if self.stride == 2:
            x = self.activation(self.upconv(x))
        return torch.cat([self.activation(self.expand1x1(x)), self.activation(self.expand3x3(x))], 1)


class Backbone(nn.Module):
    def __init__(self, input_depth: int, dropout: float):
        super().__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(input_depth, 64, kernel_size=3, stride=[1, 2], padding=1), nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Conv2d(input_depth, 64, kernel_size=1, stride=1, padding=0)
        self.fire23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=[1, 2], padding=1), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64)
        )
        self.fire45 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=[1, 2], padding=1), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128)
        )
        self.fire6789 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=[1, 2], padding=1),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.last_channels = 512

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        return y, skips, os

    def forward(self, x):
        skips = {}
        os = 1
        skip_in = self.conv1b(x)
        x = self.conv1a(x)
        skips[1] = skip_in.detach()
        os *= 2
        x, skips, os = self.run_layer(x, self.fire23, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire45, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire6789, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, dropout: float, feature_depth: int = 512):
        super().__init__()
        self.firedec10 = FireUp(feature_depth, 64, 128, 128, stride=2)
        self.firedec11 = FireUp(256, 32, 64, 64, stride=2)
        self.firedec12 = FireUp(128, 16, 32, 32, stride=2)
        self.firedec13 = FireUp(64, 16, 32, 32, stride=2)
        self.dropout = nn.Dropout2d(dropout)
        self.last_channels = 64

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)
        if feats.shape[-1] > x.shape[-1]:
            os //= 2
            feats = feats + skips[os].detach()
        return feats, skips, os

    def forward(self, x, skips):
        os = 16
        x, skips, os = self.run_layer(x, self.firedec10, skips, os)
        x, skips, os = self.run_layer(x, self.firedec11, skips, os)
        x, skips, os = self.run_layer(x, self.firedec12, skips, os)
        x, skips, os = self.run_layer(x, self.firedec13, skips, os)
        return self.dropout(x)


class SqueezeSegNet(nn.Module):
    def __init__(self, input_depth: int, nclasses: int, backbone_dropout: float, decoder_dropout: float, head_dropout: float):
        super().__init__()
        self.backbone = Backbone(input_depth, backbone_dropout)
        self.decoder = Decoder(decoder_dropout, feature_depth=self.backbone.last_channels)
        self.head = nn.Sequential(
            nn.Dropout2d(p=head_dropout), nn.Conv2d(self.decoder.last_channels, nclasses, kernel_size=3, padding=1)
        )

    def forward(self, x):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        return F.softmax(y, dim=1)


# ---------------------------------------------------------------------------
# SemanticSegmentationModel implementation
# ---------------------------------------------------------------------------

class SqueezeSegModel:
    """Pretrained SqueezeSeg (SemanticKITTI, Bonnetal checkpoint). Runs on
    CPU or Apple Silicon MPS -- whichever torch reports available; no CUDA
    dependency (there is no CUDA on this machine)."""

    name = "SqueezeSeg (SemanticKITTI, Bonnetal pretrained)"
    model_ontology = "semantickitti-20"
    provides_confidence = True  # softmax output -> per-point max-class probability

    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        with open(model_dir / "arch_cfg.yaml") as f:
            arch_cfg = yaml.safe_load(f)
        with open(model_dir / "data_cfg.yaml") as f:
            data_cfg = yaml.safe_load(f)

        sensor = arch_cfg["dataset"]["sensor"]
        self.projector = RangeImageProjector(
            height=sensor["img_prop"]["height"],
            width=sensor["img_prop"]["width"],
            fov_up_deg=sensor["fov_up"],
            fov_down_deg=sensor["fov_down"],
            img_means=sensor["img_means"],
            img_stds=sensor["img_stds"],
        )
        self.learning_map_inv: dict[int, int] = data_cfg["learning_map_inv"]  # reduced-id -> raw SemanticKITTI id
        self.raw_label_names: dict[int, str] = data_cfg["labels"]  # raw SemanticKITTI id -> name
        self.nclasses = len(self.learning_map_inv)

        input_depth = 5  # range + xyz + remission, per arch_cfg's input_depth flags (all True for this checkpoint)
        self.net = SqueezeSegNet(
            input_depth=input_depth,
            nclasses=self.nclasses,
            backbone_dropout=arch_cfg["backbone"]["dropout"],
            decoder_dropout=arch_cfg["decoder"]["dropout"],
            head_dropout=arch_cfg["head"]["dropout"],
        )

        self.net.backbone.load_state_dict(torch.load(model_dir / "backbone", map_location="cpu", weights_only=True), strict=True)
        self.net.decoder.load_state_dict(torch.load(model_dir / "segmentation_decoder", map_location="cpu", weights_only=True), strict=True)
        self.net.head.load_state_dict(torch.load(model_dir / "segmentation_head", map_location="cpu", weights_only=True), strict=True)
        self.net.eval()

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.net.to(self.device)

    def preprocess(self, points: np.ndarray) -> RangeImageProjection:
        # SemanticKITTI's own .bin remission channel is stored already
        # normalized to roughly [0,1] (a documented SemanticKITTI convention,
        # confirmed by this checkpoint's own img_means/img_stds for the
        # "signal" channel: mean 0.21, std 0.16). SemanticPOSS and RELLIS-3D
        # instead store raw sensor reflectivity in [0,255]. Feeding that
        # directly into a channel normalized for ~[0,1] input produces
        # values ~250x too large after normalization, saturating the network
        # -- verified empirically during integration: predictions collapsed
        # to one dominant class (~79% "traffic-sign") and accuracy was <1%
        # before this fix. Rescale to [0,1] whenever the input is clearly not
        # already in that range, rather than assuming a fixed per-dataset
        # convention (keeps this model correct for any future dataset too).
        pts = points
        if pts.shape[0] > 0 and float(pts[:, 3].max()) > 1.5:
            pts = pts.copy()
            pts[:, 3] = pts[:, 3] / 255.0
        return self.projector.project(pts)

    def forward(self, preprocessed: RangeImageProjection) -> np.ndarray:
        x = torch.from_numpy(preprocessed.proj_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.net(x)  # (1, nclasses, H, W)
        return probs.squeeze(0).cpu().numpy()  # (nclasses, H, W)

    def postprocess(self, points: np.ndarray, preprocessed: RangeImageProjection, model_output: np.ndarray) -> RawPrediction:
        pixel_pred = np.argmax(model_output, axis=0).astype(np.int64)  # (H, W), reduced 0..nclasses-1 ids
        pixel_conf = np.max(model_output, axis=0).astype(np.float32)  # (H, W)

        predicted_reduced = self.projector.unproject(pixel_pred, preprocessed, fallback_label=0)
        confidence = self.projector.unproject_float(pixel_conf, preprocessed, fallback=0.0)

        n = points.shape[0]
        if predicted_reduced.shape[0] != n:
            raise RuntimeError(
                f"SqueezeSeg postprocess produced {predicted_reduced.shape[0]} labels for {n} input points -- "
                f"refusing to return a misaligned prediction array."
            )

        # Map back to RAW SemanticKITTI ids (the model_ontology this class advertises) --
        # the reduced 0..19 training ids are an implementation detail of how the
        # checkpoint was trained, not the ontology we document/expose.
        lut = np.zeros(self.nclasses, dtype=np.int64)
        for reduced_id, raw_id in self.learning_map_inv.items():
            lut[reduced_id] = raw_id
        predicted_raw = lut[predicted_reduced]

        return RawPrediction(
            predicted_label_ids=predicted_raw,
            confidence=confidence,
            model_ontology=self.model_ontology,
        )

    def predict(self, points: np.ndarray) -> RawPrediction:
        pre = self.preprocess(points)
        out = self.forward(pre)
        return self.postprocess(points, pre, out)


_MODEL_SINGLETON: SqueezeSegModel | None = None


def get_squeezeseg_model(cfg: dict | None = None) -> SqueezeSegModel:
    """Loading the checkpoint takes real time; reuse one instance per
    process rather than reloading it on every request. `cfg` (if given) is
    the loaded config.yaml -- checkpoint_dir comes from
    cfg["perception"]["models"]["squeezeseg"]["checkpoint_dir"] when
    present, else falls back to the models/squeezeseg/ default (fine for
    direct/CLI use without a full config load)."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        model_dir = MODEL_DIR
        if cfg is not None:
            model_dir = Path(cfg["perception"]["models"]["squeezeseg"]["checkpoint_dir"])
        _MODEL_SINGLETON = SqueezeSegModel(model_dir=model_dir)
    return _MODEL_SINGLETON
