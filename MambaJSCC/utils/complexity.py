import time
import torch

try:
    from thop import profile
except ImportError:
    profile = None


def count_params(model, trainable_only=False):
    """
    统计模型参数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def to_million(x):
    return x / 1e6


def to_giga(x):
    return x / 1e9


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer_start():
    sync_cuda()
    return time.perf_counter()


def timer_end_ms(t0):
    sync_cuda()
    return (time.perf_counter() - t0) * 1000.0


def _profile_once(model, inputs):
    """
    用 thop 统计单个模型 MACs
    """
    if profile is None:
        raise ImportError(
            "没有安装 thop，请先运行: pip install thop"
        )

    was_training = model.training
    model.eval()

    with torch.no_grad():
        macs, params = profile(model, inputs=inputs, verbose=False)

    if was_training:
        model.train()

    return float(macs), int(params)


def profile_roi_head_macs(roi_head, device, h=896, w=1024):
    """
    ROI head 是整图输入，所以统计一次整图 MACs
    """
    dummy = torch.randn(1, 3, h, w, device=device)
    macs, _ = _profile_once(roi_head, (dummy,))
    return macs


def _prepare_decoder_input(feature, channel, config, snr):
    """
    复用你当前 JSCC 的 channel + equalizer 逻辑，
    得到 decoder 真正需要的输入形状。
    """
    received, pwr, h = channel.forward(feature, snr)

    if config.CHANNEL.TYPE == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr / 10))
        received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
    elif config.CHANNEL.TYPE == "awgn":
        pass
    else:
        raise ValueError(f"Unknown channel type: {config.CHANNEL.TYPE}")

    received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
    return received


def profile_branch_macs_per_block(
    encoder,
    decoder,
    channel,
    config,
    device,
    snr,
    block_h=128,
    block_w=128,
):
    """
    统计单个 128×128 块经过某一条 MambaJSCC 分支的 MACs。
    返回 encoder / decoder / total 的 MACs。
    """
    dummy = torch.randn(1, 3, block_h, block_w, device=device)

    enc_macs, _ = _profile_once(encoder, (dummy, snr))

    with torch.no_grad():
        feature = encoder(dummy, snr)
        decoder_input = _prepare_decoder_input(feature, channel, config, snr)

    dec_macs, _ = _profile_once(decoder, (decoder_input, snr))

    return {
        "encoder_macs": enc_macs,
        "decoder_macs": dec_macs,
        "branch_macs": enc_macs + dec_macs,
    }