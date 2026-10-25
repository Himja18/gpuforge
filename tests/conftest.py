import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="session")
def trained_ckpt(tmp_path_factory):
    """A tiny model trained for a few steps on CPU, shared by export/serve tests."""
    from gpuforge.train.train import TrainConfig, train

    out = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(model="resnet_tiny", n_synthetic=512, epochs=1, batch_size=64, workers=0,
                      amp_dtype="none", channels_last=False, log_every=4, out_dir=str(out),
                      db=str(out / "a.db"))
    summary = train(cfg)
    return out / "checkpoint.pt", summary
