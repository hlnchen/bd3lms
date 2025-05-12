import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
        weights_only=False,
    ).to("cuda")


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def generate_samples(config, logger, tokenizer):
    logger.info("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    text_samples = model.restore_model_and_sample(num_steps=config.algo.T)
    print("Text samples:", text_samples)
    print("Generative perplexity:", model.metrics.gen_ppl.compute())
    print("Entropy:", model.metrics.gen_entropy.compute())
    csv_path = config.sampling.logdir
    save_dict = {
        "gen_ppl": model.metrics.gen_ppls,
        "gen_nfes": model.metrics.gen_nfes,
        "gen_entropy": model.metrics.gen_entropies,
        "gen_lengths": model.metrics.gen_lengths,
        "samples": [[i] for i in text_samples],
        "seed": [config.seed for _ in range(len(text_samples))],
    }
    if config.sampling.var_length:
        print(text_samples)
        save_dict["samples"] = ["" for _ in range(len(text_samples))]
    utils.update_and_save_csv(save_dict, csv_path)
    return text_samples


def _ppl_eval(config, logger, tokenizer):
    logger.info("Starting Eval.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    seed = config.seed
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    L.seed_everything(seed)
    config.seed = seed
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=seed
    )
    trainer.validate(model, valid_ds)


def _train(fabric, config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
        logger.info(f"Resuming training at {ckpt_path}")
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.from_pretrained is not None and ckpt_path is None:
        logger.info(f"Loading pretrained model from {config.training.from_pretrained}")
        # load pretraining checkpoint
        if "kuleshov-group/" in config.training.from_pretrained:
            # load from hf
            model = diffusion.Diffusion(config, tokenizer=tokenizer)
            state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
                config.training.from_pretrained, trust_remote_code=True
            ).state_dict()
            model.load_state_dict(state_dict)
        else:
            model = diffusion.Diffusion.load_from_checkpoint(
                config.training.from_pretrained,
                tokenizer=tokenizer,
                config=config,
                strict=False,
            )
        # add buffers for grid search
        model.register_buffer(
            "sampling_eps_min", torch.tensor(config.training.sampling_eps_min)
        )
        model.register_buffer(
            "sampling_eps_max", torch.tensor(config.training.sampling_eps_max)
        )
    else:
        logger.info(f"Initializing new model")
        model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )

    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(fabric, config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == "sample_eval":
        config.wandb = None
        samples = generate_samples(config, logger, tokenizer)
    elif config.mode == "ppl_eval":
        config.wandb = None
        _ppl_eval(config, logger, tokenizer)
    else:
        _train(fabric, config, logger, tokenizer)


if __name__ == "__main__":
    from lightning.fabric import Fabric
    from lightning.fabric.strategies import (
        XLAFSDPStrategy,
    )  # Can also use string "xla_fsdp"
    from models.autoregressive import AR
    from models.dit import DIT

    # Ensure necessary environment variables like PJRT_DEVICE=TPU are set
    # Configuration (these would typically come from args or a config file)
    NUM_TPU_CORES_PER_HOST = "auto"  # Standard for most TPU VMs
    # NUM_HOSTS should be dynamically determined or configured based on your TPU Pod slice
    # For example, if you have a v4-32 (32 chips), and each host has 4 chips, NUM_HOSTS = 8.
    # This value is critical for Fabric to initialize the distributed environment correctly.
    # It can often be inferred by XLA if the launch mechanism sets up the environment.
    # If fabric.launch() is used, num_nodes must be passed if > 1.
    # If an external launcher like xla_dist is used, Fabric might pick it up.
    # For this example, let's assume it's known or passed.
    NUM_HOSTS = 2  # Example: training on 2 hosts

    # Define auto_wrap_policy (example: wrap TransformerEncoderLayer)
    # This is a critical parameter for FSDP.
    auto_wrap_policy_config = {AR, DIT, diffusion.Diffusion}  # NOTE: a set of Modules
    # activation_checkpointing_policy_config = {    } # Optional

    # Initialize Fabric
    # For multi-host, state_dict_type MUST be 'sharded'
    # XLAFSDPStrategy specific parameters
    strategy_params = {
        "auto_wrap_policy": auto_wrap_policy_config,
        # "activation_checkpointing_policy": activation_checkpointing_policy_config,
        "state_dict_type": "sharded",  # Crucial for multi-host TPU
        "sequential_save": False,  # Set to True to reduce host RAM during checkpointing
    }

    fabric = Fabric(
        accelerator="tpu",
        devices=NUM_TPU_CORES_PER_HOST,  # Number of TPU cores per host
        num_nodes=NUM_HOSTS,  # Number of hosts/nodes
        strategy=XLAFSDPStrategy(**strategy_params),
        precision="32-true",  # NOTE: ValueError: `precision='bf16-mixed')` is not supported in XLA. `precision` must be one of: ('32-true', '16-true', 'bf16-true').
    )
    fabric.launch(main)  # Essential for initializing the distributed environment
