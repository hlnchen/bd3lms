import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

from lightning.fabric import Fabric
from lightning.fabric.strategies import XLAFSDPStrategy 

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer)
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False)

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _train(fabric: Fabric, config, logger, model, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  train_ds, valid_ds = fabric.setup_dataloaders(train_ds, valid_ds)
  _print_batch(train_ds, valid_ds, tokenizer)
  try:
    fabric.barrier()
  except Exception as e:
    logger.error(f"Error during TPU synchronization: {e}")
    raise
  # if config.training.from_pretrained is not None and ckpt_path is None:
  #   logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
  #   # load pretraining checkpoint
  #   if 'kuleshov-group/' in config.training.from_pretrained:
  #     # load from hf
  #     model = diffusion.Diffusion(config, tokenizer=tokenizer)
  #     state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
  #         config.training.from_pretrained,
  #         trust_remote_code=True
  #     ).state_dict()
  #     model.load_state_dict(state_dict)
  #   else:
  #     model = diffusion.Diffusion.load_from_checkpoint(
  #       config.training.from_pretrained,
  #       tokenizer=tokenizer,
  #       config=config,
  #       strict=False)
  #   # add buffers for grid search
  #   model.register_buffer('sampling_eps_min', torch.tensor(
  #     config.training.sampling_eps_min))
  #   model.register_buffer('sampling_eps_max', torch.tensor(
  #     config.training.sampling_eps_max))
  # else:
  #   logger.info(f'Initializing new model')
  #   model = diffusion.Diffusion(
  #     config, tokenizer=tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

def _train_manual(fabric: Fabric, config, logger, model, tokenizer):
  logger.info('Starting Manual Training Loop.')
  
  # WandB Logger setup
  wandb_logger = None
  if config.get('wandb', None) is not None and fabric.is_global_zero:
      wandb_logger = L.pytorch.loggers.WandbLogger(
          config=omegaconf.OmegaConf.to_object(config),
          **config.wandb
      )

  # Setup dataloaders
  train_ds, val_ds = dataloader.get_dataloaders(config, tokenizer)
  train_ds = fabric.setup_dataloaders(train_ds)
  val_ds = fabric.setup_dataloaders(val_ds)

  # Setup model and optimizer
  model = fabric.setup_module(model)
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=config.optim.lr,
      betas=(config.optim.beta1, config.optim.beta2),
      eps=config.optim.eps,
      weight_decay=config.optim.weight_decay
  )
  optimizer = fabric.setup_optimizers(optimizer)

  # # Resume from checkpoint if ckpt_path is provided
  # if (config.checkpointing.resume_from_ckpt and
  #     config.checkpointing.resume_ckpt_path is not None and
  #     utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
  #   ckpt_path = config.checkpointing.resume_ckpt_path
  #   logger.info(f'Resuming training from checkpoint: {ckpt_path}')
    
  #   # fabric.load() returns a dictionary of state_dicts.
  #   # Load these into the already Fabric-setup model and optimizer.
  #   loaded_state = fabric.load(ckpt_path) 
  #   model.load_state_dict(loaded_state['model'])
  #   optimizer.load_state_dict(loaded_state['optimizer'])
  #   global_step = loaded_state.get('global_step', 0) # Ensure global_step is initialized here
  #   logger.info(f"Resumed at global_step: {global_step}")
  # else:
  global_step = 0 # Initialize global_step if not resuming
  
  # fabric.barrier() # Synchronize before training starts
  # Determine number of epochs. If max_steps is the primary goal, epochs can be large.
  # If you have config.training.num_epochs, you can use that.
  # For this example, we'll rely on max_steps.
  # num_epochs = getattr(config.training, "num_epochs", int(1e6)) # A large number if not specified
  print(f"Start training loop")
  for epoch_idx in range(config.trainer.max_steps):
    fabric.print(f"Starting Epoch {epoch_idx} (Global step: {global_step})")
    model.train()
    for batch_idx, batch in enumerate(train_ds):
      optimizer.zero_grad()
      fabric.print(f"Batch {batch_idx} training")
      fabric.print(f"Batch: {batch}")
      # Forward pass and loss calculation using model's _loss method
      loss = model.training_step(
          batch, batch_idx
      )
      print(f"Loss: {loss.item()}")
      fabric.backward(loss)
      optimizer.step()
      if batch_idx % 10 == 0:
        fabric.print(f"Epoch {epoch_idx}/{config.trainer.max_steps} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    fabric.barrier()

  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  NUM_TPU_CORES_PER_HOST = "auto"
  NUM_HOSTS = 64
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  
  # # Use the model's components for auto wrap policy
  # auto_wrap_policy_config = {
  #   "module_class": {
  #       type(block) for block in model.backbone.blocks
  #     }
  # }

  strategy_params = {
      "auto_wrap_policy": {diffusion.Diffusion},
      "state_dict_type": 'sharded', # Crucial for multi-host TPU
      "sequential_save": False, # Set to True to reduce host RAM during checkpointing
  }
  fabric = Fabric(
        accelerator="tpu",
        devices=NUM_TPU_CORES_PER_HOST, # Number of TPU cores per host
        num_nodes=NUM_HOSTS,            # Number of hosts/nodes
        strategy=XLAFSDPStrategy(**strategy_params),
        precision="32-true"          # NOTE: ValueError: `precision='bf16-mixed')` is not supported in XLA. `precision` must be one of: ('32-true', '16-true', 'bf16-true').
    )
  fabric.launch(_train_manual, config, logger, model, tokenizer)

  # if config.mode == 'sample_eval':
  #   config.wandb = None
  #   samples = generate_samples(config, logger, tokenizer)
  # elif config.mode == 'ppl_eval':
  #   config.wandb = None
  #   _ppl_eval(config, logger, tokenizer)
  # else:
  #   _train(fabric, config, logger, tokenizer)


if __name__ == '__main__':
  main()