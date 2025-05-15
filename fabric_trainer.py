import os
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Literal, Optional, Union, cast

import torch
from lightning_utilities import apply_to_collection
from tqdm import tqdm

import lightning as L
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden

from diffusion import Diffusion


class FabricTrainer:
    def __init__(
        self,
        fabric: L.Fabric,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        log_every_n_steps: int = 50,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        val_check_interval: Union[int, float] = 1.0,
        default_root_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            fabric: The Lightning Fabric instance to use.
            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.
            log_every_n_steps: How often to log within steps.
                Default: ``50``.
            accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
                Default: 1.
            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.
            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.
                Default: ``None``.
            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.
            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches.
                Default: ``1.0``.
            default_root_dir: Default path for logs and weights when no logger/checkpoint_dir passed.
                Default: ``os.getcwd()``.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """
        # Set default_root_dir if not provided
        self.default_root_dir = default_root_dir or os.getcwd()

        # Use default_root_dir for checkpoint_dir if not specified
        if checkpoint_dir == "./checkpoints" and default_root_dir is not None:
            checkpoint_dir = os.path.join(self.default_root_dir, "checkpoints")

        self.fabric = fabric
        self.global_step = 0
        self.grad_accum_steps: int = accumulate_grad_batches
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        # TODO: exmaine if these two batches are used correctly.
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        self.validation_frequency = validation_frequency
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        self.log_every_n_steps = log_every_n_steps
        self.num_sanity_val_steps = num_sanity_val_steps
        self.val_check_interval = val_check_interval
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

    def fit(
        self,
        model: Union[L.LightningModule, Diffusion],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(
            model.configure_optimizers()
        )
        assert optimizer is not None

        model = self.fabric.setup_module(model)
        # NOTE: exprimental, not sure if this is correct
        self.model = model
        model.trainer = self

        optimizer = self.fabric.setup_optimizers(optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if (
                    self.max_epochs is not None
                    and self.current_epoch >= self.max_epochs
                ):
                    self.should_stop = True

        # NOTE: exprimental, reloading distributed samplers
        train_loader, val_loader = self.model.on_train_start(
            dataloaders=[train_loader, val_loader]
        )

        # Run sanity check validation before training starts
        if self.num_sanity_val_steps > 0 and val_loader is not None:
            self._run_sanity_check(model, val_loader)

        while not self.should_stop:
            self.train_loop(
                model,
                optimizer,
                train_loader,
                limit_batches=self.limit_train_batches,
                scheduler_cfg=scheduler_cfg,
            )

            if self.should_validate:
                model.on_validation_model_zero_grad()
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            # TODO: check if the scheduler should be epoch or step
            self.step_scheduler(
                model, scheduler_cfg, level="epoch", current_value=self.current_epoch
            )

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.fabric.call("on_train_epoch_start")

        for batch_idx, batch in enumerate(train_loader):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer)

                # optimizer step runs train step internally through closure
                optimizer.step(
                    partial(
                        self.training_step,
                        model=model,
                        batch=batch,
                        batch_idx=batch_idx,
                    )
                )

                # Apply gradient clipping if specified
                if self.gradient_clip_val is not None:
                    clip_kwargs = {}
                    if self.gradient_clip_algorithm is not None:
                        clip_kwargs["norm_type"] = (
                            2.0
                            if self.gradient_clip_algorithm.lower() == "norm"
                            else float("inf")
                        )

                    self.fabric.clip_gradients(
                        model,
                        optimizer=optimizer,
                        clip_val=self.gradient_clip_val,
                        **clip_kwargs,
                    )

                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call(
                "on_train_batch_end", self._current_train_return, batch, batch_idx
            )

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(
                    model, scheduler_cfg, level="step", current_value=self.global_step
                )

            # Log metrics based on log_every_n_steps
            if self.global_step % self.log_every_n_steps == 0:
                metrics = {}
                if isinstance(self._current_train_return, torch.Tensor):
                    metrics["loss"] = self._current_train_return
                elif isinstance(self._current_train_return, Mapping):
                    metrics.update(self._current_train_return)

                if metrics and hasattr(self.fabric, "log_dict"):
                    self.fabric.log_dict(metrics, step=self.global_step)

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden(
            "validation_step", _unwrap_objects(model)
        ):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
            model.eval()
        else:
            self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        val_metrics = {}
        for batch_idx, batch in enumerate(val_loader):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            # Accumulate metrics for logging
            if isinstance(out, torch.Tensor):
                val_metrics.setdefault("val_loss", 0)
                val_metrics["val_loss"] += out.item()
            elif isinstance(out, Mapping):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        val_metrics.setdefault(f"val_{k}", 0)
                        val_metrics[f"val_{k}"] += v.item()

        # Log validation metrics at the end of validation
        if val_metrics and hasattr(self.fabric, "log_dict"):
            # Average the metrics
            for k in val_metrics:
                val_metrics[k] /= min(len(val_loader), limit_batches)
            self.fabric.log_dict(val_metrics, step=self.global_step)

        self.fabric.call("on_validation_epoch_end")

        if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
            model.train()
        else:
            self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(
        self, model: L.LightningModule, batch: Any, batch_idx: int
    ) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(
            batch, batch_idx=batch_idx
        )

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            outputs, dtype=torch.Tensor, function=lambda x: x.detach()
        )

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {"train_" + k: v for k, v in self._current_train_return.items()}
            )

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(
                {"val_" + k: v for k, v in self._current_val_return.items()}
            )

        try:
            monitor = possible_monitor_vals[
                cast(Optional[str], scheduler_cfg["monitor"])
            ]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        # For epoch-based validation
        if (
            self.validation_frequency > 0
            and self.current_epoch % self.validation_frequency == 0
        ):
            return True

        # Check if validation should run based on val_check_interval
        if isinstance(self.val_check_interval, int):
            return self.global_step % self.val_check_interval == 0
        else:
            # Only consider fraction-based validation if we have a train loader
            train_loader_defined = hasattr(self, "model") and hasattr(
                self.model, "train_dataloader"
            )
            if train_loader_defined:
                try:
                    train_loader = self.model.train_dataloader()
                    if train_loader is not None:
                        train_batches = len(train_loader)
                        epoch_step = self.global_step % train_batches
                        return epoch_step / train_batches >= self.val_check_interval
                except:
                    pass
            # If we can't determine train loader length, fall back to epoch-based validation
            return False

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.model.on_load_checkpoint(remainder)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)
        self.model.on_save_checkpoint(state)
        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            state,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(self, configure_optim_output) -> tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(
                isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                for _opt_cand in configure_optim_output
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return (
                        None,
                        self._parse_optimizers_schedulers(configure_optim_output[0])[1],
                    )

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    def _run_sanity_check(
        self, model: L.LightningModule, val_loader: torch.utils.data.DataLoader
    ):
        """Run validation sanity check before starting training.

        Args:
            model: the LightningModule to evaluate
            val_loader: the validation dataloader
        """
        if not is_overridden("validation_step", _unwrap_objects(model)):
            return

        self.fabric.call("on_validation_model_eval")

        torch.set_grad_enabled(False)

        if self.fabric.is_global_zero:
            print(f"Sanity Checking: {self.num_sanity_val_steps} validation batches")

        limit = (
            min(self.num_sanity_val_steps, len(val_loader))
            if self.num_sanity_val_steps != -1
            else len(val_loader)
        )

        iterable = self.progbar_wrapper(
            val_loader, total=limit, desc="Validation Sanity Check"
        )

        for batch_idx, batch in enumerate(iterable):
            if batch_idx >= limit and self.num_sanity_val_steps != -1:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

        if self.fabric.is_global_zero:
            print("Sanity check complete")
