from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .multi_env_gdp import MultiEnvDGP
from .utils import summary_statistics, plot_dag, random_perm


class MultiEnvDataModule(LightningDataModule):
    """
    Data module for multi-environment data.

    Attributes
    ----------
    medgp: MultiEnvDGP
        Multi-environment data generating process.
    num_samples_per_env: int
        Number of samples per environment.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the data loaders.
    intervention_targets_per_env: Tensor, shape (num_envs, num_causal_variables)
        Intervention targets per environment, with 1 indicating that the variable is intervened on.
    log_dir: Optional[Path]
        Directory to save summary statistics and plots to. Default: None.
    intervention_target_misspec: bool
        Whether to misspecify the intervention targets. If true, the intervention targets are permuted.
        I.e. the model received the wrong intervention targets. Default: False.
    intervention_target_perm: Optional[list[int]]
        Permutation of the intervention targets. If None, a random permutation is used. Only used if
        intervention_target_misspec is True. Default: None.

    Methods
    -------
    setup(stage=None) -> None
        Setup the data module. This is where the data is sampled.
    train_dataloader() -> DataLoader
        Return the training data loader.
    val_dataloader() -> DataLoader
        Return the validation data loader.
    test_dataloader() -> DataLoader
        Return the test data loader.
    """

    def __init__(
        self,
        multi_env_dgp: MultiEnvDGP,
        num_samples_per_env: int,
        batch_size: int,
        num_workers: int,
        intervention_targets_per_env: Tensor,
        log_dir: Optional[Path] = None,
        intervention_target_misspec: bool = False,
        intervention_target_perm: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.medgp = multi_env_dgp
        self.num_samples_per_env = num_samples_per_env
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.intervention_targets_per_env = intervention_targets_per_env
        self.log_dir = log_dir

        self.intervention_target_misspec = intervention_target_misspec
        latent_dim = self.medgp.latent_scm.latent_dim
        assert (
            intervention_target_perm is None
            or len(intervention_target_perm) == latent_dim
        )
        self.intervention_target_perm = intervention_target_perm

    def setup(self, stage: Optional[str] = None) -> None:
        latent_dim = self.medgp.latent_scm.latent_dim
        num_envs = self.intervention_targets_per_env.shape[0]

        x, v, u, e, intervention_targets, log_prob = self.medgp.sample(
            self.num_samples_per_env,
            intervention_targets_per_env=self.intervention_targets_per_env,
        )
        if self.intervention_target_misspec:
            assert (
                num_envs == latent_dim + 1
            ), "only works if num_envs == num_causal_variables + 1"
            if self.intervention_target_perm is None:
                perm = random_perm(latent_dim)
                self.intervention_target_perm = perm
            else:
                perm = self.intervention_target_perm

            # remember where old targets were
            idx_mask_list = []
            for i in range(latent_dim):
                idx_mask = intervention_targets[:, i] == 1
                idx_mask_list.append(idx_mask)
                intervention_targets[idx_mask, i] = 0

            # permute targets
            for i in range(latent_dim):
                intervention_targets[idx_mask_list[i], perm[i]] = 1

        dataset = TensorDataset(x, v, u, e, intervention_targets, log_prob)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.5 * (len(dataset) - train_size))
        test_size = len(dataset) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            summary_stats = summary_statistics(x, v, e, intervention_targets)
            for key, value in summary_stats.items():
                value.to_csv(self.log_dir / f"{key}_summary_stats.csv")
            plot_dag(self.medgp.adjacency_matrix, self.log_dir)
            try:
                with open(self.log_dir / "base_coeff_values.txt", "w") as f:
                    f.write(str(self.medgp.latent_scm.base_coeff_values))
            except AttributeError:
                pass
            # save mixing function coefficients
            self.medgp.mixing_function.save_coeffs(self.log_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
