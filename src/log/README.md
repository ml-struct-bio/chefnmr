WandB Offline Sync Hook (Modified)
==================================

Source: https://github.com/klieret/wandb-offline-sync-hook

Purpose:
--------
Enable Weights & Biases (WandB) logging on SLURM distributed systems where only the login node has internet access.

Modifications:
--------------
This codebase has been modified to address a known issue in the original repository:
https://github.com/klieret/wandb-offline-sync-hook/issues/115

Specific changes:
1. Modified `hooks.py` to fix the synchronization issue across different ranks.
2. Updated system environment configuration in `src.model` to ensure reliable syncing.