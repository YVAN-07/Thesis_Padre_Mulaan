# checkpoint_callback.py
"""
Checkpoint Callback
Saves PPO model checkpoints at fixed step intervals.
Compatible with Stable-Baselines3 and Webots.
"""

from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path


class CheckpointCallback(BaseCallback):
    """
    Save model checkpoint every N timesteps.
    """

    def __init__(self, checkpoint_every: int = 5000, save_dir: str = "checkpoints", verbose: int = 1):
        super().__init__(verbose=verbose)

        self.checkpoint_every = int(checkpoint_every)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.last_checkpoint_step = 0
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        # Safety: ensure model exists
        if self.model is None:
            return True

        # Save every N steps
        if (self.num_timesteps - self.last_checkpoint_step) >= self.checkpoint_every:
            self._save_checkpoint()
            self.last_checkpoint_step = self.num_timesteps

        return True

    def _save_checkpoint(self):
        self.checkpoint_count += 1

        filename = f"checkpoint_{self.checkpoint_count:04d}_step{self.num_timesteps}"
        path = self.save_dir / filename

        self.model.save(path)

        if self.verbose > 0:
            print(f"[Checkpoint] #{self.checkpoint_count} saved at step {self.num_timesteps:,}")
