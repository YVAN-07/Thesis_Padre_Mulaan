# ppo_monitor_callback.py
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.table import Table
import numpy as np


class PPOMonitorCallback(BaseCallback):
    """
    PPO training monitor callback (Webots-safe).

    Logs key PPO metrics every rollout without clearing the console.
    """

    def __init__(self, total_timesteps: int = 100_000, log_every: int = 2048):
        super().__init__()
        self.console = Console()
        self.total_timesteps = int(total_timesteps)
        self.log_every = int(log_every)
        self.last_log_step = 0

        # --- EMA smoothing (display only) ---
        self.ema_alpha = 0.1   # smoothing factor (0.05–0.2 typical)
        self.ema_ep_rew = None
        self.ema_ep_len = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step >= self.log_every:
            self._display_stats()
            self.last_log_step = self.num_timesteps
        return True

    def _display_stats(self):
        logger = self.model.logger.name_to_value

        # Clamp progress
        progress = min(100.0, 100.0 * self.num_timesteps / max(1, self.total_timesteps))

        table = Table(
            title=f"PPO Training Progress",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Core stats
        table.add_row("Timesteps", f"{self.num_timesteps:,} / {self.total_timesteps:,}")
        table.add_row("Progress", f"{progress:6.2f}%")

        # Performance
        if "time/fps" in logger:
            table.add_row("FPS", f"{logger.get('time/fps', 0):.0f}")

        # Rollout stats
        
        if "rollout/ep_rew_mean" in logger:
            raw_rew = logger.get("rollout/ep_rew_mean", 0.0)

            if self.ema_ep_rew is None:
                self.ema_ep_rew = raw_rew
            else:
                self.ema_ep_rew = (
                    self.ema_alpha * raw_rew +
                    (1.0 - self.ema_alpha) * self.ema_ep_rew
                )

            table.add_row(
                "ep_rew_mean (EMA)",
                f"{self.ema_ep_rew:.6f}",
            )

        if "rollout/ep_len_mean" in logger:
            raw_len = logger.get("rollout/ep_len_mean", 0.0)

            if self.ema_ep_len is None:
                self.ema_ep_len = raw_len
            else:
                self.ema_ep_len = (
                    self.ema_alpha * raw_len +
                    (1.0 - self.ema_alpha) * self.ema_ep_len
                )

            table.add_row(
                "ep_len_mean (EMA)",
                f"{self.ema_ep_len:.1f}",
            )

        # Losses
        if "train/value_loss" in logger:
            table.add_row(
                "value_loss",
                f"{logger.get('train/value_loss', 0.0):.6f}",
            )

        if "train/policy_gradient_loss" in logger:
            table.add_row(
                "policy_loss",
                f"{logger.get('train/policy_gradient_loss', 0.0):.6f}",
            )

        if "train/entropy_loss" in logger:
            table.add_row(
                "entropy_loss",
                f"{logger.get('train/entropy_loss', 0.0):.6f}",
            )

        if "train/approx_kl" in logger:
            table.add_row(
                "approx_kl",
                f"{logger.get('train/approx_kl', 0.0):.6f}",
            )

        if "train/clip_fraction" in logger:
            table.add_row(
                "clip_fraction",
                f"{logger.get('train/clip_fraction', 0.0):.4f}",
            )

        if "train/explained_variance" in logger:
            table.add_row(
                "explained_variance",
                f"{logger.get('train/explained_variance', 0.0):.4f}",
            )

        self.console.print(table)


class SimpleProgressCallback(BaseCallback):
    """
    Lightweight one-line PPO progress logger.
    Recommended for long Webots runs.
    """

    def __init__(self, total_timesteps: int = 100_000, log_every: int = 2048):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.log_every = int(log_every)
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step >= self.log_every:
            logger = self.model.logger.name_to_value

            reward = logger.get("rollout/ep_rew_mean", 0.0)
            fps = logger.get("time/fps", 0.0)
            progress = min(100.0, 100.0 * self.num_timesteps / max(1, self.total_timesteps))

            print(
                f"[PPO] step={self.num_timesteps:,} | "
                f"ep_rew_mean={reward: .6f} | "
                f"fps={fps:5.0f} | "
                f"{progress:5.1f}%"
            )

            self.last_log_step = self.num_timesteps

        return True
