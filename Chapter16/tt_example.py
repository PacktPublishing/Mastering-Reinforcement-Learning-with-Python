"""
This code is taken and slightly modified from:
https://www.tensortrade.org/en/latest/agents/overview.html#ray
"""
import numpy as np
from ray import tune
from ray.tune.registry import register_env
import tensortrade.env.default as default
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio


USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")


def create_env(config):
    x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
    p = Stream.source(50*np.sin(3*x) + 100,
                      dtype="float").rename("USD-TTC")

    coinbase = Exchange("coinbase", service=execute_order)(
        p
    )

    cash = Wallet(coinbase, 100000 * USD)
    asset = Wallet(coinbase, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
        p,
        p.rolling(window=10).mean().rename("fast"),
        p.rolling(window=50).mean().rename("medium"),
        p.rolling(window=100).mean().rename("slow"),
        p.log().diff().fillna(0).rename("lr")
    ])

    reward_scheme = default.rewards.SimpleProfit()

    action_scheme = default.actions.BSH(
        cash=cash,
        asset=asset
    )

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return env

register_env("TradingEnv", create_env)


analysis = tune.run(
    "PPO",
    stop={
      "episode_reward_mean": 500
    },
    config={
        "env": "TradingEnv",
        "env_config": {
            "window_size": 25
        },
        "ignore_worker_failures": True,
        "num_workers": 60,
        "num_gpus": 0,
        "observation_filter": "MeanStdFilter",
    },
    checkpoint_freq=5
)
