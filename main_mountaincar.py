import pickle
import gym

from src.models import PolicyNetwork, PolicyLR, ValueNetwork, ValueLR
from src.agents import TRPOGaussianNN
from src.algorithms import Trainer
from src.utils import Discretizer


if __name__ == "__main__":
    res_nn, res_lr = [], []
    env = gym.make("MountainCarContinuous-v0")
    for _ in range(100):
        # NN
        actor = PolicyNetwork(2, [8], 1).double()
        critic = ValueNetwork(2, [8], 1).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            gamma=0.99,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.05,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        agent, totals, timesteps = trainer.train(
            env,
            agent,
            epochs=300,
            max_steps=10000,
            update_freq=15000,
            initial_offset=10,
        )
        res_nn.append(totals)

        # LR
        discretizer = Discretizer(
            min_points=[-1.2, -0.07],
            max_points=[0.6, 0.07],
            buckets=[4, 4],
            dimensions=[[0], [1]]
        )

        actor = PolicyLR(4, 4, 1, 0.1).double()
        critic = ValueLR(4, 4, 1, 1.0).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            gamma=0.99,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.05,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, totals, _ = trainer.train(
            env,
            agent,
            epochs=300,
            max_steps=10000,
            update_freq=15000,
            initial_offset=10,
        )
        res_lr.append(totals)

    with open('results/mount_nn.pkl','wb') as f:
        pickle.dump(res_nn, f)

    with open('results/mount_lr.pkl','wb') as f:
        pickle.dump(res_lr, f)
