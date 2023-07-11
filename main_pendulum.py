import pickle

from src.environments import CustomPendulumEnv
from src.models import PolicyNetwork, PolicyLR, ValueNetwork, ValueLR
from src.agents import TRPOGaussianNN
from src.algorithms import Trainer
from src.utils import Discretizer


if __name__ == "__main__":
    res_nn, res_lr = [], []
    env = CustomPendulumEnv()
    for _ in range(100):
        # NN
        actor = PolicyNetwork(2, [16, 16], 1).double()
        critic = ValueNetwork(2, [16, 16], 1).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            gamma=0.99,
            delta=0.05,
            tau=0.9,
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, totals, _ = trainer.train(
            env,
            agent,
            epochs=2000,
            max_steps=1000,
            update_freq=15000,
            initial_offset=-1,
        )
        res_nn.append(totals)

        # LR
        discretizer = Discretizer(
            min_points=[-1, -5],
            max_points=[1, 5],
            buckets=[16, 16],
            dimensions=[[0], [1]]
        )

        actor = PolicyLR(16, 16, 4, 1.0).double()
        critic = ValueLR(16, 16, 4, 1.0).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            gamma=0.99,
            delta=0.05,
            tau=0.9,
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, totals, _ = trainer.train(
            env,
            agent,
            epochs=2000,
            max_steps=1000,
            update_freq=15000,
            initial_offset=-1,
        )
        res_lr.append(totals)

    with open('results/pend_nn.pkl','wb') as f:
        pickle.dump(res_nn, f)

    with open('results/pend_lr.pkl','wb') as f:
        pickle.dump(res_lr, f)
