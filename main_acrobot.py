import pickle

from src.environments import CustomAcrobotEnv
from src.models import PolicyNetwork, PolicyLR, ValueNetwork, ValueLR
from src.agents import TRPOGaussianNN
from src.algorithms import Trainer
from src.utils import Discretizer


if __name__ == "__main__":
    res_nn, res_lr = [], []
    env = CustomAcrobotEnv()
    for _ in range(100):
        # NN
        actor = PolicyNetwork(4, [16], 1).double()
        critic = ValueNetwork(4, [16], 1).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            gamma=0.9,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, totals, _ = trainer.train(
            env,
            agent,
            epochs=20000,
            max_steps=1000,
            update_freq=10000,
            initial_offset=-1,
        )
        res_nn.append(totals)

        # LR
        discretizer = Discretizer(
            min_points=[-1, -1, -1, -1],
            max_points=[1, 1, 1, 1],
            buckets=[2, 2, 2, 2],
            dimensions=[[0, 1], [2, 3]]
        )

        actor = PolicyLR(4, 4, 2, 0.1).double()
        critic = ValueLR(4, 4, 2, 0.1).double()

        agent = TRPOGaussianNN(
            actor,
            critic,
            discretizer,
            discretizer,
            gamma=0.9,
            delta=0.01,
            tau=0.9,
            cg_dampening=0.1,
            cg_tolerance=1e-10,
            cg_iteration=10,
        )

        trainer = Trainer(actor_opt='sgd', critic_opt='sgd')
        _, totals, _ = trainer.train(
            env,
            agent,
            epochs=20000,
            max_steps=1000,
            update_freq=10000,
            initial_offset=-1,
        )
        res_lr.append(totals)

    with open('results/acro_nn.pkl','wb') as f:
        pickle.dump(res_nn, f)

    with open('results/acro_lr.pkl','wb') as f:
        pickle.dump(res_lr, f)
