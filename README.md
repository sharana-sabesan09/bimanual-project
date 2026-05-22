# Bimanual Ball Balance

G1 humanoid (fixed base) balancing a ball on a tray attached to the right wrist.

## Setup

```bash
conda activate <your_env>
```

## Running

**Zero agent** — right arm sends all-zero position targets:
```bash
python zero_agent.py
```

**Random agent** — right arm gets random targets each step:
```bash
python random_agent.py
```

**Train** — PPO via rsl-rl (requires GPU):
```bash
python train_single_hand.py
python train_single_hand.py -n 2048 --max_iterations 2000 
```

Checkpoints and TensorBoard logs saved to `logs/ball_balance/`.

**Eval** — pass a specific checkpoint file:
```bash
python eval_single_hand.py --checkpoint logs/ball_balance/model_1000.pt
python eval_single_hand.py --checkpoint logs/ball_balance/model_1000.pt -n 8
```

**TensorBoard:**
```bash
tensorboard --logdir logs/
```
