# meshgraphnets_paddlepaddle

Work-in-progress. Re-implementing meshgraphnets using PaddlePaddle.

## How to run the demo

There is a pretrained `cylinder_flow` model in the git already.

Simply run:

```bash
python rollout.py --gpu --rollout_num 5
```

to generate a few rollouts. Then you can use:

```
python visualize_cylinder_flow.py
```

for visualization. The output videos will be in `./video/` directory.
