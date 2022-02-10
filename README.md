# Score-based image-to-image regression

This repo contains the official implementation for score-based image-to-image regression with joint diffusion. 

Required datasets: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)


The example command to train and test a model:
```
main.py --runner DiffusionRunner --config nyu_d.yml --doc folder_name
main.py --runner DiffusionRunner --test --doc model_folder -o results_folder
```

