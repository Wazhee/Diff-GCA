# Diff-GCA
This is the implementation of our Diffusion based Generative Counterfactual Augmentation Framework

## Fine-Tune SD-v1.5 with DreamBooth
```python
python train_dreambooth.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" --instance_data_dir="../CXR/datasets/rsna/" --output_dir="saved_models/" --instance_prompt="photo of a Chest X-ray" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000
```

## Troubleshooting
```python
# If issues with 'cached_downloads' library
https://github.com/easydiffusion/easydiffusion/issues/1851
```

## Install Required Libraries
```python
#Python3.10
pip install diffusers==0.25.0
pip install transformers==4.39.3
pip install accelerate
```

## Cite this work
Kulkarni et al, [*Hidden in Plain Sight*](https://arxiv.org/abs/2402.05713), MIDL 2024.
```
@article{kulkarni2024hidden,
  title={Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations},
  author={Kulkarni, Pranav and Chan, Andrew and Navarathna, Nithya and Chan, Skylar and Yi, Paul H and Parekh, Vishwa S},
  journal={arXiv preprint arXiv:2402.05713},
  year={2024}
}
