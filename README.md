# Diff-GCA
This is the implementation of our Diffusion based Generative Counterfactual Augmentation Framework

## Preliminary Results
<img width="804" alt="stable_diffusion_FT" src="https://github.com/user-attachments/assets/1d008925-6e87-4cef-8cb9-3e95ac3bda62" />

**Figure** : (a) SD-v1.5 w/o fine-tuning, (b-c) SD-v1.5 fine-tuned on RSNA dataset 

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
pip install datasets
```

## Fine-Tune SD-v1.5 with DreamBooth

#### One Concept
```python
python train_dreambooth.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" --instance_data_dir="../CXR/datasets/rsna/" --output_dir="saved_models/one_concept_db/" --instance_prompt="photo of a Chest X-ray" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000
```

#### Multiple Concepts
```python
python train_dreambooth_shivam.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5"  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000 --concepts_list="concepts.json" --output_dir="saved_models/n_concepts_db/"
```

## Fine-Tune SD-v1.5 with Textual Inversion
```python
python textual_inversion.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" --train_data_dir="../CXR/datasets/rsna/" --learnable_property="object" --placeholder_token="<x-ray>" --initializer_token="ray" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=4000 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="saved_models/textual_inversion_cxr"
```

## Fine-Tune SD-v1.5 with Custom Diffusion
```python
python train_custom_diffusion.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" --instance_data_dir="../CXR/datasets/rsna/" --output_dir="saved_models/custom_diffusion_cxr" --class_data_dir="../CXR/datasets/rsna/" --real_prior --prior_loss_weight=1.0 --class_prompt="rsna x-ray" --num_class_images=30000 --instance_prompt="photo of a <rsna> x-ray" --resolution=512 --train_batch_size=2 --learning_rate=1e-5 --lr_warmup_steps=0 --max_train_steps=4000 --scale_lr --hflip --modifier_token "<rsna>" --validation_prompt="<rsna> x-ray" --no_safe_serialization
```

## Fine-Tune SD-v1.5 with InstructPix2Pix
```python
python train_instruct_pix2pix.py --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" --dataset_name="../CXR/datasets/cxr_instruct_dataset" --resolution=256 --random_flip --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --checkpointing_steps=5000 --checkpoints_total_limit=1 --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 --conditioning_dropout_prob=0.05 --mixed_precision=fp16 --seed=42 --output_dir="saved_models/instruct-pix2pix-model"
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
