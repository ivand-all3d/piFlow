import argparse
import os
import torch
from collections import OrderedDict
from safetensors.torch import load_file
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Import a diffusers safetensors model back to LakonLab pi-Flow checkpoint format.')
    parser.add_argument(
        'config', 
        help='Config file path (required to verify model structure/metadata)')
    parser.add_argument(
        'safetensors_file', 
        help='Path to the input .safetensors file')
    parser.add_argument(
        '--out', 
        required=True,
        help='Output path for the .pth checkpoint file')
    parser.add_argument(
        '--to-ema',
        action='store_true',
        help='If specified, weights will be saved to the EMA key prefix (diffusion_ema.denoising) instead of standard (diffusion.denoising).')
    parser.add_argument(
        '--copy-to-both',
        action='store_true',
        help='If specified, weights will be duplicated to BOTH standard and EMA prefixes. Useful for initializing training.')
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    safetensors_path = args.safetensors_file
    out_path = args.out
    
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    # Load config mainly for validation and potential metadata use
    # We don't strictly need it for string manipulation, but good to ensure environment matches
    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)
    
    print(f"Loading weights from {safetensors_path}...")
    loaded_state_dict = load_file(safetensors_path)
    
    out_dict = OrderedDict()
    
    # Determine prefixes
    standard_prefix = 'diffusion.denoising.'
    ema_prefix = 'diffusion_ema.denoising.'
    
    target_prefixes = []
    
    if args.copy_to_both:
        target_prefixes = [standard_prefix, ema_prefix]
        print("Mode: Copying weights to BOTH standard and EMA slots.")
    elif args.to_ema:
        target_prefixes = [ema_prefix]
        print("Mode: Copying weights to EMA slot.")
    else:
        target_prefixes = [standard_prefix]
        print("Mode: Copying weights to standard slot.")

    print("Converting keys...")
    
    for k, v in loaded_state_dict.items():
        # 1. Reverse the LoRA naming done in the export script
        # Export: 'lora_A.default.weight' -> 'lora_A.weight'
        # Import: 'lora_A.weight' -> 'lora_A.default.weight'
        
        new_suffix = k
        if 'lora_A.weight' in k:
            new_suffix = new_suffix.replace('lora_A.weight', 'lora_A.default.weight')
        if 'lora_B.weight' in k:
            new_suffix = new_suffix.replace('lora_B.weight', 'lora_B.default.weight')
            
        # 2. Prepend the mmcv prefixes
        for prefix in target_prefixes:
            full_key = prefix + new_suffix
            out_dict[full_key] = v

    # Construct the final checkpoint dictionary required by mmcv/runner
    # Standard format is usually dict(meta=..., state_dict=...)
    checkpoint = dict(
        meta=dict(), # We leave meta empty as we are importing external weights
        state_dict=out_dict
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Saving checkpoint to {out_path}...")
    torch.save(checkpoint, out_path)
    print("Done.")

if __name__ == '__main__':
    main()