find checkpoints/gmqwen_k8_datafree_piid_4step_1gpu/ -name "iter_?0*.pth" | while read filepath; do
    # Extract the stem
    stem=$(basename "$filepath" .pth)
    
    # Run the command
    uv run pth_to_safetensors.py --ckpt "$filepath" configs/piqwen/1gpu_test.py --out-dir "checkpoints/converted/$stem"
done
