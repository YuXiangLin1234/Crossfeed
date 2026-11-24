# Vendored from UNISON and modified for CrossFeed.
# https://github.com/lizhaoqing/UNISON/blob/7608c183f9f76e8da73261c95e923376b22f1e33/unison/utils/diffusers_compat.py

def patch_transformers_deepspeed() -> None:
    import transformers

    if not hasattr(transformers, "deepspeed"):
        import transformers.integrations.deepspeed as deepspeed

        transformers.deepspeed = deepspeed
