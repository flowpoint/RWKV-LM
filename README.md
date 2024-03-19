# The RWKV Language Model (flowpoint fork)

RWKV homepage: https://www.rwkv.com/ https://wiki.rwkv.com/

pretrained models:

**RWKV-6 World** Demo: https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1 Weights-preview https://huggingface.co/BlinkDL/temp

**RWKV-5 World** Demo: https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-2 weights: https://huggingface.co/BlinkDL/rwkv-5-world

### howto pretrain

```
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade

cd RWKV-v5/
./demo-training-prepare.sh
./demo-training-run.sh
(you may want to log in to wandb first)
```

You can run your model using https://pypi.org/project/rwkv/ (use "rwkv_vocab_v20230424" instead of "20B_tokenizer.json")

Use https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/make_data.py to prepare binidx data from jsonl, and compute "--my_exit_tokens" and "--magic_prime".

The "epoch" in train.py is "mini-epoch" (not real epoch. only for convenience), and 1 mini-epoch = 40320 * ctx_len tokens.

For example, if your binidx has 1498226207 tokens and ctxlen=4096, set "--my_exit_tokens 1498226207" (this will override epoch_count), and it will be 1498226207/(40320 * 4096) = 9.07 miniepochs. The trainer will auto-exit after "--my_exit_tokens" tokens. Set "--magic_prime" to the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/4096-1 = 365776), which is "--magic_prime 365759" in this case.

simple: prepare SFT jsonl => repeat your SFT data 3 or 4 times in make_data.py. more repetition leads to overfitting.

advanced: repeat your SFT data 3 or 4 times in your jsonl (note make_data.py will shuffle all jsonl items) => add some base data (such as slimpajama) to your jsonl => and only repeat 1 times in make_data.py.

**Train RWKV-6**: use /RWKV-v5/ and add --my_testing "x060" to demo-training-prepare.sh and demo-training-run.sh

**Simple inference for RWKV-5**: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v5_demo.py

**Simple inference for RWKV-6**: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py

lm_eval: https://github.com/BlinkDL/ChatRWKV/blob/main/run_lm_eval.py

chat demo for developers: https://github.com/BlinkDL/ChatRWKV/blob/main/API_DEMO_CHAT.py

### howto finetune

Use .jsonl format for your data (see https://huggingface.co/BlinkDL/rwkv-5-world for formats).

Use https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/make_data.py to tokenizer it using World tokenizer into binidx, suitable for finetuning World models.

Rename the base checkpoint in your model folder to rwkv-init.pth, and change the training commands to use --n_layer 32 --n_embd 4096 --vocab_size 65536 --lr_init 1e-5 --lr_final 1e-5 for 7B.


```
@software{peng_bo_2021_5196578,
  author       = {PENG Bo},
  title        = {BlinkDL/RWKV-LM: 0.01},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.01},
  doi          = {10.5281/zenodo.5196577},
  url          = {https://doi.org/10.5281/zenodo.5196577}
}
```
