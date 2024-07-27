# 🕴️ finetune `LLM` 🦙 model on custom `DataSet`, using `unsloth`

## 🕴️ ONESTEP CODE : finetune.py 🦙

1. Prepare `DataSet` to match the `unsloth` format :
> [process_dataset_to_unsloth.ipynb](process_dataset_to_unsloth.ipynb)

```bash
Process Dataset -> dataset_to_unsloth.ipynb
```

2. Adapt the `config arguments` : `finetune.py` in finetune.md
> [finetune.md](finetune.md)

```python
def finetune(
  # -- PARAMETERS CONFIG -- 
  SOURCE_MODEL = "unsloth/Phi-3-mini-4k-instruct",
  DATASET = "0xZee/arxiv-math-Unsloth-tune-50k", 
  #DATASET = "ArtifactAI/arxiv-math-instruct-50k",
  MAX_STEPS = 444,
  FINETUNED_LOCAL_MODEL = "Phi-3-mini_ft_arxiv-math",
  FINETUNED_ONLINE_MODEL = "0xZee/Phi-3-mini_ft_arxiv-math",
  TEST_PROMPT = "Which compound is antiferromagnetic?", # response : common magnetic ordering in various materials.
):
```
3. Run the onestep file : `finetune.py` in finetune.md
> [finetune.md](finetune.md)
```bash
python finetune.py
```
---

## 🕴️ finetune llama3.1 🦙 model on custom DataSet

- 🏬 FineTunning Framework : `Unsloth` on `GPU Tesla T4`
- 🦙 Source Model : `models--unsloth--meta-llama-3.1-8b-bnb-4bit` Model 🕴️ 
- 💾 Training DataSet ; "yahma/alpaca-cleaned" on HuggingFace
- ⚙️ Fine-Tuned Model : 🕴️ `llama3-1_0xZee_model`
- Model saved to : https://huggingface.co/0xZee/llama3-1_0xZee_model
