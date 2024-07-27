### CODE : `finetune.py`

> INSTALL DEPENDECIES :
```bash
#%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```
> FINETUNE `PHI3` ON `PHYSICS` :

```python

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import TextStreamer
#from google.colab import userdata

def finetune(
  # -- PARAMETERS CONFIG -- 
  SOURCE_MODEL = "unsloth/Phi-3-mini-4k-instruct",
  DATASET = "0xZee/arxiv-math-Unsloth-tune-50k", # https://huggingface.co/datasets/yahma/alpaca-cleaned
  #DATASET = "ArtifactAI/arxiv-math-instruct-50k", # https://huggingface.co/datasets/yahma/alpaca-cleaned
  MAX_STEPS = 444,
  FINETUNED_LOCAL_MODEL = "Phi-3-mini_ft_arxiv-math",
  FINETUNED_ONLINE_MODEL = "0xZee/Phi-3-mini_ft_arxiv-math",
  TEST_PROMPT = "Which compound is antiferromagnetic?", # response : common magnetic ordering in various materials.
):
  
  
  
  # -- PARAMETERS CONFIG -- 
  #SOURCE_MODEL = "unsloth/Meta-Llama-3.1-8B"
  #DATASET = "yahma/alpaca-cleaned" # https://huggingface.co/datasets/yahma/alpaca-cleaned
  #MAX_STEPS = 60
  #FINETUNED_LOCAL_MODEL = "llama3-1_0xZee_model"
  #FINETUNED_ONLINE_MODEL = "0xZee/llama3-1_0xZee_model"
  #TEST_PROMPT = "List the 10 last kings in France",
  # -- ----------------- --
  
  print("-------------------------------------------------------------")
  print(" 🛠 FINETUNE MODEL ON CUSTOM DATASET 📚 (UNSLOTH)")
  print("-------------------------------------------------------------\n")
  print(" ⚙️ Config Parameters : ")
  print("-------------------------------------------------------------")
  print(f" SOURCE_MODEL :\t {SOURCE_MODEL}")
  print(f" DATASET :\t {DATASET}")
  print(f" FINETUNED_LOCAL_MODEL :\t {FINETUNED_LOCAL_MODEL}")
  print(f" FINETUNED_ONLINE_MODEL :\t {FINETUNED_ONLINE_MODEL}")
  print(f" MAX_STEPS :\t {MAX_STEPS}")
  print(f" TEST_PROMPT :\t {TEST_PROMPT}")
  #print(f" HuggingFace API :\t {HF_API}")
  print("-------------------------------------------------------------\n\n")
  
  # PREPARE
  max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
  dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
  load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
  
  # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
  fourbit_models = [
      "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
      "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
      "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
      "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
      "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
      "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
      "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
      "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
      "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
      "unsloth/Phi-3-medium-4k-instruct",
      "unsloth/gemma-2-9b-bnb-4bit",
      "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
  ] # More models at https://huggingface.co/unsloth
  
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = SOURCE_MODEL,
      max_seq_length = max_seq_length,
      dtype = dtype,
      load_in_4bit = load_in_4bit,
      # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
  )
  
  print("\n-------------------------------------------------------------")
  print(" ✅ Model and Tokenizer set up successfully ")
  
  model = FastLanguageModel.get_peft_model(
      model,
      r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
      lora_alpha = 16,
      lora_dropout = 0, # Supports any, but = 0 is optimized
      bias = "none",    # Supports any, but = "none" is optimized
      # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
      use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
      random_state = 3407,
      use_rslora = False,  # We support rank stabilized LoRA
      loftq_config = None, # And LoftQ
  )
  
  print("\n-------------------------------------------------------------")
  print(" ✅ LoRA Adapters : LoRA to optimize finetuning 1% of parameters adapted ")
  
  ##
  
  alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
  
  ### Instruction:
  {}
  
  ### Input:
  {}
  
  ### Response:
  {}"""
  
  EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
  def formatting_prompts_func(examples):
      instructions = examples["instruction"]
      inputs       = examples["input"]
      outputs      = examples["output"]
      texts = []
      for instruction, input, output in zip(instructions, inputs, outputs):
          # Must add EOS_TOKEN, otherwise your generation will go on forever!
          text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
          texts.append(text)
      return { "text" : texts, }
  pass
  
  #dataset = load_dataset(DATASET, split = "train") # from HF Hub
  dataset = load_dataset("parquet", data_dir='arxiv-math-Unsloth-tune-50k/data', split = "train") # From local
  dataset = dataset.map(formatting_prompts_func, batched = True,)
  
  print("\n-------------------------------------------------------------")
  print(" ✅ DataSet Loaded and splitted successfully")
  print(" 📚 DataSet : xxx ")
  
  ## use Huggingface TRL's SFTTrainer! More docs here: TRL SFT docs. We do 100 steps to speed things up, but you can set num_train_epochs=1 for a full run, and turn off max_steps=None. We also support TRL's DPOTrainer!
  

  
  print("\n-------------------------------------------------------------")
  print(" Setting Trainning Model")
  
  trainer = SFTTrainer(
      model = model,
      tokenizer = tokenizer,
      train_dataset = dataset,
      dataset_text_field = "text",
      max_seq_length = max_seq_length,
      dataset_num_proc = 2,
      packing = False, # Can make training 5x faster for short sequences.
      args = TrainingArguments(
          per_device_train_batch_size = 2,
          gradient_accumulation_steps = 4,
          warmup_steps = 5,
          #num_train_epochs = 1, # Set this for 1 full training run.
          #max_steps = None,
          max_steps = MAX_STEPS,
          learning_rate = 2e-4,
          fp16 = not is_bfloat16_supported(),
          bf16 = is_bfloat16_supported(),
          logging_steps = 1,
          optim = "adamw_8bit",
          weight_decay = 0.01,
          lr_scheduler_type = "linear",
          seed = 3407,
          output_dir = "outputs",
      ),
  )
  
  print("\n-------------------------------------------------------------")
  print(" ✅ Model Set up successfully")
  print(" Trainning MAX_STEPS : ", MAX_STEPS)
  
  ##
  print("\n-------------------------------------------------------------")
  print(" 📊 Memory stats before training :\n")
  
  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
  print(f"{start_gpu_memory} GB of memory reserved.")
  
  
  ##
  
  print("\n-------------------------------------------------------------")
  print(" 🌐 Start training Model :")
  print("-------------------------------------------------------------")
  
  trainer_stats = trainer.train()
  
  print("\n-------------------------------------------------------------")
  print(" ✅ training Model Finished ")
  print("-------------------------------------------------------------")
  
  #
  print("\n-------------------------------------------------------------")
  print(" 📊 Memory stats after training :\n")
  
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory         /max_memory*100, 3)
  lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
  print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
  print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
  print(f"Peak reserved memory = {used_memory} GB.")
  print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
  print(f"Peak reserved memory % of max memory = {used_percentage} %.")
  print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
  

  #
  print("\n-------------------------------------------------------------")
  print(" --- SIMPLE INFERENCE ---")
  print(" ▶️ PROMPT :\t Continue the fibonnaci sequence : 1, 1, 2, 3, 5, 8..")
  print(" 🤖 RESPONSE : ")
  # alpaca_prompt = Copied from above
  FastLanguageModel.for_inference(model) # Enable native 2x faster inference
  inputs = tokenizer(
  [
      alpaca_prompt.format(
          "Which country has the biggest GDP between", # instruction
          "USA, FRANCE, CHINA", # input
          "", # output - leave this blank for generation!
      )
  ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
  tokenizer.batch_decode(outputs)

  #
  print("\n-------------------------------------------------------------")
  print(" --- STREAM INFERENCE ---")
  print(f" ▶️ PROMPT :\t {TEST_PROMPT}")
  print(" 🤖 RESPONSE : ")

  # alpaca_prompt = Copied from above
  FastLanguageModel.for_inference(model) # Enable native 2x faster inference
  inputs = tokenizer(
  [
      alpaca_prompt.format(
          TEST_PROMPT, # instruction
          "", # input
          "", # output - leave this blank for generation!
      )
  ], return_tensors = "pt").to("cuda")

  #from transformers import TextStreamer
  text_streamer = TextStreamer(tokenizer)
  _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
  
  #
  print("\n-------------------------------------------------------------")
  print(" ✅ API HuggingFace 🗝️ for Saving Online:\n")
  #
  # HF TOKEN TO SAVE MODEL ONLINE
  #from google.colab import userdata
  HF_API = 'hf_**************************'
  
  print("\n-------------------------------------------------------------")
  print(" 💾 Saving Model to local and HuggingFace Online:\n")
  print(f" FINETUNED_LOCAL_MODEL :\t {FINETUNED_LOCAL_MODEL}")
  print(f" FINETUNED_ONLINE_MODEL 🗝️ :\t {FINETUNED_ONLINE_MODEL}")
  
  
  #
  model.save_pretrained(FINETUNED_LOCAL_MODEL) # Local saving
  tokenizer.save_pretrained(FINETUNED_LOCAL_MODEL)
  model.push_to_hub(FINETUNED_ONLINE_MODEL, token = HF_API) # Online saving
  tokenizer.push_to_hub(FINETUNED_ONLINE_MODEL, token = HF_API) # Online saving



# Run
finetune()
```

> OUTPUT :

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
-------------------------------------------------------------
 🛠 FINETUNE MODEL ON CUSTOM DATASET 📚 (UNSLOTH)
-------------------------------------------------------------

 ⚙️ Config Parameters : 
-------------------------------------------------------------
 SOURCE_MODEL :	 unsloth/Meta-Llama-3.1-8B
 DATASET :	 yahma/alpaca-cleaned
 FINETUNED_LOCAL_MODEL :	 llama3-1_finetuned_NVDA_L4
 FINETUNED_ONLINE_MODEL :	 0xZee/llama3-1_finetuned_NVDA_L4
 MAX_STEPS :	 444
 TEST_PROMPT :	 List the 10 last kings in France
-------------------------------------------------------------


==((====))==  Unsloth: Fast Llama patching release 2024.8
   \\   /|    GPU: NVIDIA L4. Max memory: 22.168 GB. Platform = Linux.
O^O/ \_/ \    Pytorch: 2.3.0+cu121. CUDA = 8.9. CUDA Toolkit = 12.1.
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.26.post1. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth

-------------------------------------------------------------
 ✅ Model and Tokenizer set up successfully 

-------------------------------------------------------------
 ✅ LoRA Adapters : LoRA to optimize finetuning 1% of parameters adapted 

-------------------------------------------------------------
 ✅ DataSet Loaded and splitted successfully
 📚 DataSet : https://huggingface.co/datasets/yahma/alpaca-cleaned

-------------------------------------------------------------
 Setting Trainning Model

-------------------------------------------------------------
 ✅ Model Set up successfully
 Trainning MAX_STEPS :  444

-------------------------------------------------------------
 📊 Memory stats before training :

GPU = NVIDIA L4. Max memory = 22.168 GB.
5.984 GB of memory reserved.

-------------------------------------------------------------
 🌐 Start training Model :
-------------------------------------------------------------
{'loss': 1.8225, 'grad_norm': 0.7166228890419006, 'learning_rate': 4e-05, 'epoch': 0.0}
{'loss': 2.2761, 'grad_norm': 1.0028027296066284, 'learning_rate': 8e-05, 'epoch': 0.0}
{'loss': 1.7235, 'grad_norm': 0.568682074546814, 'learning_rate': 0.00012, 'epoch': 0.0}
{'loss': 2.0109, 'grad_norm': 0.8697764873504639, 'learning_rate': 0.00016, 'epoch': 0.0}
{'loss': 1.7482, 'grad_norm': 0.7355954051017761, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 1.6531, 'grad_norm': 1.0432188510894775, 'learning_rate': 0.00019954441913439636, 'epoch': 0.0}
{'loss': 1.2153, 'grad_norm': 1.8806127309799194, 'learning_rate': 0.00019908883826879272, 'epoch': 0.0}
{'loss': 1.2499, 'grad_norm': 0.8655325174331665, 'learning_rate': 0.00019863325740318908, 'epoch': 0.0}
{'loss': 1.0765, 'grad_norm': 0.9465372562408447, 'learning_rate': 0.00019817767653758543, 'epoch': 0.0}
{'loss': 1.1294, 'grad_norm': 0.8750275373458862, 'learning_rate': 0.00019772209567198179, 'epoch': 0.0}
{'loss': 0.8993, 'grad_norm': 0.44296687841415405, 'learning_rate': 0.00019726651480637814, 'epoch': 0.0}
{'loss': 0.9178, 'grad_norm': 0.5278958678245544, 'learning_rate': 0.0001968109339407745, 'epoch': 0.0}
{'loss': 0.8858, 'grad_norm': 0.5283158421516418, 'learning_rate': 0.00019635535307517085, 'epoch': 0.0}
{'loss': 1.0239, 'grad_norm': 0.7033094763755798, 'learning_rate': 0.0001958997722095672, 'epoch': 0.0}
{'loss': 0.8508, 'grad_norm': 1.1653854846954346, 'learning_rate': 0.00019544419134396356, 'epoch': 0.0}
{'loss': 0.8677, 'grad_norm': 0.3868544399738312, 'learning_rate': 0.00019498861047835992, 'epoch': 0.0}
{'loss': 0.9766, 'grad_norm': 0.3736693263053894, 'learning_rate': 0.00019453302961275627, 'epoch': 0.0}
{'loss': 1.219, 'grad_norm': 0.40859293937683105, 'learning_rate': 0.00019407744874715263, 'epoch': 0.0}
{'loss': 0.9789, 'grad_norm': 0.48821914196014404, 'learning_rate': 0.00019362186788154898, 'epoch': 0.0}
{'loss': 0.8509, 'grad_norm': 0.38756129145622253, 'learning_rate': 0.00019316628701594534, 'epoch': 0.0}
{'loss': 0.8769, 'grad_norm': 0.5972719788551331, 'learning_rate': 0.0001927107061503417, 'epoch': 0.0}
{'loss': 0.9481, 'grad_norm': 0.6999387741088867, 'learning_rate': 0.00019225512528473805, 'epoch': 0.0}
{'loss': 0.849, 'grad_norm': 0.5360097885131836, 'learning_rate': 0.0001917995444191344, 'epoch': 0.0}
....
.....
......
{'loss': 0.9819, 'grad_norm': 0.3074074387550354, 'learning_rate': 1.366742596810934e-06, 'epoch': 0.07}
{'loss': 0.8503, 'grad_norm': 0.30303120613098145, 'learning_rate': 9.111617312072893e-07, 'epoch': 0.07}
{'loss': 0.7618, 'grad_norm': 0.2843836545944214, 'learning_rate': 4.5558086560364467e-07, 'epoch': 0.07}
{'loss': 0.8205, 'grad_norm': 0.30724790692329407, 'learning_rate': 0.0, 'epoch': 0.07}
{'train_runtime': 1664.6966, 'train_samples_per_second': 2.134, 'train_steps_per_second': 0.267, 'train_loss': 0.8831236177199596, 'epoch': 0.07}

-------------------------------------------------------------
 ✅ training Model Finished 
-------------------------------------------------------------

-------------------------------------------------------------
 📊 Memory stats after training :

1664.6966 seconds used for training.
27.74 minutes used for training.
Peak reserved memory = 7.258 GB.
Peak reserved memory for training = 1.274 GB.
Peak reserved memory % of max memory = 32.741 %.
Peak reserved memory for training % of max memory = 5.747 %.

-------------------------------------------------------------
 --- SIMPLE INFERENCE ---
 ▶️ PROMPT :	 Continue the fibonnaci sequence : 1, 1, 2, 3, 5, 8..
 🤖 RESPONSE : 

-------------------------------------------------------------
 --- STREAM INFERENCE ---
 ▶️ PROMPT :	 List the 10 last kings in France
 🤖 RESPONSE : 
<|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
  
  ### Instruction:
  List the 10 last kings in France
  
  ### Input:
  
  
  ### Response:
   1. Louis XVI (1774-1792)
2. Louis XV (1715-1774)
3. Louis XIV (1643-1715)
4. Louis XIII (1610-1643)
5. Henri IV (1589-1610)
6. François II (1559-1560)
7. Henri II (1547-1559)
8. François I (1515-1547)
9. Louis XII (1498-1515)
10. Louis XI (1461-1483)<|end_of_text|>

-------------------------------------------------------------
 ✅ API HuggingFace 🗝️ for Saving Online:


-------------------------------------------------------------
 💾 Saving Model to local and HuggingFace Online:

 FINETUNED_LOCAL_MODEL :	 llama3-1_finetuned_NVDA_L4
 FINETUNED_ONLINE_MODEL 🗝️ :	 0xZee/llama3-1_finetuned_NVDA_L4
Saved model to https://huggingface.co/0xZee/llama3-1_finetuned_NVDA_L4

```
