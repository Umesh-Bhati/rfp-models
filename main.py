from unsloth import FastLanguageModel, is_bfloat16_supported

import json 
from datasets import Dataset


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# Load and prepare your dataset


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

with open('./alpaca_data_cleaned.json', 'r') as f:
     data = json.load(f)


# alpaca_prompt = """### Government Audit Services Proposal Context
# You are a professional government audit services consultant tasked with preparing a comprehensive proposal in response to a detailed Request for Proposal (RFP).

# ### Request for Proposal (RFP):
# {rfp}

# ### Proposal Generation Instructions:
# - Analyze the RFP requirements meticulously
# - Develop a precise, compliant, and competitive proposal
# - Demonstrate technical expertise and understanding of audit scope
# - Provide clear, structured, and professional response

# ### Generated Proposal Response:
# {proposal}

# ### Evaluation Criteria Focus:
# - Technical Competence
# - Compliance with Government Standards
# - Cost-Effectiveness
# - Audit Methodology Clarity"""

# def formatting_prompts_func(item):
#     texts = []

   
#     rfp = item["rfp"][0].strip() if item['rfp'] else ''
#     proposal = " ".join(filter(bool, [
#     f"Proposal Content: {item['proposal'][0]}" if item["proposal"] else '',
#     f"Technical Proposal Conten: {item["technical_proposal"][0]}" if item['technical_proposal'] else '',
#     f"Cost Proposal Content: {item["cost_proposal"][0]}" if item['cost_proposal'] else ''
# ])).strip()

#     if rfp and proposal:
#             text = alpaca_prompt.format(
#                 rfp=rfp, 
#                 proposal=proposal
#             ) + tokenizer.eos_token
#             texts.append(text)      
    
#     return {"text": texts}


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

# Dataset processing
dataset = Dataset.from_list(data)
dataset = dataset.map(
    formatting_prompts_func, 
    batched=True, 
    remove_columns=dataset.column_names
)

# Initialize the model

# Apply PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Set up the trainer
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

training_args = SFTTrainer(
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
        max_steps = 60, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", 
        # Use this for WandB etc
    ),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

