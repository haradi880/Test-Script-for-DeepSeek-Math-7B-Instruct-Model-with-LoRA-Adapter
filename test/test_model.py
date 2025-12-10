"""Test Script for DeepSeek-Math 7B Instruct Model with LoRA Adapter
this is the code build by haradibots only for testing purpose

This script loads the DeepSeek-Math 7B Instruct model along with a LoRA adapter
and tests it on a challenging mathematical problem.
It demonstrates how to set up the model for inference and generate responses
to complex questions.

to use this script, ensure you have the required libraries installed:
- transformers
- peft
- torch
- kagglehub

and the link of the model is given below:
 base model: /kaggle/input/deepseek-math/pytorch/deepseek-math-7b-instruct/1
 lora adapter: /kaggle/input/math-lora-deepeseek7b-math-instruct/transformers/default/1/lora_adapter
 
"""





import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


####################  Configuration ####################


"""
this path only work in kaggle so if you are using locally then change the path accordingly
and also make sure to download the model and lora adapter from the given link above

"""
BASE_MODEL_PATH = "/kaggle/input/deepseek-math/pytorch/deepseek-math-7b-instruct/1"
LORA_ADAPTER_PATH = "/kaggle/input/math-lora-deepeseek7b-math-instruct/transformers/default/1/lora_adapter"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


############### Load base model and tokenizer ###############

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


################ Load LoRA adapter ################

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# Merge adapter for faster inference (optional)
print("Merging adapter for faster inference...")
model = model.merge_and_unload()

# Or keep it separate for testing:
# model = model.eval()


################### Test Function ###################

def test_model(question, max_new_tokens=4096):
    """Test the model with a question."""
    
    # Format prompt (same as training)
    prompt = f"Question:\n{question}\n\nSolution:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    print(f"\n{'='*50}")
    print(f"Testing model with question:")
    print(f"{question}")
    print(f"{'='*50}\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and display
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    prompt_length = len(tokenizer.encode(prompt))
    generated_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("Model's full response:")
    print("-" * 50)
    print(full_response)
    print("-" * 50)
    
    return response

if __name__ == "__main__":
    # Test 3: This is an extremely challenging problem from advanced competitive mathematics, combining Euclidean Geometry (specifically triangle centers, circles, reflections, and concurrency/cyclicity conditions) with Number Theory (Fibonacci sequences) and Calculus (finding a limit of a sequence)
    test_question_1 = "Compute Let ABC be a triangle with AB ≠ AC, circumcircle Ω, and incircle ω. Let the contact points of ω with BC, CA, and AB be D, E, and F, respectively. Let the circumcircle of AFE meet Ω at K and let the reflection of K in EF be K′. Let N denote the foot of the perpendicular from D to EF. The circle tangent to line BN and passing through B and K intersects BC again at T ≠ B.\n\nLet the sequence (F_n) be defined by F_0 = 0, F_1 = 1, and for n ≥ 2, F_n = F_{n−1} + F_{n−2}. Call triangle ABC n-tastic if BD = F_n, CD = F_{n+1}, and quadrilateral KNK′B is cyclic.\n\nAcross all n-tastic triangles, let a_n denote the maximum possible value of (CT·NB)/(BT·NE). Let α be the smallest real number such that for all sufficiently large n, a_{2n} < α. Given α = p + √q for rationals p and q, what is the remainder when p^{q·p} is divided by 99991?"
    result1 = test_model(test_question_1)



# End of script

""" build by haradibots with love """