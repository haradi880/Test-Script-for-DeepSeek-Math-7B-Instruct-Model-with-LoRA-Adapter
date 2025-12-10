# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the LoRA Adapter using Kagglehub
# This will download the adapter to your local cache
python -c "import kagglehub; print(kagglehub.model_download('haradibots/math-solving-hard-lora/transformers/default'))"

# NOTE: Update the LORA_ADAPTER_PATH in inference.py 
# to the path printed by the command above!

# 3. Run the inference script
python test/test_model.py