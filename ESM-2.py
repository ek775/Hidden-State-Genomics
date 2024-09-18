from transformers import AutoTokenizer, TFAutoModel

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = TFAutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# check the model architecture
print(model.summary())
esm_main_layer = model.layers[0]
encoder = esm_main_layer.encoder
#decoder = esm_main_layer.decoder
print(f"ESM-main-layer \n {esm_main_layer.config}")
print(f"Encoder \n {encoder.config}")
#print(f"Decoder \n {decoder.summary()}")