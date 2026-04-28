# Step to follow:
# Before interacting with your model, you need to initialize an object where you can store your conversation history.

# Initialize object to store conversation history
# Afterward, you'll do the following for each interaction with the model:
# Encode conversation history as a string
# Fetch prompt from user
# Tokenize (optimize) prompt
# Generate output from the model using prompt and history
# Decode output
# Update conversation history


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


conversation_history = []


history_string = "\n".join(conversation_history)

input_text ="hello, how are you doing?"


inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)


tokenizer.pretrained_vocab_files_map

outputs = model.generate(**inputs)
print(outputs)



response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)



#  Update conversation history

conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)



# Repeat
# Now, you can put everything in a loop and run a whole conversation!


while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
