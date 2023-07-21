
# import streamlit as st
# import torch
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer

# @st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
# def load_data():    
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
#     return tokenizer, model
# tokenizer, model = load_data()

# st.write("Welcome to the Chatbot")
# input = st.text_input('You:')

# if 'count' not in st.session_state or st.session_state.count == 6:
#  st.session_state.count = 0 
#  st.session_state.chat_history_ids = None
#  st.session_state.old_response = ''
# else:
#  st.session_state.count += 1


#  new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

# if st.session_state.count > 1:
#     bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
# else:
#     bot_input_ids = new_user_input_ids

# st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

# response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# if st.session_state.old_response == response:
#    bot_input_ids = new_user_input_ids
 
#    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
#    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# st.write(f"Chatbot: {response}")
         
# st.session_state.old_response = response

# import streamlit as st
# import torch
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer

# @st.cache_data(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
# def load_data():    
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
#     return tokenizer, model

# tokenizer, model = load_data()

# st.write("Welcome to the Chatbot")
# input = st.text_input('You:')

# if 'count' not in st.session_state or st.session_state.count == 6:
#  st.session_state.count = 0 
#  st.session_state.chat_history_ids = None
#  st.session_state.old_response = ''
# else:
#  st.session_state.count += 1

# new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

# if st.session_state.count > 1:
#     bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
# else:
#     bot_input_ids = new_user_input_ids

# st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

# response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# if st.session_state.old_response == response:
#    bot_input_ids = new_user_input_ids
 
#    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
#    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# st.write(f"Chatbot: {response}")
         
# st.session_state.old_response = response

import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress Streamlit warnings related to file uploader encoding (deprecated)
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_data(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash})
def load_data():    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_data()

st.write("Welcome to the Chatbot")
input_text = st.text_input('You:')

if 'count' not in st.session_state or st.session_state.count == 6:
    st.session_state.count = 0 
    st.session_state.chat_history_ids = None
    st.session_state.old_response = ''
else:
    st.session_state.count += 1

new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

if st.session_state.count > 1:
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
else:
    bot_input_ids = new_user_input_ids

st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if st.session_state.old_response == response:
    bot_input_ids = new_user_input_ids
 
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

st.write(f"Chatbot: {response}")
         
st.session_state.old_response = response
