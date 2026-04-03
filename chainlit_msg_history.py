import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM

'''
llm = AutoModelForCausalLM.from_pretrained(
     "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    # "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf"
)
'''

def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n" # orca
    # prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"         # this is for llama
    # print(f"Prompt created: {prompt}")
    return prompt

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()
    
    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        #"TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf"
)
    print("A new chat session has started!")

'''
history = []

question = "Which city is the capital of india?"
prompt = get_prompt(question)
answer = ""
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)

question = "And which is of the United States?"
prompt = get_prompt(question, history)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
'''