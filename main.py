from textbase import bot, Message
# from textbase.models import OpenAI
from typing import List
import os

from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent
from langchain import OpenAI
import gpt.main2 as main2
import regex as re

# Load your OpenAI API key

os.environ["OPENAI_API_KEY"] = ""

llm = OpenAI(temperature=0)

# Creating Main Agent
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=main2.tools ,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=main2.memory,
)


import regex as re
a = re.search(r'TOOLS:', conversational_agent.agent.llm_chain.prompt.template)
conversational_agent.agent.llm_chain.prompt.template=main2.new_template+conversational_agent.agent.llm_chain.prompt.template[a.start():]



@bot()
def on_message(message_history: List[Message], state: dict = None):
    
    bot_response = conversational_agent.run(message_history[-1]["content"][0]['value']) 
    response = {
        "data": {
            "messages": [
                {
                    "data_type": "STRING",
                    "value": bot_response
                }
            ],
            "state": state
        },
        "errors": [
            {
                "message": ""
            }
        ]
    }

    return {
        "status_code": 200,
        "response": response
    }