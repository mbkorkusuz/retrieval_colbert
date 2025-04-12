import chainlit as cl
from main import main


@cl.on_message
async def chat(message):
    app_instance = main()
    user_input = message.content


    #response = app_instance.invoke({"question": user_input, "history": cl.chat_context.to_openai()})
    response = app_instance.invoke({"question": user_input})

    chatbot_response = response.get("generation")

    msg = cl.Message(content= "")
    await msg.send()

    for char in chatbot_response:
        await msg.stream_token(char)
        await cl.sleep(0.01)

    await msg.update()