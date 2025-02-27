import gradio as gr
from utils.utils import respond, process_uploaded_files
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Chatbot"):
            ###############
            # Main App row:
            ###############
            with gr.Row() as app_row:
                with gr.Column(scale=1) as left_column:
                    with gr.Accordion("RAG Parameters", open=False):
                        rag_top_k_retrieval = gr.Slider(
                            minimum=0, maximum=5, value=2, step=1, interactive=True, label="Top K:", info="Number of retrieved chunks for RAG")
                        rag_search_type = gr.Dropdown(
                            choices=["Similarity search", "mmr"],
                            label="Select the search technique:",
                            info="Both methods will be applied to RecursiveCharacterSplitter",
                            value="Similarity search",
                            interactive=True
                        )
                    input_audio_block = gr.Audio(
                        sources=["microphone"],
                        label="Submit your query using voice",
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=2,
                            show_controls=True,
                        ),
                    )
                    audio_submit_btn = gr.Button(value="Submit audio")
                with gr.Column(scale=8) as right_column:
                    with gr.Row() as row_one:
                        with gr.Column(visible=False) as reference_bar:
                            ref_output = gr.Markdown(
                                label="RAG Reference Section")

                        with gr.Column() as chatbot_output:
                            chatbot = gr.Chatbot(
                                [],
                                elem_id="chatbot",
                                bubble_full_width=False,
                                height=500,
                                avatar_images=("images/user.jpg",
                                    "images/Llama.png"),
                                # render=False
                            )
                        # with gr.Column(visible=False) as full_image:
                        #     image_output = gr.Image()
                        #     # **Adding like/dislike icons
                        #     chatbot.like(UISettings.feedback, None, None)
                    ##############
                    # SECOND ROW:
                    ##############
                    with gr.Row():
                        input_txt = gr.MultimodalTextbox(interactive=True, lines=2, file_types=[
                                                         "image"], placeholder="Enter message or upload file...", show_label=False)
                    ##############
                    # Third ROW:
                    ##############
                    with gr.Row() as row_two:
                        upload_btn = gr.UploadButton(
                            "📁 Upload PDF or doc files for RAG", file_types=[
                                '.pdf',
                                '.doc'
                            ],
                            file_count="multiple")
                        clear_button = gr.ClearButton([input_txt, chatbot])

            #############
            # Process:
            #############

            txt_msg = audio_submit_btn.click(fn=respond,
                                             inputs=[chatbot, input_txt, input_audio_block],
                                             outputs=[chatbot, input_txt,
                                                      ref_output],
                                             queue=False).then(lambda: gr.Textbox(interactive=True),
                                                               None, [input_txt], queue=False)

            txt_msg = input_txt.submit(fn=respond,
                                         inputs=[chatbot, input_txt],
                                         outputs=[chatbot, input_txt,
                                                  ref_output],
                                         queue=False).then(lambda: gr.Textbox(interactive=True),
                                                           None, [input_txt], queue=False)

            file_msg = upload_btn.upload(fn=process_uploaded_files, inputs=[upload_btn, chatbot],
                                         outputs=[chatbot, input_txt], queue=False).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)


demo.launch()
exit()


response = s.invoke({"messages": ["Hello, my name is Misha"]}, config)
print(response)
print(response["messages"][-1])

response = s.invoke({"messages": ["What is my name?"]}, config)
print(response)
print(response["messages"][-1])

response = s.invoke({"messages": ["Whats the difference between titan architecture and transformers?"]}, config)
print(response)
print(response["messages"][-1])

response = s.invoke({"messages": ["What were we just talking about?"]}, config)
print(response)
print(response["messages"][-1])

print(memory.get(config)['channel_values']['messages'])


exit()

response = s.invoke({"messages": ["Whats the difference between titan architecture and transformers?"]}, config)
print(response)
print(response["messages"][-1])



from agents.main_graph import create_supervisor
s = create_supervisor(llm, tools)
response = s.invoke({"messages": ["Write an essay on this topic: Advantages and Limitations of Transformers Compared to RNNs and CNNs"]})
print(response)

print(response["messages"][-1])
print(response["messages"][-2])
exit()

from agents.sub_graph import ChatAgent
g = ChatAgent(llm).graph


from agents.sub_graph import AgenticRAG
agent = AgenticRAG(llm, tools).graph


response = agent.invoke({"messages": ["Whats the difference between titan architecture and transformers?"]})
print(response["messages"][-1])
exit()

from agents.sub_graph import EssayWriter
d = EssayWriter(llm, agent).graph
response = d.invoke({"task": "Advantages and Limitations of Transformers Compared to RNNs and CNNs"})
print(response["draft"])
exit()

query = "Whats the difference between titan architecture and transformers?"
query1 = "Give me 5 reasons to visit Canada"
gquery = "How Does the Attention Mechanism Work and Why Is It So Important for LLMs?"
response = agent.invoke({"messages": [query1]})
print(response["messages"][-1].content)

