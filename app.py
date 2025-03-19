import gradio as gr
from utils.utils import GradioHandler

handler = GradioHandler()

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
                            minimum=1, maximum=7, value=5, step=1, interactive=True, label="Top K:", info="Number of retrieved chunks for RAG")
                        rag_options = gr.CheckboxGroup(choices=["reranking", "check hallucinations"],
                                                           value=["reranking"],
                                                           label="Choose Options")
                        rag_refresh = gr.Button(value="Refresh retriever")
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
            rag_refresh.click(fn=handler.process_selected_options,
                              inputs=[rag_options, rag_top_k_retrieval, chatbot],
                              outputs=[chatbot, input_txt], queue=False).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)

            txt_msg = audio_submit_btn.click(fn=handler.respond,
                                             inputs=[chatbot, input_txt, input_audio_block],
                                             outputs=[chatbot, input_txt,
                                                      ref_output],
                                             queue=False).then(lambda: gr.Textbox(interactive=True),
                                                               None, [input_txt], queue=False)

            txt_msg = input_txt.submit(fn=handler.respond,
                                         inputs=[chatbot, input_txt],
                                         outputs=[chatbot, input_txt,
                                                  ref_output],
                                         queue=False).then(lambda: gr.Textbox(interactive=True),
                                                           None, [input_txt], queue=False)

            file_msg = upload_btn.upload(fn=handler.process_uploaded_files, inputs=[upload_btn, chatbot],
                                         outputs=[chatbot, input_txt], queue=False).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)


demo.launch()


