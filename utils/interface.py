import gradio as gr
gr.processing_utils._check_allowed = lambda path, allowed_paths: True
import io
import os
import time
import uuid
import traceback
import soundfile as sf
import torchaudio
import torch
from transformers import AutoModel, AutoProcessor, GenerationConfig, StoppingCriteria
from dataclasses import astuple
import sys


class MIMOStopper(StoppingCriteria):
    def __init__(self, stop_id: int) -> None:
        super().__init__()
        self.stop_id = stop_id

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:
        # Stop when last token of channel 0 is the stop token
        return input_ids[0, -1].item() == self.stop_id


class Inference:
    def __init__(self, model_path, codec_path=None, device='cuda'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            codec_path=codec_path if codec_path else "fnlp/MOSS-Speech-Codec",
            device=self.device,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto"
        ).eval()

    def forward(
        self,
        task: str,
        conversation_history_for_model: list, # Pass the entire conversation history formatted for the model
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        min_new_tokens: int,
        top_k: int,
        system_prompt: str,
        decoder_audio_prompt_path: str = None
    ):
        # Prepare the conversation for the processor
        full_conversation = []
        if system_prompt:
            full_conversation.append({"role": "system", "content": system_prompt})
        
        
        # Add previous turns from the formatted history
        full_conversation.extend(conversation_history_for_model)

        output_modalities = []
        if task.endswith("speech_response"):
            output_modalities.append('audio')
        if task.endswith("text_response"):
            output_modalities.append('text')

        # This should always be exactly one modality based on task
        if len(output_modalities) != 1: 
            raise ValueError("Expected exactly one output modality based on task.")

        inputs = self.processor([full_conversation], output_modalities)

        stopping_criteria = [
            MIMOStopper(self.processor.tokenizer.pad_token_id),
            MIMOStopper(
                self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ),
        ]
        
        generate_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": True, # Always true for these parameters
            "use_cache": True,
            "top_k": top_k,
        }
        generation_config = GenerationConfig(**generate_kwargs)

        token_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            generation_config=generation_config,
            stopping_criteria=stopping_criteria
        )

        results = self.processor.decode(
            token_ids.to(self.device),
            output_modalities,
            decoder_audio_prompt_path=decoder_audio_prompt_path
        )
        
        # As per requirement, always one output modality, so take the first result
        response_obj = results[0] 

        text_out = None
        audio_out = None

        if output_modalities[0] == 'audio':
            audio_out = (response_obj.sampling_rate, response_obj.audio.squeeze(0).cpu().numpy()) if response_obj.audio is not None else None
        elif output_modalities[0] == 'text':
            text_out = response_obj.generated_text if response_obj.generated_text is not None else None

        # Clean up temporary user audio file if it was created (only temporary for processor)
        # if temp_user_audio_path and os.path.exists(temp_user_audio_path):
        #     os.remove(temp_user_audio_path)

        return text_out, audio_out


class MIMOInterface:
    def __init__(self, model_path):
        self.inference = Inference(model_path, codec_path="fnlp/MOSS-Speech-Codec")
        self.audio_dir = "chat_audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        self.default_decoder_audio_prompt_path = "./assets/prompt_cn.wav"


    # ---------- Helpers ----------

    def get_system_prompt_default(self, task):
        if task.endswith("speech_response"):
            return "You are a helpful voice assistant. Answer the user's questions with spoken responses."
        elif task.endswith("text_response"):
            return "You are a helpful assistant. Answer the user's questions with text."
        else:
            return "You are a helpful assistant."

    def _unique_wav_path(self, prefix: str) -> str:
        return os.path.join(self.audio_dir, f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.wav")

    def _save_audio_numpy(self, audio_np_tuple, prefix="audio") -> str:
        """
        audio_np_tuple: (sample_rate, np.ndarray)
        Returns local .wav path.
        """
        if audio_np_tuple is None:
            return ""
        
        sr, arr = audio_np_tuple
        if len(arr.shape) > 1:
            arr = arr[:, 0] # Ensure mono
        path = self._unique_wav_path(prefix)
        sf.write(path, arr, sr, format="WAV")
        return path

    def _delete_audio_files(self, file_paths: list):
        """Deletes a list of audio files."""
        for path in file_paths:
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error deleting audio file {path}: {e}")

    # ---------- Core inference + chat sync ----------

    def process_input(
        self,
        audio_input,
        text_input,
        mode,
        temperature,
        top_p,
        repetition_penalty,
        max_new_tokens,
        min_new_tokens,
        top_k,
        system_prompt,
        history_state_tuple, # (chatbot_messages, audio_file_paths_to_delete, conversation_for_model)
        decoder_audio_prompt # numpy tuple from gradio audio component
    ):
        chatbot_messages, audio_file_paths_to_delete, conversation_for_model = history_state_tuple
        # Keep a copy of the state before any changes in case of warning/error
        original_chatbot_messages = list(chatbot_messages)
        original_audio_file_paths_to_delete = list(audio_file_paths_to_delete)
        original_conversation_for_model = list(conversation_for_model)

        # new_chatbot_message = []
        try:
            # --- Handle Decoder Audio Prompt ---
            decoder_audio_prompt_path_for_model = None
            if decoder_audio_prompt is not None:
                saved_decoder_audio_path = self._save_audio_numpy(decoder_audio_prompt, prefix="decoder_prompt")
                audio_file_paths_to_delete.append(saved_decoder_audio_path)
                decoder_audio_prompt_path_for_model = saved_decoder_audio_path
            else:
                decoder_audio_prompt_path_for_model = self.default_decoder_audio_prompt_path


            # --- Prepare User Input for Model and Display ---
            user_display_message_content = "" 
            user_audio_path_display = None 
            current_user_turn_for_model = None 

            if mode.startswith("speech_instruct"):
                if audio_input is None:
                    gr.Warning("Speech Input mode requires an audio input.")
                    return original_chatbot_messages[-1][1][0] if original_chatbot_messages else "", None, original_chatbot_messages, history_state_tuple # Return previous state
                else:
                    user_audio_path_display = self._save_audio_numpy(audio_input, prefix="user")
                    audio_file_paths_to_delete.append(user_audio_path_display)
                    user_display_message_content = "🎤 Voice message" # Consistent text for speech input

                    buffer = io.BytesIO()
                    sf.write(buffer, audio_input[1], audio_input[0], format="WAV")
                    buffer.seek(0)
                    current_user_turn_for_model = {"role": "user", "content": {'path': user_audio_path_display, 'type': 'audio/wav'}}
            else: # Text instruct modes
                txt = (text_input or "").strip()
                if not txt:
                    gr.Warning("Text Input mode requires a text input.")
                    return original_chatbot_messages[-1][1][0] if original_chatbot_messages else "", None, original_chatbot_messages, history_state_tuple # Return previous state
                else:
                    user_display_message_content = txt
                    current_user_turn_for_model = {"role": "user", "content": user_display_message_content}

            # Add user input to chatbot messages and model's conversation history
            # Always add a single entry for user turn in chatbot_messages
            if user_audio_path_display:
                # chatbot_messages.append([user_display_message_content, None]) 
                # new_chatbot_message.append([None, gr.Audio(user_audio_path_display, type='audio/wav')])
                chatbot_messages.append({'role': 'user', 'content': {'path': user_audio_path_display}})
            else:
                chatbot_messages.append({'role': 'user', 'content': user_display_message_content})
            
            if current_user_turn_for_model:
                conversation_for_model.append(current_user_turn_for_model)

            # --- Run Inference ---
            text_out, audio_out = self.inference.forward(
                task=mode,
                conversation_history_for_model=conversation_for_model,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                system_prompt=system_prompt,
                decoder_audio_prompt_path=decoder_audio_prompt_path_for_model
            )

            # --- Process Assistant Output for Display and Model History ---
            assistant_response_for_model_content = None # This will be string or dict for model history
            final_text_output_panel = None
            assistant_audio_output_panel = None
            
            # Assistant text for display/chatbot
            assistant_text_display = None
            assistant_audio_path_display = None

            if mode.endswith("speech_response"):
                if audio_out is None:
                    gr.Warning("Model failed to generate speech response.")
                    # Restore original history state if generation failed
                    return original_chatbot_messages[-1][1][0] if original_chatbot_messages else "", None, original_chatbot_messages, (original_chatbot_messages, original_audio_file_paths_to_delete, original_conversation_for_model)

                assistant_audio_output_panel = audio_out 
                saved_assistant_audio_path = self._save_audio_numpy(audio_out, prefix="assistant")
                audio_file_paths_to_delete.append(saved_assistant_audio_path)
                assistant_audio_path_display = saved_assistant_audio_path
                
                # Chatbot message for speech response mode
                # The text part is usually not needed, but can be a placeholder or empty
                # chatbot_messages.append(["🔊 Generated speech.", None]) 
                # new_chatbot_message.append([None, gr.Audio(assistant_audio_path_display, type="filepath")])
                chatbot_messages.append({'role': 'assistant', 'content': {'path': assistant_audio_path_display}})
                assistant_response_for_model_content = {'path': saved_assistant_audio_path, 'type': 'filepath'}

            elif mode.endswith("text_response"):
                if text_out is None or str(text_out).strip() == "":
                    gr.Warning("Model failed to generate text response.")
                    # Restore original history state if generation failed
                    return original_chatbot_messages[-1][1][0] if original_chatbot_messages else "", None, original_chatbot_messages, (original_chatbot_messages, original_audio_file_paths_to_delete, original_conversation_for_model)

                final_text_output_panel = text_out
                assistant_text_display = text_out

                # Chatbot message for text response mode
                chatbot_messages.append({'role': 'assistant', 'content': assistant_text_display})
                assistant_response_for_model_content = text_out

            # Add assistant's actual response to the conversation for the next model turn
            if assistant_response_for_model_content:
                conversation_for_model.append({"role": "assistant", "content": assistant_response_for_model_content})
            
            # Return updated history state tuple
            new_history_state_tuple = (chatbot_messages, audio_file_paths_to_delete, conversation_for_model)
            # Return panel outputs + chat + state
            return final_text_output_panel, assistant_audio_output_panel, chatbot_messages, new_history_state_tuple

        except Exception as e:
            traceback.print_exc()
            err = f"Error: {str(e)}"
            gr.Error(f"An unexpected error occurred: {err}")
            # Restore original history state on any unhandled exception
            return original_chatbot_messages[-1][0] if original_chatbot_messages else "", None, original_chatbot_messages, (original_chatbot_messages, original_audio_file_paths_to_delete, original_conversation_for_model)

    def _submit_with_clear(
        self, audio_in, text_in, mode, temperature, top_p, repetition_penalty, max_new_tokens, min_new_tokens, top_k,
        system_prompt, history_state, decoder_audio_prompt, clear_on_submit
    ):
        if clear_on_submit:
            _, audio_files, _ = history_state
            self._delete_audio_files(audio_files)
            history_state = ([], [], [])
        return self.process_input(
            audio_in, text_in, mode, temperature, top_p, repetition_penalty,
            max_new_tokens, min_new_tokens, top_k, system_prompt,
            history_state, decoder_audio_prompt
        )
        
    # ---------- UI factory ----------

    def create_interface(self):
        theme = gr.themes.Soft()

        with gr.Blocks(theme=theme) as demo:
            gr.HTML(
                """
                <div class="main-header">
                    <h1>🎤 MOSS-Speech Demo</h1>
                </div>
                """
            )


            mode = gr.Radio(
                [
                    ("Speech In → Speech Out", "speech_instruct_speech_response"),
                    ("Speech In → Text Out", "speech_instruct_text_response"),
                    ("Text In → Speech Out", "text_instruct_speech_response"),
                    ("Text In → Text Out", "text_instruct_text_response"),
                ],
                label="🎯 Interaction Mode",
                value="speech_instruct_speech_response",
                container=True,
                scale=1,
            )

            system_prompt = gr.Textbox(
                label="🤖 System Prompt",
                value=self.get_system_prompt_default("speech_instruct_speech_response"),
                lines=2,
                container=True,
                scale=1,
            )

            with gr.Accordion("⚙️ Generation Parameters", open=False, elem_classes="param-accordion"):
                with gr.Row():
                    temperature = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="🌡️ Temperature", info="Higher = more random")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="🎯 Top-p", info="Nucleus sampling")
                    top_k = gr.Slider(1, 100, value=20, step=1, label="🔝 Top-k", info="Candidate tokens")

                with gr.Row():
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="🔄 Repetition Penalty", info="Discourage repeats")
                    max_new_tokens = gr.Slider(1, 2000, value=500, step=1, label="📝 Max New Tokens", info="Upper bound")
                    min_new_tokens = gr.Slider(0, 100, value=0, step=1, label="📏 Min New Tokens", info="Lower bound")
                    
                decoder_audio_prompt = gr.Audio(type="numpy", value="assets/prompt_cn.wav", label="🎙️ Decoder Audio Prompt (Optional)", visible=True)

            with gr.Row():
                with gr.Column(scale=1, elem_classes="input-section"):
                    gr.Markdown("### 📥 Input")

                    audio_input = gr.Audio(type="numpy", label="🎙️ Speech Input", visible=True)

                    text_input = gr.Textbox(
                        label="🧾 Text Input",
                        placeholder="Type your question here…",
                        lines=3,
                        info="Enter text to query the assistant",
                        visible=False,
                    )

                with gr.Column(scale=1, elem_classes="output-section"):
                    gr.Markdown("### 📤 Output")

                    text_output = gr.Textbox(
                        label="📄 Text Output",
                        lines=8,
                        interactive=False,
                        info="Model-generated text response",
                        visible=False,
                    )

                    audio_output = gr.Audio(label="🔊 Speech Output", visible=True, autoplay=False)
                    
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary", elem_classes="btn-primary")
                clear_history_btn = gr.Button("🗑️ Clear All History", variant="secondary", elem_classes="btn-secondary")
                
            with gr.Row():
                clear_history_on_mode_change_checkbox = gr.Checkbox(
                    label="Clear history on mode change", value=True, interactive=True
                )
                clear_history_on_submit_checkbox = gr.Checkbox(
                    label="Clear history on each submit", value=False, interactive=True
                )

            # history_state will now be a tuple: (chatbot_messages, audio_file_paths_to_delete, conversation_for_model)
            history_state = gr.State(([], [], [])) 
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=True,
                type="messages", # Keep commented to allow [text, audio] in chatbot
                scale=1,
                label="💬 Chat History",
                show_copy_button=True
            )


            # ---------- Event handlers ----------

            submit_btn.click(
                fn=self._submit_with_clear,
                inputs=[
                    audio_input,
                    text_input,
                    mode,
                    temperature,
                    top_p,
                    repetition_penalty,
                    max_new_tokens,
                    min_new_tokens,
                    top_k,
                    system_prompt,
                    history_state, # Pass the current Gradio state tuple
                    decoder_audio_prompt,
                    clear_history_on_submit_checkbox
                ],
                outputs=[text_output, audio_output, chatbot, history_state],
            )

            def _hard_clear(current_history_state_tuple):
                _, audio_files, _ = current_history_state_tuple
                self._delete_audio_files(audio_files)
                gr.Info("Conversation history and associated audio files cleared.")
                return "", None, [], ([], [], [])

            clear_history_btn.click(
                fn=_hard_clear,
                inputs=[history_state],
                outputs=[text_output, audio_output, chatbot, history_state],
            )

            def update_interface_visibility(selected_mode):
                if selected_mode.startswith("speech_instruct"):
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            def update_output_visibility(selected_mode):
                if selected_mode.endswith("speech_response"):
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=True), gr.update(visible=False)

            def _on_mode_change(task, clear_history_on_mode_change, current_history_state_tuple):
                if clear_history_on_mode_change:
                    _, audio_files_to_delete, _ = current_history_state_tuple
                    self._delete_audio_files(audio_files_to_delete)
                    gr.Info("Interaction mode changed. History cleared.")
                    return self.get_system_prompt_default(task), [], ([], [], [])
                else:
                    gr.Info("Interaction mode changed. History preserved.")
                    # Keep existing chatbot messages and state
                    chatbot_messages, audio_files, conv_state = current_history_state_tuple
                    return self.get_system_prompt_default(task), chatbot_messages, (chatbot_messages, audio_files, conv_state)

            mode.change(
                fn=_on_mode_change,
                inputs=[mode, clear_history_on_mode_change_checkbox, history_state],
                outputs=[system_prompt, chatbot, history_state],
            )
            mode.change(
                fn=update_interface_visibility,
                inputs=[mode],
                outputs=[audio_input, text_input],
            )
            mode.change(
                fn=update_output_visibility,
                inputs=[mode],
                outputs=[text_output, audio_output],
            )

        return demo

if __name__ == "__main__":
    model_path = "fnlp/MOSS-Speech"
    
    interface = MIMOInterface(model_path)
    demo = interface.create_interface()
    demo.launch()
