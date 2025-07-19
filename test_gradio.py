import gradio as gr
import time
from datetime import datetime
from typing import List, Tuple

def simple_chatbot(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Simple chatbot that echoes messages with some basic responses."""
    
    # Simple responses based on keywords
    message_lower = message.lower()
    
    if "hello" in message_lower or "hi" in message_lower:
        response = f"Hello! How can I help you today? 😊"
    elif "how are you" in message_lower:
        response = "I'm doing great, thank you for asking! How are you?"
    elif "time" in message_lower:
        current_time = datetime.now().strftime("%H:%M:%S")
        response = f"The current time is {current_time}"
    elif "date" in message_lower:
        current_date = datetime.now().strftime("%Y-%m-%d")
        response = f"Today's date is {current_date}"
    elif "bye" in message_lower or "goodbye" in message_lower:
        response = "Goodbye! Have a great day! 👋"
    elif "weather" in message_lower:
        response = "I don't have access to real weather data, but I hope it's nice where you are! ☀️"
    elif "help" in message_lower:
        response = """I can help you with:
        - Greeting you
        - Telling you the time/date
        - Having a simple conversation
        - Just ask me anything!"""
    else:
        # Echo the message with some variation
        responses = [
            f"You said: '{message}'. That's interesting!",
            f"I heard you say '{message}'. Tell me more!",
            f"'{message}' - that's a good point. What else would you like to discuss?",
            f"Thanks for sharing: '{message}'. How can I help you further?",
        ]
        import random
        response = random.choice(responses)
    
    # Add to history
    history.append((message, response))
    
    return "", history

def clear_chat():
    """Clear the chat history."""
    return []

def simple_calculator(num1: float, operation: str, num2: float) -> str:
    """Simple calculator for the second tab."""
    try:
        if operation == "Add (+)":
            result = num1 + num2
            return f"{num1} + {num2} = {result}"
        elif operation == "Subtract (-)":
            result = num1 - num2
            return f"{num1} - {num2} = {result}"
        elif operation == "Multiply (×)":
            result = num1 * num2
            return f"{num1} × {num2} = {result}"
        elif operation == "Divide (÷)":
            if num2 == 0:
                return "Error: Cannot divide by zero!"
            result = num1 / num2
            return f"{num1} ÷ {num2} = {result}"
        else:
            return "Please select a valid operation"
    except Exception as e:
        return f"Error: {str(e)}"

def text_analyzer(text: str) -> str:
    """Analyze text and return statistics."""
    if not text.strip():
        return "Please enter some text to analyze."
    
    # Basic text analysis
    word_count = len(text.split())
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    sentence_count = len([s for s in text.split('.') if s.strip()])
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Most common words (simple implementation)
    words = text.lower().split()
    word_freq = {}
    for word in words:
        word_clean = word.strip('.,!?";:')
        if len(word_clean) > 2:  # Only count words longer than 2 characters
            word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
    
    # Get top 5 most common words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    analysis = f"""📊 **Text Analysis Results:**

📈 **Basic Statistics:**
- Words: {word_count}
- Characters (with spaces): {char_count}
- Characters (without spaces): {char_count_no_spaces}
- Sentences: {sentence_count}
- Paragraphs: {paragraph_count}

🔤 **Top 5 Most Common Words:**"""
    
    for i, (word, count) in enumerate(top_words, 1):
        analysis += f"\n{i}. '{word}' - {count} times"
    
    if not top_words:
        analysis += "\nNo significant words found."
    
    return analysis

def create_interface():
    """Create the main Gradio interface with tabs."""
    
    with gr.Blocks(
        title="Simple Gradio Chat with Tabs",
        theme=gr.themes.Soft(),
    ) as interface:
        
        gr.Markdown("""
        # 💬 Simple Gradio Multi-Tab Application
        
        This is a demonstration of a tabbed Gradio interface with multiple functionalities.
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Simple Chatbot")
                gr.Markdown("Start a conversation with our simple chatbot!")
                
                chatbot = gr.Chatbot(
                    value=[],
                    label="Chat History",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=1,
                        scale=4
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                
                # Chat functionality
                msg_input.submit(
                    fn=simple_chatbot,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                send_btn.click(
                    fn=simple_chatbot,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                clear_btn.click(
                    fn=clear_chat,
                    outputs=chatbot
                )
            
            # Calculator Tab
            with gr.Tab("🔢 Calculator"):
                gr.Markdown("### Simple Calculator")
                gr.Markdown("Perform basic mathematical operations.")
                
                with gr.Row():
                    num1 = gr.Number(
                        label="First Number",
                        value=0
                    )
                    operation = gr.Dropdown(
                        label="Operation",
                        choices=["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)"],
                        value="Add (+)"
                    )
                    num2 = gr.Number(
                        label="Second Number",
                        value=0
                    )
                
                calc_btn = gr.Button("Calculate 🧮", variant="primary")
                
                calc_result = gr.Textbox(
                    label="Result",
                    lines=2,
                    interactive=False
                )
                
                calc_btn.click(
                    fn=simple_calculator,
                    inputs=[num1, operation, num2],
                    outputs=calc_result
                )
            
            # Text Analyzer Tab
            with gr.Tab("📝 Text Analyzer"):
                gr.Markdown("### Text Analysis Tool")
                gr.Markdown("Analyze any text to get word count, character count, and more statistics.")
                
                text_input = gr.Textbox(
                    label="Enter Text to Analyze",
                    placeholder="Paste or type your text here...",
                    lines=8
                )
                
                analyze_btn = gr.Button("Analyze Text 🔍", variant="primary")
                
                analysis_output = gr.Textbox(
                    label="Analysis Results",
                    lines=12,
                    interactive=False
                )
                
                analyze_btn.click(
                    fn=text_analyzer,
                    inputs=text_input,
                    outputs=analysis_output
                )
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ### About This Application
                
                This is a simple demonstration of Gradio's capabilities featuring:
                
                **💬 Chat Tab:**
                - Interactive chatbot with basic responses
                - Chat history management
                - Simple keyword-based responses
                
                **🔢 Calculator Tab:**
                - Basic arithmetic operations
                - Error handling for edge cases
                - Clean number input interface
                
                **📝 Text Analyzer Tab:**
                - Word and character counting
                - Sentence and paragraph analysis
                - Most common words identification
                
                **🚀 Features:**
                - Tabbed interface for organization
                - Responsive design
                - Real-time interactions
                - Error handling
                
                **🛠️ Built with:**
                - [Gradio](https://gradio.app/) for the web interface
                - Python for backend logic
                - Soft theme for aesthetics
                
                **📧 Usage:**
                Simply click on any tab above to try out the different features!
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **Simple Gradio Demo** | Built with ❤️ using [Gradio](https://gradio.app/)
        """)
    
    return interface

def main():
    """Main function to run the application."""
    try:
        print("🚀 Starting Simple Gradio Chat Application...")
        
        # Create and launch the interface
        interface = create_interface()
        
        # Launch with share=True for accessibility
        interface.launch(
            share=True,
            server_name="127.0.0.1",
            server_port=7861,  # Different port to avoid conflicts
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("Please ensure Gradio is installed: pip install gradio")

if __name__ == "__main__":
    main() 