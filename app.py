"""
Gradio web interface for the RAG-based computer vision research chatbot.

Starts a browser-accessible chat UI backed by the RAG pipeline.  The LLM
and vector store are loaded once at startup; subsequent questions reuse the
same chain for low-latency responses.

Usage::

    python app.py
"""

import gradio as gr

from rag import build_rag_chain, load_llm, query

print("Initialising RAG pipeline...")
llm = load_llm()
chain, retriever = build_rag_chain(llm)
print("Ready.")


def answer_question(question: str, history: list) -> tuple[str, list]:
    """
    Handle a user question and append the answer to the conversation history.

    Skips empty submissions.  The answer is augmented with a bulleted source
    list so users can trace responses back to specific papers.

    Args:
        question: Raw text entered by the user.
        history: Current Gradio chatbot message history (list of role-dicts).

    Returns:
        ``(cleared_input, updated_history)`` suitable for Gradio output binding.
    """
    if not question.strip():
        return "", history

    answer, sources = query(chain, retriever, question)
    source_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    full_response = answer + source_text

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_response})
    return "", history


with gr.Blocks(title="CV Paper Q&A") as app:
    gr.Markdown("## Computer Vision Research Paper Q&A")
    gr.Markdown("Ask questions about the loaded research papers. Sources are shown with each answer.")

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        question_box = gr.Textbox(
            placeholder="e.g. What are the key differences between YOLO v5 and v8?",
            label="Your question",
            scale=4,
        )
        submit_btn = gr.Button("Ask", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear conversation")

    gr.Examples(
        examples=[
            "What is the main contribution of the YOLO paper?",
            "How does attention mechanism work in vision transformers?",
            "What are the tradeoffs between speed and accuracy in object detection?",
            "How do tracking algorithms handle occlusion?",
        ],
        inputs=question_box,
    )

    submit_btn.click(
        answer_question,
        inputs=[question_box, chatbot],
        outputs=[question_box, chatbot],
    )
    question_box.submit(
        answer_question,
        inputs=[question_box, chatbot],
        outputs=[question_box, chatbot],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_box])

if __name__ == "__main__":
    app.launch(share=False, theme=gr.themes.Soft())
