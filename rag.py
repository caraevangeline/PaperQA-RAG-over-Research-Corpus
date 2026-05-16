"""
RAG (Retrieval-Augmented Generation) pipeline for computer vision research Q&A.

Builds a LangChain LCEL chain that retrieves relevant paper chunks from a Chroma
vector store and feeds them as context to a local or API-based LLM.

Typical usage::

    from rag import load_llm, build_rag_chain, query

    llm = load_llm()
    chain, retriever = build_rag_chain(llm)
    answer, sources = query(chain, retriever, "What is YOLO?")
"""

import torch
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

PROMPT_TEMPLATE = """You are an expert in computer vision research. Use the following context from research papers to answer the question clearly and concisely.
If the answer is not in the context, say "I could not find this in the provided papers."

Context:
{context}

Question: {question}

Answer:"""


def load_llm(model_id: str = LLM_MODEL) -> HuggingFacePipeline:
    """
    Load a HuggingFace causal LM and wrap it in a LangChain pipeline.

    Automatically selects float16 on CUDA devices and float32 on CPU.
    ``device_map="auto"`` spreads the model across available hardware.

    Args:
        model_id: HuggingFace Hub model identifier.

    Returns:
        LangChain-compatible HuggingFacePipeline ready for chain use.
    """
    print(f"Loading LLM: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


def format_docs(docs: list[Document]) -> str:
    """
    Concatenate retrieved documents into a single context string.

    Args:
        docs: Documents returned by the retriever.

    Returns:
        Page contents joined by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm, vectorstore_dir: str = VECTORSTORE_DIR) -> tuple:
    """
    Assemble the RAG chain from retriever, prompt template, and LLM.

    The retriever uses Maximal Marginal Relevance (MMR) with k=5 to balance
    relevance and diversity among returned chunks.

    Args:
        llm: Any LangChain-compatible language model (HuggingFace, OpenAI, etc.).
        vectorstore_dir: Path to the persisted Chroma vector store.

    Returns:
        ``(chain, retriever)`` — the LCEL chain and the underlying retriever,
        both needed by :func:`query`.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5},
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(chain, retriever, question: str) -> tuple[str, list[str]]:
    """
    Run a question through the RAG chain and return a cleaned answer with sources.

    Post-processing removes the "Answer:" echo and any looping text the model
    may produce after a second "Question:" marker.

    Args:
        chain: LCEL chain returned by :func:`build_rag_chain`.
        retriever: Retriever returned by :func:`build_rag_chain`.
        question: Natural-language question from the user.

    Returns:
        ``(answer, sources)`` — cleaned answer string and a deduplicated list
        of source filenames from the retrieved documents.
    """
    raw_answer = chain.invoke(question)

    if "Question:" in raw_answer:
        raw_answer = raw_answer.split("Question:")[0].strip()
    if raw_answer.startswith("Answer:"):
        raw_answer = raw_answer[len("Answer:"):].strip()

    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
    return raw_answer, sources
