from typing import Dict, Any
from string import Template


class PromptTemplate:
    """Base class for prompt templates"""

    def __init__(self, template: str):
        self.template = Template(template)

    def format(self, **kwargs) -> str:
        """Format the template with the given parameters"""
        return self.template.safe_substitute(**kwargs)


class ContextualPrompts:
    """Collection of prompts for contextual embeddings"""

    DOCUMENT_CONTEXT = PromptTemplate("""
    <document>
    ${doc_content}
    </document>
    """)

    CHUNK_CONTEXT = PromptTemplate("""
    Here is the chunk we want to situate within the whole document
    <chunk>
    ${chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document.
    Answer only with the succinct context and nothing else.
    """)

    @classmethod
    def get_full_context_prompt(cls, doc_content: str, chunk_content: str) -> str:
        """Combine document and chunk prompts"""
        doc_part = cls.DOCUMENT_CONTEXT.format(doc_content=doc_content)
        chunk_part = cls.CHUNK_CONTEXT.format(chunk_content=chunk_content)
        return f"{doc_part}\n\n{chunk_part}"


class QAPrompts:
    """Collection of prompts for question answering"""

    ANSWER_WITH_CONTEXT = PromptTemplate("""
    Answer the question based on the provided context.

    Context:
    ${context}

    Question: ${question}

    Answer: Let me help you with that based on the context provided.
    """)

    ANSWER_WITHOUT_CONTEXT = PromptTemplate("""
    I don't have enough context to fully answer this question. However, here's what I know about ${topic}:
    """)

    @classmethod
    def get_qa_prompt_with_context(cls, context: str, question: str) -> str:
        return cls.ANSWER_WITH_CONTEXT.format(context=context, question=question)

    @classmethod
    def get_qa_prompt_without_context(cls, topic: str) -> str:
        return cls.ANSWER_WITH_CONTEXT.format(topic=topic)


class SummaryPrompts:
    """Collection of prompts for text summarization"""

    CHUNK_SUMMARY = PromptTemplate("""
    Summarize the following text in a concise way, focusing on the key points:

    ${text}

    Summary:
    """)

    DOCUMENT_SUMMARY = PromptTemplate("""
    Provide a comprehensive summary of the following document, covering the main topics and key findings:

    ${document}

    Summary:
    """)

    @classmethod
    def get_summary_prompt_for_chunk(cls, text: str) -> str:
        return cls.CHUNK_SUMMARY.format(text=text)

    @classmethod
    def get_summary_prompt_for_doc(cls, document: str) -> str:
        return cls.DOCUMENT_SUMMARY.format(document=document)

