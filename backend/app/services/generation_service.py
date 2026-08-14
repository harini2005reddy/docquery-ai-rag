import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class GenerationService:

    MODEL_NAME = "llama-3.3-70b-versatile"

    def __init__(self):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )


    def build_context(
        self,
        chunks: list[dict],
    ) -> str:

        context_parts = []

        for result in chunks:

            metadata = result.get(
                "metadata",
                {},
            )

            page = result.get(
                "page",
                metadata.get(
                    "page",
                    "Unknown",
                ),
            )

            chunk = result.get(
                "chunk",
                metadata.get(
                    "chunk",
                    "Unknown",
                ),
            )

            text = result.get(
                "text",
                "",
            )

            context_parts.append(
                f"[Page {page}, Chunk {chunk}]\n{text}"
            )

        return "\n\n---\n\n".join(
            context_parts
        )


    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:

        context = self.build_context(
            retrieved_chunks
        )

        return f"""
You are a strict document question-answering assistant.

Your task is to answer the USER QUESTION using ONLY the information
provided in the DOCUMENT CONTEXT.

STRICT RULES:

1. Use ONLY information explicitly present in the DOCUMENT CONTEXT.

2. Do NOT use outside knowledge.

3. Do NOT make assumptions or unsupported inferences.

4. If the answer cannot be found in the DOCUMENT CONTEXT, respond with EXACTLY:
Information not found in the provided document.

5. Every factual answer MUST contain at least one citation.

6. Place each citation immediately after the claim it supports.

7. Citations MUST use EXACTLY this format:
[Page X, Chunk Y]

8. Use ONLY page and chunk numbers explicitly present in the
DOCUMENT CONTEXT.

9. Do NOT generate citation formats such as:
[1]
[2]
(Source 1)
(1)

10. Do NOT write:
Sources:
References:
Source citation:

11. Keep the answer concise and directly answer the USER QUESTION.

12. Return ONLY the direct answer and its citation(s).

13. Do NOT add notes, explanations, disclaimers, verification statements,
or commentary before or after the answer.

14. Do NOT mention:
"Note:"
"This answer only uses..."
"The answer is based on..."
"The source citation..."
"According to the document context..."
"DOCUMENT CONTEXT"
"grounding rules"
"citation rules"

15. Never explain how the answer was generated.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""


    def generate_answer_stream(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ):

        if not retrieved_chunks:

            yield (
                "Information not found in the provided document."
            )

            return


        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )


        stream = self.client.chat.completions.create(
            model=self.MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0,

            max_completion_tokens=512,

            stream=True,
        )


        for chunk in stream:

            if not chunk.choices:
                continue


            content = (
                chunk.choices[0]
                .delta
                .content
            )


            if content:
                yield content