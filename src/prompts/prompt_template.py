from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class Prompt:
    base_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior reporter for 'The Batch'. Your goal is to synthesize the provided wide and detailed news segments .

        STRICT OPERATING RULES:
        1. NO REDUNDANCY: Do not create separate 'Summary' and 'Details' sections. Merge all information into one unified list.
        2. TOTAL COVERAGE: You must briefly address EVERY news item found in the context (including editorial notes or milestones mentioned).
        3. ZERO FILLER: Start immediately with facts. Do not say "The articles discuss..."
        4. NO HALLUCINATION: If a detail isn't in the context, do not mention it.
        5. NO DISCLAIMERS: Do not end with apologies or statements about missing info.
        6. NO SEPARATORS: dont add * as separator.
        7. NO WRAP-UP: Do not conclude with 'Overall...' or 'In summary...'
        """,
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                """[NEWS SEGMENTS]:
        ---
        {context}
        ---

        [USER REQUEST]: {question}

        Provide the report in bullet points below:""",
            ),
        ],
    )
    critique_chain = ChatPromptTemplate.from_template("""
            ### ROLE: Expert Technical Editor & Verifier

            Use the provided context to answer the question. Strictly ignore any parts of the context that do not directly relate to the specific topic of the question

            ### STRICT INSTRUCTIONS:
            1. NO INTROS/OUTROS: Start directly with the first bullet point. Remove "Here is the report," "The articles discuss," and "Note:".
            2. ATOMIC SYNTHESIS: Merge identical news items. If two bullets discuss the same startup or model, combine them into one dense sentence.
            3. THE GROUNDING RULE: For every statement in the 'Proposed Summary', verify it against the 'Context'.
               - If a detail is NOT in the context, DELETE it immediately.
                   - Do not use outside knowledge (e.g., don't add info about GPT-5 if it's not in the text).
            4. DENSITY: Use Bold Headers followed by one clear sentence.
            5. NO REPETITION: Ensure no two bullets say the same thing using different words.
            6. NO SEPARATORS: dont add * as separator.
            7. NO WRAP-UP: Do not conclude with 'Overall...' or 'In summary...'


            Context: {context}
            Original Question: {question}
            Proposed Summary: {answer}

            ### FINAL DENSE REPORT (NO ASTERISKS):
        """)
    query_expansion = ChatPromptTemplate.from_template(
        "Generate 2 alternative search queries for: {question}. Output only queries.",
    )

    image_desc_prompt = ChatPromptTemplate.from_template("""
                ### ROLE: Expert Technical Editor & Verifier

                ### STRICT INSTRUCTIONS:
                1. NO INTROS/OUTROS: Start directly with the first bullet point. Remove "Here is the report," "The articles discuss," and "Note:".
                2. ATOMIC SYNTHESIS: Merge identical news items. If two bullets discuss the same startup or model, combine them into one dense sentence.
                3. THE GROUNDING RULE: For every statement in the 'Proposed Summary', verify it against the 'Context'.
                   - If a detail is NOT in the context, DELETE it immediately.
                   - Do not use outside knowledge (e.g., don't add info about GPT-5 if it's not in the text).
                4. DENSITY: Use Bold Headers followed by one clear sentence.
                5. NO REPETITION: Ensure no two bullets say the same thing using different words.

                Context: {context}
                Original Question: {question}
                Proposed Summary: {answer}

                ### FINAL DENSE REPORT (NO ASTERISKS):
            """)


prompts = Prompt()
