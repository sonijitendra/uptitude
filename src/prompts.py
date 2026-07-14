"""
Prompt templates for LLM-based legal contract analysis.

Contains carefully engineered prompts for:
    - Clause extraction (termination, confidentiality, liability)
    - Contract summarization (100-150 words)

Design principles:
    - Role prompting for consistent persona
    - Structured JSON output format with schema enforcement
    - Explicit instructions to return null for absent clauses
    - Few-shot examples for improved extraction accuracy
    - Delimiters for clear input/output boundaries
    - Chain-of-thought internally (suppressed in output)
"""

# =========================================================================
# System Prompts
# =========================================================================

CLAUSE_EXTRACTION_SYSTEM_PROMPT: str = """You are an expert legal analyst specializing in contract review and clause extraction. You have 20+ years of experience analyzing commercial contracts including NDAs, service agreements, licensing agreements, and M&A documents.

Your task is to identify and extract specific clauses from legal contracts with high precision. You must:

1. Extract the EXACT text from the contract — do not paraphrase or summarize.
2. If a clause type is NOT present in the contract, return null for that field.
3. NEVER fabricate or hallucinate clause text that does not exist in the contract.
4. If multiple sections relate to a clause type, extract the most comprehensive one.
5. Think step-by-step internally but only output the final structured JSON.

You must respond ONLY with a valid JSON object matching the specified schema. Do not include any text before or after the JSON."""


SUMMARY_SYSTEM_PROMPT: str = """You are an expert legal analyst who produces concise, accurate contract summaries for executive review. You have 20+ years of experience distilling complex legal agreements into actionable briefs.

Your summaries must:
1. Be exactly 100-150 words long.
2. Cover the PURPOSE of the agreement.
3. Identify KEY OBLIGATIONS of each party.
4. Highlight NOTABLE RISKS or PENALTIES.
5. Be purely factual — no opinions, speculation, or information not in the contract.
6. NEVER include information that is not explicitly stated in the contract text.

You must respond ONLY with a valid JSON object matching the specified schema. Do not include any text before or after the JSON."""


# =========================================================================
# User Prompts — Clause Extraction
# =========================================================================

CLAUSE_EXTRACTION_USER_PROMPT: str = """Analyze the following legal contract and extract the specified clauses.

<CONTRACT>
{contract_text}
</CONTRACT>

Extract the following clause types from the contract above:

1. **Termination Clause**: Provisions describing how, when, or under what conditions the agreement can be terminated by either party. Look for sections titled "Termination", "Term and Termination", "Cancellation", or similar.

2. **Confidentiality Clause**: Provisions requiring parties to keep certain information confidential or secret. Look for sections titled "Confidentiality", "Non-Disclosure", "Proprietary Information", "Trade Secrets", or similar.

3. **Liability Clause**: Provisions limiting, excluding, or defining liability between the parties. Look for sections titled "Limitation of Liability", "Indemnification", "Liability", "Damages", or similar.

IMPORTANT RULES:
- Extract the EXACT text as it appears in the contract.
- If a clause type is not found, set its value to null.
- Do NOT invent or hallucinate any clause text.
- If there are multiple relevant sections, extract the most comprehensive one.
- Keep extracted text concise — focus on the core provision (typically 1-3 paragraphs).

Respond with ONLY this JSON structure:
{{
    "termination_clause": "<exact clause text or null>",
    "confidentiality_clause": "<exact clause text or null>",
    "liability_clause": "<exact clause text or null>"
}}"""


# =========================================================================
# User Prompts — Clause Extraction with Few-Shot Examples
# =========================================================================

CLAUSE_EXTRACTION_FEWSHOT_USER_PROMPT: str = """Analyze the following legal contract and extract the specified clauses.

Here are examples of correct extractions:

<EXAMPLE_1>
Contract snippet: "...8. TERMINATION. Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party. This Agreement will automatically terminate if either party becomes insolvent or files for bankruptcy..."
Extracted:
{{
    "termination_clause": "Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party. This Agreement will automatically terminate if either party becomes insolvent or files for bankruptcy.",
    "confidentiality_clause": null,
    "liability_clause": null
}}
</EXAMPLE_1>

<EXAMPLE_2>
Contract snippet: "...5. CONFIDENTIALITY. Each party agrees to hold in strict confidence all Confidential Information received from the other party. Confidential Information shall not be disclosed to any third party without prior written consent and shall be protected with at least the same degree of care as the receiving party uses for its own confidential information..."
Extracted:
{{
    "termination_clause": null,
    "confidentiality_clause": "Each party agrees to hold in strict confidence all Confidential Information received from the other party. Confidential Information shall not be disclosed to any third party without prior written consent and shall be protected with at least the same degree of care as the receiving party uses for its own confidential information.",
    "liability_clause": null
}}
</EXAMPLE_2>

<EXAMPLE_3>
Contract snippet: "...10. LIMITATION OF LIABILITY. IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING OUT OF THIS AGREEMENT. THE TOTAL LIABILITY OF EITHER PARTY SHALL NOT EXCEED THE AMOUNTS PAID UNDER THIS AGREEMENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM..."
Extracted:
{{
    "termination_clause": null,
    "confidentiality_clause": null,
    "liability_clause": "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING OUT OF THIS AGREEMENT. THE TOTAL LIABILITY OF EITHER PARTY SHALL NOT EXCEED THE AMOUNTS PAID UNDER THIS AGREEMENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM."
}}
</EXAMPLE_3>

Now analyze this contract:

<CONTRACT>
{contract_text}
</CONTRACT>

Extract the following clause types:

1. **Termination Clause**: Provisions for ending the agreement.
2. **Confidentiality Clause**: Provisions for protecting confidential information.
3. **Liability Clause**: Provisions limiting or defining liability.

RULES:
- Extract EXACT text from the contract.
- Return null for absent clauses.
- Do NOT hallucinate.
- Focus on the core provision (1-3 paragraphs).

Respond with ONLY this JSON:
{{
    "termination_clause": "<exact text or null>",
    "confidentiality_clause": "<exact text or null>",
    "liability_clause": "<exact text or null>"
}}"""


# =========================================================================
# User Prompts — Summarization
# =========================================================================

SUMMARY_USER_PROMPT: str = """Summarize the following legal contract in 100-150 words.

<CONTRACT>
{contract_text}
</CONTRACT>

Your summary MUST include:
1. **Purpose**: What is the agreement about? What type of contract is it?
2. **Obligations**: What are the key responsibilities of each party?
3. **Risks/Penalties**: What are the notable risks, penalties, or consequences defined?

RULES:
- Stay within 100-150 words.
- Be factual — only include information explicitly stated in the contract.
- Do NOT speculate or add information not present in the text.
- Use clear, professional language suitable for executive review.

Respond with ONLY this JSON:
{{
    "summary": "<your 100-150 word summary>"
}}"""


# =========================================================================
# Chunked Contract Prompts
# =========================================================================

CLAUSE_EXTRACTION_CHUNK_PROMPT: str = """You are analyzing a PORTION of a larger legal contract. This is chunk {chunk_index} of {total_chunks}.

<CONTRACT_CHUNK>
{chunk_text}
</CONTRACT_CHUNK>

Search this chunk for the following clause types:
1. **Termination Clause**: Provisions for ending the agreement.
2. **Confidentiality Clause**: Provisions for protecting confidential information.
3. **Liability Clause**: Provisions limiting or defining liability.

RULES:
- Extract EXACT text from this chunk.
- Return null for clause types NOT found in this specific chunk.
- Do NOT hallucinate or guess based on context.

Respond with ONLY this JSON:
{{
    "termination_clause": "<exact text or null>",
    "confidentiality_clause": "<exact text or null>",
    "liability_clause": "<exact text or null>"
}}"""


MERGE_CLAUSES_PROMPT: str = """You are merging clause extraction results from multiple chunks of the same contract.

Below are extraction results from {total_chunks} chunks of the same contract:

<CHUNK_RESULTS>
{chunk_results_json}
</CHUNK_RESULTS>

For each clause type, select the most complete and comprehensive extraction across all chunks. If a clause was found in multiple chunks, combine them into a coherent extraction preserving the original text.

If NO chunk found a particular clause type, return null for that field.

Respond with ONLY this JSON:
{{
    "termination_clause": "<best extraction or null>",
    "confidentiality_clause": "<best extraction or null>",
    "liability_clause": "<best extraction or null>"
}}"""


# =========================================================================
# Helper Functions
# =========================================================================

def format_extraction_prompt(
    contract_text: str,
    use_few_shot: bool = True,
) -> str:
    """Format the clause extraction user prompt with contract text.

    Args:
        contract_text: The cleaned contract text to analyze.
        use_few_shot: Whether to use few-shot examples.

    Returns:
        Formatted prompt string ready for LLM.
    """
    template = (
        CLAUSE_EXTRACTION_FEWSHOT_USER_PROMPT
        if use_few_shot
        else CLAUSE_EXTRACTION_USER_PROMPT
    )
    return template.format(contract_text=contract_text)


def format_summary_prompt(contract_text: str) -> str:
    """Format the summarization user prompt with contract text.

    Args:
        contract_text: The cleaned contract text to summarize.

    Returns:
        Formatted prompt string ready for LLM.
    """
    return SUMMARY_USER_PROMPT.format(contract_text=contract_text)


def format_chunk_extraction_prompt(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Format the chunk-level extraction prompt.

    Args:
        chunk_text: Text of the current chunk.
        chunk_index: 1-based index of this chunk.
        total_chunks: Total number of chunks.

    Returns:
        Formatted prompt string for chunk extraction.
    """
    return CLAUSE_EXTRACTION_CHUNK_PROMPT.format(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def format_merge_prompt(
    chunk_results_json: str,
    total_chunks: int,
) -> str:
    """Format the merge prompt for combining chunk results.

    Args:
        chunk_results_json: JSON string of all chunk results.
        total_chunks: Total number of chunks processed.

    Returns:
        Formatted prompt string for merging.
    """
    return MERGE_CLAUSES_PROMPT.format(
        chunk_results_json=chunk_results_json,
        total_chunks=total_chunks,
    )
