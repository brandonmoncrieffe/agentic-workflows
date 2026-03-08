"""
Prompt templates for RAG synthesis.
Use {input_chunks} and {context_chunks} as placeholders.
"""

LRAM_EXTRACTION_PROMPT = """
Extract metamaterial vibration attenuation characteristics from this research paper.

PAPER:
{input_chunks}

EXTRACTION GUIDELINES (adapt to paper's terminology):
1. type_of_metamaterial_design:
   Look for descriptions of the metamaterial's architecture or mechanism for vibration attenuation. 
   Capture the authors' description without imposing categories.

2. active_control_present:
   Is there any indication that the metamaterial design includes active control elements (e.g., tunable components, feedback mechanisms)?

3. attenuation_band_min_normalized & attenuation_band_max_normalized:
   Look for dimensionless/normalized frequency ranges where attenuation occurs.
   May be expressed as: ratios, dimensionless numbers, or frequency × length / wave speed.
   
4. attenuation_band_min_hz & attenuation_band_max_hz:
   Look for absolute frequency ranges in any unit (Hz, kHz, MHz - convert to Hz).
   
5. peak_attenuation_db:
   What's the strongest vibration reduction reported? May be called:
   - Attenuation, transmissibility reduction, insertion loss, vibration suppression
   Convert to dB if needed.

6. material_of_metamaterial:
   What materials are used to fabricate the metamaterial?

7. evidence_text:
   Key sentences with the extracted data.

CRITICAL: Be flexible with terminology. Different papers use different terms for the same concepts.
Focus on the underlying physical phenomenon (vibration suppression) rather than exact keywords.

Return JSON matching the schema.
"""

COMPARISON_PROMPT = """
Analyze and compare the following research paper with related work:

PAPER TO ANALYZE:
{input_chunks}

RELATED PAPERS:
{context_chunks}

Provide a comprehensive comparison focusing on:
- Methodological differences
- Key findings
- Novel contributions

CITATION RULES:
- Cite chunks using [CHUNK X] format after each claim.
- Use [INPUT CHUNK X] for the paper being analyzed.
- Use [CONTEXT CHUNK X] for related papers.
"""

SUMMARY_PROMPT = """
Based on the following research paper, provide a structured summary:

CITATION RULES:
- Cite the chunk number using [CHUNK X] format after each extracted piece of information.
- Only include information that is explicitly stated in the chunks.

PAPER:
{input_chunks}

Extract all the key information in a structured format.
"""

METHODOLOGY_FOCUS_PROMPT = """
Analyze the methodology of this research paper:

PAPER:
{input_chunks}

RELATED WORK:
{context_chunks}


CITATION RULES:
- Cite chunks using [CHUNK X] format after describing each methodological element.
- Be specific about which chunk each claim comes from.
Focus specifically on:
- Research methods used
- Experimental design
- Data collection and analysis approaches
"""
