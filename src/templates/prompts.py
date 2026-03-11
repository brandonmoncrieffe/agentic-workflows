"""
Prompt templates for RAG synthesis.
Use {input_chunks} and {context_chunks} as placeholders.
"""

LRAM_BUCKET_PROMPT = """
Provide large, structured paragraphs of the metamaterial attenuation characteristics from this research paper. 
You are an expert in analyzing research papers on metamaterials for acoustic attenuation.

EXTRACTION GUIDELINES:

1. type_of_metamaterial_design:
   Look for descriptions of the metamaterial's architecture or mechanism for frequency attenuation. 
   Capture the authors' description without imposing categories.

2. active_control_present:
   Is there any indication that the metamaterial design includes active control elements (e.g., tunable components, feedback mechanisms)?
   
3. attenuation_bands_hz:
   Identify each frequency range or specific frequency in the paper where attenuation,
   transmission loss, bandgaps, or vibration reduction are reported.

   Record each band or frequency exactly as described in the text.
   Multiple attenuation bands may exist; list them all separately.

5. peak_attenuation_db:
   What's the strongest acoustic attenuation reported? May be called:
   - Attenuation, transmissibility reduction, insertion loss, vibration suppression
   Convert to dB if needed.

5. material_of_metamaterial:
   What materials are used to fabricate the metamaterial?

6. unit_cell_information:
   What are  the key geometric and structural parameters of the 
   unit cell that define the metamaterial design? This includes the overall unit cell size 
   (lattice constant) and the internal geometry of the resonating structure. These parameters 
   describe how the repeating cell is constructed and allow the design to be compared, scaled, or reproduced.

CRITICAL: Be flexible with terminology. Different papers use different terms for the same concepts.
Focus on the underlying physical phenomenon of acoustic attenuation rather than exact keywords.

Return JSON matching the schema.

PAPER:
{input_chunks}

"""

SUMMARY_PROMPT = """
Based on the following research paper, provide a structured summary:

CITATION RULES:
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
