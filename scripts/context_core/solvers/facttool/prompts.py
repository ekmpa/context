CLAIM_EXTRACTION_SYSTEM_PROMPT = "You are a brilliant assistant."

CLAIM_EXTRACTION_USER_PROMPT = """
You are given a piece of text that includes knowledge claims. A claim is a statement that asserts something as true or false, which can be verified by humans. Your task is to accurately identify and extract every claim stated in the provided text. Then, resolve any coreference (pronouns or other referring expressions) in the claim for clarity. Each claim should be concise (less than 15 words) and self-contained.
Your response MUST be a list of dictionaries. Each dictionary should contains the key "claim", which correspond to the extracted claim (with all coreferences resolved).
You MUST only respond in the format as described below. DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED. START YOUR RESPONSE WITH '['.
[response format]:
[
  {
    "claim": "Ensure that the claim is fewer than 15 words and conveys a complete idea. Resolve any coreference (pronouns or other referring expressions) in the claim for clarity"
  }
]

Now complete the following,ONLY RESPONSE IN A LIST FORMAT, NO OTHER WORDS!!!:
[text]: {input}
[response]:
""".strip()

QUERY_GENERATION_SYSTEM_PROMPT = (
    "You are a query generator that generates effective and concise search engine queries "
    "to verify a given claim. You only response in a python list format(NO OTHER WORDS!)."
)

QUERY_GENERATION_USER_PROMPT = """
You are a query generator designed to help users verify a given claim using search engines. Your primary task is to generate a Python list of two effective and skeptical search engine queries. These queries should assist users in critically evaluating the factuality of a provided claim using search engines.
You should only respond in format as described below (a Python list of queries). PLEASE STRICTLY FOLLOW THE FORMAT. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '['.
[response format]: ['query1', 'query2']

Now complete the following(ONLY RESPONSE IN A LIST FORMAT, DO NOT RETURN OTHER WORDS!!! START YOUR RESPONSE WITH '[' AND END WITH ']'):
claim: {input}
response:
""".strip()

VERIFICATION_SYSTEM_PROMPT = "You are a brilliant assistant."

VERIFICATION_USER_PROMPT = """
You are given a piece of text. Your task is to identify whether there are any factual errors within the text.
When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences may be helpful. Some evidences may contradict to each other. You must be careful when using the evidences to judge the factuality of the given text.
The response must be a valid Python dictionary with exactly four keys: "reasoning", "factuality", "error", and "correction".
Use Python booleans for factuality (True or False), not JSON booleans.
Do not include chain-of-thought, analysis notes, markdown, code fences, or any text before/after the dictionary.
Keep "reasoning" to one concise sentence.
The following is the given text
[text]: {claim}
The following is the provided evidences
[evidences]: {evidence}
You must output exactly one dictionary and nothing else. Start with '{' and end with '}'.
[response format]:
{{
  "reasoning": "Why is the given text factual or non-factual?",
  "error": "None if the text is factual; otherwise, describe the error.",
  "correction": "The corrected text if there is an error.",
  "factuality": True
}}
""".strip()
