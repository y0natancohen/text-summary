import braintrust
from openai import AsyncOpenAI

from md_utils import get_clean_markdown_text_heading
from data_set import DATA_SET

braintrust.login(api_key='')
client = braintrust.wrap_openai(AsyncOpenAI(api_key=''))

import json

PROMPT = """\
You are judging a candidate summary against a reference summary for the same source markdown text.

[BEGIN DATA]
************
[markdown content]: {markdown_content}
************
[Summary]: {summary}
************
[END DATA]

Evaluate the summary of the given markdown content according to these criteria:
- Relavance: The summary shouldn't include irrelevant information from that appear in all websites, such as: cookies related things, privacy terms, terms and conditions, legal disclaimer, accessibility notes.
- Faithfulness: Every factual claim is supported by the source; no hallucinations.
- General Coverage: Includes the most important points from the source.
- Key-Word Coverage: Includes the most important keywords from the source.
- Conciseness: Minimal redundancy; WE WANT A SHORT SUMMARY; not too long; appropriate length for a summary.
- Coherence: Well-written and logically organized. Coherence is less important then the other criteria.

Rate the summary on a scale of 1 to 10.
"""


@braintrust.traced
async def llm_as_a_judge_numeric_rater(markdown_content, summary):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(markdown_content=markdown_content, summary=summary),
            }
        ],
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "Rate the summary on a scale of 1 to 10.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rating": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["rating"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "rate"}},
    )
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return (arguments["rating"] - 1) / 9


import asyncio

child_story = """
Once upon a time, in a tiny town tucked between two hills, there lived a little cloud named Puff.

Puff wasn’t like the other clouds.

The other clouds loved to do big, important cloud things—like making thunder booms, painting sunsets, and pouring rain into rivers. Puff wanted to help too… but every time Puff tried, only a single plip of rain would fall."""

async def main():
    i = 1
    print(DATA_SET[i]['markdown_content'], DATA_SET[i]['summary'])
    print('====================')
    print('====================')
    print('====================')
    print('====================')
    # result = await numeric_rater(DATA_SET[i]['markdown_content'], DATA_SET[i]['summary'])
    text_clean = get_clean_markdown_text_heading(DATA_SET[i]['markdown_content'])
    result = await llm_as_a_judge_numeric_rater(markdown_content=text_clean,
                                                summary=DATA_SET[i]['summary'])
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
