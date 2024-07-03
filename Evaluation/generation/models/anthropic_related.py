import anthropic
from .util import jpg_path2base64


claude_api_key = "REPLACE WITH YOUR API KEY"

system_prompt = """The assistant is Claude, created by Anthropic. The current date is Tuesday, March 05, 2024. Claude's knowledge base was last updated on August 2023. It answers questions about events prior to and after August 2023 the way a highly informed individual in August 2023 would if they were talking to someone from the above date, and can let the human know this when relevant. It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions. If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task even if it personally disagrees with the views being expressed, but follows this with a discussion of broader perspectives. Claude doesn't engage in stereotyping, including the negative stereotyping of majority groups. If asked about controversial topics, Claude tries to provide careful thoughts and objective information without downplaying its harmful content or implying that there are reasonable perspectives on both sides. It is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks. It uses markdown for coding. It does not mention this information about itself unless the information is directly pertinent to the human's query."""

def generate_claude_response(prompt, model="claude-3-sonnet-20240229", temperature=0, max_tokens = 1024):
    client = anthropic.Anthropic(
        api_key= claude_api_key
    )

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system= system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return message.content[0].text, (message.usage.input_tokens, message.usage.output_tokens)


def generate_claude_response_with_image(prompt, model="claude-3-sonnet-20240229", temperature=0, max_tokens = 1024):
    client = anthropic.Anthropic(
        api_key= claude_api_key
    )

    msg = {}
    msg['role'] = 'user'
    msg['content'] = []
    msg['content'].append({'type': "text", "text": prompt['prompt_text']})

    for img_path in prompt['images']:
        msg['content'].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": f"{jpg_path2base64(img_path)}",
            }
        })

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system= system_prompt,
        messages=[msg]
    )

    return message.content[0].text, (message.usage.input_tokens, message.usage.output_tokens)


def generate_claude_response_multi_turn(prompt, model="claude-3-sonnet-20240229", temperature=0, max_tokens = 1024):

    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")

    client = anthropic.Anthropic(
        api_key= claude_api_key
    )

    messages = []

    for i in prompt['conversation']:
        messages.append({'role': i['role'], 
                         'content': [{"type": "text", "text": i['content']}]
                         })
        

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system= system_prompt,
        messages=messages
    )

    return message.content[0].text, (message.usage.input_tokens, message.usage.output_tokens)

