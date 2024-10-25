from zhipuai import ZhipuAI

client = ZhipuAI(api_key="38b1ca6c973ab68f72e9069491688c1f.gfbXLBHc5tk2eQGd")

response = client.chat.completions.create(
    model="glm-4-flash:1800577043::nwsgohqk",
    messages=[
        {"role": "system", "content": "你是浩哥哥研发的一个知识渊博的AI助手"},
        {"role": "user", "content": "你是谁"},
    ],
    top_p=0.7,
    temperature=0.95,
    max_tokens=1024,
    tools=[{"type": "web_search", "web_search": {"search_result": True}}],
    stream=True,
)
for trunk in response:
    print(trunk)


def add(nums):
    return sum(nums)
