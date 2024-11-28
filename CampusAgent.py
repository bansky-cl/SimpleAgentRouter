def get_gpt_response(messages):
    
    import requests
    import json

    url = ""
    headers = { 
        "Content-Type": "application/json", 
        "Authorization": f"Bearer "
    }
    data = { 
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0,
        "seed": 42,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
get_gpt_response([{"role": "user", "content": "Say this is a test!"}])

def complete(prompt): # 每次回复还是不一样emm 种子固定不下来
    return get_gpt_response([{"role": "user", "content": prompt}])['choices'][0]['message']['content']

response = complete("你能告诉我学生公寓在哪儿吗？")

choices = [
    "Useful for questions related to outsider",
    "Useful for questions related to student",
    "Useful for questions related to staff",
    "Useful for questions related to teacher",
] # campus profile


def get_choice_str(choices):
    choices_str = "\n\n".join(
        [f"{idx+1}. {c}" for idx, c in enumerate(choices)]
    )
    return choices_str


choices_str = get_choice_str(choices) # context_list

router_prompt0 = "Some choices are given below. It is provided in a numbered list (1 to \
     {num_choices}), where each item in the list corresponds to a \
     summary.\n---------------------\n{context_list}\n---------------------\nUsing \
     only the choices above and not prior knowledge, return the top choices \
     (no more than {max_outputs}, but only select what is needed) that are \
     most relevant to the question: '{query_str}'\n"

def get_formatted_prompt(query_str): 
    fmt_prompt = router_prompt0.format(
        num_choices=len(choices),
        max_outputs=len(choices),
        context_list=choices_str,
        query_str=query_str,
    )
    return fmt_prompt

response = complete(get_formatted_prompt("你能告诉我学生公寓在哪儿吗？"))

query_str_en = [
    "Can you tell me where the student apartment is?",
    "Where is the visitor lobby, please? I want to visit the gymnasium.",
    "I will be sharing my teaching experience in the department today. Which conference room will it be held in?",
    "I want to go to the canteen to clean up, can you show me the way?",
]

for s_en in query_str_en:
    fmt_prompt = get_formatted_prompt(s_en)
    response = complete(fmt_prompt)
    print("User query:\n", s_en)
    print("Classification:\n",str(response))
    print("###")


query_str_zh = [
    "你能告诉我学生公寓在哪儿吗？",
    "请问访客大厅在哪，我想参观体育馆？",
    "我今天要在系里分享授课心得，请问在哪间会议厅",
    "我要去饭堂打扫卫生，可以指路吗?"
]
for s_zh in query_str_zh:
    fmt_prompt = get_formatted_prompt(s_zh)
    response = complete(fmt_prompt)
    print("用户查询：\n",s_zh)
    print("query分类：\n",str(response))
    print("###")

# 定义router输出类

from dataclasses import fields
from pydantic import BaseModel # 数据验证，json抽取
import json

class Answer(BaseModel):
    choice: int
    reason: str
    
print(json.dumps(Answer.model_json_schema(), indent=2)) 
#   The `schema` method is deprecated; use `model_json_schema` instead.Pylance

# 定义llm输出类 = router输入类

FORMAT_STR = """The output should be formatted as a JSON instance that conforms to
the JSON schema below.

Here is the output schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""

def _escape_curly_braces(input_string: str) -> str:
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces 去双重花括号
    escaped_string = input_string.replace("{", "{{").replace("}", "}}")
    return escaped_string

def _marshal_output_to_json(output: str) -> str: # 从llm的回复中提取json串
    output = output.strip()
    left = output.find("[")
    right = output.find("]")
    output = output[left : right + 1]
    return output

from typing import List

class RouterOutputParser():
    def parse(self, output: str) -> List[Answer]:
        """Parse string."""
        json_output = _marshal_output_to_json(output) # 从llm的回复中提取json串
        json_dicts = json.loads(json_output) # json转成python的dict
        answers = [Answer.parse_obj(json_dict) for json_dict in json_dicts] # 多个json生成Answer
        return answers

    def format(self, prompt_template: str) -> str: # 格式化输出
        return prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)

output_parser = RouterOutputParser() # 初始化一个解析类

def route_query( # 输入用户query，输入预定义的画像类，以及输入llm回复解析类
    query_str: str, choices: List[str], output_parser: RouterOutputParser
):
    choices_str = get_choice_str(choices)
    fmt_base_prompt = get_formatted_prompt(query_str)
    fmt_json_prompt = output_parser.format(fmt_base_prompt)
    # print(fmt_json_prompt) # 格式化后的prompt

    raw_output = complete(fmt_json_prompt) # 调用llm生成回复

    parsed = output_parser.parse(str(raw_output)) # 解析llm的输出

    return parsed

route_result = route_query(query_str="帮我写一个会议纪要模板", choices=choices, output_parser=output_parser)
print(route_result)

def get_intent_str(route_result: List[Answer]):
    intent_str = "\n\n".join(
        [f"{idx+1}. {choices[c.choice-1]} : {c.reason}" for idx, c in enumerate(route_result)]
    )
    return intent_str

# MemoAgent

class MemoAgent():
    def __init__(self, memo_prompt: str, description = "会议纪要助手，主页负责撰写会议纪要、整理会议内容"):
        self.memo_prompt = memo_prompt
        self.description = description

    def run(self, query_str: str, route_result: List[Answer]): 
        fmt_prompt = self.memo_prompt.format(query_str=query_str, intents=get_intent_str(route_result))
        # print(fmt_prompt + "\n")
        return complete(fmt_prompt)
    
memo_prompt0 = "You are a meeting assistant. The following are questions from users:'{query_str}'. \
Users may have the following identities:\n{intents}\n\
Please give corresponding responses based on the user's identity, intent and question."

memo_agent = MemoAgent(memo_prompt0)
memo_query_str = "帮我预约会议室"
memo_route_result = route_query(query_str=memo_query_str, choices=choices, output_parser=output_parser)
print(memo_agent.run(query_str=memo_query_str, route_result=memo_route_result))

# BookAgent
class BookAgent():
    def __init__(self, book_prompt: str, description = "空间预约助手，主要负责预约会议室、活动中心、体育场馆"):
        self.book_prompt = book_prompt
        self.description = description

    def run(self, query_str: str, route_result: List[Answer]):
        fmt_prompt = self.book_prompt.format(query_str=query_str, intents=get_intent_str(route_result))
        return complete(fmt_prompt)
    
book_prompt0 = "You are a booking assistant. The following are questions from users:'{query_str}'. \
Users may have the following identities:\n{intents}\n\
Please give corresponding responses based on the user's identity, intent and question."

book_agent = BookAgent(book_prompt0)
book_query_str = "帮我预约会议室"
book_route_result = route_query(query_str=book_query_str, choices=choices, output_parser=output_parser)
print(book_agent.run(query_str=book_query_str, route_result=book_route_result))

class NavigationAgent():
    def __init__(self, navigation_prompt: str, description = "导航助手，主要负责带路、指路"):
        self.navigation_prompt = navigation_prompt
        self.description = description

    def run(self, query_str: str, route_result: List[Answer]):
        fmt_prompt = self.navigation_prompt.format(query_str=query_str, intents=get_intent_str(route_result))
        return complete(fmt_prompt)
    
navigation_prompt0 = "You are a navigation assistant. The following are questions from users:'{query_str}'. \
Users may have the following identities:\n{intents}\n\
Please give corresponding responses based on the user's identity, intent and question."

navigation_agent = NavigationAgent(book_prompt0)
navigation_query_str = "带我去校长办公室"
navigation_route_result = route_query(query_str=navigation_query_str, choices=choices, output_parser=output_parser)
print(navigation_agent.run(query_str=navigation_query_str, route_result=navigation_route_result))

from typing import List, Any

output_parser = RouterOutputParser() # 初始化一个解析类

def route_query( # 输入用户query，输入预定义的画像类，以及输入llm回复解析类
    query_str: str, choices: List[str], output_parser: RouterOutputParser
):
    choices_str = get_choice_str(choices)
    fmt_base_prompt = get_formatted_prompt(query_str)
    fmt_json_prompt = output_parser.format(fmt_base_prompt)
    # print(fmt_json_prompt) # 格式化后的prompt

    raw_output = complete(fmt_json_prompt) # 调用llm生成回复

    parsed = output_parser.parse(str(raw_output)) # 解析llm的输出

    return parsed

class RouterAgent():
    def __init__(self, router_prompt: str, agent_choices: List[Any], profile_choices: List[str], output_parser: RouterOutputParser):
        self.router_prompt = router_prompt
        self.agent_choices = agent_choices
        self.agent_descriptions = [agent.description for agent in agent_choices]
        self.profile_choices = profile_choices
        self.output_parser = output_parser
    
    def _get_formatted_prompt(self, query_str: str, choices: List[str]): 
        fmt_prompt = self.router_prompt.format(
            num_choices=len(choices),
            max_outputs=len(choices),
            context_list=get_choice_str(choices),
            query_str=query_str,
        )
        return fmt_prompt
    
    def _route_query(
        self, query_str: str, choices: List[str], output_parser: RouterOutputParser # 优化，两个解析不一样才对
    ):
        fmt_base_prompt = self._get_formatted_prompt(query_str, choices)
        fmt_json_prompt = self.output_parser.format(fmt_base_prompt)
        raw_output = complete(fmt_json_prompt)
        parsed = self.output_parser.parse(str(raw_output))
        return parsed
    
    def _profile_route_query(self, query_str: str): # query-profile分类
        return self._route_query(query_str=query_str, choices=self.profile_choices, output_parser=self.output_parser)

    def _agent_route_query(self, query_str: str): # query-agent分类
        return self._route_query(query_str=query_str, choices=self.agent_descriptions, output_parser=self.output_parser)
    
    def run(self, query_str: str):
        profile_route_result = self._profile_route_query(query_str)
        agent_route_result = self._agent_route_query(query_str)
        agent_id = [x.choice for x in agent_route_result]  #  调用可能用得上的agent
        for id in agent_id:
            current_agent_response = self.agent_choices[id-1].run(query_str, profile_route_result)

            return current_agent_response # 优化：调用多个agent的时候，summary多个agent的回复再输出

router_prompt0 = "Some choices are given below. It is provided in a numbered list (1 to \
     {num_choices}), where each item in the list corresponds to a \
     summary.\n---------------------\n{context_list}\n---------------------\nUsing \
     only the choices above and not prior knowledge, return the top choices \
     (no more than {max_outputs}, but only select what is needed) that are \
     most relevant to the question: '{query_str}'\n"

profile_choices = [
    '与学生相关',
    '与教职工相关',
    '与家属相关',
    '与访客游客相关'
]
agent_choices = [memo_agent, book_agent, navigation_agent]
output_parser = RouterOutputParser()

router_agent = RouterAgent(router_prompt0, agent_choices=agent_choices, profile_choices=profile_choices, output_parser=output_parser)

router_agent._agent_route_query("带我去校长办公室")

router_agent._profile_route_query("带我去校长办公室")

print(router_agent.run("帮我取消预约下午的会议室"))

print(router_agent.run("带我去教师活动中心"))