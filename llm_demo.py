# from dashscope import Generation
# from dashscope.api_entities.dashscope_response import Role
#
# messages = []
#
# while True:
#     message = input('user:')
#     # 将输入信息加入历史对话
#     messages.append({'role': Role.USER, 'content': message})
#     # 获得模型输出结果
#     response = Generation.call(Generation.Models.qwen_v1, messages=messages, result_format='message', api_key="sk-8f6b97abc94b4fb2bc523b3edb532835")
#     print('system:'+response.output.choices[0]['message']['content'])
#     # 将输出信息加入历史对话
#     messages.append({'role': response.output.choices[0]['message']['role'], 'content': response.output.choices[0]['message']['content']})


# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import dashscope

def sample_sync_call(prompt_text):
    # prompt_text = '世界山最的遗憾是什么'
    resp = dashscope.Generation.call(
        model='qwen-plus',
        prompt=prompt_text,
        api_key='sk-8f6b97abc94b4fb2bc523b3edb532835'
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if resp.status_code == HTTPStatus.OK:
        print('system:' + resp.output.text)
        # print(resp.output)  # The output text
        # print(resp.usage)  # The usage information
    else:
        print(resp.code)  # The error code.
        print(resp.message)  # The error message.

while True:
    message = input('user:')
    sample_sync_call(message)
