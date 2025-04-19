import requests
import json
import os
from configparser import ConfigParser

# =================================配置信息=========================================
API_URL = 'http://127.0.0.1'  # RAGFlow API 地址
AUTHORIZATION = 'ragflow-gxZDExNjZjMWQ0ZDExZjBiN2RiNWFlZT'  # 替换为实际 API Key
KB_NAME = 'NYU CAS Policy'  # 目标知识库名称


# ==================================处理逻辑========================================
def get_dataset_id_by_name(name):
    """根据名称查找知识库ID"""
    headers = {"Authorization": f"Bearer {AUTHORIZATION}"}
    url = f"{API_URL}/api/v1/datasets"
    params = {'name': name}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"请求失败: {response.status_code} - {response.text}")
        return None

    data = response.json()
    if data.get('code') != 0:
        print(f"API错误: {data.get('message')}")
        return None

    datasets = data.get('data', [])
    if not datasets:
        print(f"未找到知识库: {name}")
        return None

    return datasets[0]['id']


def ask_question(question, chat_id):
    """
    向RAGflow的聊天助手发送问题并获取回答
    """
    url = f"{API_URL}/api/v1/chats/{chat_id}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTHORIZATION}"
    }
    data = {
        "question": question,
        "stream": True  # 使用流式响应
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()  # 检查HTTP错误

        # 处理流式响应
        for line in response.iter_lines():
            if line:
                line_json = json.loads(line)
                if line_json.get("code") == 0 and line_json.get("data"):
                    answer_data = line_json["data"]
                    if isinstance(answer_data, dict) and "answer" in answer_data:
                        print(answer_data["answer"], end="", flush=True)
                    elif answer_data is True:
                        # 流式响应结束标志
                        break
        print()  # 流式响应结束后换行

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")


def main():
    # 获取知识库ID
    dataset_id = get_dataset_id_by_name(KB_NAME)
    if not dataset_id:
        return

    # 创建聊天会话
    chat_id = dataset_id  # 假设使用知识库ID作为聊天助手ID

    # 用户输入问题
    user_question = input("请输入您的问题（输入'exit'退出）：")
    while user_question.lower() != 'exit':
        ask_question(user_question, chat_id)
        user_question = input("\n请输入您的问题（输入'exit'退出）：")


if __name__ == "__main__":
    main()