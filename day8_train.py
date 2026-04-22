import ollama

def simple_chat():
    print("="*50)
    print("【基础LLM对话】")
    print("="*50)

    response=ollama.chat(
        model="qwen2:0.5b",
        messages=[{"role":"user","content":"你好，请简单介绍一下什么是大语言模型(LLM)"}]
    )

    print("用户：你好，请简单介绍一下什么是大语言模型(LLM)")
    print(f"LLM:{response['message']['content']}\n")

def prompt_agent_demo():
    print("=" * 50)
    print("【Prompt工程 - 结构化输出】")
    print("=" * 50)

    prompt="""
    你是一个AI助手，请完成以下任务
    1.用3句话总结大模型agent的核心功能
    2.输出格式严格为JSON，包含key:agent_ability
    3.语言简洁，专业易懂
    """

    response=ollama.chat(
        model="qwen2:0.5b",
        messages=[{"role":"user","content":prompt}]
    )

    print(f"LLM结构化输出\n{response['message']['content']}\n")

def mini_agent_demo():
    print("=" * 50)
    print("【迷你Agent - 问答助手】")
    print("=" * 50)

    while(True):
        user_input=input("请输入你的问题(输入q退出):")
        if user_input=="q":
            print("Agent退出！")
            break

        response=ollama.chat(
            model="qwen2:0.5b",
            messages=[{"role":"user","content":user_input}]
        )
        print(f"Agent:{response['message']['content']}\n")

if __name__=="__main__":
    simple_chat()
    prompt_agent_demo()
    mini_agent_demo()