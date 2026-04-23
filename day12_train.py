import ollama

def structured_ai_chat():
    print("=" * 60)
    print("📋 结构化输出AI助手 | q退出 | clear清空记忆")
    print("=" * 60)

    messages=[
        {
            "role": "system",
            "content": """你是严格遵守指令的AI助手，必须遵守以下规则：
    1. 回答只围绕用户问题，不添加任何多余闲聊、解释、开场白
    2. 要求列表输出时，只用1.2.3.编号，不换行、不啰嗦
    3. 要求JSON输出时，只返回纯JSON字符串，无其他任何文字
    4. 回答简洁精准，禁止长篇大论
    """
        }
    ]
    while True:
        user_input=input("你: ").strip()
        if user_input.lower()=="q":
            print("✅ 对话结束")
            break
        if user_input.lower()=="clear":
            messages = [messages[0]]
            print("🧹 已清空对话历史\n")
            continue
        messages.append({"role":"user","content":user_input})

        print("AI: ",end="",flush=True)
        ai_reply=""
        response=ollama.chat(
            model="qwen2:0.5b",
            messages=messages,
            stream=True
        )

        for chunk in response:
            content=chunk["message"]["content"]
            ai_reply+=content
            print(content,end="",flush=True)
        print("\n")
        messages.append({"role":"assistant","content":ai_reply})

if __name__=="__main__":
    structured_ai_chat()